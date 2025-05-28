import torch

torch.manual_seed(456)
torch.cuda.manual_seed(456)

import torch.nn as nn
import torch.nn.functional as F
import open_clip

from src.utils import bench


from src.augmix import AugMixKornia, ImageTransform, kornia_preprocess, kornia_random_crop
from src.data import ImagenetA


class ClipWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        class_labels: dict,
        prompt: str = "a photo of a {}",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.class_labels = class_labels
        self.prompt = prompt

        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        self.model = model
        self.logit_scale = model.logit_scale.exp()


        # Precompute text features
        with torch.no_grad():
            prompts = torch.cat(
                [self.tokenizer(prompt.format(c)) for c in class_labels.values()]
            ).to(device)
            self.text_features = model.encode_text(prompts, normalize=True)

    def forward(self, x: torch.Tensor) -> int:
        with torch.no_grad(), torch.autocast("cuda"):
            image_features = self.model.encode_image(x, normalize=True)
            logits = self.logit_scale * image_features @ self.text_features.t()
            
            best_label_confidence = logits.topk(5, dim=1)
            best_labels_idx = logits.topk(5, dim=1).indices
            print(f"Initial top-5 labels indices: {best_labels_idx[0]}")
            print(f"Initial top-5 confidence: {best_label_confidence[0][0]}")
            
            # Get unique indices from top-k predictions
            unique_best_labels_idx = best_labels_idx.unique()
            
            # Create mapping between reduced indices and original indices
            idx_mapping = {new_idx: original_idx for new_idx, original_idx in enumerate(unique_best_labels_idx.tolist())}
            reverse_idx_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(unique_best_labels_idx.tolist())}
            
            # Reduce the class labels to the unique indices
            class_labels = list(self.class_labels.values())
            reduced_class_labels = {i: class_labels[i] for i in unique_best_labels_idx.tolist()}
            
            # Get prompts and features for reduced set
            prompts = torch.cat(
                [self.tokenizer(self.prompt.format(c)) for c in reduced_class_labels.values()]
            ).to(device)
            text_features = self.model.encode_text(prompts, normalize=True)
            
            # Compute logits for reduced set
            reduced_logits = self.logit_scale * image_features @ text_features.t()
            
            # Create full logits tensor with original dimensions
            full_logits = torch.full((logits.shape[0], len(self.class_labels)), -float('inf'), device=device)
            
            # Fill in the values from reduced_logits to their original positions
            for new_idx, original_idx in idx_mapping.items():
                full_logits[:, original_idx] = reduced_logits[:, new_idx]
            
            # Now you can get the top predictions from the full logits
            final_topk = full_logits.topk(5, dim=1)
            final_indices = final_topk.indices
            
            print(f"Final top-5 labels indices: {final_indices[0]}")
            print(f"Final top-5 confidence: {final_topk[0][0]}")
            exit()            
            
            marginal_prob = F.softmax(logits, dim=1).mean(0)
            pred_class = marginal_prob.argmax().item()
        return int(pred_class)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    augmenter = ImageTransform(
        model_transform=kornia_preprocess,
        custom_transform=kornia_random_crop,
        n_views=63,
    )

    dataloader, dataset = ImagenetA(augmenter)

    # Load the CLIP model
    clip_model, _, _ = open_clip.create_model_and_transforms(
        # model_name="ViT-B-32", pretrained="datacomp_xl_s13b_b90k", device=device#, force_quick_gelu=True
        model_name="ViT-B-16",
        pretrained="openai",
        device=device,
        force_quick_gelu=True,
    )
    clip_model.eval()

    # Create a ClipSkeleton instance
    wrapper_clip = ClipWrapper(
        clip_model, class_labels=dataset.class_code_to_label, device=device
    ).to(device)

    bench(wrapper_clip, dataloader, device, reduce=None, comment="", visualize=False)
