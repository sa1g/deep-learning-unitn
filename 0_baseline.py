import torch

torch.manual_seed(456)
torch.cuda.manual_seed(456)

import torch.nn as nn
import torch.nn.functional as F
import open_clip

from src.utils import bench


from src.augmix import AugMixKornia, ImageTransform, kornia_preprocess
from src.data import ResnetA


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
            marginal_prob = F.softmax(logits, dim=1).mean(0)
            pred_class = marginal_prob.argmax().item()
        return int(pred_class)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    augmenter = ImageTransform(
        model_transform=kornia_preprocess,
        n_views=0,
    )

    dataloader, dataset = ResnetA(augmenter)

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
