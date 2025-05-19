import torch

torch.manual_seed(456)
torch.cuda.manual_seed(456)

import torch.nn as nn
import torch.nn.functional as F
import open_clip


from src.augmix import AugMixKornia, ImageTransform, kornia_preprocess
from src.data import ResnetA
from src.utils import bench


class ClipWrapper(nn.Module):
    def __init__(
        self, model: nn.Module, prompt: str = "a photo of a {}", device: str = "cuda"
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        self.model = model
        self.logit_scale = model.logit_scale.data
        # self.logit_scale = model.log

        self.prompt = prompt

    # @torch.compile()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prompt = torch.cat(
            [
                self.tokenizer(self.prompt.format(c))
                for c in dataset.class_code_to_label.values()
            ]
        ).to(device)

        with torch.no_grad(), torch.autocast("cuda"):
            image_features = self.model.encode_image(x)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            text_features = self.model.encode_text(prompt)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        marginal_prob = F.softmax(logits, dim=1).mean(0)
        pred_class = marginal_prob.argmax().item()

        return pred_class


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kornia_augmix = AugMixKornia()

    augmenter = ImageTransform(
        model_transform=kornia_preprocess,
        # custom_transform=kornia_augmix,
        n_views=0,
    )

    dataloader, dataset = ResnetA(augmenter)

    # Load the CLIP model
    clip_model, _, preprocess_val = open_clip.create_model_and_transforms(
        # model_name="ViT-B-16", pretrained="datacomp_xl_s13b_b90k", device=device#, force_quick_gelu=True
        model_name="ViT-B-16", pretrained="openai", device=device, force_quick_gelu=True
    )
    clip_model.eval()

    # Create a ClipSkeleton instance
    wrapper_clip = ClipWrapper(clip_model).to(device)

    accuracy, latency = bench(wrapper_clip, dataloader, device, reduce=30)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Latency: {latency * 1000:.2f} ms")
