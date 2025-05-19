# Accuracy: 49.37%
# Latency: 141.43 ms

import clip.model
import torch

torch.manual_seed(456)
torch.cuda.manual_seed(456)

import torch.nn as nn
import torch.nn.functional as F
import clip


from src.augmix import AugMixKornia, ImageTransform, kornia_preprocess
from src.data import ResnetA
from src.utils import bench




class ClipWrapper(nn.Module):
    def __init__(
        self, model: clip.model.CLIP, prompt: str = "a photo of a {}", device: str = None
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.logit_scale = model.logit_scale.data

        self.prompt = prompt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Returns the predicted class.
        """
        prompt = torch.cat(
            [
                clip.tokenize(self.prompt.format(c))
                for c in dataset.class_code_to_label.values()
            ]
        ).to(device)

        with torch.no_grad():
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
    clip_model, _ = clip.load("ViT-B/16", device=device)
    clip_model.eval()

    # Create a ClipSkeleton instance
    wrapper_clip = ClipWrapper(clip_model).to(device)

    accuracy, latency = bench(wrapper_clip, dataloader, device, reduce=30)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Latency: {latency * 1000:.2f} ms")
