# Accuracy: 49.37%
# Latency: 141.43 ms

import clip.model
import torch

torch.manual_seed(42)
torch.cuda.manual_seed(42)

import torch.nn as nn
import torch.nn.functional as F
import clip


from src.augmix import AugMixKornia, ImageTransform, kornia_preprocess
from src.data import ImagenetA
from src.utils import bench


class ClipWrapper(nn.Module):
    def __init__(
        self,
        model: clip.model.CLIP,
        class_labels: dict,
        prompt: str = "a photo of a {}",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.model = model
        self.logit_scale = model.logit_scale.exp()

        with torch.no_grad():
            prompts = torch.cat([
                clip.tokenize(prompt.format(c)) for c in class_labels.values()
            ]).to(device)
            self.text_features = model.encode_text(prompts)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor) -> int:
        """
        Forward pass through the model.

        Returns the predicted class.
        """

        with torch.no_grad():
            image_features = self.model.encode_image(x)
            image_features /= image_features.norm(dim=-1, keepdim=True)
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

    dataloader, dataset = ImagenetA(augmenter)

    # Load the CLIP model
    clip_model, _ = clip.load("ViT-B/16", device=device, jit=True)
    clip_model.eval()

    # Create a ClipSkeleton instance
    wrapper_clip = ClipWrapper(clip_model, class_labels=dataset.class_code_to_label, device=device).to(device)

    bench(wrapper_clip, dataloader, device, reduce=None, comment="baseline clip1", visualize=False)
