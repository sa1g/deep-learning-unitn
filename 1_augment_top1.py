from typing import Optional
import torch


torch.manual_seed(456)
torch.cuda.manual_seed(456)

import torch.nn as nn
import torch.nn.functional as F
import open_clip


from src.augmix import (
    AugMixKornia,
    ImageTransform,
    kornia_preprocess,
    kornia_random_crop,
)
from src.data import ResnetA
from src.utils import bench


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

        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        self.model: open_clip.model.CLIP = model
        self.logit_scale = model.logit_scale.data.exp()
        # self.logit_scale = model.log

        with torch.no_grad():
            prompts = torch.cat(
                [self.tokenizer(prompt.format(c)) for c in class_labels.values()]
            ).to(device)
            self.text_features = model.encode_text(prompts, normalize=True)

    def select_confident_samples(
        self, logits: torch.Tensor, top: float = 0.1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Selects the top-k samples with the lowest entropy from the logits.

        Args:
            logits (torch.Tensor): The logits from the model.
            top (float): The fraction of samples to select.
                For example, if top=0.1, it selects the top 10% of samples.
        Returns:
            torch.Tensor: The selected logits.
            torch.Tensor: The indices of the selected samples.

        [Reference](https://github.com/azshue/TPT/blob/63ecbace79694205d7884e63fdc3137a200f0b0e/tpt_classification.py#L41C5-L41C11)
        """
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        idx = torch.argsort(batch_entropy, descending=False)[
            : int(batch_entropy.size()[0] * top)
        ]

        return logits[idx], idx

    def forward(self, x: torch.Tensor) -> int:
        with torch.no_grad(), torch.autocast("cuda"):
            image_features = self.model.encode_image(x, normalize=True)

            initial_image_features = image_features[-1:]
            filtered_image_features = image_features[:-1:]

            # filter logits
            initial_logits = (
                self.logit_scale * initial_image_features @ self.text_features.t()
            )
            filtered_logits = (
                self.logit_scale * filtered_image_features @ self.text_features.t()
            )

            # Get top k logits
            selected_logits, _ = self.select_confident_samples(
                # filtered_logits, top=1 / filtered_logits.shape[0]
                filtered_logits,
                top=0.1,
            )

            # selected_logits = selected_logits.mean(0, keepdim=True)

            # final_logits = selected_logits
            final_logits = torch.cat((selected_logits, initial_logits), dim=0)

            marginal_prob = F.softmax(final_logits, dim=1).mean(0)
            pred_class = int(marginal_prob.argmax().item())

        return pred_class


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    augmenter = ImageTransform(
        model_transform=kornia_preprocess,
        custom_transform=kornia_random_crop,
        n_views=63,
        device="cpu"
    )

    dataloader, dataset = ResnetA(augmenter, num_workers=6)

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
