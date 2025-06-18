import torch.nn as nn
import torch.nn.functional as F
import torch

from src.models.common import ClipCommon


class ClipTop10(ClipCommon):
    def __init__(
        self,
        model: nn.Module,
        class_labels: dict,
        prompt: str = "a photo of a {}",
        device: str = "cuda",
    ):
        super().__init__(model, class_labels, prompt, device)

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
