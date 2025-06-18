import clip.model
import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenAIClip(nn.Module):
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
            prompts = torch.cat(
                [clip.tokenize(prompt.format(c)) for c in class_labels.values()]
            ).to(device)
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
