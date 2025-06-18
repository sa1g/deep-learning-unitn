import open_clip
import torch
import torch.nn as nn


def select_confident_samples(
    logits: torch.Tensor, top: float = 0.1
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


class ClipCommon(nn.Module):
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

        return select_confident_samples(logits, top)
