from typing import Optional
import torch


torch.manual_seed(456)
torch.cuda.manual_seed(456)
torch.randn(456).to("cuda")  # warm-up

import torch.nn as nn
import torch.nn.functional as F
import open_clip


from src.augmix import (
    AugMixKornia,
    ImageTransform,
    kornia_preprocess,
    kornia_random_crop,
)
from src.data import ImagenetA
from src.utils import bench


class TNT(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        class_labels: dict,
        prompt: str = "a photo of a {}",
        device: str = "cuda",
        tnt_steps: int = 3,
        top_k: float = 0.1,
        epsilon: float = 1/255,
        lr: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 1.0,
        temperature: float = 7e-3, 
    ):
        super().__init__()
        self.device = device
        self.model: open_clip.model.CLIP = model
        self.logit_scale = model.logit_scale.data.exp()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        self.tnt_steps = tnt_steps
        self.top_k = top_k
        self.eps = epsilon
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

        with torch.no_grad():
            prompts = torch.cat(
                [self.tokenizer(prompt.format(c)) for c in class_labels.values()]
            ).to(device)
            self.text_features = model.encode_text(prompts, normalize=True)

        self.noise = None

    def reset(self):
        self.noise = None

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
        x = x.to(self.device)

        if self.noise is None:
            self.noise = torch.randn_like(x[0], requires_grad=True, device=self.device)#, dtype=torch.float16)
            self.noise.data = self.noise.clamp(-self.eps, self.eps)

        self.noise.requires_grad = True

        with torch.autocast(self.device):#, dtype=torch.float16):
            for _ in range(self.tnt_steps):
                # x_aug = x + torch.clamp(self.noise, 0, 1)[None, ...]
                x_aug = x + self.noise[None, ...].clamp(0, 1)

                image_features = self.model.encode_image(x_aug, normalize=True)
                logits = self.logit_scale * image_features @ self.text_features.t()

                # Select top-k logits
                top_logits, top_idx = self.select_confident_samples(logits, top=self.top_k)
                top_features = image_features[top_idx]

                # Entropy loss
                prob = F.softmax(top_logits, dim=1).mean(dim=0)
                entropy_loss = -(prob * prob.log()).sum()

                # Inter-view consistency loss
                pairwise_dist = torch.cdist(top_features, top_features, p=2)
                inter_view_loss = pairwise_dist.sum()

                # Total loss
                loss = self.alpha * entropy_loss + self.beta * inter_view_loss
                loss.backward()

                # Update noise
                with torch.no_grad():
                    grad = self.noise.grad
                    self.noise -= self.lr * grad.sign()
                    self.noise.clamp_(-self.eps, self.eps)
                    self.noise.requires_grad = True
                    self.noise.grad = None

        with torch.no_grad(), torch.autocast(self.device):
            x_aug = x + self.noise[None, ...].clamp(0, 1)
            image_features = self.model.encode_image(x_aug, normalize=True)
            logits = self.logit_scale * image_features @ self.text_features.t()

            selected_logits, _ = self.select_confident_samples(logits[:-1], top=self.top_k)
            final_logits = torch.cat((selected_logits, logits[-1:]), dim=0)
            probs = F.softmax(final_logits/self.temperature, dim=1).mean(dim=0)
            pred_class = int(probs.argmax().item())

        return pred_class



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    augmenter = ImageTransform(
        model_transform=kornia_preprocess,
        custom_transform=kornia_random_crop,
        n_views=63,
        device="cpu"
    )

    dataloader, dataset = ImagenetA(augmenter, num_workers=6)

    # Load the CLIP model
    clip_model, _, _ = open_clip.create_model_and_transforms(
        # model_name="ViT-B-32", pretrained="datacomp_xl_s13b_b90k", device=device#, force_quick_gelu=True
        model_name="ViT-B-16",
        pretrained="openai",
        device=device,
        force_quick_gelu=True,
    )
    clip_model.eval()

    # Set the model to evaluation mode
    for param in clip_model.parameters():
        param.requires_grad = False

    # Create a ClipSkeleton instance
    wrapper_clip = TNT(
        clip_model, class_labels=dataset.class_code_to_label, device=device, tnt_steps = 1,
    ).to(device)


    # print(f"Model parameters: {sum(p.numel() for p in wrapper_clip.parameters())}")
    # print(f"trainable parameters: {sum(p.numel() for p in wrapper_clip.parameters() if p.requires_grad)}")
    # exit()

    bench(wrapper_clip, dataloader, device, reduce=None, comment="tnt-top1", visualize=False)
