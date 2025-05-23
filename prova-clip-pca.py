import time
from typing import Tuple
import numpy as np
import torch
from tqdm import tqdm

torch.manual_seed(456)
torch.cuda.manual_seed(456)

import torch.nn as nn
import torch.nn.functional as F
import open_clip

from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

from src.augmix import AugMixKornia, ImageTransform, kornia_preprocess
from src.data import ResnetA

# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def bench(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    comment: str,
    reduce: int | None = None,
):
    """Benchmark the model on the dataset.

    The model must return logits.
    """

    # board = SummaryWriter(comment=comment)

    total = 0
    correct = 0
    masked_correct = 0

    start = time.time()

    total_tqdm = reduce if reduce is not None else len(dataloader)
    # ░▒█
    # ascii=" ▖▘▝▗▚▞█"
    # ascii=' >='
    for image, label in tqdm(dataloader, total=total_tqdm, ascii=" ▖▘▝▗▚▞█"):
        this_start = time.time()

        image = image.to(device)

        # start1 = time.time()
        pred_class, masked_class = model(image)
        del image
        # print(f"model: {(time.time() - start1) * 1000:.2f} ms")

        total += 1
        correct += int((pred_class == label))
        masked_correct += int((masked_class == label))

        if reduce:
            if total > reduce:
                break

        c_a = correct / total
        m_a = masked_correct / total
        print(f"Correct: {c_a*100:.2f}%, Masked Correct: {m_a*100:.2f}%")

        # break
        # board.add_scalar("accuracy", correct / total, total)
        # board.add_scalar("masked_accuracy", masked_correct / total, total)
        # board.add_scalar("dbg/label/predict_class", pred_class, total)
        # board.add_scalar("dbg/label/label", label, total)

    end = time.time()

    accuracy = correct / total
    latency = (end - start) / total  # ms

    # board.add_scalar("metrics/latency (ms)", latency)
    # board.add_scalar("metrics/accuracy", accuracy)
    # board.add_scalar("metrics/masked_accuracy", masked_correct / total)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Masked Accuracy: {masked_correct / total * 100:.2f}%")
    print(f"Latency: {latency * 1000:.2f} ms")

    return accuracy, latency


class ClipWrapper(nn.Module):
    def __init__(
        self, model: nn.Module, prompt: str = "a photo of a {}", device: str = "cuda"
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        self.model: open_clip.model.CLIP = model
        self.logit_scale = model.logit_scale.data
        # self.logit_scale = model.log

        self.prompt = prompt
        self.pca = PCA(n_components=1)
        self.pca3 = PCA(n_components=3)

        self.model.visual.output_tokens = True

    def encode_image(self, x, normalize: bool = False) -> torch.Tensor:
        pooled, tokens = self.model.visual(x)

        if self.model.visual.output_tokens:
            return F.normalize(pooled, dim=-1) if normalize else pooled, tokens

        return F.normalize(pooled, dim=-1) if normalize else pooled

    def compute_pca_masks(
        self, x: torch.Tensor, threshold: float = 0.3
    ) -> torch.Tensor:
        """
        x: torch.Tensor of shape [B, 3, 224, 224]
        Returns:
            mask: torch.Tensor of shape [B, 3, 224, 224]
        """
        B = x.shape[0]
        masks = []

        # Get patch tokens for each image in batch
        _, all_tokens = self.encode_image(x, normalize=True)  # [B, 196, 768]

        all_tokens = all_tokens.cpu().numpy()  # convert to numpy for PCA

        for i in range(B):
            patch_embeddings = all_tokens[i]  # [196, 768]

            # Step 1: Foreground PCA mask (1 component)
            fg_scores = minmax_scale(self.pca.fit_transform(patch_embeddings)).reshape(
                14, 14
            )
            mask = fg_scores > threshold

            mask = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0)
            mask = (
                F.interpolate(mask, size=(x.shape[2], x.shape[3]), mode="bicubic")
                .expand(1, 3, -1, -1)
                .to(x.device)
            )
            
            masks.append(mask)

        # Step 5: Stack into batch
        mask_batch = torch.cat(masks, dim=0).to(x.device)  # [B, 3, 224, 224]
        return mask_batch

    def forward(self, x: torch.Tensor) -> Tuple[int, int]:
        prompt = torch.cat(
            [
                self.tokenizer(self.prompt.format(c))
                for c in dataset.class_code_to_label.values()
            ]
        ).to(device)

        with torch.no_grad(), torch.autocast("cuda"):
            mask = self.compute_pca_masks(x)  # [B, 3, 224, 224]


            # # # # _, tokens = self.encode_image(x, normalize=True)
            # # # # patch_embeddings = tokens.squeeze(0).cpu().numpy()

            # # # # # Foreground PCA mask
            # # # # fg_scores = minmax_scale(self.pca.fit_transform(patch_embeddings)).reshape(
            # # # #     14, 14
            # # # # )
            # # # # mask = fg_scores > 0.3

            # # # # # PCA of foreground only
            # # # # # masked_patches = patch_embeddings[mask.ravel()]
            # # # # # fg_rgb = minmax_scale(self.pca3.fit_transform(masked_patches))

            # # # # # # Fill only masked areas in full grid
            # # # # # fg_scores = np.zeros((14 * 14, 3), dtype=np.float32)
            # # # # # fg_scores[mask.ravel()] = fg_rgb
            # # # # # fg_scores = fg_scores.reshape(14, 14, 3)

            # # # # # resize both mask and fg_result to x size
            # # # # mask = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0)
            # # # # mask = (
            # # # #     F.interpolate(mask, size=(x.shape[2], x.shape[3]), mode="bicubic")
            # # # #     .expand(1, 3, -1, -1)
            # # # #     .to(x.device)
            # # # # )

            # fg_result = np.moveaxis(fg_scores, -1, 0)
            # fg_result = torch.tensor(fg_result).float().unsqueeze(0).to(x.device)
            # fg_result = F.interpolate(
            #     fg_result, size=(x.shape[2], x.shape[3]), mode="bicubic"
            # ).to(x.device)

            # # Original Image
            # plt.subplot(1, 6, 1)
            # plt.imshow(x.squeeze(0).permute(1, 2, 0).cpu().numpy())
            # plt.title("Original Image")
            # plt.axis("off")

            # # Foreground PCA Mask
            # plt.subplot(1, 6, 2)
            # plt.imshow(fg_scores)
            # plt.title("Foreground PCA Mask")
            # plt.axis("off")
            # # Foreground PCA Result
            # plt.subplot(1, 6, 3)
            # plt.imshow(fg_result.squeeze(0).permute(1, 2, 0).cpu().numpy())
            # plt.title("Foreground PCA Result")
            # plt.axis("off")
            # # Foreground PCA with Image
            # plt.subplot(1, 6, 4)
            # plt.imshow(
            #     (x * mask).squeeze(0).permute(1, 2, 0).cpu().numpy()
            # )
            # plt.title("Masked Image")
            # plt.axis("off")
            # # Foreground PCA 3 with Image
            # plt.subplot(1, 6, 5)
            # plt.imshow(
            #     (x * fg_result).squeeze(0).permute(1, 2, 0).cpu().numpy()
            # )
            # plt.title("Masked Image Result")
            # plt.axis("off")

            # plt.show()

            # x1 = x * masks.squeeze(0)
            x1 = x * mask

            original_class = self.get_class(x, prompt, self.encode_image)
            masked_class = self.get_class(x1, prompt, self.encode_image)

            # exit()

        return original_class, masked_class

    def get_class(self, x, prompt, image_encoder) -> int:
        image_features, _ = image_encoder(x, normalize=True)

        text_features = self.model.encode_text(prompt, normalize=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        marginal_prob = F.softmax(logits, dim=1).mean(0)
        pred_class: int = int(marginal_prob.argmax().item())

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
    clip_model, _, _ = open_clip.create_model_and_transforms(
        # model_name="ViT-B-32", pretrained="datacomp_xl_s13b_b90k", device=device#, force_quick_gelu=True
        model_name="ViT-B-16",
        pretrained="openai",
        device=device,
        force_quick_gelu=True,
    )
    clip_model.eval()

    # Create a ClipSkeleton instance
    wrapper_clip = ClipWrapper(clip_model).to(device)

    accuracy, latency = bench(
        wrapper_clip, dataloader, device, reduce=30, comment="zero shot clip AAAA"
    )
