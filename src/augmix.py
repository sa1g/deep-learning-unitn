import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import numpy as np
import kornia
import kornia.augmentation as K
import kornia.enhance as Ke


class AugMixKornia(nn.Module):
    def __init__(
        self,
        severity: int = 3,
        width: int = 3,
        depth: int = -1,
        alpha: float = 1.0,
        mixture_width: int = 3,
        chain_depth: int = 3,
        all_ops: bool = True,
        device: Optional[str] = None,
    ):
        """
        AugMix implementation using Kornia with closer fidelity to the original paper.

        Args:
            severity: Severity level of augmentations (1-10)
            width: Width of augmentation chain (not used directly, kept for compatibility)
            depth: Depth of augmentation chain (-1 for random between 1-3)
            alpha: Dirichlet distribution parameter for mixing weights
            mixture_width: Number of augmentation chains to mix
            chain_depth: Number of operations in each chain
            all_ops: Whether to use all augmentation operations
            device: Device to run on (cuda/cpu)
        """
        super().__init__()

        self.severity = severity
        self.alpha = alpha
        self.mixture_width = mixture_width
        self.chain_depth = chain_depth if depth <= 0 else depth
        self.all_ops = all_ops
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Define augmentation operations
        self.augmentations = self._get_augmentations()

    def _get_augmentations(self) -> List[nn.Module]:
        """Create a list of augmentation operations that will be randomly applied"""
        severity_factor = self.severity / 10.0

        if self.all_ops:
            # Full set of augmentations similar to original AugMix
            return [
                # AutoContrast
                K.ColorJitter(
                    brightness=0.1 * self.severity, contrast=0.1 * self.severity, p=1.0
                ),
                # Equalize
                Ke.equalize,
                # Posterize
                K.RandomPosterize(bits=max(1, 8 - self.severity), p=1.0),
                # Rotate
                K.RandomRotation(
                    degrees=(-30 * severity_factor, 30 * severity_factor), p=1.0
                ),
                # Solarize
                K.RandomSolarize(
                    thresholds=0.5, additions=(0.0, 0.1 * self.severity), p=1.0
                ),
                # Shear
                K.RandomAffine(
                    degrees=0,
                    shear=(-15 * severity_factor, 15 * severity_factor),
                    p=1.0,
                ),
                # Translate
                K.RandomAffine(
                    degrees=0,
                    translate=(0.1 * severity_factor, 0.1 * severity_factor),
                    p=1.0,
                ),
                # ColorJitter
                K.ColorJitter(
                    brightness=0.1 * self.severity,
                    contrast=0.1 * self.severity,
                    saturation=0.1 * self.severity,
                    hue=0.1,
                    p=1.0,
                ),
            ]
        else:
            # Simplified version
            return [
                K.ColorJitter(
                    brightness=0.1 * self.severity, contrast=0.1 * self.severity, p=1.0
                ),
                Ke.equalize,
                K.RandomAffine(
                    degrees=(-15 * severity_factor, 15 * severity_factor), p=1.0
                ),
            ]

    def _apply_augmentation_chain(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply a random sequence of augmentations to an image.

        Args:
            image: Input image tensor (C, H, W)

        Returns:
            Augmented image tensor (C, H, W)
        """
        # Randomly select augmentations for this chain
        op_indices = np.random.choice(
            len(self.augmentations), size=self.chain_depth, replace=True
        )

        augmented = image  # Don't clone immediately
        for op_idx in op_indices:
            augmented = self.augmentations[op_idx](augmented)

        return augmented.squeeze(0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply AugMix to a batch of images.

        Args:
            images: Input batch of images (B, C, H, W) or (C, H, W)

        Returns:
            Augmented batch (same shape as input)
        """
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Input validation
            if not isinstance(images, torch.Tensor):
                images = K.image_to_tensor(images)

            if images.dim() == 3:
                images = images.unsqueeze(0)

            # Move to device if needed
            if images.device != self.device:
                images = images.to(self.device)

            batch_size = images.shape[0]

            # Sample mixing weights from Dirichlet distribution
            weights = (
                torch.from_numpy(
                    np.random.dirichlet([self.alpha] * self.mixture_width, size=batch_size)
                )
                .float()
                .to(self.device)
            )  # Shape (B, mixture_width)

            # Sample weights for mixing with original
            mix_weights = (
                torch.from_numpy(
                    np.random.dirichlet([self.alpha, self.alpha], size=batch_size)
                )
                .float()
                .to(self.device)
            )  # Shape (B, 2)

            # Generate augmented versions for each mixture component
            # Pre-allocate memory for augmented versions
            augmented = torch.empty(
                (self.mixture_width, batch_size, *images.shape[1:]), device=self.device
            )

            for i in range(self.mixture_width):
                augmented[i] = self._apply_augmentation_chain(images)

            # Weighted sum of augmented versions
            mixed = torch.einsum("mbchw,bm->bchw", augmented, weights).to(self.device)

            # Final mix with original image
            result = (
                mix_weights[:, 0:1, None, None] * images
                + mix_weights[:, 1:2, None, None] * mixed
            )

            result = result.squeeze(0) if result.shape[0] == 1 else result

        return result


def kornia_random_crop(images: torch.Tensor) -> torch.Tensor:
    """
    Applies random crop to a batch of images using Kornia's RandomResizedCrop.
    Preserves the original image size while randomly cropping a portion.
    """
    b, c, h, w = images.shape

    # Create random crop transform that:
    # 1. Crops between 50% and 100% of original area
    # 2. Maintains original aspect ratio
    # 3. Resizes back to original dimensions
    transform = K.RandomResizedCrop(
        size=(h, w),
        # scale=(0.5, 1.0),  # Crop between 50% and 100% of original area
        # ratio=(1.0, 1.0),  # Maintain original aspect ratio
        resample=kornia.constants.Resample.BICUBIC,
        same_on_batch=False,  # Different crop for each image in batch
    )

    return transform(images)


kornia_preprocess = nn.Sequential(
    K.SmallestMaxSize(
        224,
        resample=kornia.constants.Resample.BICUBIC,
    ),
    K.CenterCrop(
        size=(224, 224),
        resample=kornia.constants.Resample.BICUBIC,
    ),
    kornia.enhance.Normalize(
        mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]),
        std=torch.tensor([0.26862954, 0.26130258, 0.27577711]),
    ),
)


class ImageTransform(nn.Module):
    def __init__(self, model_transform, custom_transform=None, n_views=63, device="cuda"):
        super().__init__()
        self.model_transform = model_transform
        self.custom_transform = custom_transform
        self.n_views = n_views
        self.device = device

        self.eval()
        # self.model_transform.eval()
        # self.custom_transform.eval() if custom_transform is not None else None

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply the model transform and custom transform to the image.
        """
        with torch.no_grad(), torch.cuda.amp.autocast():
            torch.cuda.empty_cache()
            
            image = image.to(self.device)

            if self.custom_transform is not None:
                with torch.no_grad():
                    views = torch.empty((self.n_views+1, *image.shape), device=self.device)
                    views[:-1] = self.custom_transform(image.repeat(self.n_views, 1, 1, 1))
                    views[-1] = image
                    return self.model_transform(views)
            else:
                with torch.no_grad():
                    return self.model_transform(image)
