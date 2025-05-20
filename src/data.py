import os

import torchvision 
import torch
from torch.utils.data import Dataset, DataLoader

from src.augmix import ImageTransform

class ImageNetADataset(Dataset):

    """
    Custom Dataset class for the ImageNet-A dataset.

    Set the `transform` parameter so that images work with your model.
    Example usage:
    ```python
        model, transform = clip.load("ViT-B/32")
        dataset = ImageNetADataset(<path>, transform=transform)
    ```
    ----

    The dataset is organized into subdirectories, each named with a class code (e.g., "n01614925").
    Each subdirectory contains images belonging to that class. The dataset also includes a README.txt file that maps class codes to human-readable names.

    The dataset is expected to be structured as follows:
    ```
    datasets/imagenet-a/
        n01440764/
            image1.jpg
            image2.jpg
            ...
        n01614925/
            image1.jpg
            image2.jpg
            ...
        ...
        README.txt
    ```

    """

    def __init__(self, root_dir="datasets/imagenet-a", transform=None):
        """
        Args:
            root_dir (str): Root directory of the ImageNet-A dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.__download_if_needed()

        # Load mapping from class codes (e.g., "n01614925") to human-readable names
        readme_path = os.path.join(root_dir, "README.txt")
        self.class_code_to_label = self._load_class_mapping(readme_path)

        # Filter valid class directories that match the mapping
        self.class_codes = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
                and d in self.class_code_to_label
            ]
        )

        # Map class codes to indices
        self.class_code_to_idx = {
            code: idx for idx, code in enumerate(self.class_codes)
        }

        # Collect all image file paths and corresponding labels
        self.samples = self._gather_samples()

        # Inverse mapping from label index to class name
        self.idx_to_label = {
            idx: self.class_code_to_label[code]
            for code, idx in self.class_code_to_idx.items()
        }

    def __download_if_needed(self):
        """
        Check if the dataset is already downloaded. If not, download it.
        """
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(
                f"Dataset not found at {self.root_dir}. Please download it first."
            )

    def _load_class_mapping(self, readme_path):
        """
        Load class code to human-readable name mapping from README.txt.
        Skips header lines and parses lines in format: 'n01440764 tench'.
        """
        mapping = {}
        with open(readme_path, "r") as file:
            lines = file.readlines()[12:]  # Skip first 12 header lines
            for line in lines:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    code, name = parts
                    mapping[code] = name
        return mapping

    def _gather_samples(self):
        """
        Walk through each class directory to gather image paths and corresponding labels.
        """
        samples = []
        for class_code in self.class_codes:
            class_dir = os.path.join(self.root_dir, class_code)
            for filename in os.listdir(class_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(class_dir, filename)
                    label = self.class_code_to_idx[class_code]
                    samples.append((image_path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load image and return dictionary containing image, label index, and class name.

        Returns:
            image (tensor)
            label (tensor)
        """
        image_path, label = self.samples[idx]

        image = torchvision.io.read_image(image_path).float() / 255.0

        if image.shape[0] == 1:  # Grayscale → RGB
            image = image.repeat(3, 1, 1)

        elif image.shape[0] == 4:  # RGBA → RGB
            image = image[:3, :, :]

        elif image.shape[0] != 3:
            raise ValueError(f"Unsupported number of channels: {image.shape[0]}")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)

    def get_class_name(self, idx):
        """
        Get human-readable class name for a given index.
        """
        return self.idx_to_label[idx]
    
def ResnetA(
    augmenter: ImageTransform,
    root_dir="datasets/imagenet-a",
):
    """
    Create a DataLoader for the ImageNet-A dataset. Defaults to 1 element per batch.
    Non modifiable. No shuffling.
    Args:
        augmenter (callable):
        root_dir (str): Root directory of the ImageNet-A dataset.

    Returns:
        dataloader (DataLoader): DataLoader for the ImageNet-A dataset.
        dataset (ImageNetADataset): The underlying dataset object.
    """

    def collate_fn(batch):
        """
        Custom collate function to handle the batch of images and labels.
        """

        images = batch[0][0]

        if images.ndim == 3:
            images = images.unsqueeze(0)

        labels = batch[0][1]

        return images, labels

    dataset = ImageNetADataset(root_dir=root_dir, transform=augmenter)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    return dataloader, dataset