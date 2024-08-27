from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import logging


class ImageNetA(Dataset):
    """

    Variables:
        classes: dict mapping class indexes to class names
        data: list of tuples (class index, image name)
    """

    def __init__(self, path, transform=None, target_transform=None, n_px=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        self.n_px = n_px

        if transform is not None and n_px is None:
            raise ValueError("If transform is not None, n_px must be provided")


        # Read the labels file
        # The first 11 lines contain infos about the dataset that are not useful here.
        with open(os.path.join(path, "README.txt"), "r") as f:
            labels = f.readlines()[12:]

        # Create a dictionary to map the class names to their indexes
        self.classes = {}
        for line in labels:
            if line.strip():
                parts = line.strip().split()
                numeric_id = parts[0]
                name = " ".join(parts[1:]).lower()
                self.classes[int(numeric_id[1:])] = name

        # Read the images and their labels
        self.data = []
        for cl in self.classes.keys():
            for img in os.listdir(os.path.join(path, f"n{str(cl).zfill(8)}")):
                self.data.append((cl, img))

        logging.info(
            f"Loaded {len(self.data)} images and {len(self.classes)} classes from ImageNet-A"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        cl, file_name = self.data[index]
        path = os.path.join(self.path, f"n{str(cl).zfill(8)}", file_name)

        img = Image.open(path).convert("RGB")

        cl = torch.tensor(cl)

        if self.transform:
            transform = self.transform(self.n_px)
            img = transform(img)
        
        if self.target_transform:
            cl = self.target_transform(cl)
        
        return img, cl

    def idx_to_class(self, idx):
        return {str(self.classes[idx])}