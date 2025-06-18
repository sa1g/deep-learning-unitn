# setup argparse
import argparse

import torch

import open_clip
import clip

from src.utils import bench
from src.augmix import ImageTransform, kornia_preprocess, kornia_random_crop
from src.data import ImagenetA

from src.models.open_clip import OpenClip
from src.models.clip import OpenAIClip
from src.models.top10 import ClipTop10
from src.models.tpt import TPT, TPTTop10
from src.models.tnt import TNT, TNTTop10


def parse_args():
    parser = argparse.ArgumentParser(description="Different TTA methods.")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name the method: 'openclip', 'clip', 'top10', 'tpt', 'tpt_top10', 'tnt', 'tnt_top10'",
    )
    parser.add_argument(
        "--reduce-dataset",
        type=int,
        default=None,
        help="Reduce the dataset to N samples. Default is None (no reduction).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    reduce = args.reduce_dataset

    # This code is ugly, cmon no one will ever read this.

    if args.name == "openclip" or args.name == "clip":
        augmenter = ImageTransform(
            model_transform=kornia_preprocess,
            n_views=0,
        )
    else:
        augmenter = ImageTransform(
            model_transform=kornia_preprocess,
            custom_transform=kornia_random_crop,
            n_views=63,
            device=device,
        )

    dataloader, dataset = ImagenetA(augmenter)

    if args.name == "openclip":
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name="ViT-B-16",
            pretrained="openai",
            device=device,
            force_quick_gelu=True,
        )
        clip_model.eval()  # type: ignore

        # Create a ClipSkeleton instance
        wrapper_clip = OpenClip(
            clip_model, class_labels=dataset.class_code_to_label, device=device  # type: ignore
        ).to(device)

    elif args.name == "clip":
        clip_model, _ = clip.load("ViT-B/16", device=device, jit=True)
        clip_model.eval()

        # Create a ClipSkeleton instance
        wrapper_clip = OpenAIClip(
            clip_model, class_labels=dataset.class_code_to_label, device=device
        ).to(device)

    elif args.name == "top10":
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name="ViT-B-16",
            pretrained="openai",
            device=device,
            force_quick_gelu=True,
        )
        clip_model.eval()  # type: ignore

        # Create a ClipSkeleton instance
        wrapper_clip = ClipTop10(
            clip_model, class_labels=dataset.class_code_to_label, device=device  # type: ignore
        ).to(device)

    elif args.name == "tpt":
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name="ViT-B-16",
            pretrained="openai",
            device=device,
            force_quick_gelu=True,
        )
        clip_model.eval()  # type: ignore

        wrapper_clip = TPT(
            arch="ViT-B-16",  # type: ignore
            pretrained="openai",
            class_names=dataset.class_code_to_label.values(),  # type: ignore
            tta_steps=1,
            lr=5e-3,
        )

    elif args.name == "tpt_top10":
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name="ViT-B-16",
            pretrained="openai",
            device=device,
            force_quick_gelu=True,
        )
        clip_model.eval()  # type: ignore

        wrapper_clip = TPTTop10(
            arch="ViT-B-16",  # type: ignore
            pretrained="openai",
            class_names=dataset.class_code_to_label.values(),  # type: ignore
            tta_steps=1,
            lr=5e-3,
        )

    elif args.name == "tnt":
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name="ViT-B-16",
            pretrained="openai",
            device=device,
            force_quick_gelu=True,
        )
        clip_model.eval()  # type: ignore

        for param in clip_model.parameters():  # type: ignore
            param.requires_grad = False

        wrapper_clip = TNT(
            clip_model,  # type: ignore
            class_labels=dataset.class_code_to_label,
            device=device,
            tnt_steps=1,
        ).to(device)

    elif args.name == "tnt_top10":
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name="ViT-B-16",
            pretrained="openai",
            device=device,
            force_quick_gelu=True,
        )
        clip_model.eval()  # type: ignore

        for param in clip_model.parameters():  # type: ignore
            param.requires_grad = False

        wrapper_clip = TNTTop10(
            clip_model,  # type: ignore
            class_labels=dataset.class_code_to_label,
            device=device,
            tnt_steps=1,
        ).to(device)
    else:
        raise ValueError(
            f"Unknown method name: {args.name}. Choose from 'openclip', 'clip', 'top10', 'tpt', 'tpt_top10', 'tnt', 'tnt_top10'."
        )

    bench(
        wrapper_clip,
        dataloader,
        device,
        reduce=reduce,
        comment=args.name,
        visualize=False,
    )
