import torch

torch.manual_seed(456)
torch.cuda.manual_seed(456)

from src.augmix import AugMixKornia, ImageTransform, kornia_preprocess, kornia_random_crop
from src.data import ResnetA
from src.models import TPT

from src.utils import bench

no = None

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # kornia_augmix = AugMixKornia()

    augmenter = ImageTransform(
        model_transform=kornia_preprocess,
        custom_transform=kornia_random_crop,
        n_views=64,
    )

    dataloader, dataset = ResnetA(augmenter)
# datacomp_xl_s13b_b90k
    my_tpt = TPT(
        arch="ViT-B-16",
        pretrained="openai",
        class_names=dataset.class_code_to_label.values(),
        tta_steps=1,
        lr=5e-3,
    )

    # print(f"leanable params: {sum(p.numel() for p in my_tpt.parameters() if p.requires_grad)}")
    # print(f"total params: {sum(p.numel() for p in my_tpt.parameters())}")

    accuracy, latency = bench(my_tpt, dataloader, device, reduce=200, comment="tpt 1 step 63 - check")

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Latency: {latency * 1000:.2f} ms")
