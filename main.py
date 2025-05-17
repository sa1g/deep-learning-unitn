from src.augmix import AugMixKornia, ImageTransform, kornia_preprocess
from src.data import ResnetA
from src.models import TPT

# from scalene.scalene_profiler import enable_profiling

import torch

from src.utils import bench

torch.manual_seed(42)
torch.cuda.manual_seed(42)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kornia_augmix = AugMixKornia()

    augmenter = ImageTransform(
        model_transform=kornia_preprocess,
        custom_transform=kornia_augmix,
        n_views=63,
    )

    dataloader, dataset = ResnetA(augmenter)

    my_tpt = TPT(
        class_names=dataset.class_code_to_label.values(), tta_steps=1, lr=0.005
    )
    
    accuracy, latency = bench(my_tpt, dataloader, device, reduce=2500)
        
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Latency: {latency * 1000:.2f} ms")
