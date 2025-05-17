import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
from tqdm import tqdm

def bench(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    reduce: int | None = None,
):
    """Benchmark the model on the dataset.

    The model must return logits.
    """

    total = 0
    correct = 0

    start = time.time()

    for image, label in tqdm(dataloader):
        image = image.to(device)
        label = label.to(device)

        # with torch.no_grad():
        logits = model(image)

        # pred_class = logits.argmax(dim=-1)
        marginal_prob = F.softmax(logits, dim=1).mean(0)
        pred_class = marginal_prob.argmax().item()

        total += 1
        correct += int((pred_class == label).max().item())

        if reduce:
            if total > reduce:
                break

    end = time.time()

    accuracy = correct / total
    latency = (end - start) / total  # ms

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Latency: {latency * 1000:.2f} ms")

    return accuracy, latency
