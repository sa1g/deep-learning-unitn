import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


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

    board = SummaryWriter(comment=comment)

    total = 0
    correct = 0

    start = time.time()

    total_tqdm = reduce if reduce is not None else len(dataloader)
    # ░▒█
    # ascii=" ▖▘▝▗▚▞█"
    # ascii=' >='
    for image, label in tqdm(dataloader, total=total_tqdm, ascii=" ▖▘▝▗▚▞█"):
        this_start = time.time()

        image = image.to(device)

        # start1 = time.time()
        pred_class = model(image)
        del image
        # print(f"model: {(time.time() - start1) * 1000:.2f} ms")

        total += 1
        correct += int((pred_class == label))

        if reduce:
            if total > reduce:
                break

        # break
        board.add_scalar("accuracy", correct / total, total)
        board.add_scalar("metrics/latency (ms)", (time.time() - this_start) * 1000, total)
        board.add_scalar("label/predict_class", pred_class, total)
        board.add_scalar("label/label", label, total)


    end = time.time()

    accuracy = correct / total
    latency = (end - start) / total  # ms

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Latency: {latency * 1000:.2f} ms")

    return accuracy, latency
