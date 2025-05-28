from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
from tqdm import tqdm
import gc

from torch.utils.tensorboard import SummaryWriter

def bench(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    comment: str,
    reduce: Optional[int | None] = None,
    visualize: Optional[bool] = False,
):
    """Benchmark the model on the dataset.

    The model must return logits.
    """

    board = SummaryWriter(comment=comment)

    total = 0
    correct = 0

    times = []

    total_tqdm = reduce if reduce is not None else len(dataloader)
    # ░▒█
    # ascii=" ▖▘▝▗▚▞█"
    # ascii=' >='
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for image, label in tqdm(dataloader, total=total_tqdm, ascii=" ▖▘▝▗▚▞█"):
        image = image.to(device)

        start_event.record()
        pred_class = model(image)
        end_event.record()
        torch.cuda.synchronize()

        times.append(start_event.elapsed_time(end_event))

        total += 1
        correct += int((pred_class == label))

        if reduce:
            if total > reduce:
                break

        # break
        board.add_scalar("accuracy", correct / total, total)
        board.add_scalar("dbg/label/predict_class", pred_class, total)
        board.add_scalar("dbg/label/label", label, total)

        running_accuracy = correct / total

        if visualize:
            print(f"[{label} || {pred_class}] | Acc: [{running_accuracy*100:.2f}%]")

    accuracy = correct / total
    latency = (np.array(times).sum() / total).item()  # ms

    board.add_scalar("metrics/latency (ms)", latency)
    board.add_scalar("metrics/accuracy", accuracy)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Latency: {latency:.2f} ms")

    return accuracy, latency
