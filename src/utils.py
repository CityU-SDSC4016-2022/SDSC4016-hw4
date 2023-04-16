import csv
import math
from pathlib import Path

from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from model import Classifier


def get_adamw(model: Classifier, learning_rate: int):
    """Get the AdamW optimizer."""
    return AdamW(model.parameters(), lr=learning_rate)


def get_cos_scheduler(optimizer: Optimizer, warmup_steps: int, training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
    """Get the Cosine Annealing Scheduler."""
    def lr_lambda(current_step: int):
        # Warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # decadence
        progress = float(current_step - warmup_steps) / float(max(1, training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def save_results(results: list[list[str]], path: str, accuracy: float):
    """Save the results to a CSV file."""
    file = Path(path)
    path = f"{file.stem}_{accuracy * 100:.4f}{file.suffix}"
    with open(path, 'w', newline='', encoding="utf8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)
