import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import Classifier
from utils import get_adamw, get_cos_scheduler


def model_fn(batch: list, model: Classifier, criterion: nn.CrossEntropyLoss, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward a batch through the model."""
    mels, labels = batch
    mels = mels.to(device)
    labels = labels.to(device)

    outs = model(mels)
    loss = criterion(outs, labels)

    # Get the speaker id with highest probability.
    preds = outs.argmax(1)
    # Compute accuracy.
    accuracy = torch.mean((preds == labels).float())

    return loss, accuracy


def validating(dataloader: DataLoader, model: Classifier, criterion: nn.CrossEntropyLoss, device: torch.device):
    """Validate on validation set."""

    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
            loss=f"{running_loss / (i+1):.5f}",
            accuracy=f"{running_accuracy / (i+1):.5f}",
        )

    pbar.close()
    model.train()

    return running_accuracy / len(dataloader)


def training(train_loader: DataLoader, valid_loader: DataLoader, model: Classifier, device: torch.device, lr: int, steps: list[int], save_path: str):
    """Train on training set."""

    # Check if all 4 steps setting exist.
    if len(steps) != 4:
        return None

    # Local data
    best_accuracy = -1.0
    best_state_dict = None
    criterion = nn.CrossEntropyLoss()
    optimizer = get_adamw(model, lr)
    scheduler = get_cos_scheduler(optimizer, steps[0], steps[3])
    pbar = tqdm(total=steps[1], ncols=0, desc="Train", unit=" step")

    # Training block
    train_iterator = iter(train_loader)
    for step in range(steps[3]):
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # Updata model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Log
        pbar.update()
        pbar.set_postfix(loss=f"{batch_loss:.5f}", accuracy=f"{batch_accuracy:.5f}", step=step + 1)

        # Do validation
        if (step + 1) % steps[1] == 0:
            pbar.close()

            valid_accuracy = validating(valid_loader, model, criterion, device)

            # keep the best model
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()

            pbar = tqdm(total=steps[1], ncols=0, desc="Train", unit=" step")

        # Save the best model
        if (step + 1) % steps[2] == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.5f})")

    pbar.close()
