"""
src/evaluate.py
Evaluation metrics: accuracy, BLEU score, loss.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

from src.utils.logger import get_logger

logger = get_logger("evaluate")


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on a validation/test set.

    Args:
        model: SLT model
        dataloader: Val or test DataLoader
        criterion: Loss function (CrossEntropyLoss)
        device: torch device

    Returns:
        dict with keys: 'val_loss', 'val_acc'
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for frames, keypoints, labels in pbar:
            frames    = frames.to(device)
            keypoints = keypoints.to(device)
            labels    = labels.to(device)

            out    = model(frames, keypoints, mode="finetune")
            logits = out["logits"]  # (B, vocab_size)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.shape[0]

    val_loss = total_loss / max(len(dataloader), 1)
    val_acc  = correct / max(total, 1)

    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
    }
