"""
src/evaluate_test.py
Comprehensive test set evaluation with detailed metrics.

Usage:
    python src/evaluate_test.py --checkpoint models/best_model.pt
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from src.models.slt_model import SLTModel
from src.dataset import build_dataloaders
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("evaluate_test")


class TestEvaluator:
    """Comprehensive test evaluation."""

    def __init__(self, cfg, checkpoint_path: str):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Eval] Device: {self.device}")

        # Load model
        self.model = SLTModel(cfg).to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        logger.info(f"[Eval] Loaded checkpoint: {checkpoint_path}")

        # Load data
        self.loaders = build_dataloaders(cfg)
        self.test_loader = self.loaders.get("test")

        # Results
        self.results = {}

    def evaluate(self) -> dict:
        """Run full test evaluation."""
        if self.test_loader is None:
            logger.error("[Eval] No test data available")
            return {}

        all_preds = []
        all_labels = []
        all_probs = []

        logger.info("[Eval] Running inference on test set...")
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluating")
            for frames, keypoints, labels in pbar:
                frames = frames.to(self.device)
                keypoints = keypoints.to(self.device)
                labels = labels.to(self.device)

                out = self.model(frames, keypoints, mode="finetune")
                logits = out["logits"]  # (B, vocab_size)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Compute metrics
        self.results = {
            "accuracy": float(accuracy_score(all_labels, all_preds)),
            "precision": float(precision_score(all_labels, all_preds, average="weighted", zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds, average="weighted", zero_division=0)),
            "f1": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
            "predictions": all_preds.tolist(),
            "labels": all_labels.tolist(),
        }

        return self.results

    def print_report(self) -> None:
        """Print detailed evaluation report."""
        if not self.results:
            logger.error("[Eval] No results to print")
            return

        logger.info("=" * 60)
        logger.info("TEST EVALUATION REPORT")
        logger.info("=" * 60)

        acc = self.results["accuracy"]
        prec = self.results["precision"]
        rec = self.results["recall"]
        f1 = self.results["f1"]

        logger.info(f"\n📊 OVERALL METRICS")
        logger.info(f"  Accuracy:  {acc:.4f}")
        logger.info(f"  Precision: {prec:.4f}")
        logger.info(f"  Recall:    {rec:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")

        # Per-class metrics
        all_labels = np.array(self.results["labels"])
        all_preds = np.array(self.results["predictions"])

        logger.info(f"\n📈 PER-CLASS METRICS")
        classes = sorted(np.unique(all_labels))
        for cls in classes:
            mask = all_labels == cls
            cls_acc = (all_preds[mask] == cls).sum() / mask.sum()
            logger.info(f"  Class {cls:3d}: accuracy={cls_acc:.4f} "
                       f"(n={mask.sum()})")

        logger.info(f"\n{len(classes)} total classes")
        logger.info("=" * 60)

    def save_results(self, save_path: str = "logs/test_results.json") -> None:
        """Save results to JSON."""
        if not self.results:
            logger.error("[Eval] No results to save")
            return

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"[Eval] Results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on test set")
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--base", default="configs/base_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config, base_path=args.base)
    evaluator = TestEvaluator(cfg, args.checkpoint)

    evaluator.evaluate()
    evaluator.print_report()
    evaluator.save_results()


if __name__ == "__main__":
    main()
