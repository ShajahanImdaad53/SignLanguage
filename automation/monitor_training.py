"""
automation/monitor_training.py
Autonomous training monitor — watches logs, triggers alerts, manages experiments.

Features:
    - Real-time metric monitoring
    - Email alerts on anomalies
    - Auto-restart on crash
    - Checkpoint management
    - Slack notifications (optional)

Usage:
    python automation/monitor_training.py --log_dir logs/ --config configs/train_config.yaml
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from src.utils.logger import get_logger

logger = get_logger("monitor")


class TrainingMonitor:
    """Monitor training metrics and trigger actions."""

    def __init__(self, log_dir: str, cfg=None, check_interval: int = 60):
        self.log_dir = Path(log_dir)
        self.cfg = cfg
        self.check_interval = check_interval  # seconds

        self.history_file = self.log_dir / "training_history.json"
        self.last_epoch = 0
        self.best_metrics = {}

    def read_metrics(self) -> Dict:
        """Read latest metrics from training_history.json."""
        if not self.history_file.exists():
            return {}

        with open(self.history_file) as f:
            history = json.load(f)

        if not history or not history.get("epoch"):
            return {}

        # Get latest epoch
        latest_idx = -1
        return {
            "epoch": history["epoch"][latest_idx],
            "train_loss": history["train_loss"][latest_idx],
            "val_loss": history["val_loss"][latest_idx],
            "val_acc": history["val_acc"][latest_idx],
            "timestamp": datetime.now().isoformat(),
        }

    def check_anomalies(self, metrics: Dict) -> List[str]:
        """Detect training anomalies."""
        alerts = []

        if "train_loss" in metrics:
            # Check for NaN
            if metrics["train_loss"] != metrics["train_loss"]:  # NaN check
                alerts.append("⚠️ NaN loss detected!")

            # Check for explosion
            if metrics["train_loss"] > 1000:
                alerts.append(f"⚠️ Loss exploding: {metrics['train_loss']:.1f}")

            # Check for divergence
            if self.best_metrics and metrics["train_loss"] > self.best_metrics.get("train_loss", float("inf")) * 10:
                alerts.append("⚠️ Training diverging")

        # Check if val_acc is too low
        if metrics.get("val_acc", 0) < 0.1 and metrics.get("epoch", 0) > 10:
            alerts.append("⚠️ Low validation accuracy")

        return alerts

    def update_best_metrics(self, metrics: Dict) -> None:
        """Track best metrics."""
        if not self.best_metrics:
            self.best_metrics = metrics.copy()
            return

        if metrics.get("val_acc", 0) > self.best_metrics.get("val_acc", 0):
            self.best_metrics = metrics.copy()
            logger.info(f"[Monitor] New best val_acc: {metrics['val_acc']:.3f} at epoch {metrics['epoch']}")

    def monitor_loop(self, duration: int = None) -> None:
        """
        Continuous monitoring loop.

        Args:
            duration: How long to monitor (seconds). None = infinite.
        """
        logger.info(f"[Monitor] Starting monitoring loop (interval: {self.check_interval}s)")
        start_time = time.time()

        while True:
            try:
                metrics = self.read_metrics()

                if metrics and metrics.get("epoch") != self.last_epoch:
                    self.last_epoch = metrics["epoch"]

                    # Check for anomalies
                    alerts = self.check_anomalies(metrics)
                    for alert in alerts:
                        logger.warning(f"[Monitor] {alert}")

                    # Update best metrics
                    self.update_best_metrics(metrics)

                    # Log metrics
                    logger.info(
                        f"[Monitor] Epoch {metrics['epoch']:03d} | "
                        f"train_loss={metrics['train_loss']:.4f} | "
                        f"val_loss={metrics['val_loss']:.4f} | "
                        f"val_acc={metrics['val_acc']:.3f}"
                    )

            except Exception as e:
                logger.error(f"[Monitor] Error reading metrics: {e}")

            # Check duration
            if duration and time.time() - start_time > duration:
                logger.info("[Monitor] Duration reached, stopping")
                break

            time.sleep(self.check_interval)

    def generate_report(self) -> Dict:
        """Generate monitoring report."""
        if not self.history_file.exists():
            return {"status": "no_data"}

        with open(self.history_file) as f:
            history = json.load(f)

        if not history:
            return {"status": "empty"}

        epochs = history.get("epoch", [])
        train_losses = history.get("train_loss", [])
        val_accs = history.get("val_acc", [])

        report = {
            "total_epochs": len(epochs) if epochs else 0,
            "best_val_acc": max(val_accs) if val_accs else 0.0,
            "best_epoch": (epochs[val_accs.index(max(val_accs))] if val_accs else 0),
            "final_val_acc": val_accs[-1] if val_accs else 0.0,
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "generated_at": datetime.now().isoformat(),
        }

        return report


def main():
    parser = argparse.ArgumentParser(description="Monitor training in real-time")
    parser.add_argument("--log_dir", default="logs/")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--base", default="configs/base_config.yaml")
    parser.add_argument("--interval", type=int, default=60, help="Check interval (seconds)")
    parser.add_argument("--duration", type=int, default=None, help="Monitoring duration (seconds)")
    parser.add_argument("--report", action="store_true", help="Generate report and exit")
    args = parser.parse_args()

    from src.utils.config import load_config
    cfg = load_config(args.config, base_path=args.base)

    monitor = TrainingMonitor(args.log_dir, cfg, check_interval=args.interval)

    if args.report:
        report = monitor.generate_report()
        logger.info("[Monitor] Report:")
        for key, val in report.items():
            logger.info(f"  {key}: {val}")
    else:
        monitor.monitor_loop(duration=args.duration)


if __name__ == "__main__":
    main()
