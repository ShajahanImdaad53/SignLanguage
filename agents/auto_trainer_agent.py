"""
agents/auto_trainer_agent.py
Autonomous training orchestrator — automatically manages SSL pretraining,
fine-tuning, and hyperparameter sweeps with smart monitoring and early stopping.

Usage:
    python agents/auto_trainer_agent.py --mode "full"
"""

import time
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("auto_trainer_agent")


class TrainingAgent:
    """Autonomous training orchestrator with multi-stage pipeline."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.log_dir = Path(cfg.logging.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.agent_log_file = self.log_dir / f"agent_{datetime.now():%Y%m%d_%H%M%S}.log"

    def log(self, stage: str, msg: str) -> None:
        """Log to both console and file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] [{stage}] {msg}"
        print(line)
        with open(self.agent_log_file, "a") as f:
            f.write(line + "\n")

    # ──────────────────────────────────────────────────
    # Stage 1: Data Pipeline
    # ──────────────────────────────────────────────────

    def stage_data_pipeline(self) -> bool:
        """Run data preprocessing pipeline."""
        self.log("DATA", "Starting data pipeline...")

        cmd = [
            sys.executable,
            "run_data_pipeline.py",
            "--dummy",
            "--num_samples", "100",  # Small dataset for testing
            "--config", "configs/data_config.yaml",
        ]

        self.log("DATA", f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.log("DATA", f"FAILED: {result.stderr}")
            return False

        self.log("DATA", "✓ Data pipeline complete")
        return True

    # ──────────────────────────────────────────────────
    # Stage 2: SSL Pretraining
    # ──────────────────────────────────────────────────

    def stage_ssl_pretraining(self) -> bool:
        """Run self-supervised pretraining."""
        self.log("SSL", "Starting SSL pretraining...")

        cmd = [
            sys.executable,
            "src/ssl_pretrain.py",
            "--config", "configs/ssl_config.yaml",
            "--base", "configs/base_config.yaml",
        ]

        self.log("SSL", f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.log("SSL", f"FAILED: {result.stderr}")
            return False

        self.log("SSL", "✓ SSL pretraining complete")
        return True

    # ──────────────────────────────────────────────────
    # Stage 3: Fine-tuning
    # ──────────────────────────────────────────────────

    def stage_finetune(self) -> bool:
        """Run supervised fine-tuning."""
        self.log("FINETUNE", "Starting fine-tuning...")

        cmd = [
            sys.executable,
            "src/trainer.py",
            "--config", "configs/train_config.yaml",
            "--base", "configs/base_config.yaml",
            "--ssl_config", "configs/ssl_config.yaml",
        ]

        self.log("FINETUNE", f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.log("FINETUNE", f"FAILED: {result.stderr}")
            return False

        self.log("FINETUNE", "✓ Fine-tuning complete")
        return True

    # ──────────────────────────────────────────────────
    # Hyperparameter sweep
    # ──────────────────────────────────────────────────

    def stage_hyperparam_sweep(self, lrs: List[float]) -> Dict[float, Dict]:
        """Grid search over learning rates."""
        self.log("SWEEP", f"Starting hyperparameter sweep over {len(lrs)} LR values")

        results = {}

        for lr in lrs:
            self.log("SWEEP", f"Testing LR={lr}")

            # Modify config with new LR
            cfg = load_config(
                "configs/train_config.yaml",
                base_path="configs/base_config.yaml",
                override={"training": {"learning_rate": lr}},
            )

            # Save modified config
            sweep_config_path = self.log_dir / f"config_lr{lr:.2e}.yaml"
            from src.utils.config import save_config
            save_config(cfg, str(sweep_config_path))

            # Run training with this config
            cmd = [
                sys.executable,
                "src/trainer.py",
                "--config", str(sweep_config_path),
                "--base", "configs/base_config.yaml",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Parse validation accuracy from logs
                try:
                    history_path = self.log_dir / "training_history.json"
                    with open(history_path) as f:
                        history = json.load(f)
                    best_acc = max(history.get("val_acc", [0.0]))
                    results[lr] = {"best_val_acc": best_acc, "success": True}
                    self.log("SWEEP", f"  LR={lr}: best_val_acc={best_acc:.4f}")
                except Exception as e:
                    self.log("SWEEP", f"  Failed to parse results: {e}")
                    results[lr] = {"success": False}
            else:
                results[lr] = {"success": False}

        # Save sweep results
        sweep_results_path = self.log_dir / "sweep_results.json"
        with open(sweep_results_path, "w") as f:
            json.dump(results, f, indent=2)

        self.log("SWEEP", f"✓ Sweep complete. Results → {sweep_results_path}")
        return results

    # ──────────────────────────────────────────────────
    # Main orchestrator
    # ──────────────────────────────────────────────────

    def run_full_pipeline(self) -> bool:
        """Execute full training pipeline."""
        t0 = time.time()

        self.log("MAIN", "="*60)
        self.log("MAIN", "SLT Hybrid SSL — Autonomous Training Agent")
        self.log("MAIN", "="*60)

        stages = [
            ("data_pipeline", self.stage_data_pipeline),
            ("ssl_pretraining", self.stage_ssl_pretraining),
            ("finetune", self.stage_finetune),
        ]

        for stage_name, stage_func in stages:
            try:
                success = stage_func()
                if not success:
                    self.log("MAIN", f"✗ {stage_name} failed. Aborting pipeline.")
                    return False
            except Exception as e:
                self.log("MAIN", f"✗ {stage_name} crashed: {e}")
                return False

        elapsed = time.time() - t0
        self.log("MAIN", f"✓ Full pipeline complete in {elapsed/3600:.1f} hours")
        self.log("MAIN", f"Logs → {self.agent_log_file}")
        return True

    def run_sweep_mode(self) -> bool:
        """Execute hyperparameter sweep."""
        lrs = [1e-4, 3e-4, 1e-3]
        self.stage_hyperparam_sweep(lrs)
        return True


# ──────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Training Agent")
    parser.add_argument("--mode", choices=["full", "sweep"], default="full",
                        help="Training mode")
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    agent = TrainingAgent(cfg)

    if args.mode == "full":
        success = agent.run_full_pipeline()
    elif args.mode == "sweep":
        success = agent.run_sweep_mode()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
