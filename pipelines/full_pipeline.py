"""
pipelines/full_pipeline.py
End-to-end pipeline: data → SSL pretrain → fine-tune → evaluate.

This is the single command to reproduce a full experiment:
    python pipelines/full_pipeline.py --config configs/train_config.yaml

Stages:
    1. Data pipeline (frames + keypoints + splits)
    2. SSL pretraining
    3. Supervised fine-tuning
    4. Test set evaluation
    5. Save experiment report
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from src.utils.config import load_config, save_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("full_pipeline")


def run_step(name: str, cmd: list) -> None:
    """Run a sub-process step, exit on failure."""
    print(f"\n{'='*60}")
    print(f"  STAGE: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        logger.error(f"[Pipeline] Stage '{name}' FAILED (exit code {result.returncode})")
        sys.exit(result.returncode)
    logger.info(f"[Pipeline] Stage '{name}' completed in {elapsed:.1f}s ✓")


def main():
    parser = argparse.ArgumentParser(description="Full SLT Training Pipeline")
    parser.add_argument("--config",     default="configs/train_config.yaml")
    parser.add_argument("--ssl_config", default="configs/ssl_config.yaml")
    parser.add_argument("--base",       default="configs/base_config.yaml")
    parser.add_argument("--dummy",      action="store_true",
                        help="Use dummy data (no real videos needed)")
    parser.add_argument("--skip_ssl",   action="store_true",
                        help="Skip SSL pretraining (use random init)")
    parser.add_argument("--skip_data",  action="store_true",
                        help="Skip data pipeline (assume splits already exist)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config, base_path=args.base)

    # Experiment folder
    exp_name = cfg.logging.get("experiment_name", "exp") + \
               f"_{datetime.now():%Y%m%d_%H%M%S}"
    exp_dir  = Path("experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    save_config(cfg, str(exp_dir / "config_snapshot.yaml"))
    logger.info(f"[Pipeline] Experiment: {exp_dir}")

    t_start = time.time()

    # ── Stage 1: Data ────────────────────────────────────────
    if not args.skip_data:
        data_cmd = [sys.executable, "run_data_pipeline.py",
                    "--config", "configs/data_config.yaml",
                    "--base",   args.base]
        if args.dummy:
            data_cmd.append("--dummy")
        run_step("Data Pipeline", data_cmd)
    else:
        logger.info("[Pipeline] Skipping data stage (--skip_data)")

    # ── Stage 2: SSL Pretraining ─────────────────────────────
    if not args.skip_ssl:
        ssl_cmd = [sys.executable, "-m", "src.ssl_pretrain",
                   "--config", args.ssl_config,
                   "--base",   args.base,
                   "--seed",   str(args.seed)]
        run_step("SSL Pretraining", ssl_cmd)
    else:
        logger.info("[Pipeline] Skipping SSL pretraining (--skip_ssl)")

    # ── Stage 3: Fine-tuning ──────────────────────────────────
    run_step("Supervised Fine-tuning", [
        sys.executable, "-m", "src.trainer",
        "--config",     args.config,
        "--ssl_config", args.ssl_config,
        "--base",       args.base,
        "--seed",       str(args.seed),
    ])

    # ── Stage 4: Test Evaluation ──────────────────────────────
    run_step("Test Evaluation", [
        sys.executable, "-m", "src.evaluate",
        "--config",     args.config,
        "--base",       args.base,
        "--checkpoint", "models/best_model.pt",
        "--split",      "test",
    ])

    # ── Stage 5: Save experiment report ──────────────────────
    total_time = time.time() - t_start
    report = {
        "experiment": exp_name,
        "config":     args.config,
        "ssl_config": args.ssl_config,
        "seed":       args.seed,
        "dummy_data": args.dummy,
        "skip_ssl":   args.skip_ssl,
        "total_time_sec": round(total_time, 1),
    }

    # Include test results if available
    test_results_path = Path("logs/test_results.json")
    if test_results_path.exists():
        with open(test_results_path) as f:
            report["test_results"] = json.load(f)

    report_path = exp_dir / "experiment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total time : {total_time/60:.1f} minutes")
    print(f"  Report     : {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
