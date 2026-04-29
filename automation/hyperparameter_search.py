"""
automation/hyperparameter_search.py
Automated hyperparameter search using grid search.

Features:
    - Grid search over hyperparameter space
    - Parallel execution
    - Results tracking
    - Best model selection

Usage:
    python automation/hyperparameter_search.py \
        --base configs/base_config.yaml \
        --output results/hparam_search.json
"""

import argparse
import json
import subprocess
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils.logger import get_logger

logger = get_logger("hparam_search")


# Hyperparameter search space
SEARCH_SPACE = {
    "learning_rate": [0.0001, 0.0003, 0.001],
    "batch_size": [8, 16, 32],
    "hidden_dim": [128, 256],
    "num_layers": [2, 4],
}


def create_experiment_config(base_path: str, hparams: dict, experiment_name: str) -> str:
    """
    Create experiment config with specific hyperparameters.

    Returns:
        Path to created config file
    """
    from src.utils.config import load_config, save_config

    cfg = load_config(base_path, base_path=base_path)

    # Update hyperparameters
    cfg["training"]["learning_rate"] = hparams["learning_rate"]
    cfg["training"]["batch_size"] = hparams["batch_size"]
    cfg["model"]["hidden_dim"] = hparams["hidden_dim"]
    cfg["model"]["num_layers"] = hparams["num_layers"]
    cfg["logging"]["experiment_name"] = experiment_name

    # Save
    exp_path = Path("configs/experiments") / f"{experiment_name}.yaml"
    save_config(cfg, str(exp_path))

    return str(exp_path)


def run_experiment(experiment_name: str, config_path: str, base_path: str) -> dict:
    """
    Run a single experiment and return results.

    Returns:
        dict with results
    """
    logger.info(f"[Search] Starting experiment: {experiment_name}")

    try:
        # Run training
        result = subprocess.run(
            [
                "python", "src/trainer.py",
                "--config", config_path,
                "--base", base_path,
            ],
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )

        if result.returncode != 0:
            logger.error(f"[Search] Experiment {experiment_name} failed")
            return {
                "experiment_name": experiment_name,
                "status": "failed",
                "error": result.stderr[:200],
            }

        # Read results
        metrics_path = Path("logs/test_metrics.json")
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            return {
                "experiment_name": experiment_name,
                "status": "success",
                "metrics": metrics,
            }
        else:
            return {
                "experiment_name": experiment_name,
                "status": "incomplete",
            }

    except subprocess.TimeoutExpired:
        logger.error(f"[Search] Experiment {experiment_name} timed out")
        return {
            "experiment_name": experiment_name,
            "status": "timeout",
        }
    except Exception as e:
        logger.error(f"[Search] Experiment {experiment_name} error: {e}")
        return {
            "experiment_name": experiment_name,
            "status": "error",
            "error": str(e),
        }


def grid_search(
    base_path: str,
    output_dir: str = "results/",
    max_workers: int = 2,
) -> None:
    """
    Run grid search over hyperparameter space.

    Args:
        base_path: Path to base config
        output_dir: Where to save results
        max_workers: Number of parallel experiments
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all hyperparameter combinations
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    combinations = list(product(*values))

    logger.info(f"[Search] Grid search over {len(combinations)} combinations")
    logger.info(f"[Search] Search space: {SEARCH_SPACE}")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for idx, combo in enumerate(combinations):
            hparams = dict(zip(keys, combo))
            exp_name = f"grid_exp_{idx:03d}"

            # Create config
            config_path = create_experiment_config(base_path, hparams, exp_name)

            # Submit job
            future = executor.submit(run_experiment, exp_name, config_path, base_path)
            futures[future] = (exp_name, hparams)

        # Collect results
        for i, future in enumerate(as_completed(futures)):
            exp_name, hparams = futures[future]
            try:
                result = future.result()
                result["hyperparameters"] = hparams
                results.append(result)

                status = result.get("status", "unknown")
                logger.info(f"[Search] [{i+1}/{len(combinations)}] {exp_name}: {status}")

                if status == "success":
                    val_acc = result.get("metrics", {}).get("top1_accuracy", 0.0)
                    logger.info(f"         Val accuracy: {val_acc:.3f}")

            except Exception as e:
                logger.error(f"[Search] Error collecting results: {e}")

    # Save results
    results_file = output_dir / "grid_search_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Find best
    successful = [r for r in results if r.get("status") == "success"]
    if successful:
        best = max(successful, key=lambda x: x.get("metrics", {}).get("top1_accuracy", 0))
        logger.info(f"\n[Search] ✓ COMPLETE")
        logger.info(f"  Best experiment: {best['experiment_name']}")
        logger.info(f"  Best hyperparameters: {best['hyperparameters']}")
        logger.info(f"  Best val accuracy: {best['metrics']['top1_accuracy']:.3f}")

    logger.info(f"  Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter grid search")
    parser.add_argument("--base", default="configs/base_config.yaml")
    parser.add_argument("--output", default="results/")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    grid_search(args.base, output_dir=args.output, max_workers=args.workers)


if __name__ == "__main__":
    main()
