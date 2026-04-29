"""
agents/data_watcher_agent.py
File-watcher agent — automatically processes any new .mp4 dropped
into data/raw_videos/ without manual intervention.

How it works:
    1. Watches data/raw_videos/ for new .mp4 files
    2. When a new file appears, runs frame extraction + keypoint extraction
    3. Appends the new record to data/interim/processed_videos.csv
    4. Optionally triggers a split rebuild

Usage:
    python agents/data_watcher_agent.py
    python agents/data_watcher_agent.py --watch_dir /path/to/videos --rebuild_splits
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("data_watcher_agent")


class VideoHandler(FileSystemEventHandler):
    """
    Reacts to new video files in the watched directory.
    Processes them through frame extraction and keypoint extraction.
    """

    def __init__(self, cfg, rebuild_splits: bool = False):
        super().__init__()
        self.cfg            = cfg
        self.rebuild_splits = rebuild_splits

        # Import processing modules lazily (heavy deps)
        from download_and_process_video import process_video, KeypointExtractor
        self.process_video = process_video
        self.extractor     = KeypointExtractor()

        self.processed_csv = Path(cfg.data.interim_dir) / "processed_videos.csv"
        self.processed_csv.parent.mkdir(parents=True, exist_ok=True)

        # Track already-processed IDs to avoid duplicates
        self.seen_ids = self._load_seen_ids()
        logger.info(f"[Watcher] Ready. Already seen: {len(self.seen_ids)} videos")

    def _load_seen_ids(self) -> set:
        if not self.processed_csv.exists():
            return set()
        import pandas as pd
        df = pd.read_csv(self.processed_csv)
        return set(df["video_id"].tolist())

    def on_created(self, event):
        if isinstance(event, FileCreatedEvent) and not event.is_directory:
            path = Path(event.src_path)
            if path.suffix.lower() in (".mp4", ".avi", ".mov"):
                # Brief delay to ensure file is fully written
                time.sleep(2)
                self._handle_new_video(path)

    def _handle_new_video(self, video_path: Path) -> None:
        video_id = video_path.stem

        if video_id in self.seen_ids:
            logger.info(f"[Watcher] Already processed: {video_id}")
            return

        logger.info(f"[Watcher] New video detected: {video_path.name}")

        try:
            rec = self.process_video(str(video_path), video_id, self.cfg, self.extractor)
            # Derive label from filename convention: LABEL_XXXXX.mp4
            parts = video_id.split("_")
            rec["gloss"]     = parts[0] if parts else video_id
            rec["timestamp"] = datetime.now().isoformat()

            # Append to processed CSV
            self._append_record(rec)
            self.seen_ids.add(video_id)

            logger.info(f"[Watcher] Processed and saved: {video_id}")

            if self.rebuild_splits:
                self._rebuild_splits()

        except Exception as e:
            logger.error(f"[Watcher] Failed to process {video_path.name}: {e}")

    def _append_record(self, rec: dict) -> None:
        """Append one record to the processed_videos.csv."""
        file_exists = self.processed_csv.exists()
        with open(self.processed_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rec.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(rec)

    def _rebuild_splits(self) -> None:
        """Trigger split rebuild via subprocess."""
        import subprocess, sys
        logger.info("[Watcher] Rebuilding train/val/test splits...")
        subprocess.run([
            sys.executable, "create_test_data.py",
            f"--source={self.processed_csv}",
        ])


# ─────────────────────────────────────────────
# Polling fallback (no watchdog)
# ─────────────────────────────────────────────

class PollingWatcher:
    """Simple polling-based watcher for environments without watchdog."""

    def __init__(self, watch_dir: str, handler: VideoHandler, interval: int = 10):
        self.watch_dir = Path(watch_dir)
        self.handler   = handler
        self.interval  = interval
        self.seen_files: set = set()

    def start(self) -> None:
        logger.info(f"[Watcher] Polling {self.watch_dir} every {self.interval}s "
                    f"(install watchdog for real-time detection)")
        while True:
            try:
                current = set(self.watch_dir.glob("*.mp4")) | \
                          set(self.watch_dir.glob("*.avi")) | \
                          set(self.watch_dir.glob("*.mov"))
                new_files = current - self.seen_files
                for fp in new_files:
                    self.handler._handle_new_video(fp)
                self.seen_files = current
            except KeyboardInterrupt:
                logger.info("[Watcher] Stopped by user")
                break
            except Exception as e:
                logger.error(f"[Watcher] Polling error: {e}")
            time.sleep(self.interval)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Data Watcher Agent")
    parser.add_argument("--watch_dir",      default=None,
                        help="Directory to watch (default: from config)")
    parser.add_argument("--rebuild_splits", action="store_true",
                        help="Rebuild splits after each new video")
    parser.add_argument("--interval",       type=int, default=10,
                        help="Polling interval in seconds (fallback mode)")
    parser.add_argument("--config",         default="configs/data_config.yaml")
    parser.add_argument("--base",           default="configs/base_config.yaml")
    args = parser.parse_args()

    cfg       = load_config(args.config, base_path=args.base)
    watch_dir = args.watch_dir or cfg.data.raw_videos_dir
    Path(watch_dir).mkdir(parents=True, exist_ok=True)

    handler = VideoHandler(cfg, rebuild_splits=args.rebuild_splits)

    if WATCHDOG_AVAILABLE:
        logger.info(f"[Watcher] Watching {watch_dir} in real-time mode (watchdog)")
        observer = Observer()
        observer.schedule(handler, str(watch_dir), recursive=False)
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            logger.info("[Watcher] Stopped by user")
        observer.join()
    else:
        logger.warning("[Watcher] watchdog not installed. Using polling mode.")
        logger.warning("          Install with: pip install watchdog")
        poller = PollingWatcher(watch_dir, handler, interval=args.interval)
        poller.start()


if __name__ == "__main__":
    main()
