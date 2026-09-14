"""
CSV logging for per-episode training metrics.

Writes one row per episode with standardized columns so that analysis
in Jupyter notebooks is straightforward:

    import pandas as pd
    df = pd.read_csv("runs/my_run/train_history.csv")
    df["episode_return"].rolling(100).mean().plot()
"""

from __future__ import annotations

import csv
from pathlib import Path


class CSVLogger:
    """Logs per-episode training metrics to a CSV file.

    Supports append mode for resumed training sessions — the header
    is only written once, and new rows are appended.
    """

    COLUMNS: list[str] = [
        "episode",
        "episode_return",
        "episode_length",
        "success",
        "collisions",
        "epsilon",
        "episode_time_sec",
    ]

    def __init__(self, path: str | Path, append: bool = False):
        """
        Open (or create) a CSV file for logging.

        Args:
            path: Path to the CSV file.
            append: If True, open in append mode (no header written).
                    Used when resuming training.
        """
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        self._file = open(self._path, mode, newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS)

        # Write header only for new files
        if not append:
            self._writer.writeheader()
            self._file.flush()

    def log(self, metrics: dict) -> None:
        """Append one row of metrics to the CSV."""
        self._writer.writerow(metrics)

    def flush(self) -> None:
        """Force write buffered data to disk."""
        self._file.flush()

    def close(self) -> None:
        """Close the CSV file."""
        self._file.close()

    def __enter__(self) -> CSVLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
