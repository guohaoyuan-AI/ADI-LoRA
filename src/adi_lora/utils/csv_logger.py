"""Small append-only CSV logger."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional


def append_csv(path: str | Path, row: Mapping[str, object], fieldnames: Optional[Iterable[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(row.keys())
    else:
        fieldnames = list(fieldnames)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(dict(row))


def write_csv(path: str | Path, rows: list[Mapping[str, object]], fieldnames: Optional[Iterable[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    else:
        fieldnames = list(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
