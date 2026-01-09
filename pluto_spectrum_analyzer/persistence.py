"""Persistence helpers for analyzer state and calibration.

Stores and retrieves state/calibration JSON files for the application. This
module must not import UI or SDR classes; it only handles filesystem I/O.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAL_PATH = os.path.join(ROOT_DIR, "spectrum-monitor-calibration.json")
STATE_PATH = os.path.join(ROOT_DIR, "spectrum-monitor-state.json")

# Keep a short recent list to avoid bloating state files.
RECENT_URI_LIMIT = 5


@dataclass
class Calibration:
    dbfs_to_dbm_offset: Optional[float] = None
    external_gain_db: float = 0.0

    @classmethod
    def load(cls) -> "Calibration":
        # Load calibration from JSON (if present); tolerate parse errors.
        if not os.path.exists(CAL_PATH):
            return cls()
        try:
            with open(CAL_PATH, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return cls()
        return cls(
            dbfs_to_dbm_offset=data.get("dbfs_to_dbm_offset"),
            external_gain_db=float(data.get("external_gain_db", 0.0)),
        )

    def save(self) -> None:
        # Persist calibration for future runs.
        data = {
            "dbfs_to_dbm_offset": self.dbfs_to_dbm_offset,
            "external_gain_db": self.external_gain_db,
        }
        with open(CAL_PATH, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def has_calibration(self) -> bool:
        return self.dbfs_to_dbm_offset is not None


def load_state() -> Dict:
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(data: Dict) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def update_recent_uris(recent: list[str], uri: str) -> list[str]:
    uri = uri.strip()
    if not uri:
        return recent
    filtered = [entry for entry in recent if entry != uri]
    filtered.insert(0, uri)
    return filtered[:RECENT_URI_LIMIT]
