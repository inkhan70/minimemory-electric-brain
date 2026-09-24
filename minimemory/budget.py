"""Local resource budgets for optional network research and storage growth."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict


class ResourceBudget:
    def __init__(self, *, network_mb: float = 0.0, storage_mb: float = 0.0) -> None:
        if network_mb < 0 or storage_mb < 0:
            raise ValueError("budgets must be >= 0")
        self.network_bytes = int(network_mb * 1024 * 1024)
        self.storage_bytes = int(storage_mb * 1024 * 1024)
        self.network_used = 0

    def allow_network(self, estimated_bytes: int) -> bool:
        if estimated_bytes < 0:
            raise ValueError("estimated_bytes must be >= 0")
        return self.network_bytes == 0 or self.network_used + estimated_bytes <= self.network_bytes

    def consume_network(self, actual_bytes: int) -> None:
        if actual_bytes < 0:
            raise ValueError("actual_bytes must be >= 0")
        self.network_used += int(actual_bytes)

    def storage_status(self, path: str) -> Dict[str, int | bool]:
        target = Path(path).expanduser()
        usage = shutil.disk_usage(target.parent if target.parent.exists() else Path.cwd())
        return {
            "limit_bytes": self.storage_bytes,
            "free_bytes": int(usage.free),
            "allowed": self.storage_bytes == 0 or target.exists() or usage.free >= self.storage_bytes,
        }
