from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CoreBundle:
    root: Path

    @property
    def eval_data_dir(self) -> Path:
        return self.root / "eval_data"

    @property
    def meta_csv(self) -> Path:
        return self.root / "eval_meta_data.csv"

    def require(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(str(self.root))
        if not self.eval_data_dir.is_dir():
            raise FileNotFoundError(str(self.eval_data_dir))
        if not self.meta_csv.is_file():
            raise FileNotFoundError(str(self.meta_csv))

    def dataset_path(self, dataset_uri: str) -> Path:
        # Guard against path traversal. dataset_uris are bundle-relative paths like
        # "world_knowledge/arc_easy.jsonl".
        rel = Path(dataset_uri)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"invalid dataset_uri (must be relative): {dataset_uri!r}")
        return self.eval_data_dir / rel

    def load_jsonl(self, dataset_uri: str) -> list[dict]:
        path = self.dataset_path(dataset_uri)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        out: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
        if not out:
            raise RuntimeError(f"no records in {path}")
        return out

    def load_random_baselines(self) -> dict[str, float]:
        """Return mapping task_label -> random baseline (percent, 0-100)."""
        if not self.meta_csv.is_file():
            raise FileNotFoundError(str(self.meta_csv))
        out: dict[str, float] = {}
        with self.meta_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = (row.get("Eval Task") or "").strip()
                baseline = (row.get("Random baseline") or "").strip()
                if not label or not baseline:
                    continue
                try:
                    out[label] = float(baseline)
                except ValueError:
                    continue
        if not out:
            raise RuntimeError(f"no baselines read from {self.meta_csv}")
        return out

