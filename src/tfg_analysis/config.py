from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Rutas principales del proyecto."""

    root: Path = Path(".")
    tracking_dir: Path = Path("tracking_partidos")
    eventing_dir: Path = Path("eventing_partidos")
    sequences_dir: Path = Path("sequences_partidos")
    outputs_dir: Path = Path("outputs")

    def resolve(self) -> "ProjectPaths":
        root = self.root.resolve()
        return ProjectPaths(
            root=root,
            tracking_dir=(root / self.tracking_dir).resolve(),
            eventing_dir=(root / self.eventing_dir).resolve(),
            sequences_dir=(root / self.sequences_dir).resolve(),
            outputs_dir=(root / self.outputs_dir).resolve(),
        )

    def ensure_dirs(self) -> None:
        paths = self.resolve()
        paths.tracking_dir.mkdir(parents=True, exist_ok=True)
        paths.eventing_dir.mkdir(parents=True, exist_ok=True)
        paths.sequences_dir.mkdir(parents=True, exist_ok=True)
        paths.outputs_dir.mkdir(parents=True, exist_ok=True)
        (paths.outputs_dir / "figures").mkdir(parents=True, exist_ok=True)
        (paths.outputs_dir / "tables").mkdir(parents=True, exist_ok=True)
        (paths.outputs_dir / "reports").mkdir(parents=True, exist_ok=True)


FIELD_LENGTH_M = 102.0
FIELD_WIDTH_M = 64.0
