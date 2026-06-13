from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MotionLibraryLayout:
    root_dir: Path
    motion_dir: Path
    amp_dataset_dir: Path


def dir_has_npz(path: str | Path) -> bool:
    path = Path(path)
    return path.is_dir() and any(child.suffix == ".npz" for child in path.iterdir())


def resolve_motion_library(path: str | Path) -> MotionLibraryLayout:
    root_dir = Path(path).resolve()
    walk_dir = root_dir / "Walk"

    has_walk = dir_has_npz(walk_dir)
    has_local_npz = dir_has_npz(root_dir)

    if has_walk:
        motion_dir = walk_dir
        amp_dataset_dir = root_dir
    elif has_local_npz:
        motion_dir = root_dir
        amp_dataset_dir = root_dir
    else:
        raise FileNotFoundError(
            "No AMP motion files found. Expected either .npz files directly in "
            f"{root_dir} or subfolder Walk/."
        )

    return MotionLibraryLayout(
        root_dir=root_dir,
        motion_dir=motion_dir,
        amp_dataset_dir=amp_dataset_dir,
    )
