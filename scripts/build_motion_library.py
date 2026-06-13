"""Build a clear Z1 motion library from raw retargeted clips.

This script keeps four different representations so the library is easier to
understand and inspect:

- `retargeted-full/`
  Raw retargeted qpos clips copied into this repo.
- `trimmed_qpos/`
  Human-picked trimmed qpos clips used to define the active library.
- `curated/full_length_vel/`
  Full-length retargeted clips compiled into AMP `_vel.npz` format for review.
- `curated/amp/`
  The active trimmed-and-compiled AMP library used for training.
- `curated/phase1/`
  A builder-managed forward-biased subset used for the Phase 1 locomotion warm start.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from _motion_compile import compile_qpos_to_amp
from _motion_compile import default_xml
from _motion_compile import normalize_quat
from _motion_compile import pack_qpos_to_amp
from _motion_compile import resample_qpos
from _motion_quality import MotionQualityMetrics
from _motion_quality import MotionQualityEvaluator


@dataclass(frozen=True)
class ClipSpec:
    output_group: str
    output_name: str
    source_file: str
    clip_kind: str
    semantic_label: str
    training_sets: tuple[str, ...] = ("phase2", "unified")
    start_frame: int | None = None
    end_frame: int | None = None
    static_frame: int | None = None
    static_length: int | None = None
    enabled: bool = True
    note: str = ""


@dataclass(frozen=True)
class SourceProcessSpec:
    joint_smooth_window: int = 1
    root_pos_smooth_window: int = 1
    root_quat_smooth_window: int = 1
    arm_fix: str | None = None
    pop_repair_threshold: float | None = None
    pop_repair_passes: int = 0


@dataclass(frozen=True)
class OutputRepairStep:
    kind: str
    joint_mode: str = "leg"
    alpha: float = 0.25
    radius: int = 2
    passes: int = 1
    threshold: float | None = None
    pad: int = 1


ACTIVE_CLIPS: tuple[ClipSpec, ...] = (
    ClipSpec(
        output_group="Walk",
        output_name="walk1_subject2_stand_to_walk_1332_1530_natural",
        source_file="walk1_subject2.npz",
        clip_kind="range",
        semantic_label="walk_transition",
        training_sets=("phase1", "phase2", "unified"),
        start_frame=1332,
        end_frame=1530,
        note="Stand-to-walk transition from walk1_subject2.",
    ),
    ClipSpec(
        output_group="Walk",
        output_name="walk1_subject2_straight_sidewalk_walk_right_4690_4765",
        source_file="walk1_subject2.npz",
        clip_kind="range",
        semantic_label="walk_lateral",
        start_frame=4690,
        end_frame=4765,
        note="Rightward sidewalk segment from walk1_subject2.",
    ),
    ClipSpec(
        output_group="Walk",
        output_name="walk1_subject2_straight_backward_walk_normal_3265_3340",
        source_file="walk1_subject2.npz",
        clip_kind="range",
        semantic_label="walk_backward",
        start_frame=3265,
        end_frame=3340,
        note="Backward walk segment from walk1_subject2.",
    ),
    ClipSpec(
        output_group="Walk",
        output_name="walk1_subject2_straight_walk_fast_2940_3015",
        source_file="walk1_subject2.npz",
        clip_kind="range",
        semantic_label="walk_forward",
        start_frame=2940,
        end_frame=3015,
        note="Fast straight walk loop from walk1_subject2.",
    ),
    ClipSpec(
        output_group="Walk",
        output_name="walk2_subject1_walk_normal_6540_6629",
        source_file="walk2_subject1.npz",
        clip_kind="range",
        semantic_label="walk_forward",
        training_sets=("phase1", "phase2", "unified"),
        start_frame=6540,
        end_frame=6629,
        note="Normal-speed walk window from walk2_subject1.",
    ),
    ClipSpec(
        output_group="Walk",
        output_name="walk2_subject1_walk_transition_6690_6870",
        source_file="walk2_subject1.npz",
        clip_kind="range",
        semantic_label="walk_transition",
        training_sets=("phase2", "unified"),
        start_frame=6690,
        end_frame=6870,
        note="Walk transition segment mined from walk2_subject1.",
    ),
    ClipSpec(
        output_group="Walk",
        output_name="walk1_subject5_walk_rightturn_510_599",
        source_file="walk1_subject5.npz",
        clip_kind="range",
        semantic_label="turn_walk",
        start_frame=510,
        end_frame=599,
        note="Right-turn walking segment from walk1_subject5 after left-arm repair.",
    ),
    ClipSpec(
        output_group="Walk",
        output_name="walk1_subject5_walk_turnaround_thenwalk_5622_5711",
        source_file="walk1_subject5.npz",
        clip_kind="range",
        semantic_label="turn_walk",
        start_frame=5622,
        end_frame=5711,
        note="Turn-around then walk segment from walk1_subject5 after left-arm repair.",
    ),
    ClipSpec(
        output_group="Walk",
        output_name="walk1_subject5_stand_turn_6341_6430",
        source_file="walk1_subject5.npz",
        clip_kind="range",
        semantic_label="upright_turn",
        start_frame=6341,
        end_frame=6430,
        note="Stand-in-place turning segment from walk1_subject5 after left-arm repair.",
    ),
    ClipSpec(
        output_group="Walk",
        output_name="aiming1_subject1_relaxed_upright_1408_1497",
        source_file="aiming1_subject1.npz",
        clip_kind="range",
        semantic_label="upright",
        training_sets=("phase1", "phase2", "unified"),
        start_frame=1408,
        end_frame=1497,
        note="Upright segment with both arms relaxed out of the aiming pose.",
    ),
    ClipSpec(
        output_group="Walk",
        output_name="aiming1_subject4_walk_forward_90_179",
        source_file="aiming1_subject4.npz",
        clip_kind="range",
        semantic_label="walk_forward",
        training_sets=("phase2", "unified"),
        start_frame=90,
        end_frame=179,
        note="Forward-moving upright segment from aiming1_subject4 with arms relaxed out of the locked aiming pose.",
    ),
    ClipSpec(
        output_group="Recovery",
        output_name="fallAndGetUp1_subject1",
        source_file="fallAndGetUp1_subject1.npz",
        clip_kind="range",
        semantic_label="recovery",
        start_frame=408,
        end_frame=1953,
        note="Recovery trim copied from the G1 reference motion.",
    ),
    ClipSpec(
        output_group="Recovery",
        output_name="fallAndGetUp1_subject4",
        source_file="fallAndGetUp1_subject4.npz",
        clip_kind="range",
        semantic_label="recovery",
        start_frame=40,
        end_frame=520,
        note="Early fall-and-recover cycle with the dead front T-pose frames removed.",
    ),
    ClipSpec(
        output_group="Recovery",
        output_name="fallAndGetUp2_subject2",
        source_file="fallAndGetUp2_subject2.npz",
        clip_kind="range",
        semantic_label="recovery",
        start_frame=530,
        end_frame=1020,
        note="Later fall-and-recover cycle from fallAndGetUp2_subject2 chosen to avoid the shaky early ground phase.",
    ),
    ClipSpec(
        output_group="Recovery",
        output_name="fallAndGetUp2_subject2_recover_990_1340",
        source_file="fallAndGetUp2_subject2.npz",
        clip_kind="range",
        semantic_label="recovery",
        start_frame=990,
        end_frame=1340,
        note="Second recovery trim from fallAndGetUp2_subject2 that captures a separate clean get-up cycle.",
    ),
    ClipSpec(
        output_group="Recovery",
        output_name="pushAndStumble1_subject2_recover_520_1180",
        source_file="pushAndStumble1_subject2.npz",
        clip_kind="range",
        semantic_label="disturbance_recovery",
        start_frame=520,
        end_frame=1180,
        note="Push-and-stumble recovery segment trimmed from the clean early phase of subject2.",
    ),
    ClipSpec(
        output_group="Recovery",
        output_name="pushAndStumble1_subject5_recover_520_1180",
        source_file="pushAndStumble1_subject5.npz",
        clip_kind="range",
        semantic_label="disturbance_recovery",
        start_frame=520,
        end_frame=1180,
        note="Push-and-stumble recovery segment from subject5 with local arm relaxation to remove locked-arm retarget frames.",
    ),
)

DEFERRED_CANDIDATES: tuple[ClipSpec, ...] = ()

SOURCE_PROCESS: dict[str, SourceProcessSpec] = {
    "aiming1_subject1.npz": SourceProcessSpec(arm_fix="relaxed_bilateral_idle"),
    "aiming1_subject4.npz": SourceProcessSpec(arm_fix="relaxed_bilateral_idle"),
    "walk1_subject2.npz": SourceProcessSpec(joint_smooth_window=1, root_pos_smooth_window=1, root_quat_smooth_window=1),
    "walk1_subject5.npz": SourceProcessSpec(
        joint_smooth_window=1,
        root_pos_smooth_window=1,
        root_quat_smooth_window=1,
        arm_fix="left_from_right_tempered",
    ),
    "walk2_subject1.npz": SourceProcessSpec(joint_smooth_window=1, root_pos_smooth_window=1, root_quat_smooth_window=1),
    "fallAndGetUp1_subject1.npz": SourceProcessSpec(pop_repair_threshold=0.85, pop_repair_passes=2),
    "fallAndGetUp1_subject4.npz": SourceProcessSpec(pop_repair_threshold=0.90, pop_repair_passes=2),
    "fallAndGetUp2_subject2.npz": SourceProcessSpec(pop_repair_threshold=0.85, pop_repair_passes=2),
    "pushAndStumble1_subject2.npz": SourceProcessSpec(pop_repair_threshold=0.55, pop_repair_passes=2),
    "pushAndStumble1_subject5.npz": SourceProcessSpec(arm_fix="relaxed_bilateral_guarded", pop_repair_threshold=0.55, pop_repair_passes=2),
}


QUALITY_GATES: dict[str, float] = {
    "upright": 0.15,
    "upright_turn": 0.30,
    "walk_backward": 0.50,
    "walk_lateral": 0.50,
    "turn_walk": 0.50,
    "walk_transition": 0.35,
    "walk_forward": 0.50,
    "recovery": 0.45,
    "disturbance_recovery": 0.40,
}


OUTPUT_REPAIRS: dict[str, tuple[OutputRepairStep, ...]] = {
    "walk1_subject2_stand_to_walk_1332_1530_natural": (
        OutputRepairStep(kind="midpoint", joint_mode="all", alpha=0.50, radius=3, passes=3),
    ),
    "walk2_subject1_walk_normal_6540_6629": (
        OutputRepairStep(kind="midpoint", joint_mode="all", alpha=0.50, radius=3, passes=3),
    ),
    "walk2_subject1_walk_transition_6690_6870": (
        OutputRepairStep(kind="midpoint", joint_mode="all", alpha=0.50, radius=3, passes=3),
    ),
    "fallAndGetUp1_subject1": (
        OutputRepairStep(kind="jump", joint_mode="all", threshold=0.50, pad=1),
        OutputRepairStep(kind="midpoint", joint_mode="all", alpha=0.40, radius=3, passes=3),
    ),
    "fallAndGetUp1_subject4": (
        OutputRepairStep(kind="jump", joint_mode="all", threshold=0.48, pad=1),
        OutputRepairStep(kind="midpoint", joint_mode="all", alpha=0.40, radius=3, passes=3),
    ),
    "fallAndGetUp2_subject2": (
        OutputRepairStep(kind="jump", joint_mode="all", threshold=0.48, pad=1),
        OutputRepairStep(kind="midpoint", joint_mode="all", alpha=0.40, radius=3, passes=3),
    ),
    "fallAndGetUp2_subject2_recover_990_1340": (
        OutputRepairStep(kind="jump", joint_mode="all", threshold=0.48, pad=1),
        OutputRepairStep(kind="midpoint", joint_mode="all", alpha=0.40, radius=3, passes=3),
    ),
    "walk1_subject2_straight_walk_fast_2940_3015": (
        OutputRepairStep(kind="midpoint", joint_mode="all", alpha=0.50, radius=3, passes=3),
    ),
    "walk1_subject5_walk_rightturn_510_599": (
        OutputRepairStep(kind="midpoint", joint_mode="all", alpha=0.50, radius=3, passes=3),
    ),
    "walk1_subject5_walk_turnaround_thenwalk_5622_5711": (
        OutputRepairStep(kind="midpoint", joint_mode="all", alpha=0.50, radius=3, passes=3),
    ),
}

RECOVERY_STAGE_ENTRY_WINDOWS: dict[str, dict[str, tuple[tuple[int, int], ...]]] = {
    "fallAndGetUp1_subject1": {
        "stage1": (
            (482, 509),
            (736, 753),
        ),
        "stage2": (
            (437, 451),
            (1956, 1974),
        ),
    },
    "fallAndGetUp1_subject4": {
        "stage1": ((612, 632),),
        "stage2": ((718, 729),),
    },
    "fallAndGetUp2_subject2": {
        "stage0": ((396, 578),),
    },
    "fallAndGetUp2_subject2_recover_990_1340": {
        "stage0": ((284, 352),),
    },
}


def _flatten_stage_windows(
    stage_windows: dict[str, tuple[tuple[int, int], ...]],
) -> tuple[tuple[int, int], ...]:
    flattened: list[tuple[int, int]] = []
    for stage_name in ("stage0", "stage1", "stage2"):
        flattened.extend(stage_windows.get(stage_name, ()))
    return tuple(flattened)


RECOVERY_ENTRY_WINDOWS: dict[str, tuple[tuple[int, int], ...]] = {
    clip_name: _flatten_stage_windows(stage_windows)
    for clip_name, stage_windows in RECOVERY_STAGE_ENTRY_WINDOWS.items()
}

FORCE_PROMOTE_ALL_TO_AMP = True


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_source_dir() -> Path:
    return _repo_root() / "src" / "assets" / "motions" / "z1" / "retargeted-full"


def _default_curated_root() -> Path:
    return _repo_root() / "src" / "assets" / "motions" / "z1" / "curated"


def _default_trimmed_root() -> Path:
    return _repo_root() / "src" / "assets" / "motions" / "z1" / "trimmed_qpos"


def _default_amp_root() -> Path:
    return _default_curated_root() / "amp"


def _default_phase1_root() -> Path:
    return _default_curated_root() / "phase1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=_default_source_dir())
    parser.add_argument("--curated-root", type=Path, default=_default_curated_root())
    parser.add_argument("--trimmed-root", type=Path, default=_default_trimmed_root())
    parser.add_argument("--amp-root", type=Path, default=_default_amp_root())
    parser.add_argument("--phase1-root", type=Path, default=_default_phase1_root())
    parser.add_argument("--output-fps", type=int, default=50)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def _load_raw_motion(path: Path) -> tuple[np.ndarray, float]:
    data = np.load(path, allow_pickle=True)
    if "qpos" not in data:
        raise ValueError(f"{path} does not contain a qpos array.")
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    fps_value = data.get("fps", 30)
    fps = float(np.asarray(fps_value).reshape(-1)[0])
    return qpos, fps


def _odd(window: int) -> int:
    return window if window % 2 == 1 else window + 1


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = _odd(max(1, int(window)))
    if window <= 1 or values.shape[0] < 3:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def _smooth_columns(values: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(values)
    for idx in range(values.shape[1]):
        out[:, idx] = _smooth_1d(values[:, idx], window)
    return out


def _continuous_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = quat_wxyz.copy()
    for i in range(1, quat.shape[0]):
        if np.dot(quat[i - 1], quat[i]) < 0.0:
            quat[i] *= -1.0
    return quat


def _smooth_quat(quat_wxyz: np.ndarray, window: int) -> np.ndarray:
    quat = _continuous_quat(quat_wxyz)
    smoothed = _smooth_columns(quat, window)
    return normalize_quat(smoothed)


def _canonical_left_from_right(right_arm: np.ndarray) -> np.ndarray:
    """Approximate a natural left-arm swing from the right arm.

    The mapping is fitted from healthy walking clips where both arms move
    plausibly. It is used only for walk1_subject5, whose left arm is retargeted
    into a folded, nearly unusable configuration.
    """

    signs = np.asarray([1.0, -1.0, -1.0, 1.0, 1.0], dtype=np.float64)
    bias = np.asarray([0.0785, -0.1000, 0.2100, -0.1290, 0.0], dtype=np.float64)
    return right_arm * signs + bias


def _tempered_left_from_right(right_arm: np.ndarray) -> np.ndarray:
    """Build a restrained left-arm swing from the healthy right arm signal.

    The older canonical mapping produced a valid-but-exaggerated left arm on
    walk1_subject5. This version keeps the swing but recenters it around a
    healthier mean pose with lower lateral exaggeration.
    """

    target = np.asarray([0.30, 0.10, 0.12, 0.48, 0.0], dtype=np.float64)
    gains = np.asarray([0.70, -0.15, -0.08, 0.12, 1.0], dtype=np.float64)
    centered = right_arm - right_arm.mean(axis=0, keepdims=True)
    return target + centered * gains


def _relaxed_idle_arms(left_arm: np.ndarray, right_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower both arms into a neutral pose while preserving slight motion."""

    relaxed_left = np.asarray([0.31, 0.12, 0.18, 0.37, 0.0], dtype=np.float64)
    relaxed_right = np.asarray([0.21, -0.17, 0.02, 0.45, 0.0], dtype=np.float64)

    left_centered = left_arm - left_arm.mean(axis=0, keepdims=True)
    right_centered = right_arm - right_arm.mean(axis=0, keepdims=True)
    motion_gain = 0.15
    return (
        relaxed_left + motion_gain * left_centered,
        relaxed_right + motion_gain * right_centered,
    )


def _guarded_recovery_arms(left_arm: np.ndarray, right_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep both arms clear of the legs during recovery-style clips.

    This uses a slightly wider, higher guard pose than the idle relaxer so the
    retargeted forearms do not fold inward through the thighs during stumble
    and recovery motions.
    """

    guarded_left = np.asarray([0.34, 0.22, 0.10, 0.42, 0.0], dtype=np.float64)
    guarded_right = np.asarray([0.14, -0.28, -0.06, 0.50, 0.0], dtype=np.float64)

    left_centered = left_arm - left_arm.mean(axis=0, keepdims=True)
    right_centered = right_arm - right_arm.mean(axis=0, keepdims=True)
    motion_gain = 0.12
    return (
        guarded_left + motion_gain * left_centered,
        guarded_right + motion_gain * right_centered,
    )


def _repair_joint_pops(
    joints: np.ndarray,
    *,
    threshold: float,
    passes: int,
) -> tuple[np.ndarray, list[list[int]]]:
    repaired = joints.copy()
    touched_frames: set[int] = set()

    for _ in range(max(0, passes)):
        prev = repaired[:-2]
        curr = repaired[1:-1]
        nxt = repaired[2:]
        interp = 0.5 * (prev + nxt)
        dev = np.abs(curr - interp)
        jump_left = np.abs(curr - prev)
        jump_right = np.abs(nxt - curr)
        mask = (dev > threshold) & ((jump_left > threshold) | (jump_right > threshold))
        if not np.any(mask):
            continue
        frame_hits = np.where(np.any(mask, axis=1))[0]
        for frame_idx in frame_hits:
            touched_frames.add(int(frame_idx) + 1)
        curr = curr.copy()
        curr[mask] = interp[mask]
        repaired[1:-1] = curr

    if not touched_frames:
        return repaired, []

    grouped: list[list[int]] = []
    ordered = sorted(touched_frames)
    start = ordered[0]
    end = ordered[0]
    for frame_idx in ordered[1:]:
        if frame_idx <= end + 2:
            end = frame_idx
            continue
        grouped.append([max(0, start - 1), end + 1])
        start = frame_idx
        end = frame_idx
    grouped.append([max(0, start - 1), end + 1])
    return repaired, grouped


def _process_source_qpos(source_name: str, qpos: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    spec = SOURCE_PROCESS.get(source_name, SourceProcessSpec())
    processed = qpos.copy()
    left_slice = slice(7 + 14, 7 + 19)
    right_slice = slice(7 + 19, 7 + 24)
    repair_windows: list[list[int]] = []

    if spec.arm_fix == "left_from_right_canonical":
        right_arm = processed[:, right_slice]
        repaired_left = _canonical_left_from_right(right_arm)
        processed[:, left_slice] = repaired_left
    elif spec.arm_fix == "left_from_right_tempered":
        right_arm = processed[:, right_slice]
        repaired_left = _tempered_left_from_right(right_arm)
        processed[:, left_slice] = repaired_left
    elif spec.arm_fix == "relaxed_bilateral_idle":
        repaired_left, repaired_right = _relaxed_idle_arms(processed[:, left_slice], processed[:, right_slice])
        processed[:, left_slice] = repaired_left
        processed[:, right_slice] = repaired_right
    elif spec.arm_fix == "relaxed_bilateral_guarded":
        repaired_left, repaired_right = _guarded_recovery_arms(processed[:, left_slice], processed[:, right_slice])
        processed[:, left_slice] = repaired_left
        processed[:, right_slice] = repaired_right

    if spec.pop_repair_threshold is not None and spec.pop_repair_passes > 0:
        repaired_joints, repair_windows = _repair_joint_pops(
            processed[:, 7:],
            threshold=spec.pop_repair_threshold,
            passes=spec.pop_repair_passes,
        )
        processed[:, 7:] = repaired_joints

    if spec.root_pos_smooth_window > 1:
        processed[:, :3] = _smooth_columns(processed[:, :3], spec.root_pos_smooth_window)
    if spec.root_quat_smooth_window > 1:
        processed[:, 3:7] = _smooth_quat(processed[:, 3:7], spec.root_quat_smooth_window)
    else:
        processed[:, 3:7] = normalize_quat(_continuous_quat(processed[:, 3:7]))
    if spec.joint_smooth_window > 1:
        processed[:, 7:] = _smooth_columns(processed[:, 7:], spec.joint_smooth_window)

    process_report = {
        "repair_windows": repair_windows,
        "process_spec": asdict(spec),
    }
    return processed, process_report


def _build_qpos_clip(qpos: np.ndarray, spec: ClipSpec) -> np.ndarray:
    if spec.clip_kind == "full":
        clip = qpos.copy()
    elif spec.clip_kind == "range":
        assert spec.start_frame is not None
        assert spec.end_frame is not None
        clip = qpos[spec.start_frame : spec.end_frame + 1].copy()
    elif spec.clip_kind == "static":
        assert spec.static_frame is not None
        assert spec.static_length is not None
        frame = qpos[spec.static_frame : spec.static_frame + 1].copy()
        clip = np.repeat(frame, spec.static_length, axis=0)
    else:
        raise ValueError(f"Unsupported clip kind: {spec.clip_kind}")

    if clip.shape[0] < 2:
        raise ValueError(f"{spec.output_name} produced fewer than 2 frames.")
    return clip


def _write_qpos_clip(path: Path, qpos: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        qpos=qpos.astype(np.float32),
        fps=np.asarray([fps], dtype=np.float32),
    )


def _compile_vel_clip(
    *,
    qpos_file: Path,
    output_file: Path,
    input_fps: float,
    output_fps: int,
) -> None:
    data = np.load(qpos_file, allow_pickle=True)
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    compile_qpos_to_amp(
        qpos=qpos,
        input_fps=float(input_fps),
        output_fps=output_fps,
        output_file=output_file,
        xml_file=default_xml(),
    )


def _compile_output_qpos(
    *,
    qpos: np.ndarray,
    output_file: Path,
    output_fps: int,
) -> None:
    pack_qpos_to_amp(
        qpos=qpos,
        output_fps=output_fps,
        output_file=output_file,
        xml_file=default_xml(),
    )


def _landing_joint_indices(joint_mode: str, foot_idx: int) -> np.ndarray:
    if joint_mode == "distal":
        joint_map = {0: (3, 4, 5), 1: (9, 10, 11)}
    elif joint_mode == "leg":
        joint_map = {0: (0, 1, 2, 3, 4, 5), 1: (6, 7, 8, 9, 10, 11)}
    elif joint_mode == "leg_waist":
        joint_map = {0: (0, 1, 2, 3, 4, 5, 12, 13), 1: (6, 7, 8, 9, 10, 11, 12, 13)}
    elif joint_mode == "all":
        joint_map = {0: tuple(range(24)), 1: tuple(range(24))}
    else:
        raise ValueError(f"Unsupported landing repair joint mode: {joint_mode}")
    return np.asarray(joint_map[foot_idx], dtype=np.int64) + 7


def _repair_landing_interp(
    qpos: np.ndarray,
    *,
    landing_frames: tuple[np.ndarray, ...],
    step: OutputRepairStep,
) -> np.ndarray:
    repaired = qpos.copy()
    for _ in range(max(1, step.passes)):
        for foot_idx, frames in enumerate(landing_frames):
            joint_indices = _landing_joint_indices(step.joint_mode, foot_idx)
            for frame_idx in frames:
                start = max(0, int(frame_idx) - step.radius)
                end = min(repaired.shape[0] - 1, int(frame_idx) + step.radius)
                if end - start < 2:
                    continue
                segment = repaired[start : end + 1, joint_indices]
                trend = np.linspace(segment[0], segment[-1], segment.shape[0])
                repaired[start : end + 1, joint_indices] = (1.0 - step.alpha) * segment + step.alpha * trend
    return repaired


def _repair_landing_midpoint(
    qpos: np.ndarray,
    *,
    landing_frames: tuple[np.ndarray, ...],
    step: OutputRepairStep,
) -> np.ndarray:
    repaired = qpos.copy()
    for _ in range(max(1, step.passes)):
        for foot_idx, frames in enumerate(landing_frames):
            joint_indices = _landing_joint_indices(step.joint_mode, foot_idx)
            for frame_idx in frames:
                start = max(1, int(frame_idx) - step.radius)
                end = min(repaired.shape[0] - 2, int(frame_idx) + step.radius)
                for idx in range(start, end + 1):
                    target = 0.5 * (repaired[idx - 1, joint_indices] + repaired[idx + 1, joint_indices])
                    repaired[idx, joint_indices] = (1.0 - step.alpha) * repaired[idx, joint_indices] + step.alpha * target
    return repaired


def _jump_repair_indices(step: OutputRepairStep) -> slice:
    if step.joint_mode == "all":
        return slice(7, None)
    if step.joint_mode == "legs":
        return slice(7, 21)
    raise ValueError(f"Unsupported jump repair joint mode: {step.joint_mode}")


def _repair_jump_frames(
    qpos: np.ndarray,
    *,
    step: OutputRepairStep,
) -> np.ndarray:
    if step.threshold is None:
        raise ValueError("Jump repair requires a threshold.")
    repaired = qpos.copy()
    cols = _jump_repair_indices(step)
    joint_delta = np.abs(np.diff(repaired[:, cols], axis=0))
    hit_frames = np.argwhere(joint_delta > step.threshold)
    ranked_hits = sorted(
        ((float(joint_delta[frame_idx, joint_idx]), int(frame_idx), int(joint_idx)) for frame_idx, joint_idx in hit_frames),
        reverse=True,
    )
    touched: set[tuple[int, int]] = set()
    col_offset = 7 if cols.start is None else cols.start
    for _, frame_idx, joint_idx in ranked_hits:
        if (frame_idx, joint_idx) in touched:
            continue
        start = max(0, frame_idx - step.pad)
        end = min(repaired.shape[0] - 1, frame_idx + 1 + step.pad)
        if end - start < 2:
            continue
        col_idx = col_offset + joint_idx
        repaired[start : end + 1, col_idx] = np.linspace(
            repaired[start, col_idx],
            repaired[end, col_idx],
            end - start + 1,
        )
        for local_frame in range(start, end):
            touched.add((local_frame, joint_idx))
    return repaired


def _apply_output_repairs(
    *,
    qpos: np.ndarray,
    clip_name: str,
    quality_evaluator: MotionQualityEvaluator,
    output_fps: int,
) -> tuple[np.ndarray, MotionQualityMetrics, dict[str, object]]:
    repaired_qpos = qpos.copy()
    quality_before = quality_evaluator.evaluate(repaired_qpos, output_fps)
    repair_steps = OUTPUT_REPAIRS.get(clip_name, ())
    repair_report: dict[str, object] = {
        "output_fps": output_fps,
        "repair_applied": bool(repair_steps),
        "quality_before_repair": quality_before.to_dict(),
        "steps": [],
    }

    if not repair_steps:
        return repaired_qpos, quality_before, repair_report

    landing_frames_cache: tuple[np.ndarray, ...] | None = None
    for step in repair_steps:
        if step.kind in {"midpoint", "interp"}:
            if landing_frames_cache is None:
                landing_frames_cache = quality_evaluator.landing_frames(repaired_qpos)
            if step.kind == "midpoint":
                repaired_qpos = _repair_landing_midpoint(
                    repaired_qpos,
                    landing_frames=landing_frames_cache,
                    step=step,
                )
            else:
                repaired_qpos = _repair_landing_interp(
                    repaired_qpos,
                    landing_frames=landing_frames_cache,
                    step=step,
                )
            step_report = {
                "step": asdict(step),
                "landing_frames": [frames.tolist() for frames in landing_frames_cache],
            }
        elif step.kind == "jump":
            repaired_qpos = _repair_jump_frames(
                repaired_qpos,
                step=step,
            )
            step_report = {"step": asdict(step)}
        else:
            raise ValueError(f"Unsupported output repair kind: {step.kind}")
        step_report["quality_after_step"] = quality_evaluator.evaluate(repaired_qpos, output_fps).to_dict()
        repair_report["steps"].append(step_report)

    quality_after = quality_evaluator.evaluate(repaired_qpos, output_fps)
    repair_report["quality_after_repair"] = quality_after.to_dict()
    return repaired_qpos, quality_after, repair_report


def _prepare_output_qpos(
    *,
    clip_qpos: np.ndarray,
    input_fps: float,
    output_fps: int,
    clip_name: str,
    quality_evaluator: MotionQualityEvaluator,
) -> tuple[np.ndarray, MotionQualityMetrics, dict[str, object]]:
    output_qpos = resample_qpos(clip_qpos, input_fps=float(input_fps), output_fps=float(output_fps))
    return _apply_output_repairs(
        qpos=output_qpos,
        clip_name=clip_name,
        quality_evaluator=quality_evaluator,
        output_fps=output_fps,
    )


def _quality_gate_result(semantic_label: str, max_mean_stance_slip: float) -> dict[str, object]:
    threshold = QUALITY_GATES[semantic_label]
    passed = max_mean_stance_slip <= threshold
    return {
        "semantic_label": semantic_label,
        "stance_slip_threshold": threshold,
        "passed": passed,
        "reason": (
            f"max_mean_stance_slip {max_mean_stance_slip:.4f} <= {threshold:.4f}"
            if passed
            else f"max_mean_stance_slip {max_mean_stance_slip:.4f} > {threshold:.4f}"
        ),
    }


def _promotion_result(
    *,
    quality_gate: dict[str, object],
) -> dict[str, object]:
    forced = FORCE_PROMOTE_ALL_TO_AMP and not bool(quality_gate["passed"])
    promoted = bool(quality_gate["passed"]) or forced
    return {
        "promoted_to_amp": promoted,
        "forced": forced,
        "reason": (
            "passed local quality gate"
            if bool(quality_gate["passed"])
            else "forced promotion override enabled for sim-stage training"
        ),
    }


def _clean_outputs(trimmed_root: Path, curated_root: Path, amp_root: Path, phase1_root: Path) -> None:
    legacy_trimmed_root = curated_root / "trimmed_qpos"
    for target in (
        trimmed_root,
        legacy_trimmed_root,
        curated_root / "full_length_vel",
        amp_root,
        phase1_root,
        curated_root / "deferred",
    ):
        if target.exists():
            shutil.rmtree(target)
    manifest = curated_root / "library_manifest.json"
    if manifest.exists():
        manifest.unlink()


def _compile_full_length_sources(
    *,
    source_dir: Path,
    full_length_dir: Path,
    output_fps: int,
    quality_evaluator: MotionQualityEvaluator,
) -> list[dict[str, object]]:
    compiled_entries: list[dict[str, object]] = []
    full_length_dir.mkdir(parents=True, exist_ok=True)

    for source_file in sorted(source_dir.glob("*.npz")):
        raw_qpos, raw_fps = _load_raw_motion(source_file)
        processed_qpos, process_report = _process_source_qpos(source_file.name, raw_qpos)
        qpos_copy = full_length_dir / "_qpos_cache" / source_file.name
        output_vel = full_length_dir / f"{source_file.stem}_vel.npz"
        _write_qpos_clip(qpos_copy, processed_qpos, raw_fps)
        _compile_vel_clip(
            qpos_file=qpos_copy,
            output_file=output_vel,
            input_fps=raw_fps,
            output_fps=output_fps,
        )
        compiled_entries.append(
            {
                "source_file": str(source_file),
                "output_vel_file": str(output_vel),
                "input_fps": raw_fps,
                "num_frames": int(processed_qpos.shape[0]),
                "process_spec": process_report["process_spec"],
                "repair_windows": process_report["repair_windows"],
                "source_quality_raw": quality_evaluator.evaluate(raw_qpos, raw_fps).to_dict(),
                "source_quality_processed": quality_evaluator.evaluate(processed_qpos, raw_fps).to_dict(),
            }
        )

    cache_dir = full_length_dir / "_qpos_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    return compiled_entries


def build_library(args: argparse.Namespace) -> Path:
    source_dir = args.source_dir.resolve()
    curated_root = args.curated_root.resolve()
    trimmed_root = args.trimmed_root.resolve()
    amp_root = args.amp_root.resolve()
    phase1_root = args.phase1_root.resolve()
    full_length_dir = curated_root / "full_length_vel"
    quality_evaluator = MotionQualityEvaluator(xml_file=default_xml())

    if args.clean:
        _clean_outputs(trimmed_root, curated_root, amp_root, phase1_root)
    curated_root.mkdir(parents=True, exist_ok=True)
    amp_root.mkdir(parents=True, exist_ok=True)
    phase1_root.mkdir(parents=True, exist_ok=True)
    trimmed_root.mkdir(parents=True, exist_ok=True)

    full_length_entries = _compile_full_length_sources(
        source_dir=source_dir,
        full_length_dir=full_length_dir,
        output_fps=args.output_fps,
        quality_evaluator=quality_evaluator,
    )

    manifest: dict[str, object] = {
        "source_dir": str(source_dir),
        "curated_root": str(curated_root),
        "trimmed_qpos_root": str(trimmed_root),
        "active_amp_root": str(amp_root),
        "phase1_amp_root": str(phase1_root),
        "full_length_review_root": str(full_length_dir),
        "compiler_xml": str(default_xml()),
        "output_fps": args.output_fps,
        "force_promote_all_to_amp": FORCE_PROMOTE_ALL_TO_AMP,
        "active_clips": [],
        "phase1_clips": [],
        "deferred_clips": [],
        "deferred_candidates": [asdict(spec) for spec in DEFERRED_CANDIDATES],
        "full_length_converted": full_length_entries,
    }

    for spec in ACTIVE_CLIPS:
        source_file = source_dir / spec.source_file
        if not source_file.exists():
            raise FileNotFoundError(f"Missing source clip: {source_file}")

        raw_qpos, raw_fps = _load_raw_motion(source_file)
        processed_qpos, process_report = _process_source_qpos(spec.source_file, raw_qpos)
        clip_qpos = _build_qpos_clip(processed_qpos, spec)
        output_qpos, output_quality_metrics, landing_repair_report = _prepare_output_qpos(
            clip_qpos=clip_qpos,
            input_fps=raw_fps,
            output_fps=args.output_fps,
            clip_name=spec.output_name,
            quality_evaluator=quality_evaluator,
        )

        trimmed_qpos_file = trimmed_root / spec.output_group / f"{spec.output_name}.npz"
        _write_qpos_clip(trimmed_qpos_file, clip_qpos, raw_fps)
        quality_gate = _quality_gate_result(spec.semantic_label, output_quality_metrics.max_mean_stance_slip)
        promotion = _promotion_result(quality_gate=quality_gate)
        output_root = amp_root
        output_vel_file = output_root / spec.output_group / f"{spec.output_name}_vel.npz"
        phase1_output_vel_file = None
        if "phase1" in spec.training_sets:
            phase1_output_vel_file = phase1_root / spec.output_group / f"{spec.output_name}_vel.npz"

        entry = {
            **asdict(spec),
            "source_path": str(source_file),
            "trimmed_qpos_file": str(trimmed_qpos_file),
            "output_vel_file": str(output_vel_file),
            "phase1_output_vel_file": str(phase1_output_vel_file) if phase1_output_vel_file is not None else None,
            "input_fps": raw_fps,
            "output_fps": args.output_fps,
            "num_source_frames": int(raw_qpos.shape[0]),
            "num_clip_frames": int(clip_qpos.shape[0]),
            "num_output_frames": int(output_qpos.shape[0]),
            "process_spec": process_report["process_spec"],
            "repair_windows": process_report["repair_windows"],
            "quality_metrics": output_quality_metrics.to_dict(),
            "landing_repair": landing_repair_report,
            "recovery_entry_windows": [
                [int(start), int(end)] for start, end in RECOVERY_ENTRY_WINDOWS.get(spec.output_name, ())
            ],
            "recovery_stage0_entry_windows": [
                [int(start), int(end)]
                for start, end in RECOVERY_STAGE_ENTRY_WINDOWS.get(spec.output_name, {}).get("stage0", ())
            ],
            "recovery_stage1_entry_windows": [
                [int(start), int(end)]
                for start, end in RECOVERY_STAGE_ENTRY_WINDOWS.get(spec.output_name, {}).get("stage1", ())
            ],
            "recovery_stage2_entry_windows": [
                [int(start), int(end)]
                for start, end in RECOVERY_STAGE_ENTRY_WINDOWS.get(spec.output_name, {}).get("stage2", ())
            ],
            "quality_gate": quality_gate,
            "promotion": promotion,
        }

        output_vel_file.parent.mkdir(parents=True, exist_ok=True)
        _compile_output_qpos(
            qpos=output_qpos,
            output_file=output_vel_file,
            output_fps=args.output_fps,
        )
        if phase1_output_vel_file is not None:
            phase1_output_vel_file.parent.mkdir(parents=True, exist_ok=True)
            _compile_output_qpos(
                qpos=output_qpos,
                output_file=phase1_output_vel_file,
                output_fps=args.output_fps,
            )

        if promotion["promoted_to_amp"]:
            manifest["active_clips"].append(entry)
            if phase1_output_vel_file is not None:
                manifest["phase1_clips"].append(entry)
            if not bool(quality_gate["passed"]):
                manifest["deferred_clips"].append(entry)
        else:
            manifest["deferred_clips"].append(entry)

    manifest_path = curated_root / "library_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="ascii")
    return manifest_path


def main() -> None:
    args = parse_args()
    manifest_path = build_library(args)
    print(f"Wrote library manifest: {manifest_path}")
    print(f"Trimmed qpos root: {args.trimmed_root.resolve()}")
    print(f"Full-length AMP root: {(args.curated_root / 'full_length_vel').resolve()}")
    print(f"Active AMP library root: {args.amp_root.resolve()}")
    print(f"Phase 1 AMP library root: {args.phase1_root.resolve()}")


if __name__ == "__main__":
    main()
