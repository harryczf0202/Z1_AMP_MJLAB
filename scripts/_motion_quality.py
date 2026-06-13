from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass(frozen=True)
class MotionQualityMetrics:
    fps: float
    num_frames: int
    left_stance_mean_slip: float
    right_stance_mean_slip: float
    left_stance_max_slip: float
    right_stance_max_slip: float
    left_stance_samples: int
    right_stance_samples: int
    max_mean_stance_slip: float
    max_joint_delta: float
    p99_joint_delta: float
    p999_joint_delta: float
    root_z_min: float
    root_z_max: float
    left_landing_count: int
    right_landing_count: int
    landing_oscillation_p99: float
    landing_oscillation_max: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class MotionQualityEvaluator:
    def __init__(
        self,
        *,
        xml_file: Path,
        foot_site_names: tuple[str, str] = ("l_foot_center", "r_foot_center"),
        stance_height_threshold: float = 0.045,
    ) -> None:
        self._model = mujoco.MjModel.from_xml_path(str(xml_file))
        self._data = mujoco.MjData(self._model)
        self._stance_height_threshold = stance_height_threshold
        self._site_ids = tuple(
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            for site_name in foot_site_names
        )

    def site_positions(self, qpos: np.ndarray) -> np.ndarray:
        site_positions = np.zeros((qpos.shape[0], len(self._site_ids), 3), dtype=np.float64)
        for frame_idx in range(qpos.shape[0]):
            self._data.qpos[:] = qpos[frame_idx]
            mujoco.mj_forward(self._model, self._data)
            for site_idx, mujoco_site_id in enumerate(self._site_ids):
                site_positions[frame_idx, site_idx] = self._data.site_xpos[mujoco_site_id]
        return site_positions

    def landing_frames(self, qpos: np.ndarray) -> tuple[np.ndarray, ...]:
        site_positions = self.site_positions(qpos)
        contact_mask = site_positions[:, :, 2] < self._stance_height_threshold
        return tuple(
            np.where(contact_mask[1:, foot_idx] & ~contact_mask[:-1, foot_idx])[0] + 1
            for foot_idx in range(len(self._site_ids))
        )

    def evaluate(self, qpos: np.ndarray, fps: float) -> MotionQualityMetrics:
        site_positions = self.site_positions(qpos)
        foot_xy_velocity = np.linalg.norm(np.diff(site_positions[:, :, :2], axis=0), axis=2) * fps
        stance_height = site_positions[:-1, :, 2]
        contact_mask = site_positions[:, :, 2] < self._stance_height_threshold

        stance_stats: list[tuple[float, float, int]] = []
        for foot_idx in range(len(self._site_ids)):
            stance_mask = stance_height[:, foot_idx] < self._stance_height_threshold
            if np.any(stance_mask):
                slip_values = foot_xy_velocity[stance_mask, foot_idx]
                stance_stats.append((float(slip_values.mean()), float(slip_values.max()), int(slip_values.size)))
            else:
                stance_stats.append((0.0, 0.0, 0))

        joint_delta = np.abs(np.diff(qpos[:, 7:], axis=0))
        max_joint_delta = float(joint_delta.max()) if joint_delta.size else 0.0
        p99_joint_delta = float(np.percentile(joint_delta, 99)) if joint_delta.size else 0.0
        p999_joint_delta = float(np.percentile(joint_delta, 99.9)) if joint_delta.size else 0.0

        landing_frames = tuple(
            np.where(contact_mask[1:, foot_idx] & ~contact_mask[:-1, foot_idx])[0] + 1
            for foot_idx in range(len(self._site_ids))
        )
        joint_second_diff = np.abs(qpos[:-2, 7:] - 2.0 * qpos[1:-1, 7:] + qpos[2:, 7:])
        landing_oscillation_p99 = 0.0
        landing_oscillation_max = 0.0
        if joint_second_diff.size:
            landing_mask = np.zeros(joint_second_diff.shape[0], dtype=bool)
            for frames in landing_frames:
                for frame_idx in frames:
                    start = max(0, int(frame_idx) - 2)
                    end = min(joint_second_diff.shape[0], int(frame_idx) + 3)
                    landing_mask[start:end] = True
            if np.any(landing_mask):
                landing_values = joint_second_diff[landing_mask]
                landing_oscillation_p99 = float(np.percentile(landing_values, 99))
                landing_oscillation_max = float(landing_values.max())

        return MotionQualityMetrics(
            fps=float(fps),
            num_frames=int(qpos.shape[0]),
            left_stance_mean_slip=stance_stats[0][0],
            right_stance_mean_slip=stance_stats[1][0],
            left_stance_max_slip=stance_stats[0][1],
            right_stance_max_slip=stance_stats[1][1],
            left_stance_samples=stance_stats[0][2],
            right_stance_samples=stance_stats[1][2],
            max_mean_stance_slip=max(stance_stats[0][0], stance_stats[1][0]),
            max_joint_delta=max_joint_delta,
            p99_joint_delta=p99_joint_delta,
            p999_joint_delta=p999_joint_delta,
            root_z_min=float(qpos[:, 2].min()),
            root_z_max=float(qpos[:, 2].max()),
            left_landing_count=int(landing_frames[0].size),
            right_landing_count=int(landing_frames[1].size),
            landing_oscillation_p99=landing_oscillation_p99,
            landing_oscillation_max=landing_oscillation_max,
        )
