from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms


@dataclass
class BodyAMPMotion:
    amp_obs: torch.Tensor
    next_amp_obs: torch.Tensor
    reset_state: torch.Tensor
    num_frames: int


class BodyAMPLoader:
    """Expert AMP loader for body-based motion features from compiled `_vel.npz` files."""

    def __init__(
        self,
        motion_dir: str | Path,
        *,
        body_names: Sequence[str],
        anchor_name: str,
        joint_names: Sequence[str] | None = None,
        device: str = "cpu",
        group_weights: dict[str, float] | None = None,
    ) -> None:
        self.motion_dir = Path(motion_dir)
        self.body_names = tuple(body_names)
        self.anchor_name = anchor_name
        self.joint_names = tuple(joint_names) if joint_names is not None else None
        self.device = device
        self._group_weights = group_weights or {}

        motions, motion_paths = self._load_all_motions()
        if not motions:
            raise FileNotFoundError(f"No .npz AMP motions found in {self.motion_dir}")

        self.all_obs = torch.cat([motion.amp_obs for motion in motions], dim=0)
        self.all_next_obs = torch.cat([motion.next_amp_obs for motion in motions], dim=0)
        self.all_states = torch.cat([motion.reset_state for motion in motions], dim=0)

        # Build per-frame weights with optional group scaling.
        per_frame_weights_list: list[torch.Tensor] = []
        for motion, motion_path in zip(motions, motion_paths):
            group_name = self._resolve_group_name(motion_path)
            group_scale = self._group_weights.get(group_name, 1.0)
            clip_weight = group_scale / motion.num_frames
            per_frame_weights_list.append(
                torch.full((motion.num_frames,), clip_weight, device=self.device)
            )
        per_frame = torch.cat(per_frame_weights_list)
        self.per_frame_weights = per_frame / per_frame.sum()
        self.per_frame_weights_reset = self.per_frame_weights.clone()

        sample_state = motions[0].reset_state
        self._reset_dims = [4, sample_state.shape[1] // 2 - 5, sample_state.shape[1] // 2 - 5, 3, 3]
        # overwrite with correct joint dim after construction
        joint_dim = (sample_state.shape[1] - 10) // 2
        self._reset_dims = [4, joint_dim, joint_dim, 3, 3]

    def _load_all_motions(self) -> tuple[list[BodyAMPMotion], list[Path]]:
        motions: list[BodyAMPMotion] = []
        paths: list[Path] = []
        for motion_path in sorted(self.motion_dir.rglob("*.npz")):
            motions.append(self._load_motion(motion_path))
            paths.append(motion_path)
        return motions, paths

    @staticmethod
    def _resolve_group_name(motion_path: Path) -> str:
        """Return the immediate parent directory name as the group (e.g. 'Walk')."""
        return motion_path.parent.name

    def _load_motion(self, motion_path: Path) -> BodyAMPMotion:
        data = np.load(motion_path, allow_pickle=True)

        joint_pos = np.asarray(data["joint_pos"])
        joint_vel = np.asarray(data["joint_vel"])
        body_pos_w = np.asarray(data["body_pos_w"])
        body_quat_w = np.asarray(data["body_quat_w"])
        body_lin_vel_w = np.asarray(data["body_lin_vel_w"])
        body_ang_vel_w = np.asarray(data["body_ang_vel_w"])

        motion_joint_names = tuple(data["joint_names"].tolist()) if "joint_names" in data else ()
        motion_body_names = tuple(data["body_names"].tolist()) if "body_names" in data else ()

        if motion_joint_names and joint_pos.shape[-1] > len(motion_joint_names):
            joint_pos = joint_pos[:, -len(motion_joint_names):]
        if motion_joint_names and joint_vel.shape[-1] > len(motion_joint_names):
            joint_vel = joint_vel[:, -len(motion_joint_names):]

        if motion_body_names and motion_body_names[0] == "world":
            motion_body_names = motion_body_names[1:]
            body_pos_w = body_pos_w[:, 1:, :]
            body_quat_w = body_quat_w[:, 1:, :]
            body_lin_vel_w = body_lin_vel_w[:, 1:, :]
            body_ang_vel_w = body_ang_vel_w[:, 1:, :]

        if self.joint_names and motion_joint_names:
            joint_index = [motion_joint_names.index(name) for name in self.joint_names]
            joint_pos = joint_pos[:, joint_index]
            joint_vel = joint_vel[:, joint_index]

        body_index = [motion_body_names.index(name) for name in self.body_names]
        anchor_index = motion_body_names.index(self.anchor_name)

        amp_obs = self._build_amp_obs(
            body_pos_w=torch.tensor(body_pos_w[:, body_index], dtype=torch.float32, device=self.device),
            body_quat_w=torch.tensor(body_quat_w[:, body_index], dtype=torch.float32, device=self.device),
            body_lin_vel_w=torch.tensor(body_lin_vel_w[:, body_index], dtype=torch.float32, device=self.device),
            body_ang_vel_w=torch.tensor(body_ang_vel_w[:, body_index], dtype=torch.float32, device=self.device),
            anchor_pos_w=torch.tensor(body_pos_w[:, anchor_index], dtype=torch.float32, device=self.device),
            anchor_quat_w=torch.tensor(body_quat_w[:, anchor_index], dtype=torch.float32, device=self.device),
        )

        next_index = torch.clamp(torch.arange(amp_obs.shape[0], device=self.device) + 1, max=amp_obs.shape[0] - 1)
        next_amp_obs = amp_obs[next_index]

        root_quat = torch.tensor(body_quat_w[:, 0], dtype=torch.float32, device=self.device)
        root_lin_vel = torch.tensor(body_lin_vel_w[:, 0], dtype=torch.float32, device=self.device)
        root_ang_vel = torch.tensor(body_ang_vel_w[:, 0], dtype=torch.float32, device=self.device)
        joint_pos_t = torch.tensor(joint_pos, dtype=torch.float32, device=self.device)
        joint_vel_t = torch.tensor(joint_vel, dtype=torch.float32, device=self.device)
        reset_state = torch.cat([root_quat, joint_pos_t, joint_vel_t, root_lin_vel, root_ang_vel], dim=-1)

        return BodyAMPMotion(
            amp_obs=amp_obs,
            next_amp_obs=next_amp_obs,
            reset_state=reset_state,
            num_frames=amp_obs.shape[0],
        )

    def _build_amp_obs(
        self,
        *,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        body_lin_vel_w: torch.Tensor,
        body_ang_vel_w: torch.Tensor,
        anchor_pos_w: torch.Tensor,
        anchor_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        num_frames, num_bodies = body_pos_w.shape[:2]

        pos_b, ori_b = subtract_frame_transforms(
            anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
            anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
            body_pos_w,
            body_quat_w,
        )
        ori_mat = matrix_from_quat(ori_b)
        ori_6d = ori_mat[..., :2].reshape(num_frames, -1)

        body_lin_vel_b = quat_apply_inverse(
            body_quat_w.reshape(-1, 4),
            body_lin_vel_w.reshape(-1, 3),
        ).reshape(num_frames, num_bodies, 3)
        body_ang_vel_b = quat_apply_inverse(
            body_quat_w.reshape(-1, 4),
            body_ang_vel_w.reshape(-1, 3),
        ).reshape(num_frames, num_bodies, 3)

        return torch.cat(
            [
                pos_b.reshape(num_frames, -1),
                ori_6d,
                body_lin_vel_b.reshape(num_frames, -1),
                body_ang_vel_b.reshape(num_frames, -1),
            ],
            dim=-1,
        )

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        for _ in range(num_mini_batch):
            idx = torch.multinomial(self.per_frame_weights, mini_batch_size, replacement=True)
            yield self.all_obs[idx], self.all_next_obs[idx]

    def get_state_for_reset(self, number_of_samples: int):
        idx = torch.multinomial(self.per_frame_weights_reset, number_of_samples, replacement=True)
        full = self.all_states[idx]
        return torch.split(full, self._reset_dims, dim=1)
