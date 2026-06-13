from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch


class MotionLoader:
    def __init__(
        self,
        motion_dir: str,
        tgt_body_indexes: Sequence[int],
        tgt_anchor_indexes: int,
        feet_indexes: int,
        device: str = "cpu",
    ):
        self.motion_data: list[dict] = self._load_dir(motion_dir, device)
        assert len(self.motion_data) > 0, f"No npz files found in: {motion_dir}"

        self.motion_names = [m["motion_name"] for m in self.motion_data]
        if not self.motion_data:
            raise ValueError(f"No motion data loaded from: {motion_dir}")

        default_motion = self.motion_data[0]
        self.fps = default_motion["fps"]
        self._dof_pos = default_motion["dof_pos"]
        self._dof_vel = default_motion["dof_vel"]
        self._body_pos_w = default_motion["body_pos_w"]
        self._body_quat_w = default_motion["body_quat_w"]
        self._body_lin_vel_w = default_motion["body_lin_vel_w"]
        self._body_ang_vel_w = default_motion["body_ang_vel_w"]

        self._body_indexes = tgt_body_indexes
        self._anchor_indexes = tgt_anchor_indexes
        self._feet_indexes = feet_indexes
        self.time_step_total = self._dof_pos.shape[0]
        self.motion_total_time = self.time_step_total / self.fps

    @staticmethod
    def _load_dir(dir_path: str, device: str) -> list[dict]:
        """Load AMP motions and normalize them to the simulator convention.

        The curated Z1 files store:
        - `joint_pos` / `joint_vel` with floating-base state included
        - body arrays with a leading `world` entry

        AMP reset code only needs actuated joint state because root pose and
        velocity are read from the body tensors. We therefore strip the extra
        root-state dimensions and remove the synthetic world body.
        """

        assert os.path.isdir(dir_path), f"Not a directory: {dir_path}"
        result = []
        for filename in sorted(os.listdir(dir_path)):
            if not filename.endswith(".npz"):
                continue

            motion_name = os.path.splitext(filename)[0]
            data = np.load(os.path.join(dir_path, filename), allow_pickle=True)

            joint_pos = np.asarray(data["joint_pos"])
            joint_vel = np.asarray(data["joint_vel"])
            body_pos_w = np.asarray(data["body_pos_w"])
            body_quat_w = np.asarray(data["body_quat_w"])
            body_lin_vel_w = np.asarray(data["body_lin_vel_w"])
            body_ang_vel_w = np.asarray(data["body_ang_vel_w"])

            joint_names = tuple(data["joint_names"].tolist()) if "joint_names" in data else ()
            body_names = tuple(data["body_names"].tolist()) if "body_names" in data else ()

            if joint_names:
                num_actuated = len(joint_names)
                if joint_pos.shape[-1] > num_actuated:
                    joint_pos = joint_pos[:, -num_actuated:]
                if joint_vel.shape[-1] > num_actuated:
                    joint_vel = joint_vel[:, -num_actuated:]

            if body_names and body_names[0] == "world":
                body_names = body_names[1:]
                body_pos_w = body_pos_w[:, 1:, :]
                body_quat_w = body_quat_w[:, 1:, :]
                body_lin_vel_w = body_lin_vel_w[:, 1:, :]
                body_ang_vel_w = body_ang_vel_w[:, 1:, :]

            result.append(
                {
                    "motion_name": motion_name,
                    "fps": float(np.asarray(data["fps"]).reshape(-1)[0]),
                    "joint_names": joint_names,
                    "body_names": body_names,
                    "dof_pos": torch.tensor(joint_pos, dtype=torch.float32, device=device),
                    "dof_vel": torch.tensor(joint_vel, dtype=torch.float32, device=device),
                    "body_pos_w": torch.tensor(body_pos_w, dtype=torch.float32, device=device),
                    "body_quat_w": torch.tensor(body_quat_w, dtype=torch.float32, device=device),
                    "body_lin_vel_w": torch.tensor(body_lin_vel_w, dtype=torch.float32, device=device),
                    "body_ang_vel_w": torch.tensor(body_ang_vel_w, dtype=torch.float32, device=device),
                }
            )
        return result

    def _get_motion_data(self, motion_index: int = None):
        if motion_index is None:
            return {
                "body_pos_w": self._body_pos_w,
                "body_quat_w": self._body_quat_w,
                "body_lin_vel_w": self._body_lin_vel_w,
                "body_ang_vel_w": self._body_ang_vel_w,
                "dof_pos": self._dof_pos,
                "dof_vel": self._dof_vel,
            }
        assert 0 <= motion_index < len(self.motion_data), (
            f"Motion index {motion_index} out of range [0, {len(self.motion_data)})"
        )
        return self.motion_data[motion_index]

    def tgt_body_pos_w(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_pos_w"][:, self._body_indexes, :]

    def tgt_body_quat_w(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_quat_w"][:, self._body_indexes, :]

    def tgt_body_lin_vel_w(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_lin_vel_w"][:, self._body_indexes, :]

    def tgt_body_ang_vel_w(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_ang_vel_w"][:, self._body_indexes, :]

    def tgt_anchor_pos_w(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_pos_w"][:, self._anchor_indexes]

    def tgt_anchor_quat_w(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_quat_w"][:, self._anchor_indexes]

    def tgt_anchor_lin_vel_w(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_lin_vel_w"][:, self._anchor_indexes]

    def tgt_anchor_ang_vel_w(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_ang_vel_w"][:, self._anchor_indexes]

    def tgt_dof_pos(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["dof_pos"]

    def tgt_dof_vel(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["dof_vel"]

    def tgt_feet_pos_w(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_pos_w"][:, self._feet_indexes]

    def tgt_root_pos(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_pos_w"][:, 0, :]

    def tgt_root_quat(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_quat_w"][:, 0, :]

    def tgt_root_lin_vel(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_lin_vel_w"][:, 0, :]

    def tgt_root_ang_vel(self, motion_index: int = None) -> torch.Tensor:
        data = self._get_motion_data(motion_index)
        return data["body_ang_vel_w"][:, 0, :]

    def sample_random_frames(self, num_samples: int) -> dict[str, torch.Tensor]:
        all_motions = self.motion_data
        motion_indices = torch.randint(0, len(all_motions), (num_samples,))

        result_root_pos = []
        result_root_quat = []
        result_root_lin_vel = []
        result_root_ang_vel = []
        result_joint_pos = []
        result_joint_vel = []

        for i in range(num_samples):
            motion = all_motions[motion_indices[i].item()]
            num_frames = motion["dof_pos"].shape[0]
            frame_idx = torch.randint(0, num_frames, (1,)).item()

            result_root_pos.append(motion["body_pos_w"][frame_idx, 0, :])
            result_root_quat.append(motion["body_quat_w"][frame_idx, 0, :])
            result_root_lin_vel.append(motion["body_lin_vel_w"][frame_idx, 0, :])
            result_root_ang_vel.append(motion["body_ang_vel_w"][frame_idx, 0, :])
            result_joint_pos.append(motion["dof_pos"][frame_idx])
            result_joint_vel.append(motion["dof_vel"][frame_idx])

        return {
            "root_pos": torch.stack(result_root_pos),
            "root_quat": torch.stack(result_root_quat),
            "root_lin_vel": torch.stack(result_root_lin_vel),
            "root_ang_vel": torch.stack(result_root_ang_vel),
            "joint_pos": torch.stack(result_joint_pos),
            "joint_vel": torch.stack(result_joint_vel),
        }
