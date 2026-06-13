"""Play a trained Z1 AMP policy."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import mjlab
import mjlab.tasks  # noqa: F401
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.registry import load_rl_cfg
from mjlab.tasks.registry import load_runner_cls
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer import ViserPlayViewer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.tasks  # noqa: F401
from src.tasks.amp_loco.motion_layout import resolve_motion_library


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_motion_dir() -> Path:
    return _repo_root() / "src" / "assets" / "motions" / "z1" / "curated" / "amp"


def _configure_motion_folder(env_cfg, agent_cfg, motion_folder: Path) -> None:
    layout = resolve_motion_library(motion_folder)
    env_cfg.events["init_motion_loader"].params["motion_dir"] = str(layout.motion_dir)
    env_cfg.events["reset_from_motion"].params["motion_dir"] = str(layout.motion_dir)
    if hasattr(agent_cfg, "amp_motion_files"):
        agent_cfg.amp_motion_files = str(layout.amp_dataset_dir)
    if hasattr(agent_cfg, "dataset"):
        agent_cfg.dataset["amp_data_path"] = str(layout.amp_dataset_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Z1-AMP-Flat")
    parser.add_argument("--checkpoint-file", type=Path)
    parser.add_argument("--motion-folder", type=Path, default=_default_motion_dir())
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
    parser.add_argument("--agent", choices=("trained", "zero", "random"), default="trained")
    parser.add_argument(
        "--stabilize-head",
        action="store_true",
        help="Counter-rotate joint_hy against torso yaw during playback to keep the head facing straighter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_torch_backends()

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    _configure_motion_folder(env_cfg, agent_cfg, args.motion_folder)
    env_cfg.scene.num_envs = args.num_envs
    if not hasattr(env_cfg.sim, "dt"):
        env_cfg.sim.dt = env_cfg.sim.mujoco.timestep

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args.agent == "trained":
        if args.checkpoint_file is None:
            raise ValueError("--checkpoint-file is required for --agent trained")
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=args.device)
        runner.load(str(args.checkpoint_file), load_optimizer=False)
        policy = runner.get_inference_policy(device=args.device)
    elif args.agent == "zero":
        action_shape = env.unwrapped.action_space.shape

        class ZeroPolicy:
            def __call__(self, obs):
                del obs
                return torch.zeros(action_shape, device=env.unwrapped.device)

        policy = ZeroPolicy()
    else:
        action_shape = env.unwrapped.action_space.shape

        class RandomPolicy:
            def __call__(self, obs):
                del obs
                return 2.0 * torch.rand(action_shape, device=env.unwrapped.device) - 1.0

        policy = RandomPolicy()

    if args.stabilize_head:
        base_policy = policy
        robot = env.unwrapped.scene["robot"]
        joint_names = tuple(robot.joint_names)
        head_joint_idx = joint_names.index("joint_hy")
        torso_joint_idx = joint_names.index("joint_wy")
        action_term = env.unwrapped.action_manager._terms["joint_pos"]
        action_target_names = tuple(action_term.target_names)
        head_action_idx = action_target_names.index("joint_hy")
        action_scale = action_term.scale
        action_offset = action_term.offset
        head_limit = 0.6981

        class HeadStabilizedPolicy:
            def __call__(self, obs):
                actions = base_policy(obs)
                torso_yaw = robot.data.joint_pos[:, torso_joint_idx]
                desired_head_yaw = torch.clamp(-torso_yaw, min=-head_limit, max=head_limit)
                if isinstance(action_offset, torch.Tensor):
                    head_offset = action_offset[:, head_action_idx]
                else:
                    head_offset = torch.full_like(desired_head_yaw, float(action_offset))
                if isinstance(action_scale, torch.Tensor):
                    head_scale = action_scale[:, head_action_idx]
                else:
                    head_scale = torch.full_like(desired_head_yaw, float(action_scale))
                target_action = (desired_head_yaw - head_offset) / torch.clamp(head_scale, min=1e-6)
                actions[:, head_action_idx] = torch.clamp(target_action, min=-1.0, max=1.0)
                return actions

        policy = HeadStabilizedPolicy()

    viewer = args.viewer
    if viewer == "auto":
        viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"

    if viewer == "native":
        NativeMujocoViewer(env, policy).run()
    else:
        ViserPlayViewer(env, policy).run()

    env.close()


if __name__ == "__main__":
    main()
