"""Headless evaluation for a trained Z1 AMP policy."""

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
from mjlab.utils.torch import configure_torch_backends

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


def _to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().item())
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Z1-AMP-Flat")
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--motion-folder", type=Path, default=_default_motion_dir())
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
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
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(str(args.checkpoint_file), load_optimizer=False)
    policy = runner.get_inference_policy(device=args.device)
    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]

    obs, _ = env.get_observations()
    reward_sum = 0.0
    done_count = 0
    action_abs_sum = 0.0
    root_height_sum = 0.0
    lin_vel_error_sum = 0.0
    yaw_vel_error_sum = 0.0
    commanded_speed_sum = 0.0
    actual_speed_sum = 0.0
    term_episode_len_sum = 0.0
    term_episode_count = 0
    extras_accumulator: dict[str, float] = {}
    extras_counts: dict[str, int] = {}

    for _ in range(args.steps):
        episode_lengths_before_step = raw_env.episode_length_buf.detach().clone()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, rewards, dones, infos = env.step(actions)

        reward_sum += float(rewards.mean().item())
        done_count += int(dones.sum().item())
        action_abs_sum += float(actions.abs().mean().item())

        root_height = robot.data.body_link_pos_w[:, 0, 2]
        root_height_sum += float(root_height.mean().item())

        command = raw_env.command_manager.get_command("twist")
        if command is not None:
            commanded_xy = command[:, :2]
            commanded_yaw = command[:, 2]
            actual_xy = robot.data.root_link_lin_vel_b[:, :2]
            actual_yaw = robot.data.root_link_ang_vel_b[:, 2]

            lin_vel_error_sum += float(
                torch.norm(commanded_xy - actual_xy, dim=1).mean().item()
            )
            yaw_vel_error_sum += float(
                torch.abs(commanded_yaw - actual_yaw).mean().item()
            )
            commanded_speed_sum += float(torch.norm(commanded_xy, dim=1).mean().item())
            actual_speed_sum += float(torch.norm(actual_xy, dim=1).mean().item())

        if dones.any():
            done_lengths = episode_lengths_before_step[dones]
            term_episode_len_sum += float(done_lengths.float().mean().item()) * int(dones.sum().item())
            term_episode_count += int(dones.sum().item())

        if isinstance(infos, dict):
            for key, value in infos.items():
                if isinstance(value, (int, float, torch.Tensor)):
                    extras_accumulator[key] = extras_accumulator.get(key, 0.0) + _to_float(value)
                    extras_counts[key] = extras_counts.get(key, 0) + 1

    print(f"task={args.task}")
    print(f"steps={args.steps}")
    print(f"num_envs={args.num_envs}")
    print(f"mean_reward={reward_sum / args.steps:.6f}")
    print(f"done_count={done_count}")
    print(f"fall_rate={done_count / (args.steps * args.num_envs):.6f}")
    print(f"mean_abs_action={action_abs_sum / args.steps:.6f}")
    print(f"mean_root_height={root_height_sum / args.steps:.6f}")
    print(f"mean_lin_vel_tracking_error={lin_vel_error_sum / args.steps:.6f}")
    print(f"mean_yaw_vel_tracking_error={yaw_vel_error_sum / args.steps:.6f}")
    print(f"mean_commanded_speed={commanded_speed_sum / args.steps:.6f}")
    print(f"mean_actual_speed={actual_speed_sum / args.steps:.6f}")
    if term_episode_count > 0:
        print(f"mean_episode_len_at_done={term_episode_len_sum / term_episode_count:.6f}")
    else:
        print("mean_episode_len_at_done=nan")
    if infos:
        print(f"extras_keys={sorted(infos.keys())}")
    if extras_accumulator:
        for key in sorted(extras_accumulator):
            print(f"info_mean[{key}]={extras_accumulator[key] / extras_counts[key]:.6f}")

    env.close()


if __name__ == "__main__":
    main()
