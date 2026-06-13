"""Export a trained Z1 AMP checkpoint to ONNX."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Z1-AMP-Flat")
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--motion-folder", type=Path, default=_default_motion_dir())
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_torch_backends()

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(args.task)
    _configure_motion_folder(env_cfg, agent_cfg, args.motion_folder)
    if not hasattr(env_cfg.sim, "dt"):
        env_cfg.sim.dt = env_cfg.sim.mujoco.timestep

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(str(args.checkpoint_file), load_optimizer=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(runner, "_export_policy_to_onnx"):
        runner._export_policy_to_onnx(str(args.output_dir), "policy.onnx")
    elif hasattr(runner, "export_policy_to_onnx"):
        runner.export_policy_to_onnx(str(args.output_dir), "policy.onnx")
    else:
        raise RuntimeError("Runner does not expose an ONNX exporter.")

    print(f"Exported policy to: {args.output_dir / 'policy.onnx'}")
    env.close()


if __name__ == "__main__":
    main()
