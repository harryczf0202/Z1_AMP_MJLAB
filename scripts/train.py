"""Train AMP+PPO for Z1."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import MethodType

import mjlab
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.registry import load_rl_cfg
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.os import dump_yaml
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


def _make_log_dir(experiment_name: str, run_name: str | None) -> Path:
    log_root = _repo_root() / "logs" / "rsl_rl" / experiment_name
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{run_name}" if run_name else ""
    log_dir = log_root / f"{stamp}{suffix}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _disable_push_robot(env_cfg) -> None:
    env_cfg.events.pop("push_robot", None)


def _set_push_interval(env_cfg, interval_range_s: tuple[float, float]) -> None:
    push_term = env_cfg.events.get("push_robot")
    if push_term is not None:
        push_term.interval_range_s = interval_range_s


def _set_push_velocity_scale(env_cfg, scale: float) -> None:
    push_term = env_cfg.events.get("push_robot")
    if push_term is None:
        return
    velocity_range = push_term.params.get("velocity_range", {})
    for axis, bounds in tuple(velocity_range.items()):
        low, high = bounds
        velocity_range[axis] = (low * scale, high * scale)


def _format_hms(total_seconds: float) -> str:
    total_seconds = max(0.0, float(total_seconds))
    whole_seconds = int(total_seconds)
    hours, rem = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _patch_runner_time_logging(runner) -> None:
    original_log = runner.log
    writer_disabled_after_error = False

    def patched_log(self, locs: dict, width: int = 80, pad: int = 35):
        nonlocal writer_disabled_after_error
        if "it" in locs and "tot_iter" in locs:
            locs = dict(locs)
            locs["num_learning_iterations"] = max(int(locs["tot_iter"]), 0)

        total_iterations = int(locs.get("tot_iter", locs.get("num_learning_iterations", 0)))
        completed_iterations = int(locs.get("it", 0)) + 1
        remaining_iterations = max(total_iterations - completed_iterations, 0)
        iteration_time = float(locs["collection_time"]) + float(locs["learn_time"])
        eta_seconds = remaining_iterations * iteration_time
        expected_total_time = max(0.0, float(self.tot_time) + iteration_time)

        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                original_log(locs, width=width, pad=pad)
        except PermissionError as exc:
            if writer_disabled_after_error:
                raise
            writer_disabled_after_error = True
            writer = getattr(self, "writer", None)
            if writer is not None:
                with contextlib.suppress(Exception):
                    writer.close()
            self.writer = None
            print(
                "[train.py] TensorBoard writer hit a PermissionError and was disabled; "
                f"training will continue without new event-file writes. Details: {exc}"
            )
            return
        log_output = buffer.getvalue()

        log_output = re.sub(
            r"(?m)^(\s*Total time:\s+).*$",
            lambda match: f"{match.group(1)}{_format_hms(expected_total_time)}",
            log_output,
        )
        log_output = re.sub(
            r"(?m)^(\s*ETA:\s+).*$",
            lambda match: f"{match.group(1)}{_format_hms(eta_seconds)}",
            log_output,
        )
        print(log_output, end="")

    runner.log = MethodType(patched_log, runner)




def _apply_phase_preset(args: argparse.Namespace, env_cfg, agent_cfg) -> None:
    _configure_motion_folder(env_cfg, agent_cfg, args.motion_folder)
    if args.experiment_name is None:
        agent_cfg.experiment_name = "z1_amp_unified"
    if args.run_name is None:
        agent_cfg.run_name = "loco_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Z1-AMP-Flat")
    parser.add_argument("--motion-folder", type=Path, default=_default_motion_dir())
    parser.add_argument(
        "--training-phase",
        choices=("unified",),
        default="unified",
        help="Training preset configuration",
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--experiment-name")
    parser.add_argument("--run-name")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--resume-weights-only",
        action="store_true",
        help="Resume actor/critic/discriminator weights without restoring optimizer state.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")

    configure_torch_backends()

    env_cfg = load_env_cfg(args.task)
    agent_cfg = load_rl_cfg(args.task)

    _apply_phase_preset(args, env_cfg, agent_cfg)

    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.experiment_name:
        agent_cfg.experiment_name = args.experiment_name
    if args.run_name:
        agent_cfg.run_name = args.run_name

    if not hasattr(env_cfg.sim, "dt"):
        env_cfg.sim.dt = env_cfg.sim.mujoco.timestep

    log_dir = _make_log_dir(agent_cfg.experiment_name, agent_cfg.run_name)

    env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=args.device,
        render_mode="rgb_array" if args.video else None,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), str(log_dir), args.device)
    runner.add_git_repo_to_log(__file__)
    _patch_runner_time_logging(runner)

    if args.resume_checkpoint:
        runner.load(
            str(args.resume_checkpoint),
            load_optimizer=not args.resume_weights_only,
            weights_only=args.resume_weights_only,
        )

    dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )
    env.close()


if __name__ == "__main__":
    main()
