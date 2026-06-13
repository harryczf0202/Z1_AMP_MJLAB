import os
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import List

from mjlab.rl import RslRlModelCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.rl import RslRlPpoAlgorithmCfg

from src.robots.z1 import AMP_ANCHOR_NAME
from src.robots.z1 import AMP_BODY_NAMES
from src.robots.z1 import JOINT_NAMES


_MOTION_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    os.pardir,
    os.pardir,
    os.pardir,
    os.pardir,
    "src",
    "assets",
    "motions",
    "z1",
    "curated",
    "amp",
)


@dataclass
class RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    style_weight: float = 0.10
    amp_motion_files: str = ""
    min_normalized_std: List[float] = field(default_factory=lambda: [0.05] * len(JOINT_NAMES))
    max_normalized_std: List[float] = field(default_factory=lambda: [1.0] * len(JOINT_NAMES))
    amp_body_names: tuple = ()
    amp_anchor_name: str = ""
    discriminator: dict[str, Any] = field(
        default_factory=lambda: {
            "hidden_dims": [512, 256],
            "reward_scale": 0.25,
            "loss_type": "BCEWithLogits",
            "empirical_normalization": True,
        }
    )
    dataset: dict[str, Any] = field(
        default_factory=lambda: {
            "amp_data_path": "",
            "datasets": {},
            "slow_down_factor": 1,
            "velocity_representation": "body_fixed",
            "amp_joint_names": list(JOINT_NAMES),
            "amp_body_names": list(AMP_BODY_NAMES),
            "amp_anchor_name": AMP_ANCHOR_NAME,
        }
    )


@dataclass
class RslRlAmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    amp_replay_buffer_size: int = 100000
    class_name: str = "AMP_PPO"


def z1_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    return RslRlAmpRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.6,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlAmpPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=1.0e-4,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=0.5,
            amp_replay_buffer_size=100000,
            normalize_advantage_per_mini_batch=True,
        ),
        experiment_name="z1_amp_unified",
        logger="tensorboard",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=20001,
        amp_motion_files=os.path.normpath(_MOTION_DATA_DIR),
        min_normalized_std=[0.05] * len(JOINT_NAMES),
        max_normalized_std=[1.0] * len(JOINT_NAMES),
        amp_body_names=AMP_BODY_NAMES,
        amp_anchor_name=AMP_ANCHOR_NAME,
        discriminator={
            "hidden_dims": [512, 256],
            "reward_scale": 0.25,
            "loss_type": "BCEWithLogits",
            "empirical_normalization": True,
        },
        dataset={
            "amp_data_path": "",
            "datasets": {},
            "slow_down_factor": 1,
            "velocity_representation": "body_fixed",
            "amp_joint_names": list(JOINT_NAMES),
            "amp_body_names": list(AMP_BODY_NAMES),
            "amp_anchor_name": AMP_ANCHOR_NAME,
        },
    )
