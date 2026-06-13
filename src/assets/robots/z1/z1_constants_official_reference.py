"""Official-reference Z1 constants derived from the MagicBot Z1 SDK manual.

This module is intentionally not wired into training yet.

Purpose:
- keep a hardware-oriented reference profile beside the current training baseline
- record which values are directly supported by the official SDK/manual
- provide a future drop-in starting point for sim2real-oriented experiments

Important:
- joint limits and torque ceilings below are grounded in the official document
- stiffness, damping, and armature are still placeholders for simulator tuning
- do not switch training to this file until the baseline config has been tested
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from src import SRC_PATH
from src.robots.z1 import DEFAULT_STANDING_JOINT_POS


OFFICIAL_REFERENCE_SOURCE = (
    "magicbot_z1_sdk_en_1.2.4-doc1.pdf "
    "(Joint motor: page 17, joint limits: page 18, mechanical specs: pages 20-21)"
)

Z1_XML: Path = SRC_PATH / "assets" / "robots" / "z1" / "xmls" / "z1.xml"
assert Z1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, Z1_XML.parent / "meshes", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(Z1_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


# These limits are copied from the official SDK manual so we keep an explicit
# hardware reference even though the active XML already carries near-matching limits.
OFFICIAL_SDK_JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "left_hip_pitch_joint": (-2.7925, 2.7925),
    "left_hip_roll_joint": (-0.5240, 2.9670),
    "left_hip_yaw_joint": (-2.7925, 2.7925),
    "left_knee_joint": (0.0000, 2.6530),
    "left_ankle_pitch_joint": (-0.8300, 0.5240),
    "left_ankle_roll_joint": (-0.2620, 0.2620),
    "right_hip_pitch_joint": (-2.7925, 2.7925),
    "right_hip_roll_joint": (-2.9670, 0.5240),
    "right_hip_yaw_joint": (-2.7925, 2.7925),
    "right_knee_joint": (0.0000, 2.6530),
    "right_ankle_pitch_joint": (-0.8300, 0.5240),
    "right_ankle_roll_joint": (-0.2620, 0.2620),
    "left_shoulder_pitch_joint": (-2.8800, 2.8800),
    "left_shoulder_roll_joint": (-0.1750, 2.2515),
    "left_shoulder_yaw_joint": (-2.6180, 2.6180),
    "left_elbow_joint": (-0.9600, 1.7000),
    "left_wrist_yaw_joint": (-2.6180, 2.6180),
    "right_shoulder_pitch_joint": (-2.8800, 2.8800),
    "right_shoulder_roll_joint": (-2.2515, 0.1750),
    "right_shoulder_yaw_joint": (-2.6180, 2.6180),
    "right_elbow_joint": (-0.9600, 1.7000),
    "right_wrist_yaw_joint": (-2.6180, 2.6180),
}


OFFICIAL_SDK_MECHANICAL_SPECS = {
    "robot_mass_kg": 40.0,
    "standing_size_mm": (1369.0, 422.0, 200.0),
    "folded_size_mm": (730.0, 422.0, 395.0),
    "single_leg_dof": 6,
    "single_arm_dof": 5,
    "head_dof": 1,
    "waist_dof": 1,
    "motor_description": (
        "Low-inertia, high-speed, high-overload permanent magnet synchronous motor"
    ),
    "joint_motor_max_torque_nm": 130.0,
    "z1_basic_max_knee_torque_nm": 100.0,
}


# Effort limits below are intentionally closer to the official hardware envelope.
# They are the main document-grounded part of this reference module.
# The remaining actuator gains are still conservative placeholders until we do
# either real-robot identification or explicit simulator retuning.
HIP_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "JOINT_HIP_PITCH_.*",
        "JOINT_HIP_ROLL_.*",
        "JOINT_HIP_YAW_.*",
    ),
    stiffness=180.0,
    damping=12.0,
    effort_limit=130.0,
    armature=0.01,
)

KNEE_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=("JOINT_KNEE_PITCH_.*",),
    stiffness=180.0,
    damping=12.0,
    effort_limit=100.0,
    armature=0.01,
)

ANKLE_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "JOINT_ANKLE_PITCH_.*",
        "JOINT_ANKLE_ROLL_.*",
    ),
    stiffness=120.0,
    damping=8.0,
    effort_limit=80.0,
    armature=0.008,
)

TORSO_HEAD_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=("joint_wy", "joint_hy"),
    stiffness=60.0,
    damping=5.0,
    effort_limit=40.0,
    armature=0.004,
)

ARM_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=("joint_la.*", "joint_ra.*"),
    stiffness=55.0,
    damping=5.0,
    effort_limit=45.0,
    armature=0.004,
)


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.75),
    joint_pos=DEFAULT_STANDING_JOINT_POS,
    joint_vel={".*": 0.0},
)


FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=("l_foot", "r_foot"),
    contype=1,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.8,),
)


Z1_ARTICULATION_OFFICIAL_REFERENCE = EntityArticulationInfoCfg(
    actuators=(
        HIP_ACTUATOR,
        KNEE_ACTUATOR,
        ANKLE_ACTUATOR,
        TORSO_HEAD_ACTUATOR,
        ARM_ACTUATOR,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_z1_robot_cfg_official_reference() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FEET_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=Z1_ARTICULATION_OFFICIAL_REFERENCE,
    )


Z1_ACTION_SCALE_OFFICIAL_REFERENCE: dict[str, float] = {}
for actuator in Z1_ARTICULATION_OFFICIAL_REFERENCE.actuators:
    assert isinstance(actuator, BuiltinPositionActuatorCfg)
    assert actuator.effort_limit is not None
    for name in actuator.target_names_expr:
        Z1_ACTION_SCALE_OFFICIAL_REFERENCE[name] = 0.25 * actuator.effort_limit / actuator.stiffness
