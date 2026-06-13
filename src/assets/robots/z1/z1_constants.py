from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from src import SRC_PATH
from src.robots.z1 import DEFAULT_STANDING_JOINT_POS


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

#stiffness: (KP)
# Position spring gain.
# Bigger value means the joint fights harder to reach the target angle.
# Too low: floppy, delayed, collapses.
# Too high: twitchy, chatter, unstable contact.

# damping: (KD)
# Velocity-resisting term.
# Bigger value removes oscillation and overshoot.
# Too low: shaking, bouncing, vibrating after foot contact.
# Too high: sluggish, heavy, hard to move.

# effort_limit:
# Maximum actuator torque/force allowed.
# This caps how hard the controller can push.
# Too low: robot cannot stand or recover.
# Too high: can look explosive and unrealistic.

# armature:
# Extra rotational inertia added at the joint.
# In practice this helps represent motor/gear reflected inertia and often improves sim stability.
# Too low: joints can feel too light and noisy.
# Too high: motion feels heavy and slow.

# -----------------------------------------

#tuning guide:
# If you want the values to be “more correct”

# Best workflow:
# use real Z1 actuator torque limits if you have them
# keep XML inertias/masses as the physical base
# tune stand first
# tune foot landing second
# only then start long AMP training

# A very normal tuning loop is:
# if it collapses: raise effort_limit or stiffness
# if it shakes after landing: raise damping a bit or lower stiffness
# if it looks too rigid: lower stiffness
# if it feels too sluggish: lower damping or armature

# So the honest answer is: these values are chosen by a mix of
# robot physical model
# actuator capability assumptions
# simulator stability needs
# empirical locomotion tuning

LEG_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "JOINT_HIP_PITCH_.*",
        "JOINT_HIP_ROLL_.*",
        "JOINT_HIP_YAW_.*",
        "JOINT_KNEE_PITCH_.*",
    ),
    stiffness=180.0,
    damping=12.0,
    effort_limit=180.0,
    armature=0.01,
)

ANKLE_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "JOINT_ANKLE_PITCH_.*",
        "JOINT_ANKLE_ROLL_.*",
    ),
    stiffness=120.0,
    damping=8.0,
    effort_limit=120.0,
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


Z1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        LEG_ACTUATOR,
        ANKLE_ACTUATOR,
        TORSO_HEAD_ACTUATOR,
        ARM_ACTUATOR,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_z1_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FEET_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=Z1_ARTICULATION,
    )


Z1_ACTION_SCALE: dict[str, float] = {}
for actuator in Z1_ARTICULATION.actuators:
    assert isinstance(actuator, BuiltinPositionActuatorCfg)
    assert actuator.effort_limit is not None
    for name in actuator.target_names_expr:
        Z1_ACTION_SCALE[name] = 0.25 * actuator.effort_limit / actuator.stiffness
