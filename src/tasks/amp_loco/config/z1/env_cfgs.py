import os
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch
from mjlab.sensor import ContactSensorCfg
from mjlab.sensor import RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.assets.robots import Z1_ACTION_SCALE
from src.assets.robots import get_z1_robot_cfg
from src.robots.z1 import AMP_ANCHOR_NAME
from src.robots.z1 import AMP_BODY_NAMES
from src.robots.z1 import FEET_GEOM_NAMES
from src.robots.z1 import FEET_SITE_NAMES
from src.robots.z1 import ROOT_BODY_NAME
from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg
from src.tasks.amp_loco.motion_layout import resolve_motion_library


def _set_reward_weight(cfg: ManagerBasedRlEnvCfg, reward_name: str, weight: float) -> None:
    if reward_name in cfg.rewards:
        cfg.rewards[reward_name].weight = weight


def _set_reward_param(cfg: ManagerBasedRlEnvCfg, reward_name: str, param_name: str, value) -> None:
    if reward_name in cfg.rewards:
        cfg.rewards[reward_name].params[param_name] = value


def z1_amp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_amp_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 0
    cfg.sim.contact_sensor_maxmatch = 96
    cfg.sim.nconmax = 48
    cfg.scene.entities = {"robot": get_z1_robot_cfg()}

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            sensor.frame.name = ROOT_BODY_NAME

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(LINK_ANKLE_ROLL_L|LINK_ANKLE_ROLL_R)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern=ROOT_BODY_NAME, entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern=ROOT_BODY_NAME, entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, self_collision_cfg)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = Z1_ACTION_SCALE

    cfg.viewer.body_name = AMP_ANCHOR_NAME

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 0.95
    twist_cmd.ranges.lin_vel_x = (-0.25, 1.5)
    twist_cmd.ranges.lin_vel_y = (-0.6, 0.6)
    twist_cmd.ranges.ang_vel_z = (-1.2, 1.2)

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = FEET_GEOM_NAMES
    cfg.events["base_com"].params["asset_cfg"].body_names = (AMP_ANCHOR_NAME,)
    
    motion_base = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "assets",
            "motions",
            "z1",
            "curated",
            "amp",
        )
    )
    layout = resolve_motion_library(motion_base)

    cfg.events["init_motion_loader"].params["motion_dir"] = str(layout.motion_dir)
    cfg.events["reset_from_motion"].params["motion_dir"] = str(layout.motion_dir)

    cfg.rewards["track_anchor_linear_velocity"].params["anchor_cfg"].body_names = (AMP_ANCHOR_NAME,)
    cfg.rewards["track_anchor_angular_velocity"].params["anchor_cfg"].body_names = (AMP_ANCHOR_NAME,)
    cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (ROOT_BODY_NAME,)
    cfg.rewards["torso_orientation_l2"].params["body_cfg"].body_names = (AMP_ANCHOR_NAME,)
    

    cfg.rewards["foot_slip"].params["asset_cfg"].site_names = FEET_SITE_NAMES

    cfg.observations["critic"].terms["body_pos_b"].params["anchor_cfg"].body_names = (AMP_ANCHOR_NAME,)
    cfg.observations["critic"].terms["body_pos_b"].params["body_cfg"].body_names = AMP_BODY_NAMES
    cfg.observations["critic"].terms["body_ori_b"].params["anchor_cfg"].body_names = (AMP_ANCHOR_NAME,)
    cfg.observations["critic"].terms["body_ori_b"].params["body_cfg"].body_names = AMP_BODY_NAMES
    cfg.observations["amp"].terms["body_pos_b"].params["anchor_cfg"].body_names = (AMP_ANCHOR_NAME,)
    cfg.observations["amp"].terms["body_pos_b"].params["body_cfg"].body_names = AMP_BODY_NAMES
    cfg.observations["amp"].terms["body_ori_b"].params["anchor_cfg"].body_names = (AMP_ANCHOR_NAME,)
    cfg.observations["amp"].terms["body_ori_b"].params["body_cfg"].body_names = AMP_BODY_NAMES
    cfg.observations["amp"].terms["body_lin_vel_b"].params["anchor_cfg"].body_names = (AMP_ANCHOR_NAME,)
    cfg.observations["amp"].terms["body_lin_vel_b"].params["body_cfg"].body_names = AMP_BODY_NAMES
    cfg.observations["amp"].terms["body_ang_vel_b"].params["anchor_cfg"].body_names = (AMP_ANCHOR_NAME,)
    cfg.observations["amp"].terms["body_ang_vel_b"].params["body_cfg"].body_names = AMP_BODY_NAMES

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )
        cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.0

    return cfg


def z1_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = z1_amp_rough_env_cfg(play=play)
    cfg.sim.njmax = 512
    cfg.sim.mujoco.ccd_iterations = 0
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.sim.nconmax = None

    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = tuple(
        sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
    )

    if play:
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-0.10, 1.25)
        twist_cmd.ranges.lin_vel_y = (-0.35, 0.35)
        twist_cmd.ranges.ang_vel_z = (-0.8, 0.8)

    return cfg
