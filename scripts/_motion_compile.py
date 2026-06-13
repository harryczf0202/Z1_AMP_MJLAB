from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_xml() -> Path:
    return repo_root() / "src" / "assets" / "robots" / "z1" / "xmls" / "z1.xml"


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.clip(norm, 1.0e-8, None)
    return quat / norm


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta_0 = float(np.arccos(dot))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * t
    sin_theta = float(np.sin(theta))
    s0 = float(np.sin(theta_0 - theta) / sin_theta_0)
    s1 = float(sin_theta / sin_theta_0)
    return s0 * q0 + s1 * q1


def resample_qpos(qpos: np.ndarray, input_fps: float, output_fps: float) -> np.ndarray:
    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    duration = (qpos.shape[0] - 1) * input_dt
    times = np.arange(0.0, duration, output_dt, dtype=np.float64)
    src_times = np.arange(qpos.shape[0], dtype=np.float64) * input_dt

    out = np.zeros((times.shape[0], qpos.shape[1]), dtype=np.float64)
    for i, t in enumerate(times):
        if t >= duration:
            out[i] = qpos[-1]
            continue
        idx0 = int(np.floor(t / input_dt))
        idx1 = min(idx0 + 1, qpos.shape[0] - 1)
        blend = (t - src_times[idx0]) / input_dt

        out[i, :3] = (1.0 - blend) * qpos[idx0, :3] + blend * qpos[idx1, :3]
        out[i, 3:7] = slerp(qpos[idx0, 3:7], qpos[idx1, 3:7], float(blend))
        out[i, 7:] = (1.0 - blend) * qpos[idx0, 7:] + blend * qpos[idx1, 7:]

    out[:, 3:7] = normalize_quat(out[:, 3:7])
    return out


def compute_qvel(model: mujoco.MjModel, qpos: np.ndarray, dt: float) -> np.ndarray:
    qvel = np.zeros((qpos.shape[0], model.nv), dtype=np.float64)
    tmp = np.zeros(model.nv, dtype=np.float64)

    mujoco.mj_differentiatePos(model, tmp, dt, qpos[0], qpos[1])
    qvel[0] = tmp
    for i in range(1, qpos.shape[0] - 1):
        mujoco.mj_differentiatePos(model, tmp, 2.0 * dt, qpos[i - 1], qpos[i + 1])
        qvel[i] = tmp
    mujoco.mj_differentiatePos(model, tmp, dt, qpos[-2], qpos[-1])
    qvel[-1] = tmp
    return qvel


def body_kinematics(model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray) -> tuple[np.ndarray, ...]:
    data = mujoco.MjData(model)
    nframe = qpos.shape[0]
    nbody = model.nbody

    body_pos_w = np.zeros((nframe, nbody, 3), dtype=np.float32)
    body_quat_w = np.zeros((nframe, nbody, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((nframe, nbody, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((nframe, nbody, 3), dtype=np.float32)
    body_names = []
    for i in range(nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        body_names.append("world" if name is None else name)

    vel6 = np.zeros(6, dtype=np.float64)
    for f in range(nframe):
        data.qpos[:] = qpos[f]
        data.qvel[:] = qvel[f]
        mujoco.mj_forward(model, data)

        body_pos_w[f] = data.xpos.astype(np.float32)
        body_quat_w[f] = data.xquat.astype(np.float32)

        for b in range(nbody):
            if b == 0:
                continue
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 0)
            body_ang_vel_w[f, b] = vel6[:3].astype(np.float32)
            body_lin_vel_w[f, b] = vel6[3:].astype(np.float32)

    return body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, np.asarray(body_names)


def joint_names(model: mujoco.MjModel) -> np.ndarray:
    names: list[str] = []
    for j in range(1, model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        names.append("" if name is None else name)
    return np.asarray(names)


def compile_qpos_to_amp(
    *,
    qpos: np.ndarray,
    input_fps: float,
    output_fps: int,
    output_file: Path,
    xml_file: Path | None = None,
) -> None:
    qpos = resample_qpos(qpos, input_fps=float(input_fps), output_fps=float(output_fps))
    pack_qpos_to_amp(
        qpos=qpos,
        output_fps=output_fps,
        output_file=output_file,
        xml_file=xml_file,
    )


def pack_qpos_to_amp(
    *,
    qpos: np.ndarray,
    output_fps: int,
    output_file: Path,
    xml_file: Path | None = None,
) -> None:
    qpos = np.asarray(qpos, dtype=np.float64).copy()
    qpos[:, 3:7] = normalize_quat(qpos[:, 3:7])
    model = mujoco.MjModel.from_xml_path(str(xml_file or default_xml()))
    if qpos.shape[1] != model.nq:
        raise ValueError(f"Input qpos dim {qpos.shape[1]} does not match model.nq {model.nq}")

    qvel = compute_qvel(model, qpos, dt=1.0 / float(output_fps))
    body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, body_names_arr = body_kinematics(model, qpos, qvel)
    joint_names_arr = joint_names(model)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_file,
        fps=np.asarray([float(output_fps)], dtype=np.float32),
        joint_pos=qpos.astype(np.float32),
        joint_vel=qvel.astype(np.float32),
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        joint_names=joint_names_arr,
        body_names=body_names_arr,
    )
