# AMP Locomotion Guide

This guide explains the `src/tasks/amp_loco/` package.

## What `amp_loco` focuses on

`amp_loco` is the **actual AMP locomotion stack** for this repo.

Its job is to train a policy that:

- follows commanded walking velocities
- stays stable in simulation
- uses motion clips as a behavior prior
- produces more natural-looking locomotion than a purely reward-engineered controller

So this package is not just "walk at this speed."

It is:

- walk at this speed
- while resembling the reference motion distribution
- while resetting from motion frames
- while training under an AMP runner

## The core idea

In a normal velocity task, the policy learns from reward terms only.

In `amp_loco`, the policy still gets task rewards, but the training setup also uses:

- AMP motion files in `assets/motions/z1/curated/amp/`
- AMP observation features
- motion-based resets
- AMP-specific runner config

That is why this folder exists separately from `velocity/`.

## Folder structure

- `__init__.py`
- `amp_env_cfg.py`
- `ampmotion_loader.py`
- `config/__init__.py`
- `config/z1/__init__.py`
- `config/z1/env_cfgs.py`
- `config/z1/rl_cfg.py`
- `mdp/__init__.py`
- `mdp/command.py`
- `mdp/events.py`
- `mdp/metrics.py`
- `mdp/observations.py`
- `mdp/rewards.py`
- `mdp/terminations.py`
- `mdp/terrain.py`
- `rl/__init__.py`
- `rl/runner.py`
- `rl/wrapper.py`

## File roles

### `__init__.py`

Imports the config package so task registration side effects happen.

This is part of what makes `import src.tasks` register the task.

### `amp_env_cfg.py`

This is the AMP base environment factory.

It looks structurally similar to the velocity task factory, but it adds AMP-specific pieces.

Main things it defines:

- actor observations
- critic observations
- a separate `amp` observation group
- velocity commands
- startup and reset events for motion loading
- locomotion rewards tuned for AMP training
- terminations
- metrics

The biggest difference from `velocity_env_cfg.py` is the extra `amp` observation stream and motion-reset logic hooks.

Current important detail:

- `base_ang_vel` and `base_lin_vel` now come from robot state helpers (`mdp.base_ang_vel`, `mdp.base_lin_vel`)
- they do **not** depend on legacy built-in sensor names like `robot/imu_ang_vel` or `robot/imu_lin_vel`
- this was necessary because the Z1 scene exposes different sensor names than the G1-style templates

### `ampmotion_loader.py`

This file loads curated `_vel.npz` motion files.

It:

- scans a motion directory
- loads the arrays
- strips floating-base entries from joint arrays when needed
- removes a synthetic `world` body if present
- stores everything as tensors
- provides helpers to sample random frames

This is the bridge between your offline motion files and the training environment.

Without this file, AMP would have no reference motion data to use.

### `config/__init__.py`

Imports robot-specific task registrations.

### `config/z1/__init__.py`

Registers the actual Z1 AMP task:

- `Z1-AMP-Flat`

It also points the task at:

- the Z1 env config
- the Z1 AMP PPO runner config
- the custom AMP runner class

This is the file that makes the task visible to `train.py`, `play.py`, and `eval.py`.

### `config/z1/env_cfgs.py`

This file specializes the generic AMP env for Z1.

It wires in:

- the Z1 robot config
- Z1 action scaling
- Z1 anchor/body/foot names
- Z1 motion directory
- Z1 contact sensor setup
- flat vs rough environment variants

Recent practical changes in this file:

- CCD is kept off for MuJoCo Warp compatibility
- a `self_collision` contact sensor is created so the `self_collisions` reward term can run
- the feet-ground contact sensor remains separate from self-collision monitoring

This is where the generic AMP task becomes your actual Z1 task.

### `config/z1/rl_cfg.py`

This file defines the AMP PPO runner configuration for Z1.

It sets:

- actor and critic network sizes
- PPO hyperparameters
- body-based AMP dataset metadata
- discriminator settings
- action standard deviation floor
- AMP body and anchor names

This is the training-side counterpart to the environment config.

Important update:

- this config now carries explicit `dataset` and `discriminator` sections because `amp_rsl_rl` expects them
- the dataset section now includes:
  - `amp_data_path`
  - `amp_joint_names`
  - `amp_body_names`
  - `amp_anchor_name`
- the actor now uses log-space std instead of raw scalar std
- the runner clamps policy std to a configured valid range
- AMP style mixing and discriminator reward scale were reduced to improve stability

That is what lets the custom body-based AMP expert loader reconstruct the same AMP observation shape as the environment.

### `mdp/__init__.py`

Re-exports AMP task MDP helpers from this package and also inherits default MDP utilities from `mjlab.envs.mdp`.

### `mdp/command.py`

This file is where AMP-specific command logic would live.

Right now it mainly establishes the command-related module scaffold and shared typing/import structure.

### `mdp/events.py`

This file is one of the most important AMP-specific files.

It implements:

- `MotionResetManager`
- startup motion loading
- reset from random motion frames
- optional delayed-reset grouping

This is what makes episode starts come from reference motions instead of only random pose resets.

That is a major difference from the plain `velocity` task.

### `mdp/metrics.py`

Adds AMP-specific metrics, especially for delayed-reset monitoring.

Right now it includes mean delay-step logging for delayed termination environments.

### `mdp/observations.py`

This file builds body-centric observations used by AMP.

It computes:

- body positions relative to an anchor body
- body orientations relative to an anchor body
- body linear velocities
- body angular velocities

These are exactly the kinds of signals AMP needs to compare simulated behavior to motion-reference behavior.

For the current Z1 AMP stack, this file is the heart of the discriminator representation.

The active AMP observation size is:

- `39` body-position values
- `78` body-orientation values
- `39` body linear velocity values
- `39` body angular velocity values
- total `195`

### `mdp/rewards.py`

This file contains the locomotion reward terms used in the AMP task.

They still reward:

- command tracking
- upright, stable motion
- low slip
- smooth foot behavior

But they are written with AMP task behavior in mind, including delayed-reset reward handling for certain environments.

Important point:

AMP does **not** mean "no normal rewards."  
It means normal task rewards plus AMP-style motion prior.

### `mdp/terminations.py`

Defines `DelayedTerminationManager`.

This allows a subset of environments to delay reset after failure conditions instead of resetting instantly.

That is useful when you want motion-reset/recovery behavior to be less abrupt.

### `mdp/terrain.py`

Defines a rough-terrain generator used by the AMP task variants.

### `rl/__init__.py`

Exports the custom AMP runner class.

### `rl/runner.py`

This file customizes the AMP runner.

Its current special jobs are:

- replacing the backend's simple expert dataset loader with a repo-local body-based loader
- making the expert AMP feature vector match the environment AMP feature vector exactly
- exporting the policy to ONNX
- attaching metadata

This file became important because the default backend AMP loader expected a smaller joint/base AMP observation, while this repo uses a richer body-based one.

So `rl/runner.py` is now part of the actual AMP representation path, not just packaging.

### `rl/body_amp_loader.py`

This file is the custom expert-motion loader used by the current training path.

It reads the compiled `_vel.npz` files directly and builds:

- body positions relative to the anchor
- body orientation 6D features
- body linear velocities in body frame
- body angular velocities in body frame

It also builds:

- the expert AMP observation stream
- the next-step expert AMP observation stream
- reset-state samples for AMP PPO internals

This file exists because the off-the-shelf `amp_rsl_rl` expert loader expected a simpler `.npy` dataset format that did not match the body-based Z1 AMP observation layout.

### `rl/wrapper.py`

Currently empty placeholder.

It is there as a natural future place for AMP-specific wrapper logic if needed later.

## How `amp_loco` thinks

The training loop logic is roughly:

1. load motion clips
2. create locomotion environment with velocity commands
3. reset many envs from random motion frames
4. give the policy actor/critic observations
5. give AMP its own motion-comparison observation stream
6. optimize with PPO + AMP-specific training logic

More specifically in the current repo:

1. `ampmotion_loader.py` loads `_vel.npz` files for reset logic
2. `body_amp_loader.py` loads the same `_vel.npz` files for discriminator expert samples
3. `mdp/observations.py` computes body-based AMP observations from live simulation state
4. `rl/runner.py` verifies expert and policy AMP observation dimensions match
5. AMP PPO trains on the matched body-based representation

So this package is trying to solve both:

- task success
- motion naturalness

## Relation to `velocity`

This is the easiest practical distinction:

- `velocity` asks: can the robot follow the locomotion command well?
- `amp_loco` asks: can the robot follow the locomotion command well **and** move like the reference motions?

Another useful way to say it:

- `velocity` focuses on **control objective**
- `amp_loco` focuses on **control objective + motion style prior**

And inside this repo's current implementation:

- `velocity` remains the reward-engineered locomotion baseline
- `amp_loco` is now a body-based AMP system, not a reduced joint-only AMP port

## Which one matters most for your project

For your Z1 walking-from-NPZ project:

- `amp_loco` is the main task family
- `velocity` is supporting context and a useful baseline/reference

So if your end goal is:

> train Z1 to walk naturally from curated motion clips

the most important folder is this one.
