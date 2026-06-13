# Velocity Task Guide

This guide explains the `src/tasks/velocity/` package.

## What `velocity` focuses on

This package focuses on **velocity-command locomotion**.

The robot is asked to follow a command like:

- walk forward
- move sideways
- turn left or right
- stand still

The policy is trained using hand-designed locomotion rewards rather than motion imitation.

That means this package is about:

- tracking commanded velocities
- staying upright
- maintaining a useful gait
- reducing slip and harsh foot impacts
- using curriculum to expand speed range and terrain difficulty

## What `velocity` is not

It is not AMP by itself.

It does not load motion clips.
It does not reset from reference motion frames.
It does not add an AMP discriminator input group.

So this is the "classical locomotion task" side of the project.

## Folder structure

- `__init__.py`
- `velocity_env_cfg.py`
- `mdp/__init__.py`
- `mdp/curriculums.py`
- `mdp/observations.py`
- `mdp/rewards.py`
- `mdp/terminations.py`
- `mdp/velocity_command.py`

## File roles

### `__init__.py`

Simple package marker and short description.

It tells you this package is for velocity tracking environments.

### `velocity_env_cfg.py`

This is the main base environment factory for a velocity task.

It defines:

- sensors like terrain scan
- actor observations
- critic observations
- joint position action interface
- `twist` command generation
- reset and perturbation events
- reward terms
- termination rules
- curriculum terms
- simulation defaults

This is the central "task recipe" for non-AMP locomotion.

Important idea:

the policy is rewarded for following commanded motion, not for matching a recorded motion clip.

### `mdp/__init__.py`

Re-exports the MDP helper functions so config files can refer to them through one namespace.

### `mdp/velocity_command.py`

This file defines the command generator used by the task.

It handles:

- sampling linear and angular velocity commands
- optional heading control
- some standing environments
- optional initialization with commanded velocities
- GUI joystick control in the viewer
- debug visualization of command arrows and actual motion arrows

This file is important because it defines what "walk at this speed" actually means for the environment.

### `mdp/observations.py`

This file adds locomotion-specific observations such as:

- foot height
- foot air time
- foot contact state
- foot contact force features
- gait phase signal

These are especially useful for shaping gait timing and terrain-aware behavior.

### `mdp/rewards.py`

This file is the main locomotion reward toolbox.

It contains rewards and penalties for:

- linear velocity tracking
- angular velocity tracking
- upright body orientation
- body angular velocity
- angular momentum
- feet air time
- gait timing
- foot clearance
- foot slip
- soft landing
- posture regularization
- stand-still behavior
- self-collision cost

This is the main difference from AMP:

`velocity` tries to build natural walking mostly through reward engineering.

### `mdp/curriculums.py`

This file changes task difficulty over training.

It currently handles:

- terrain level progression
- velocity command range expansion
- reward-weight staging helper

This is how the task can start easier and become harder later.

### `mdp/terminations.py`

This file contains termination helpers.

Currently it provides `illegal_contact(...)`, which can be used when certain contact events should end an episode.

## How the velocity task thinks

The logic is roughly:

1. sample a `twist` command
2. let the robot act
3. measure how well it follows the command
4. reward stable, useful gait behavior
5. penalize bad contacts, slip, harsh landing, and unstable motion
6. gradually widen the task through curriculum

That is classical locomotion training.

## Relation to `amp_loco`

`velocity` and `amp_loco` are siblings, but `amp_loco` is the one used for your actual AMP walking task.

The easiest way to think about them is:

- `velocity` teaches **what motion objective to solve**
- `amp_loco` teaches **how to solve it while looking like reference motion**

So:

- `velocity` = command tracking with engineered locomotion rewards
- `amp_loco` = command tracking plus motion-style prior from AMP

## Why this folder still matters even if you train AMP

Even when you use AMP, this folder is still useful because:

- many locomotion rewards and command ideas come from this style of task design
- it is a good baseline for comparing AMP vs non-AMP walking
- it helps explain what AMP is adding on top of normal locomotion training

If your AMP policy behaves strangely, this package is also a good mental reference for what a plain velocity-tracking task would have done instead.
