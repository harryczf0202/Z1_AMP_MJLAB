# Tasks Guide

This guide explains the role of the `src/tasks/` package and how the two task families in this repo fit together.

## What `src/tasks/` is for

`src/tasks/` is the task-registration and task-logic layer for the repo.

It is where you define:

- what the environment observes
- what actions the policy outputs
- what commands the robot is asked to follow
- what rewards shape the behavior
- when episodes reset or terminate
- how a task is registered so `train.py`, `play.py`, and `eval.py` can find it

In this repo there are two task families:

1. `velocity/`
2. `amp_loco/`

They are related, but they are not the same thing.

## The difference between `velocity` and `amp_loco`

### `velocity`

`velocity` focuses on classic command-tracking locomotion.

The idea is:

- sample a target command like forward speed, sideways speed, and yaw rate
- reward the robot for following that command
- add hand-designed rewards for gait, foot clearance, slip, landing, posture, and stability

So `velocity` is mostly:

- command following
- locomotion shaping
- gait engineering through rewards and curriculum

It does **not** depend on motion clips.

### `amp_loco`

`amp_loco` focuses on AMP locomotion.

The idea is:

- still command the robot to walk at target velocities
- but also use reference motion clips as a style prior
- reset episodes from motion frames
- train with AMP-specific observations and an AMP runner/discriminator path

So `amp_loco` is:

- command following
- plus imitation-style motion prior
- plus motion-based reset logic
- plus AMP-specific training machinery

Short version:

- `velocity` = reward-engineered locomotion
- `amp_loco` = reward-engineered locomotion + motion imitation prior

## Their relation in this repo

Conceptually, `amp_loco` is the more specialized stack.

`velocity` gives the general locomotion ideas:

- twist commands
- terrain curriculum
- gait-related rewards
- locomotion observations

`amp_loco` takes that same kind of locomotion problem and changes the training objective:

- adds motion loader
- adds motion-reset startup/reset events
- adds AMP observation group
- adds custom AMP runner config
- uses motion clips to make walking look more natural

So if you ask:

> which one is trying to make Z1 walk naturally from motion data?

that is `amp_loco`.

If you ask:

> which one is the more standard locomotion reward toolbox?

that is `velocity`.

## File in this folder

### `__init__.py`

This file auto-imports task packages so registration side effects happen.

What it does:

- imports packages under `src.tasks`
- skips blacklisted helper packages like `utils` and raw `.mdp`
- makes task registration discoverable by the runner scripts

Practical meaning:

- when `import src.tasks` runs, task packages like `amp_loco` get imported
- that eventually triggers `register_mjlab_task(...)`

## Current task status in this repo

Right now the main registered training task is under:

- `src/tasks/amp_loco/config/z1/`

That is where `Z1-AMP-Flat` is registered.

The `velocity/` package in this repo is mainly a reusable locomotion base and reference layer.  
The active Z1 AMP training path is the `amp_loco/` family.
