# Z1-AMP-MJLAB Guide

This repository is configured to train a locomotion policy for the MagicLab Z1 robot.

## Layout

The repository contains the following curated motion data under the assets folder:

- `src/assets/motions/z1/curated/amp/Walk/`
- `src/assets/motions/z1/curated/library_manifest.json`


## Runtime defaults

All runtime scripts default to using the curated Walk and Run motion library:

- `src/assets/motions/z1/curated/amp/`

This includes:

- `scripts/train.py`
- `scripts/play.py`
- `scripts/eval.py`
- `scripts/export_policy.py`

## Current training pattern

The repository trains a locomotion policy:

- `Walk/` supplies upright resets and locomotion expert data for the AMP discriminator.

## Where to read next

- `README.md`
- `docs/motion_library.md`
- `docs/pipeline.md`
- `docs/training_commands.md`
- `docs/checkpoint_guide.md`
