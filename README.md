# Z1-AMP-MJLAB

Standalone AMP+PPO training repo for Z1 locomotion in simulation.

## Active layout

The repository is configured to train using a lightweight curated motion library:

- `src/assets/motions/z1/curated/amp/Walk/`: Curated locomotion NPZ files (upright transitions, walking, running)
- `src/assets/motions/z1/curated/library_manifest.json`: Build metadata manifest file

Active task:

- `Z1-AMP-Flat`

## Typical flow

Smoke test training:

```powershell
python scripts/train.py --task Z1-AMP-Flat --device cuda:0 --num-envs 1 --max-iterations 1 --experiment-name z1_amp_unified_smoke --run-name smoke
```

Train from scratch:

```powershell
python scripts/train.py --task Z1-AMP-Flat --device cuda:0 --num-envs 256 --max-iterations 20000 --experiment-name z1_amp_unified --run-name loco_v1
```

Play checkpoint:

```powershell
python scripts/play.py --task Z1-AMP-Flat --checkpoint-file <path-to-model.pt>
```

See:

- `docs/pipeline.md`
- `docs/training_commands.md`
- `docs/motion_library.md`

## Acknowledgements

This repository is adapted from the following source projects:
- [AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab): The original G1 locomotion task model logic and training configuration.
- [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab): The underlying physics simulation interface and reinforcement learning framework.
- [Open-X-Humanoid/TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab): The basis for the `rsl_rl` AMP implementation used in this project.
- [OmniRetarget / holosoma](https://github.com/amazon-far/holosoma): The framework used for retargeting the motion files for Z1.


