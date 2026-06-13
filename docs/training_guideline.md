# Training Guideline

This file explains the training pipeline for `Z1-AMP-MJLAB`.

## Summary

The active training pipeline is simplified to train locomotion skills from scratch:

1. Use the pre-compiled `Walk` motion library located at `src/assets/motions/z1/curated/amp/Walk/`.
2. Run training from scratch using the default single-phase training command.
3. Inspect and evaluate the checkpoint in play mode.

---

## Recommended workflow

### Step 1: Train from scratch

Run the training script targeting the unified locomotion task:

```powershell
python scripts/train.py `
  --task Z1-AMP-Flat `
  --device cuda:0 `
  --num-envs 256 `
  --max-iterations 20000 `
  --experiment-name z1_amp_unified `
  --run-name loco_v1
```

### Step 2: Inspect the result

Verify the policy's gait, stability, and velocity tracking using `play.py`:

```powershell
python scripts/play.py `
  --task Z1-AMP-Flat `
  --checkpoint-file logs/rsl_rl/z1_amp_unified/<run_dir>/model_20000.pt
```

---

## What to watch for during training

Bad signs:

- Immediate collapse
- Severe oscillation or joint jitter
- Large foot slip
- Shaking on landing
- High joint acceleration losses

---

## Important rule about constants

Try not to change actuator constants, action scales, contact behavior, or reward definitions in the middle of a training run series. If you change environment dynamics, old checkpoints will not transfer cleanly.
