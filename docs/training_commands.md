# Training Commands

All commands assume:

```powershell
cd "F:\amp for hardwaree\Z1_AMP_MJLAB"
```

## Recommended training strategy

Train the locomotion policy using the pre-compiled `Walk` motion library:

```powershell
python scripts/train.py `
  --task Z1-AMP-Flat `
  --device cuda:0 `
  --num-envs 256 `
  --max-iterations 20000 `
  --experiment-name z1_amp_unified `
  --run-name loco_v1
```

## Small sanity runs

Run a small training run to verify GPU, environment, and file loading correctness:

```powershell
python scripts/train.py `
  --task Z1-AMP-Flat `
  --device cuda:0 `
  --num-envs 64 `
  --max-iterations 50 `
  --experiment-name z1_amp_unified `
  --run-name sanity_run
```

## Checkpoint playback

Play back a trained policy visually in MuJoCo:

```powershell
python scripts/play.py `
  --task Z1-AMP-Flat `
  --checkpoint-file "<CHECKPOINT_PATH>"
```
