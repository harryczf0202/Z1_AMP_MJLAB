# Checkpoint Guide

This guide describes how checkpoints are structured, saved, and loaded in `Z1_AMP_MJLAB`.

## Log layout

Training logs and checkpoints are written to:

```text
logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/
```

Example:

```text
logs/rsl_rl/z1_amp_unified/2026-06-14_12-00-00_loco_v1/
```

## What to look for

Inside a run folder, the main files are:

- Checkpoint `.pt` files (e.g. `model_20000.pt`)
- `params/env.yaml`
- `params/agent.yaml`
- `policy.onnx` (if exported)

## Safest rule

Always:

1. Find the newest run folder for your experiment or run name.
2. List the `.pt` files inside it.
3. Use the newest checkpoint.

## Find the newest run folder

Example for experiment `z1_amp_unified`:

```powershell
Get-ChildItem "logs\rsl_rl\z1_amp_unified" -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 5 FullName, LastWriteTime
```

Filter by run name if needed:

```powershell
Get-ChildItem "logs\rsl_rl\z1_amp_unified" -Directory |
  Where-Object { $_.Name -like "*loco_v1*" } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 5 FullName, LastWriteTime
```

## List checkpoints inside one run folder

```powershell
Get-ChildItem "<RUN_FOLDER>" -Recurse -Filter *.pt |
  Sort-Object LastWriteTime -Descending |
  Select-Object FullName, LastWriteTime
```

## One-command newest checkpoint lookup

```powershell
$checkpoint = Get-ChildItem "logs\rsl_rl\z1_amp_unified" -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 |
  ForEach-Object {
    Get-ChildItem $_.FullName -Recurse -Filter *.pt |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 1 -ExpandProperty FullName
  }

$checkpoint
```

## Use the checkpoint

Resume training:

```powershell
python scripts/train.py `
  --task Z1-AMP-Flat `
  --device cuda:0 `
  --num-envs 256 `
  --max-iterations 30000 `
  --experiment-name z1_amp_unified `
  --run-name loco_v2 `
  --resume-checkpoint "$checkpoint"
```

Play:

```powershell
python scripts/play.py `
  --task Z1-AMP-Flat `
  --checkpoint-file "$checkpoint"
```

Eval:

```powershell
python scripts/eval.py `
  --task Z1-AMP-Flat `
  --checkpoint-file "$checkpoint" `
  --steps 1000 `
  --num-envs 256
```

Export:

```powershell
python scripts/export_policy.py `
  --task Z1-AMP-Flat `
  --checkpoint-file "$checkpoint" `
  --output-dir export
```
