# Pipeline

## Goal

Train a Z1 AMP policy for locomotion in simulation.

## Motion path

Curated outputs:

- `src/assets/motions/z1/curated/amp/` (specifically the `Walk/` subfolder containing NPZ motion files)
- `src/assets/motions/z1/curated/library_manifest.json`

> [!NOTE]
> All raw retargeted motions, intermediate trimmed review files, full-length review files, and recovery clips have been deleted/excluded to maintain a lightweight repository.

## Training path

The training pipeline uses the body-based AMP stack:

- Resets and resets-from-motions sample from `Walk/`
- AMP discriminator expert samples come from `curated/amp/Walk/` recursively.

## Main commands

Train:

```powershell
python scripts/train.py --task Z1-AMP-Flat --device cuda:0 --num-envs 256 --max-iterations 20000 --experiment-name z1_amp_unified --run-name loco_v1
```

Play:

```powershell
python scripts/play.py --task Z1-AMP-Flat --checkpoint-file <path-to-model.pt>
```

Eval:

```powershell
python scripts/eval.py --task Z1-AMP-Flat --checkpoint-file <path-to-model.pt>
```

Export:

```powershell
python scripts/export_policy.py --task Z1-AMP-Flat --checkpoint-file <path-to-model.pt> --output-dir export
```

## Runtime note

By default, the training and execution scripts load motion data from:

- `src/assets/motions/z1/curated/amp/`
