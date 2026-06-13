# Hardware Notes

## Baseline machine

- GPU: RTX 3050 6 GB
- RAM: 64 GB

## Safe starting defaults

- `--num-envs 256`
- headless training
- flat terrain first
- separate training and playback runs

## Suggested env-count sweep

- 256
- 512
- 768
- 1024

Move upward only after memory and step time look stable.

## Practical note

This repo is structured for Linux/WSL first. Windows-native runtime may still
need extra dependency work depending on the local MuJoCo and `mjlab` install.
