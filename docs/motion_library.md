# Motion Library

## Active layout

The active AMP library root is:

```text
src/assets/motions/z1/curated/amp/
```

Subfolder:

- `Walk/`

## Curated files

- `src/assets/motions/z1/curated/amp/Walk/`
  Active compiled locomotion AMP library used for training.
- `src/assets/motions/z1/curated/library_manifest.json`
  Build report and quality metrics manifest record.


## Current training usage

During training:

- Normal resets sample from `Walk/`
- The AMP expert dataset reads the `curated/amp/` root (specifically `Walk/`) recursively to learn locomotion style regularizations.
