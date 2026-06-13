# Curriculum

The policy is trained on the curated `Walk/` locomotion dataset.

## Current training setup:

- Single locomotion policy trained from scratch.
- `Walk/` provides upright resets and locomotion behavior.
- Style regularization is applied via the AMP discriminator reading the locomotion clips.
