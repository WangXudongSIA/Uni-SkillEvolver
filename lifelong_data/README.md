Uni-SkillEvolver data placeholders
==================================

This directory is intentionally light for now. Put future lifelong task assets here:

- `instructions/`: text, json, or jsonl instruction samples used by TSDA to build each skill semantic subspace.
- `tasks/`: task sequence metadata once the LRSL datasets are prepared.

The current code path can already run with `--lifelong_instruction_path` or a literal
`--lifelong_instruction_text "instruction one||instruction two"` string. If neither is
provided, the training script falls back to the dataset name as a minimal semantic seed.
