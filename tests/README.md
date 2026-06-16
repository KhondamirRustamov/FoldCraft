# FoldCraft test suite

This suite was added to give the project a regression safety net before we
refactor or tune the design pipeline. Upstream FoldCraft shipped **no tests**.

## What is covered

These are **characterization tests**: they assert the *current* behavior of the
GPU-free helper functions so that refactors and performance/accuracy work do not
silently change residue selection, interface detection, or scoring.

Covered today (all CPU-only, run on a laptop in seconds):

- `biopython_utils.set_range` — hotspot/mask range parsing
- `biopython_utils.calculate_percentages` — secondary-structure fractions
- `biopython_utils.validate_design_sequence` — composition notes
- `biopython_utils.calculate_clash_score` — clash counting (real PDB fixtures)
- `biopython_utils.hotspot_residues` — interface detection (synthetic complex)
- `biopython_utils.target_pdb_rmsd` — CA RMSD after superposition
- `cmap_utils` — fold-conditioned cmap assembly + binder_mask (differential test
  against an independent re-implementation, plus adversarial mask inputs)

## Fixed behaviors the tests lock

- **`set_range` is inclusive of both endpoints.** `"39-45"` -> `39..45`,
  `"80-81"` -> `[80, 81]`, `"80-80"` -> `[80]`. (Previously the upper bound was
  exclusive, silently narrowing every hotspot/mask window by one residue — which
  is accuracy-relevant, since it decides which residues the cmap loss conditions
  on.)

## What is NOT covered yet

The two design drivers (`FoldCraft.py`, `FoldCraft_binder.py`) are monolithic
`main()` functions requiring a GPU, AlphaFold2 weights, `colabdesign`, and the
side-loaded `BindCraft` package (`from BindCraft.functions import *`, which is
neither vendored nor declared). They have no importable seams. Covering them
requires the planned refactor into testable units; that work is tracked
separately.

## Running

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python -m pytest
```
