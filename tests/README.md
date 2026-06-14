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

## Known issues the tests pin (not yet fixed)

Some tests document behavior that is arguably a **latent bug**. They are named
and commented with `BUG:` so the behavior is locked but flagged:

- **`set_range` upper bound is exclusive.** `"39-45"` expands to `39..44`,
  dropping residue 45; `"80-81"` yields only `[80]`; `"80-80"` yields `[]`.
  This silently narrows every hotspot/mask window by one residue at the top and
  is accuracy-relevant (it decides which residues the cmap loss conditions on).
  Fixing it will change design inputs, so it must be done deliberately, with the
  test updated in the same commit and a baseline re-run to confirm the effect.

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
