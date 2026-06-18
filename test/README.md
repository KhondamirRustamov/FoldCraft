# Experimental

Code here is **experimental** and **not part of the validated FoldCraft pipeline**.
It is not covered by the paper's benchmarks and may change or break without notice.
For published results, use `FoldCraft.py` in the repository root.

## Contents

- `FoldCraft_binder.py` — length-variable de novo binder design (exploratory
  linear / miniprotein binder runs, e.g. Nipah, KEAP1).
- `bindcraft_deps.py` — locates a BindCraft checkout and exposes the PyRosetta
  helpers (`pr_relax`, `score_interface`) and `DAlphaBall.gcc` that the binder
  pipeline reuses.
- `install_foldcraft_binder.sh` — clones BindCraft into `test/BindCraft` and
  verifies it.

## Dependencies

`FoldCraft_binder.py` reuses helpers from
[BindCraft](https://github.com/martinpacesa/BindCraft), which is not
pip-installable. The main `install_foldcraft.sh` does **not** install it (the
validated `FoldCraft.py` doesn't need it). Install it once for this pipeline:

```bash
bash test/install_foldcraft_binder.sh
# optionally pin a revision:
BINDCRAFT_COMMIT=<sha> bash test/install_foldcraft_binder.sh
```

This clones BindCraft to `test/BindCraft` (git-ignored). Alternatively, point
`bindcraft_deps.py` at an existing checkout with `export BINDCRAFT_PATH=/path/to/BindCraft`.

## Usage

These scripts import `biopython_utils` from the repository root, so run them
**from the repo root**:

```bash
python test/FoldCraft_binder.py --help
```
