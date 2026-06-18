# FoldCraft Examples

Ready-to-run examples that reproduce the design runs from the
[FoldCraft preprint](https://www.biorxiv.org/content/10.1101/2025.07.02.662497v1.abstract):

1. **Fold-conditioned binder design** against PD-L1 across six folds
   (Top7, β-barrel, Ig-like, TIM-barrel, α-solenoid, Ankyrin).
2. **VHH nanobody design** against four targets (PD-L1, PD-1, IFNAR2, EGFR).

## Folder layout

```
examples/
├── templates/   # 6 monomeric fold templates (binder fold references)
├── targets/     # 4 target structures (AF2 predictions, trimmed & renumbered from 1)
├── cmaps/       # precomputed conditioned contact maps (.npy) + masks, one pair per run
└── scripts/     # one runnable script per design job + run_all.sh
```

### `templates/` — fold templates (Supplementary Table S1)

| File | Fold | PDB |
|------|------|-----|
| `1qys1.pdb` | Top7 | 1QYS |
| `6d0t1.pdb` | β-barrel | 6D0T |
| `3sd21.pdb` | Ig-like | 3SD2 (42–118) |
| `5bvl_af2.pdb` | TIM barrel | 5BVL |
| `3jx81.pdb` | α-solenoid | 3JX8 (172–269) |
| `5aao1.pdb` | Ankyrin repeat | 5AAO (12–136) |

### `targets/` — target structures (Supplementary Table S2)

| File | Target | Source |
|------|--------|--------|
| `pd-l1-1.pdb` | PD-L1 | AF2 prediction, trimmed to 20–131 |
| `pd-1.pdb` | PD-1 | AF2 prediction, trimmed to 32–146 |
| `ifnar.pdb` | IFNAR2 | AF2 prediction of 2LAG, trimmed to 8–110 |
| `egfr.pdb` | EGFR | AF2 prediction, trimmed to 334–507 |

> **Note on residue numbering.** All PDBs here are renumbered starting at residue 1.
> The `--target_hotspots` / `--binder_hotspots` in the scripts use this
> renumbered-from-1 convention (extracted directly from the conditioned cmaps in
> `cmaps/`), **not** the original PDB/UniProt numbering printed in the paper's
> supplementary tables.

### `cmaps/` — conditioned contact maps

For each run there is a pair of NumPy arrays:

- `<name>.npy` — the fold-conditioned template contact map (binder intra-chain
  contacts + masked target + defined target↔binder inter-chain contacts).
- `<name>_mask.npy` — the binary mask restricting the loss to the relevant
  contact regions.

These are provided for reference / inspection. `FoldCraft.py` builds the cmap at
runtime from the template + hotspots, so you do **not** need to pass these files
to reproduce a run.

## Prerequisites

Install FoldCraft and activate the environment (see the repository root README):

```bash
bash install_foldcraft.sh --cuda '12.4' --pkg_manager 'conda'
conda activate FoldCraft
```

## Running the examples

Each script `cd`s to the repository root automatically, so you can launch it from
anywhere. Results are written to `examples/outputs/<job_name>/`.

Run a single job:

```bash
bash examples/scripts/design_1qys_pd_l1.sh     # Top7 fold vs PD-L1
bash examples/scripts/design_vhh_egfr.sh       # VHH nanobody vs EGFR
```

Run everything (six fold jobs + four VHH jobs, sequentially):

```bash
bash examples/scripts/run_all.sh
```

## What each script runs

**Fold-conditioned binder design (vs PD-L1)** — target hotspots
`30-34,50-54,69-76` (5BVL: `29-35,49-55,68-77`):

| Script | Fold | Binder template | Binder hotspots |
|--------|------|-----------------|-----------------|
| `design_1qys_pd_l1.sh` | Top7 | `1qys1.pdb` | `26-40,58-71` |
| `design_6d0t_pd_l1.sh` | β-barrel | `6d0t1.pdb` | `12-15,25-31,40-43` |
| `design_3sd2_pd_l1.sh` | Ig-like | `3sd21.pdb` | `17-24,39-46` |
| `design_5bvl_pd_l1.sh` | TIM barrel | `5bvl_af2.pdb` | `42-65,68-87,89-113` |
| `design_3jx8_pd_l1.sh` | α-solenoid | `3jx81.pdb` | `7-13,26-33,46-53,64-71,83-88` |
| `design_5aao_pd_l1.sh` | Ankyrin | `5aao1.pdb` | `15-26,48-58,81-91,116-124` |

**VHH nanobody design** (`--vhh`; binder template ignored, VHH framework used):

| Script | Target | Template | Target hotspots |
|--------|--------|----------|-----------------|
| `design_vhh_pd_l1.sh` | PD-L1 | `pd-l1-1.pdb` | `30-34,50-54,69-76` |
| `design_vhh_pd_1.sh` | PD-1 | `pd-1.pdb` | `44-48,81-88,103-106` |
| `design_vhh_ifnar.sh` | IFNAR2 | `ifnar.pdb` | `44-46,73-76,87-91` |
| `design_vhh_egfr.sh` | EGFR | `egfr.pdb` | `106-107,128-131,138-153` |

All jobs use `--num_designs 40`. Tune any run by editing its script — common knobs:

```
--num_designs <N>           # number of design trajectories
--sample --target_success N # keep designing until N successful designs
--mpnn_weight soluble|original
--redesign_method non-interface|full
--mpnn_sampling_temp 0.0-1.0
```

See the repository root `README.md` for the full list of `FoldCraft.py` options.
