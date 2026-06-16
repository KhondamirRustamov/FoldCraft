#!/bin/bash
# FoldCraft fold-conditioned binder design
# Fold: Top7  |  Binder template: 1QYS  |  Target: PD-L1
# Hotspots extracted from conditioned cmap custom_template/fold_con/cmaps/1qys_pd-l1.npy
# (renumbered-from-1 PDB numbering)
cd "$(dirname "$0")/../.." || exit 1

python FoldCraft.py \
    --output_folder examples/outputs/design_1qys_pd_l1 \
    --binder_template examples/templates/1qys1.pdb \
    --target_template examples/targets/pd-l1-1.pdb \
    --target_hotspots '30-34,50-54,69-76' \
    --binder_hotspots '26-40,58-71' \
    --num_designs 40
