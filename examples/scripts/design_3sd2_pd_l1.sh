#!/bin/bash
# FoldCraft fold-conditioned binder design
# Fold: Ig-like  |  Binder template: 3SD2 (trimmed to 42-118)  |  Target: PD-L1
# Hotspots extracted from conditioned cmap custom_template/fold_con/cmaps/3sd2_pd-l1.npy
# (renumbered-from-1 PDB numbering)
cd "$(dirname "$0")/../.." || exit 1

python FoldCraft.py \
    --output_folder examples/outputs/design_3sd2_pd_l1 \
    --binder_template examples/templates/3sd21.pdb \
    --target_template examples/targets/pd-l1-1.pdb \
    --target_hotspots '30-34,50-54,69-76' \
    --binder_hotspots '17-24,39-46' \
    --num_designs 40
