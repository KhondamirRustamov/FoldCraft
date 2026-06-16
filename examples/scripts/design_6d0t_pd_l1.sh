#!/bin/bash
# FoldCraft fold-conditioned binder design
# Fold: beta-barrel  |  Binder template: 6D0T  |  Target: PD-L1
# Hotspots extracted from conditioned cmap custom_template/fold_con/cmaps/6d0t_pd-l1.npy
# (renumbered-from-1 PDB numbering)
cd "$(dirname "$0")/../.." || exit 1

python FoldCraft.py \
    --output_folder examples/outputs/design_6d0t_pd_l1 \
    --binder_template examples/templates/6d0t1.pdb \
    --target_template examples/targets/pd-l1-1.pdb \
    --target_hotspots '30-34,50-54,69-76' \
    --binder_hotspots '12-15,25-31,40-43' \
    --num_designs 40
