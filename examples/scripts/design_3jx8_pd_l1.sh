#!/bin/bash
# FoldCraft fold-conditioned binder design
# Fold: alpha-solenoid  |  Binder template: 3JX8 (trimmed to 172-269)  |  Target: PD-L1
# Hotspots extracted from conditioned cmap custom_template/fold_con/cmaps/3jx8_pd-l1.npy
# (renumbered-from-1 PDB numbering)
cd "$(dirname "$0")/../.." || exit 1

python FoldCraft.py \
    --output_folder examples/outputs/design_3jx8_pd_l1 \
    --binder_template examples/templates/3jx81.pdb \
    --target_template examples/targets/pd-l1-1.pdb \
    --target_hotspots '30-34,50-54,69-76' \
    --binder_hotspots '7-13,26-33,46-53,64-71,83-88' \
    --num_designs 40
