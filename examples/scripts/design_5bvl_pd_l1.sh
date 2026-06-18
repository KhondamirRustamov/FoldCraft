#!/bin/bash
# FoldCraft fold-conditioned binder design
# Fold: TIM barrel  |  Binder template: 5BVL  |  Target: PD-L1
# Hotspots extracted from conditioned cmap custom_template/fold_con/cmaps/5bvl_pd-l1.npy
# (renumbered-from-1 PDB numbering)
cd "$(dirname "$0")/../.." || exit 1

python FoldCraft.py \
    --output_folder examples/outputs/design_5bvl_pd_l1 \
    --binder_template examples/templates/5bvl_af2.pdb \
    --target_template examples/targets/pd-l1-1.pdb \
    --target_hotspots '29-35,49-55,68-77' \
    --binder_hotspots '42-65,68-87,89-113' \
    --num_designs 40
