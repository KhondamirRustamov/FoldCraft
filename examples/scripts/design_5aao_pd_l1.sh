#!/bin/bash
# FoldCraft fold-conditioned binder design
# Fold: Ankyrin repeat  |  Binder template: 5AAO (trimmed to 12-136)  |  Target: PD-L1
# Hotspots extracted from conditioned cmap custom_template/fold_con/cmaps/5aao_pd-l1.npy
# (renumbered-from-1 PDB numbering)
cd "$(dirname "$0")/../.." || exit 1

python FoldCraft.py \
    --output_folder examples/outputs/design_5aao_pd_l1 \
    --binder_template examples/templates/5aao1.pdb \
    --target_template examples/targets/pd-l1-1.pdb \
    --target_hotspots '30-34,50-54,69-76' \
    --binder_hotspots '15-26,48-58,81-91,116-124' \
    --num_designs 40
