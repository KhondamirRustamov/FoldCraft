#!/bin/bash
# FoldCraft VHH nanobody design (VHH framework, binder template ignored)
# Target: PD-1  |  Target hotspots from custom_template/vhh/vhh_pd_1.npy (renumbered-from-1)
cd "$(dirname "$0")/../.." || exit 1

python FoldCraft.py \
    --vhh \
    --output_folder examples/outputs/design_vhh_pd_1 \
    --target_template examples/targets/pd-1.pdb \
    --target_hotspots '44-48,81-88,103-106' \
    --num_designs 40
