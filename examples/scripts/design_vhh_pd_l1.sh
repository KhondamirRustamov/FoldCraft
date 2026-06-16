#!/bin/bash
# FoldCraft VHH nanobody design (VHH framework, binder template ignored)
# Target: PD-L1  |  Target hotspots from custom_template/vhh/vhh_pd_l1.npy (renumbered-from-1)
cd "$(dirname "$0")/../.." || exit 1

python FoldCraft.py \
    --vhh \
    --output_folder examples/outputs/design_vhh_pd_l1 \
    --target_template examples/targets/pd-l1-1.pdb \
    --target_hotspots '30-34,50-54,69-76' \
    --num_designs 40
