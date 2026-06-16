#!/bin/bash
# FoldCraft VHH nanobody design (VHH framework, binder template ignored)
# Target: IFNAR2 (AF2 of 2LAG)  |  Target hotspots from custom_template/vhh/vhh_ifnar.npy (renumbered-from-1)
cd "$(dirname "$0")/../.." || exit 1

python FoldCraft.py \
    --vhh \
    --output_folder examples/outputs/design_vhh_ifnar \
    --target_template examples/targets/ifnar.pdb \
    --target_hotspots '44-46,73-76,87-91' \
    --num_designs 40
