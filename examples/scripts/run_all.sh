#!/bin/bash
# Run all FoldCraft example jobs sequentially.
DIR="$(dirname "$0")"

# Fold-conditioned binder design vs PD-L1 (six folds)
for s in design_1qys_pd_l1 design_6d0t_pd_l1 design_3sd2_pd_l1 \
         design_5bvl_pd_l1 design_3jx8_pd_l1 design_5aao_pd_l1; do
    echo "=== Running $s ==="
    bash "$DIR/$s.sh"
done

# VHH nanobody design vs four targets
for s in design_vhh_pd_l1 design_vhh_pd_1 design_vhh_ifnar design_vhh_egfr; do
    echo "=== Running $s ==="
    bash "$DIR/$s.sh"
done
