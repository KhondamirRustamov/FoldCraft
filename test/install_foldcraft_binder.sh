#!/bin/bash
# =============================================================================
# Install the BindCraft dependency for the experimental FoldCraft binder
# pipeline (test/FoldCraft_binder.py).
#
# BindCraft (https://github.com/martinpacesa/BindCraft) provides the PyRosetta
# relax + interface-scoring helpers (pr_relax, score_interface) and the bundled
# DAlphaBall.gcc binary. It is NOT pip-installable, so it is cloned next to this
# script (test/BindCraft) where bindcraft_deps.py looks for it by default.
#
# This is separate from the main install_foldcraft.sh on purpose: the validated
# FoldCraft pipeline (FoldCraft.py) does not need BindCraft — only the
# experimental binder script does.
#
# Usage:
#   bash test/install_foldcraft_binder.sh
#   BINDCRAFT_COMMIT=<sha> bash test/install_foldcraft_binder.sh   # pin a revision
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINDCRAFT_DIR="${SCRIPT_DIR}/BindCraft"
BINDCRAFT_URL="https://github.com/martinpacesa/BindCraft"
BINDCRAFT_COMMIT="${BINDCRAFT_COMMIT:-}"

echo "Installing BindCraft into ${BINDCRAFT_DIR}"
if [ ! -d "${BINDCRAFT_DIR}" ]; then
    git clone "${BINDCRAFT_URL}" "${BINDCRAFT_DIR}" \
        || { echo "Error: Failed to clone BindCraft"; exit 1; }
else
    echo "BindCraft already present, skipping clone."
fi

# Pin a specific revision for reproducibility if requested.
if [ -n "${BINDCRAFT_COMMIT}" ]; then
    git -C "${BINDCRAFT_DIR}" fetch --all --quiet || true
    git -C "${BINDCRAFT_DIR}" checkout "${BINDCRAFT_COMMIT}" \
        || { echo "Error: Failed to checkout BindCraft commit ${BINDCRAFT_COMMIT}"; exit 1; }
fi

# Verify the expected layout.
[ -f "${BINDCRAFT_DIR}/functions/__init__.py" ] \
    || { echo "Error: BindCraft checkout is missing functions/__init__.py"; exit 1; }

# DAlphaBall.gcc (interface holes scoring) ships with BindCraft; ensure executable.
[ -f "${BINDCRAFT_DIR}/functions/DAlphaBall.gcc" ] \
    || { echo "Error: BindCraft is missing functions/DAlphaBall.gcc"; exit 1; }
chmod +x "${BINDCRAFT_DIR}/functions/DAlphaBall.gcc" \
    || echo "Warning: could not chmod +x DAlphaBall.gcc"

# Confirm bindcraft_deps.py can locate the checkout.
python -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from bindcraft_deps import find_bindcraft_repo; print('BindCraft at', find_bindcraft_repo())" \
    || { echo "Error: BindCraft not importable via bindcraft_deps"; exit 1; }

echo "BindCraft dependency installed for the FoldCraft binder pipeline."
