"""Locate and expose the BindCraft utilities that FoldCraft's PyRosetta pipeline needs.

``FoldCraft_binder.py`` (in this ``test/`` directory) reuses two functions from
Martin Pacesa's BindCraft (https://github.com/martinpacesa/BindCraft):

  - ``pr_relax(pdb_file, relaxed_pdb_path)``
  - ``score_interface(pdb_file, binder_chain="B")``

plus PyRosetta itself (conventionally imported as ``pr``) and the bundled
``DAlphaBall.gcc`` binary used for interface "holes" scoring.

BindCraft is a collection of scripts, not a pip-installable package, so it has
to be cloned and placed on ``sys.path`` (the import is ``BindCraft.functions``,
which resolves as a namespace package when BindCraft's *parent* directory is on
the path). Run ``test/install_foldcraft_binder.sh`` to clone it next to this
file (``test/BindCraft``). Upstream FoldCraft relied on an implicit
``from BindCraft.functions import *`` that (a) was never installed and (b)
silently shadowed FoldCraft's own ``biopython_utils`` helpers. This module makes
the dependency explicit and fails with an actionable message when it is missing.

The path-resolution logic here is intentionally free of any PyRosetta/BindCraft
import so it can be unit-tested on a machine without the GPU design stack.
"""
import os
import sys

BINDCRAFT_REPO_URL = "https://github.com/martinpacesa/BindCraft"


def _bindcraft_candidates(explicit_path=None):
    """Ordered list of directories that might be (or contain) the BindCraft repo."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    env = os.environ.get("BINDCRAFT_PATH")
    if env:
        candidates.append(env)
    # Installer default: a ``BindCraft`` checkout next to this file (test/BindCraft).
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "BindCraft"))
    return candidates


def _is_bindcraft_repo(path):
    """True if ``path`` is a BindCraft checkout (has functions/__init__.py)."""
    return os.path.isfile(os.path.join(path, "functions", "__init__.py"))


def find_bindcraft_repo(explicit_path=None):
    """Return the absolute path to the BindCraft repository directory.

    A candidate matches if it either *is* a BindCraft checkout or *contains* a
    ``BindCraft`` subdirectory that is one. Raises ``ImportError`` with install
    instructions if none match.
    """
    for cand in _bindcraft_candidates(explicit_path):
        cand = os.path.abspath(cand)
        if _is_bindcraft_repo(cand):
            return cand
        nested = os.path.join(cand, "BindCraft")
        if _is_bindcraft_repo(nested):
            return nested
    searched = [os.path.abspath(c) for c in _bindcraft_candidates(explicit_path)]
    raise ImportError(
        "FoldCraft's binder pipeline requires BindCraft, which was not found.\n"
        "Run test/install_foldcraft_binder.sh, or clone it manually:\n"
        f"    git clone {BINDCRAFT_REPO_URL}\n"
        "and/or set the BINDCRAFT_PATH environment variable to its location.\n"
        f"Searched: {searched}"
    )


def ensure_bindcraft_importable(explicit_path=None):
    """Put BindCraft's parent dir on ``sys.path`` so ``import BindCraft`` works.

    Returns the BindCraft repository path. Idempotent.
    """
    repo = find_bindcraft_repo(explicit_path)
    parent = os.path.dirname(repo)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return repo


def dalphaball_path(explicit_path=None):
    """Absolute path to the DAlphaBall.gcc binary bundled with BindCraft.

    Falls back to a ``./DAlphaBall.gcc`` in the current directory if present
    (preserving upstream FoldCraft's behavior). Raises FileNotFoundError with
    guidance otherwise.
    """
    repo = find_bindcraft_repo(explicit_path)
    bundled = os.path.join(repo, "functions", "DAlphaBall.gcc")
    if os.path.isfile(bundled):
        return bundled
    cwd_copy = os.path.abspath("DAlphaBall.gcc")
    if os.path.isfile(cwd_copy):
        return cwd_copy
    raise FileNotFoundError(
        "DAlphaBall.gcc not found. It ships with BindCraft at "
        "functions/DAlphaBall.gcc and is required for interface holes scoring. "
        "Run test/install_foldcraft_binder.sh to set it up.\n"
        f"Looked for: {bundled} and {cwd_copy}"
    )
