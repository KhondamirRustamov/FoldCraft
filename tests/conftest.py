"""Shared fixtures for the FoldCraft test suite.

These tests are *characterization* tests: they lock in the CURRENT behavior of
the GPU-free helper functions so we can refactor and tune the design pipeline
without silently changing residue selection, interface detection, or scoring.
Where current behavior looks like a latent bug, the test documents it
explicitly (see ``test_biopython_utils.py``) rather than asserting the
"intended" behavior — so a future fix will trip the test and force a conscious
decision.
"""
import os
import sys

import pytest
from Bio.PDB import PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

# Make the repo root importable so `import biopython_utils` works regardless of
# where pytest is invoked from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

@pytest.fixture(scope="session")
def pdb_1qys():
    """Path to a real single-chain protein structure (Top7 / 1QYS template)."""
    path = os.path.join(REPO_ROOT, "examples", "templates", "1qys1.pdb")
    assert os.path.exists(path), f"missing test fixture: {path}"
    return path


@pytest.fixture(scope="session")
def pdb_pdl1():
    """Path to a real single-chain protein structure (PD-L1 target)."""
    path = os.path.join(REPO_ROOT, "examples", "targets", "pd-l1-1.pdb")
    assert os.path.exists(path), f"missing test fixture: {path}"
    return path


def _add_residue(chain, resseq, resname, atoms):
    """Add a residue with the given (name, (x, y, z)) atoms to a chain."""
    res = Residue((" ", resseq, " "), resname, "")
    for i, (atom_name, coord) in enumerate(atoms):
        atom = Atom(
            atom_name,
            coord,
            0.0,          # bfactor
            1.0,          # occupancy
            " ",          # altloc
            atom_name,    # fullname
            i,            # serial_number
            element=atom_name[0],
        )
        res.add(atom)
    chain.add(res)
    return res


@pytest.fixture()
def two_chain_complex(tmp_path):
    """A minimal 2-chain complex written to disk, with a known interface.

    Layout (CA-only, coordinates in Angstrom):
      - chain A (target): residue 1 ALA at the origin.
      - chain B (binder): residue 1 GLY ~3 A from chain A  -> interface contact
                          residue 2 LEU ~20 A away          -> NOT a contact

    With a 4.0 A cutoff, ``hotspot_residues(pdb, "B")`` should report exactly
    binder residue 1 (GLY -> "G"). This pins interface detection, which drives
    the non-interface ProteinMPNN redesign in the design pipeline.
    """
    structure = Structure("mini")
    model = Model(0)
    structure.add(model)

    chain_a = Chain("A")
    chain_b = Chain("B")
    model.add(chain_a)
    model.add(chain_b)

    _add_residue(chain_a, 1, "ALA", [("CA", (0.0, 0.0, 0.0))])
    _add_residue(chain_b, 1, "GLY", [("CA", (3.0, 0.0, 0.0))])
    _add_residue(chain_b, 2, "LEU", [("CA", (20.0, 0.0, 0.0))])

    out = tmp_path / "mini_complex.pdb"
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out))
    return str(out)
