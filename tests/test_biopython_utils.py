"""Tests for the GPU-free helpers in biopython_utils.py.

Covers hotspot/mask range parsing (``set_range``, inclusive of both endpoints),
sequence-composition notes, clash counting, interface detection, and CA-RMSD
superposition.
"""
import pytest

import biopython_utils as bu


# ---------------------------------------------------------------------------
# set_range  -- parses hotspot/mask range strings like "39-45,80,90-102"
# This drives every hotspot/mask selection, so its semantics are
# accuracy-critical: it decides which residues the cmap loss conditions on.
# ---------------------------------------------------------------------------
class TestSetRange:
    def test_single_residue(self):
        assert bu.set_range("80") == [80]

    def test_comma_separated_singletons(self):
        assert bu.set_range("80,90,102") == [80, 90, 102]

    def test_range_is_upper_bound_INCLUSIVE(self):
        # Fixed: "39-45" now includes the endpoint 45 (was exclusive, which
        # silently dropped the last residue of every hotspot/mask window).
        assert bu.set_range("39-45") == [39, 40, 41, 42, 43, 44, 45]
        assert 45 in bu.set_range("39-45")

    def test_mixed_ranges_and_singletons(self):
        # Singletons keep their value; ranges include both endpoints.
        assert bu.set_range("14-30,80-81,90-102") == (
            list(range(14, 31)) + [80, 81] + list(range(90, 103))
        )

    def test_single_width_range(self):
        # "80-81" -> both endpoints; "80-80" -> the single residue 80.
        assert bu.set_range("80-81") == [80, 81]
        assert bu.set_range("80-80") == [80]

    def test_order_preserved(self):
        assert bu.set_range("90-92,10") == [90, 91, 92, 10]


# ---------------------------------------------------------------------------
# validate_design_sequence -- composition notes for a designed sequence
# ---------------------------------------------------------------------------
class TestValidateDesignSequence:
    def test_clean_high_absorption_sequence_no_notes(self):
        # Tryptophan-rich sequence => high extinction => no absorption warning.
        settings = {"omit_AAs": ""}
        notes = bu.validate_design_sequence("WWWWYYYY", 0, settings)
        assert notes == ""

    def test_clash_note(self):
        settings = {"omit_AAs": ""}
        notes = bu.validate_design_sequence("WWWWYYYY", 3, settings)
        assert "clashes" in notes

    def test_omitted_aa_flagged(self):
        settings = {"omit_AAs": "C"}
        notes = bu.validate_design_sequence("WWCWYY", 0, settings)
        assert "Contains: C" in notes

    def test_low_absorption_suggests_tryptophan(self):
        settings = {"omit_AAs": ""}
        notes = bu.validate_design_sequence("AAAAGGGG", 0, settings)
        assert "tryptophane" in notes.lower()


# ---------------------------------------------------------------------------
# calculate_clash_score -- C-alpha / heavy-atom clash counting on a real PDB
# ---------------------------------------------------------------------------
class TestCalculateClashScore:
    def test_real_structure_has_no_ca_clashes(self, pdb_1qys):
        # A well-formed deposited structure should have no CA-CA clashes
        # between non-adjacent residues at the 2.4 A default threshold.
        assert bu.calculate_clash_score(pdb_1qys, only_ca=True) == 0

    def test_returns_int(self, pdb_pdl1):
        score = bu.calculate_clash_score(pdb_pdl1, only_ca=True)
        assert isinstance(score, int)


# ---------------------------------------------------------------------------
# hotspot_residues -- interface residue detection (drives MPNN redesign mask)
# ---------------------------------------------------------------------------
class TestHotspotResidues:
    def test_detects_only_close_binder_residue(self, two_chain_complex):
        result = bu.hotspot_residues(two_chain_complex, binder_chain="B")
        # Only binder residue 1 (GLY) is within 4.0 A of chain A.
        assert result == {1: "G"}

    def test_wider_cutoff_includes_far_residue(self, two_chain_complex):
        # At a 25 A cutoff both binder residues contact chain A.
        result = bu.hotspot_residues(
            two_chain_complex, binder_chain="B", atom_distance_cutoff=25.0
        )
        assert set(result.keys()) == {1, 2}
        assert result[2] == "L"


# ---------------------------------------------------------------------------
# target_pdb_rmsd -- CA RMSD after superposition
# ---------------------------------------------------------------------------
class TestTargetPdbRmsd:
    def test_self_alignment_is_zero(self, pdb_1qys):
        # Aligning a structure's chain A against itself must give ~0 RMSD.
        rmsd = bu.target_pdb_rmsd(pdb_1qys, pdb_1qys, "A")
        assert rmsd == 0.0
