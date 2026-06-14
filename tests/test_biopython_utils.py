"""Characterization tests for biopython_utils.py.

Goal: lock the CURRENT behavior of the GPU-free helpers before we refactor or
tune the design pipeline. Several tests document behavior that is arguably a
bug (notably the exclusive upper bound in ``set_range``); they are marked with
a `BUG:` note. When we deliberately fix one, the corresponding test should be
updated in the same commit so the change is explicit and reviewed.
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

    def test_range_is_upper_bound_EXCLUSIVE(self):
        # BUG (characterized, not endorsed): "39-45" expands to 39..44 and
        # DROPS the endpoint 45, because the implementation uses
        # range(start, end) with no +1. A user typing "39-45" almost certainly
        # means residues 39 through 45 inclusive. This silently narrows every
        # hotspot/mask window by one residue at the top.
        assert bu.set_range("39-45") == [39, 40, 41, 42, 43, 44]
        assert 45 not in bu.set_range("39-45")

    def test_mixed_ranges_and_singletons(self):
        # Singletons keep their value; ranges drop their endpoint.
        assert bu.set_range("14-30,80-81,90-102") == (
            list(range(14, 30)) + [80] + list(range(90, 102))
        )

    def test_single_width_range_yields_empty(self):
        # BUG (characterized): "80-81" yields only [80]; "80-80" yields [].
        assert bu.set_range("80-81") == [80]
        assert bu.set_range("80-80") == []

    def test_order_preserved(self):
        assert bu.set_range("90-92,10") == [90, 91, 10]


# ---------------------------------------------------------------------------
# calculate_percentages -- pure arithmetic for secondary-structure fractions
# ---------------------------------------------------------------------------
class TestCalculatePercentages:
    def test_basic_split(self):
        # total=10, helix=5, sheet=3 -> loop=2
        assert bu.calculate_percentages(10, 5, 3) == (50.0, 30.0, 20.0)

    def test_zero_total_is_safe(self):
        assert bu.calculate_percentages(0, 0, 0) == (0, 0, 0)

    def test_all_helix(self):
        assert bu.calculate_percentages(4, 4, 0) == (100.0, 0.0, 0.0)

    def test_rounding_to_two_decimals(self):
        # 1/3 -> 33.33
        h, s, l = bu.calculate_percentages(3, 1, 1)
        assert h == 33.33 and s == 33.33 and l == 33.33


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
