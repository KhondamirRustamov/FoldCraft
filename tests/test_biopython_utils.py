"""Tests for the GPU-free helpers in biopython_utils.py.

Covers hotspot/mask range parsing (``set_range``, inclusive of both endpoints),
sequence-composition notes, clash counting, interface detection, and CA-RMSD
superposition.
"""
import os

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


# ---------------------------------------------------------------------------
# iter_until_target -- budget cap for the --sample design loop. Regression
# guard for the overrun where the final MPNN batch kept saving every passing
# sample past --target_success (the outer `while passed < target` only stops
# *between* trajectories). The adversarial case is a batch of all-passing
# samples with little remaining budget: the old uncapped `enumerate` accepted
# the whole batch; iter_until_target must stop at exactly the target.
# ---------------------------------------------------------------------------
class TestIterUntilTarget:
    def test_all_passing_batch_stops_at_exact_budget(self):
        # The bug: a 5-sample batch, target 2 -> old code accepted all 5.
        accepted = []
        for _i, item in bu.iter_until_target("abcde", lambda: len(accepted), 2):
            accepted.append(item)          # every sample "passes the gate"
        assert accepted == ["a", "b"]      # exactly target, not the whole batch

    def test_failures_do_not_consume_budget(self):
        accepted, seen = [], []
        for i, _item in bu.iter_until_target(range(10), lambda: len(accepted), 2):
            seen.append(i)
            if i in (1, 4):                # only these clear the success gate
                accepted.append(i)
        assert accepted == [1, 4]          # gate failures don't burn budget
        assert seen == [0, 1, 2, 3, 4]     # stopped right after the 2nd acceptance

    def test_yields_indices_like_enumerate(self):
        assert list(bu.iter_until_target("xy", lambda: 0, 10)) == [(0, "x"), (1, "y")]

    def test_already_at_target_yields_nothing(self):
        # No remaining budget -> no per-item work (no AF2 prediction) is started.
        assert list(bu.iter_until_target("abc", lambda: 5, 2)) == []

    def test_lazy_stop_skips_remaining_work(self):
        # Proves the cap stops *before* doing per-item work past the target,
        # not just before recording it (the compute-saving property).
        touched = []
        accepted = []
        def gen():
            for ch in "abcde":
                touched.append(ch)
                yield ch
        for _i, item in bu.iter_until_target(gen(), lambda: len(accepted), 2):
            accepted.append(item)
        assert touched == ["a", "b"]       # c/d/e never pulled from the source


# ---------------------------------------------------------------------------
# write_atomic -- stream to <path>.partial, promote to <path> only on finalize.
# Regression guard for the scheduler interaction: writing results.csv
# incrementally made a preempted chunk's partial table look like a finished
# chunk (results.csv existing == done), so it was skipped on resume and merged
# as complete. The final file must appear only when the run actually finished.
# ---------------------------------------------------------------------------
class TestWriteAtomic:
    def test_partial_until_finalized(self, tmp_path):
        final = str(tmp_path / "results.csv")
        bu.write_atomic(final, lambda p: open(p, "w").write("a,b\n1,2\n"))
        assert not os.path.exists(final)              # incomplete -> no results.csv
        assert os.path.exists(final + ".partial")     # progress preserved
        bu.write_atomic(final, lambda p: open(p, "w").write("a,b\n1,2\n3,4\n"),
                        finalize=True)
        assert os.path.exists(final)                  # atomically promoted
        assert not os.path.exists(final + ".partial") # partial consumed by rename
        assert open(final).read().count("\n") == 3    # final holds the full table

    def test_crash_before_finalize_leaves_no_final(self, tmp_path):
        # Simulated preemption: several incremental writes, never finalized. The
        # scheduler must NOT see results.csv (else it skips/merges as done).
        final = str(tmp_path / "results.csv")
        for n in range(3):
            bu.write_atomic(final, lambda p, n=n: open(p, "w").write("row\n" * (n + 1)))
        assert not os.path.exists(final)
        assert os.path.exists(final + ".partial")
        assert open(final + ".partial").read() == "row\nrow\nrow\n"  # latest checkpoint
