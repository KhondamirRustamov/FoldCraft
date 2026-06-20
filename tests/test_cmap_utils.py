"""Tests for the fold-conditioned cmap assembly (cmap_utils.py).

The centerpiece is a *differential* test: ``_reference_cmap`` is an independent
re-implementation of the assembly, and we assert that
``assemble_fold_conditioned_cmap`` produces bit-identical output across a range
of inputs (standard, default-hotspot, masked, and VHH-like cases).

There are also explicit small-case value tests covering the index conventions
(inclusive set_range, residue R -> array index R-1) and adversarial binder_mask
inputs (terminal and out-of-range residues).
"""
import numpy as np
import pytest

import cmap_utils as cu
from biopython_utils import set_range


# ---------------------------------------------------------------------------
# Independent re-implementation of the cmap assembly, used as a differential
# cross-check against assemble_fold_conditioned_cmap (standard/non-VHH path,
# which subsumes the VHH path as a special case).
# ---------------------------------------------------------------------------
def _reference_cmap(load_np, target_len, binder_len, target_hotspots,
                    binder_hotspots, binder_mask):
    load_np = np.array(load_np, copy=True)  # avoid cross-test mutation
    if binder_mask != '':
        for r in set_range(binder_mask):  # 1-based residue -> index R-1
            load_np[r - 1, :] = 0.
            load_np[:, r - 1] = 0.

    target_hotspots_np = np.array(set_range(target_hotspots))

    fc_cmap = np.zeros((target_len + binder_len, target_len + binder_len))

    if binder_hotspots == '':
        # empty == all binder residues, 1-based (1..binder_len); must match an
        # explicit "1-<binder_len>". (Previously 0-based here too, which made the
        # differential test tautological -- it agreed with the same bug.)
        cdr_range = np.array([range(1, binder_len + 1)]) + target_len
    else:
        cdr_range = np.array(set_range(binder_hotspots)) + target_len

    fc_cmap[-binder_len:, -binder_len:] = load_np

    for i in target_hotspots_np:
        for x in cdr_range:
            fc_cmap[x - 1, i - 1] = 1.
            fc_cmap[i - 1, x - 1] = 1.

    return fc_cmap


# Deterministic pseudo-random binder cmap (no Math.random/global-seed needed).
def _fake_binder_cmap(binder_len, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.random((binder_len, binder_len))
    return (m + m.T) / 2.0  # symmetric, like a real contact map


# (target_len, binder_len, target_hotspots, binder_hotspots, binder_mask)
DIFFERENTIAL_CASES = [
    (5, 4, "2-4", "1-3", ""),            # explicit binder hotspots
    (5, 4, "2-4", "", ""),               # default: all binder residues
    (6, 5, "1-3,5", "2-4", "2-3"),       # with a binder mask
    (8, 6, "3-6", "1-2,4-6", ""),        # multiple hotspot ranges
    (10, 127, "5-9,20", "26-35,55-59,102-116", ""),  # VHH-like geometry
    (4, 3, "1", "1", ""),                # singletons
]


@pytest.mark.parametrize(
    "target_len,binder_len,t_hot,b_hot,b_mask", DIFFERENTIAL_CASES
)
def test_matches_reference_implementation(target_len, binder_len, t_hot,
                                          b_hot, b_mask):
    binder_cmap = _fake_binder_cmap(binder_len)
    expected = _reference_cmap(binder_cmap, target_len, binder_len,
                               t_hot, b_hot, b_mask)
    got = cu.assemble_fold_conditioned_cmap(
        binder_cmap, target_len, binder_len, t_hot, b_hot, b_mask
    )
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# Explicit value/structure checks
# ---------------------------------------------------------------------------
class TestStructure:
    def test_shape_is_total_length(self):
        bc = np.zeros((4, 4))
        out = cu.assemble_fold_conditioned_cmap(bc, 5, 4, "2-3", "1-2", "")
        assert out.shape == (9, 9)

    def test_binder_block_placed_bottom_right(self):
        bc = np.full((3, 3), 0.7)
        out = cu.assemble_fold_conditioned_cmap(bc, 4, 3, "1", "", "")
        # bottom-right 3x3 block holds the binder cmap values.
        np.testing.assert_array_equal(out[-3:, -3:], bc)

    def test_hotspot_contacts_are_symmetric(self):
        bc = np.zeros((4, 4))
        out = cu.assemble_fold_conditioned_cmap(bc, 5, 4, "2-4", "1-3", "")
        np.testing.assert_array_equal(out, out.T)

    def test_empty_binder_hotspots_equals_all_residues_explicit(self):
        # Documented meaning of empty binder_hotspots = "all binder residues",
        # so it must equal passing the full 1..binder_len range explicitly. The
        # old 0-based default did NOT (it shifted the interface block up a row);
        # this pins the corrected 1-based behavior independently of the reference.
        bc = _fake_binder_cmap(4)
        default = cu.assemble_fold_conditioned_cmap(bc, 5, 4, "2-4", "", "")
        explicit = cu.assemble_fold_conditioned_cmap(bc, 5, 4, "2-4", "1-4", "")
        np.testing.assert_array_equal(default, explicit)

    def test_empty_default_conditions_every_binder_residue_not_last_target(self):
        # Adversarial small case: target_len=5, binder_len=4, target hotspot "3"
        # (col index 2). Correct binder rows are 5,6,7,8; the bug wrote 4,5,6,7 --
        # a spurious contact on target row 4 and none on binder row 8.
        bc = np.zeros((4, 4))
        out = cu.assemble_fold_conditioned_cmap(bc, 5, 4, "3", "", "")
        for r in (5, 6, 7, 8):                      # every binder residue conditioned
            assert out[r, 2] == 1.0 and out[2, r] == 1.0
        assert out[4, 2] == 0.0 and out[2, 4] == 0.0  # no spurious last-target contact

    def test_known_contact_entries(self):
        # set_range is inclusive: target_hotspots="2-3" -> [2, 3];
        # binder_hotspots="1-3" -> [1, 2, 3]; cdr_range = [1,2,3] + 5 = [6,7,8].
        # contacts set at (x-1, i-1) and (i-1, x-1) for i in {2,3}, x-1 in {5,6,7}.
        bc = np.zeros((4, 4))
        out = cu.assemble_fold_conditioned_cmap(bc, 5, 4, "2-3", "1-3", "")
        for r, c in [(5, 1), (1, 5), (6, 1), (1, 6), (7, 1)]:
            assert out[r, c] == 1.0
        # target residue 3 is now included (inclusive bound) -> (.,2) contacts present
        assert out[6, 2] == 1.0 and out[2, 6] == 1.0
        np.testing.assert_array_equal(out, out.T)  # symmetry preserved


class TestApplyBinderMask:
    def test_masks_correct_residues_one_based(self):
        # binder_mask is 1-based residue numbers -> 0-based positions (R -> R-1).
        bc = np.ones((5, 5))
        masked = cu.apply_binder_mask(bc, "1-2")  # residues 1,2 -> indices 0,1
        for idx in (0, 1):
            assert np.all(masked[idx, :] == 0) and np.all(masked[:, idx] == 0)
        # unmasked indices (2, 3, 4) untouched at their intersections
        assert masked[2, 2] == 1 and masked[3, 4] == 1 and masked[4, 4] == 1
        # input not mutated
        assert np.all(bc == 1)

    def test_terminal_range_does_not_crash(self):
        # Regression: inclusive set_range makes "1-3" -> [1,2,3]; the old code
        # used those as 0-based indices and raised IndexError on a 3-residue cmap.
        bc = np.ones((3, 3))
        masked = cu.apply_binder_mask(bc, "1-3")  # residues 1,2,3 -> indices 0,1,2
        assert np.all(masked == 0)

    def test_out_of_range_residue_raises(self):
        bc = np.ones((3, 3))
        with pytest.raises(ValueError):
            cu.apply_binder_mask(bc, "1-4")  # residue 4 > 3-residue binder
        with pytest.raises(ValueError):
            cu.apply_binder_mask(bc, "0-2")  # residue 0 invalid (1-based)

    def test_empty_mask_is_identity_copy(self):
        bc = np.arange(9).reshape(3, 3).astype(float)
        out = cu.apply_binder_mask(bc, "")
        np.testing.assert_array_equal(out, bc)
        assert out is not bc  # still a copy


class TestBinarize:
    def test_positive_entries_become_one(self):
        a = np.array([[0.0, 0.3], [2.5, 0.0]])
        out = cu.binarize_cmap(a)
        np.testing.assert_array_equal(out, np.array([[0.0, 1.0], [1.0, 0.0]]))

    def test_does_not_mutate_input(self):
        a = np.array([[0.0, 0.3]])
        cu.binarize_cmap(a)
        assert a[0, 1] == 0.3
