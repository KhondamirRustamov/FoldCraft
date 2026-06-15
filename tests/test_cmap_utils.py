"""Tests for the fold-conditioned cmap extraction (cmap_utils.py).

The centerpiece is a *differential* test: ``_reference_cmap`` reproduces the
original inline logic from FoldCraft.py verbatim, and we assert the extracted
``assemble_fold_conditioned_cmap`` produces bit-identical output across a range
of inputs (standard, default-hotspot, masked, and VHH-like cases). This is the
evidence that the extraction is behavior-preserving.

There are also explicit small-case value tests, so a future change to the index
semantics (e.g. fixing set_range's exclusive bound) trips a concrete assertion,
not just the differential check.
"""
import os
import sys

import numpy as np
import pytest

# Make the repo root importable so `import cmap_utils` works when this test runs
# on its own (independently of a test-suite conftest.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cmap_utils as cu
from biopython_utils import set_range


# ---------------------------------------------------------------------------
# Verbatim reference: the original inline assembly from FoldCraft.py main().
# (The standard/non-VHH path, which subsumes the VHH path as a special case.)
# ---------------------------------------------------------------------------
def _reference_cmap(load_np, target_len, binder_len, target_hotspots,
                    binder_hotspots, binder_mask):
    load_np = np.array(load_np, copy=True)  # avoid cross-test mutation
    if binder_mask != '':
        binder_mask = set_range(binder_mask)
        for i in binder_mask:
            load_np[i, :] = 0.
            load_np[:, i] = 0.

    target_hotspots_np = np.array(set_range(target_hotspots))

    fc_cmap = np.zeros((target_len + binder_len, target_len + binder_len))

    if binder_hotspots == '':
        cdr_range = np.array([range(0, binder_len)]) + target_len
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

    def test_known_contact_entries(self):
        # target_len=5, target_hotspots="2-3" -> set_range gives [2] (exclusive!)
        # binder_hotspots="1-3" -> [1, 2]; cdr_range = [1,2] + 5 = [6, 7]
        # contacts set at (x-1, i-1) and (i-1, x-1):
        #   i=2 -> i-1=1 ; x in {6,7} -> x-1 in {5,6}
        # => entries (5,1),(1,5),(6,1),(1,6) == 1
        bc = np.zeros((4, 4))
        out = cu.assemble_fold_conditioned_cmap(bc, 5, 4, "2-3", "1-3", "")
        for r, c in [(5, 1), (1, 5), (6, 1), (1, 6)]:
            assert out[r, c] == 1.0
        # target residue 3 was dropped by the exclusive bound -> no (.,2) contacts
        assert out[6, 2] == 0.0 and out[2, 6] == 0.0


class TestApplyBinderMask:
    def test_masks_rows_and_cols_without_mutating_input(self):
        bc = np.ones((4, 4))
        masked = cu.apply_binder_mask(bc, "1-3")  # set_range -> [1, 2]
        assert np.all(masked[1, :] == 0) and np.all(masked[:, 1] == 0)
        assert np.all(masked[2, :] == 0) and np.all(masked[:, 2] == 0)
        # row/col 0 and 3 untouched at their intersection
        assert masked[0, 3] == 1 and masked[3, 0] == 1
        # input not mutated
        assert np.all(bc == 1)

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
