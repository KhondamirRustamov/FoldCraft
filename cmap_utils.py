"""Pure (GPU-free) construction of the fold-conditioned contact map.

The fold-conditioned cmap is the artifact the design loss conditions on, so it
is the single most accuracy-relevant *deterministic* step in the pipeline. This
module extracts its assembly out of ``FoldCraft.py``'s ``main()`` so it can be
unit-tested and shared by both the standard and VHH paths (which previously
duplicated the same logic inline).

IMPORTANT: this is a *behavior-preserving* extraction, not a cleanup. The index
arithmetic is reproduced exactly as in the original inline code, including:

  - ``set_range``'s exclusive upper bound (``"39-45"`` -> 39..44);
  - the residue-number -> array-index ``-1`` offsets;
  - the quirky default branch ``np.array([range(0, binder_len)])`` (a 2-D
    ``(1, binder_len)`` array that drives NumPy fancy-indexing in the contact
    loop) used when no binder hotspots are given.

Any change to these semantics alters design inputs and must be made
deliberately and measured against the design baseline -- not folded into this
extraction. See tests/test_cmap_utils.py, which pins the behavior with a
verbatim reference implementation.
"""
import numpy as np

from biopython_utils import set_range


def apply_binder_mask(binder_cmap, binder_mask):
    """Zero out masked binder rows/columns in the binder's intra-chain cmap.

    Mirrors the original inline behavior, including using ``set_range``'s
    1-based residue numbers directly as 0-based indices into ``binder_cmap``.
    Operates on a copy; the input array is not mutated (the original mutated
    ``af_binder.aux['cmap']`` in place, but nothing downstream reads it again,
    so this is value-equivalent for the pipeline).
    """
    binder_cmap = np.array(binder_cmap, copy=True)
    if binder_mask != '':
        for i in set_range(binder_mask):
            binder_cmap[i, :] = 0.
            binder_cmap[:, i] = 0.
    return binder_cmap


def assemble_fold_conditioned_cmap(binder_cmap, target_len, binder_len,
                                   target_hotspots, binder_hotspots,
                                   binder_mask=''):
    """Build the ``(target_len + binder_len)`` square fold-conditioned cmap.

    Parameters
    ----------
    binder_cmap : array (binder_len x binder_len)
        The binder's intra-chain contact map. In the standard path this comes
        from an AlphaFold2 prediction of the binder monomer; in the VHH path it
        is loaded from ``framework/vhh.npy``.
    target_len, binder_len : int
        Chain lengths. The binder block is placed in the bottom-right corner.
    target_hotspots, binder_hotspots : str
        Residue-range strings (e.g. ``"36-41,84-88"``). If ``binder_hotspots``
        is the empty string, all binder residues are used (the original
        default).
    binder_mask : str, optional
        Residue ranges in the binder to zero out before assembly. The VHH path
        does not use a binder mask; pass ``''`` to match it.

    Returns
    -------
    np.ndarray
        The fold-conditioned cmap (float; target<->binder hotspot contacts
        marked ``1.0``).
    """
    binder_cmap = apply_binder_mask(binder_cmap, binder_mask)

    target_hotspots_np = np.array(set_range(target_hotspots))

    fc_cmap = np.zeros((target_len + binder_len, target_len + binder_len))

    if binder_hotspots == '':
        cdr_range = np.array([range(0, binder_len)]) + target_len
    else:
        cdr_range = np.array(set_range(binder_hotspots)) + target_len

    fc_cmap[-binder_len:, -binder_len:] = binder_cmap

    for i in target_hotspots_np:
        for x in cdr_range:
            fc_cmap[x - 1, i - 1] = 1.
            fc_cmap[i - 1, x - 1] = 1.

    return fc_cmap


def binarize_cmap(fc_cmap):
    """Return the binary contact mask (any positive entry -> 1).

    This is the ``cond_cmap_mask`` the design loop loads alongside the cmap.
    Returns a copy; does not mutate the input.
    """
    mask = np.array(fc_cmap, copy=True)
    mask[mask > 0] = 1
    return mask
