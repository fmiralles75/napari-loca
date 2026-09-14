#!/usr/bin/env python3
"""network_threshold_diagnostics.py

Standalone diagnostic for napari-organelle-contact-analyzer's Morphology
Network preprocessing knobs: Fill Holes Up To (px), Prune Spurs Under
(px), Collapse Bridges Under (px).

Run this in the SAME Python environment you run napari / the plugin in --
it needs numpy, scipy, scikit-image, and skan (the same libraries the
plugin itself uses), plus matplotlib if you want the histogram plots. It
will NOT run inside a Claude conversation's sandbox, which doesn't have
these installed.

WHAT IT DOES
------------
1. Loads a thresholded (binary) mask -- ideally the exact mask you
   analyzed in the plugin (see "GETTING A MASK" below).
2. Reproduces the plugin's own ``_fill_small_holes`` hole-finding logic
   exactly, to get the real distribution of fully-enclosed background
   hole sizes in your mask -- informs Fill Holes Up To.
3. Hole-fills the mask (using either your suggested or specified cutoff),
   skeletonizes it, and runs ``skan.summarize()`` exactly as the plugin's
   ``_skan_network_analysis`` does, to get the real distribution of
   junction-to-endpoint branch lengths ("spur candidates") -- informs
   Prune Spurs Under.
4. Prunes those spurs (using either your suggested or specified cutoff,
   again mirroring the plugin's own pixel-removal-and-re-run logic
   exactly) and re-runs skan, to get the real, *post-prune* distribution
   of junction-to-junction branch lengths ("bridge candidates") -- informs
   Collapse Bridges Under. Using the post-prune population matters: some
   borderline bridges only look like bridges before nearby spurs are
   cleared out, since removing a spur can drop a junction's degree and
   reclassify what's touching it.
5. Plots histograms of all three distributions (saved as one PNG) and
   marks a suggested cutoff on each, computed via Otsu's method -- the
   same "split the distribution into a low and a high population" idea
   used for image intensity thresholding, applied here to 1D size/length
   values instead of pixel brightness.
6. Prints the suggested Fill Holes Up To / Prune Spurs Under / Collapse
   Bridges Under values, chained together automatically by default (each
   suggested value feeds into computing the next distribution) so one run
   gives you a complete, internally-consistent starting point for all
   three -- override any of them with a specific value via the flags
   below to test something else instead.

These suggestions are a *starting point*, not a mandate. Otsu assumes a
roughly bimodal split between "noise-scale" and "real" values, which
won't hold for every image (e.g. a very clean mask with almost no holes,
or a network so dense that short bridges are the norm rather than the
exception). Look at the saved histogram plot too: if there's a clear
valley between two humps, the suggested cutoff should sit right in it; if
the distribution is one smooth blob with no real second population, treat
the number with a lot more skepticism -- it may be telling you there's
nothing here to cut off in the first place (e.g. a genuinely tangled,
complex network, not noise).

GETTING A MASK
--------------
In napari, after running Analyze in the plugin: use the "Body Labels
Ch N" layer (Thresholding section -> Show Body Labels) and export it,
e.g. in napari's Python/IPython console:

    import numpy as np
    np.save("mask.npy", viewer.layers["Body Labels Ch 1"].data > 0)

or via napari's File > Save Selected Layer(s)... as a .tif. This script
also accepts a raw (non-binary) image and will Otsu-threshold it itself
as a rough approximation via --auto-threshold, but that won't exactly
match whatever threshold method/settings you used in the plugin (auto
method choice, manual override, ROI restriction, Z-range restriction,
etc.) -- a real exported mask is strongly preferred for suggestions you
actually trust.

USAGE
-----
    python network_threshold_diagnostics.py mask.npy
    python network_threshold_diagnostics.py mask.tif --z-xy-ratio 4.0
    python network_threshold_diagnostics.py raw_image.tif --auto-threshold
    python network_threshold_diagnostics.py mask.npy --fill-holes-px 10 --prune-spurs-px 3
    python network_threshold_diagnostics.py mask.npy --out diagnostics/

Requires: numpy, scipy, scikit-image, skan (matplotlib optional, for the
saved histogram plot).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------
# Core logic, deliberately kept line-for-line equivalent to the
# plugin's own _widget.py (_fill_small_holes / _skan_network_analysis)
# so the suggested cutoffs are calibrated to exactly what the plugin
# will actually do with them, not an approximation of it.
# --------------------------------------------------------------------


def _border_labels(bg_labels: np.ndarray, ndim: int) -> set:
    """Background component ids that touch the array border anywhere
    -- i.e. are *not* fully-enclosed holes."""
    border: set = set()
    for axis in range(ndim):
        for edge in (0, -1):
            slicer = [slice(None)] * ndim
            slicer[axis] = edge
            border.update(np.unique(bg_labels[tuple(slicer)]).tolist())
    border.discard(0)
    return border


def hole_size_distribution(binary_mask: np.ndarray) -> np.ndarray:
    """Sizes (pixel/voxel counts) of every fully-enclosed background
    component in ``binary_mask`` -- the exact population the plugin's
    _fill_small_holes draws its "hole" candidates from, with no size
    cap applied here. Empty array if there are no enclosed holes."""
    from scipy.ndimage import label as ndi_label

    ndim = binary_mask.ndim
    struct = np.ones((3,) * ndim, dtype=int)
    bg_labels, n_bg = ndi_label(~binary_mask, structure=struct)
    if n_bg == 0:
        return np.array([])
    border = _border_labels(bg_labels, ndim)
    sizes = np.bincount(bg_labels.ravel(), minlength=n_bg + 1)
    hole_sizes = [sizes[lbl] for lbl in range(1, n_bg + 1) if lbl not in border]
    return np.array(hole_sizes, dtype=float)


def fill_small_holes(binary_mask: np.ndarray, max_hole_size: float) -> np.ndarray:
    """Identical behavior to the plugin's _fill_small_holes: fill only
    fully-enclosed background components at or below max_hole_size."""
    from scipy.ndimage import label as ndi_label

    if max_hole_size <= 0 or not np.any(binary_mask):
        return binary_mask
    ndim = binary_mask.ndim
    struct = np.ones((3,) * ndim, dtype=int)
    bg_labels, n_bg = ndi_label(~binary_mask, structure=struct)
    if n_bg == 0:
        return binary_mask
    border = _border_labels(bg_labels, ndim)
    sizes = np.bincount(bg_labels.ravel(), minlength=n_bg + 1)
    fill_ids = [
        lbl for lbl in range(1, n_bg + 1)
        if lbl not in border and sizes[lbl] <= max_hole_size
    ]
    if not fill_ids:
        return binary_mask
    filled = binary_mask.copy()
    filled[np.isin(bg_labels, fill_ids)] = True
    return filled


def _col(df, *names):
    for name in names:
        if name in df.columns:
            return df[name].to_numpy()
    raise KeyError(f"None of {names} found in columns {list(df.columns)}")


def _run_skan(skel_img: np.ndarray, spacing: Tuple[float, ...]):
    """Mirrors the plugin's _skan_network_analysis._run_skan exactly:
    returns (sk_obj, df) or (None, None) if there's nothing to analyze."""
    from skan import Skeleton, summarize

    if not np.any(skel_img):
        return None, None
    sk_obj = Skeleton(skel_img, spacing=spacing)
    if sk_obj.n_paths == 0:
        return sk_obj, None
    try:
        df = summarize(sk_obj, separator="_")
    except TypeError:
        df = summarize(sk_obj)
    return sk_obj, df.reset_index(drop=True)


def prune_spurs(
    skel: np.ndarray, sk_obj, df, spacing: Tuple[float, ...], prune_length: float
) -> Tuple[np.ndarray, object, Optional[object]]:
    """Mirrors the plugin's spur-pruning block in _skan_network_analysis
    exactly: removes junction-to-endpoint (branch_type==1) branches
    shorter than prune_length from the skeleton pixel array (keeping the
    junction-end pixel), then re-runs skan once. Returns
    (pruned_skel, new_sk_obj, new_df)."""
    if prune_length <= 0 or df is None:
        return skel, sk_obj, df

    branch_dist = _col(df, "branch_distance", "branch-distance")
    branch_type = _col(df, "branch_type", "branch-type")
    node_src = _col(df, "node_id_src", "node-id-src").astype(int)
    node_dst = _col(df, "node_id_dst", "node-id-dst").astype(int)
    ndim = skel.ndim
    src_coord_cols = [
        _col(df, f"image_coord_src_{d}", f"image-coord-src-{d}") for d in range(ndim)
    ]
    dst_coord_cols = [
        _col(df, f"image_coord_dst_{d}", f"image-coord-dst-{d}") for d in range(ndim)
    ]
    spur_positions = np.where((branch_type == 1) & (branch_dist < prune_length))[0]
    if len(spur_positions) == 0:
        return skel, sk_obj, df

    node_counts: Dict[int, int] = {}
    for nid in np.concatenate([node_src, node_dst]):
        nid = int(nid)
        node_counts[nid] = node_counts.get(nid, 0) + 1

    remove_mask = np.zeros_like(skel, dtype=bool)
    for pos in spur_positions:
        path_coords = sk_obj.path_coordinates(int(pos))
        src_id, dst_id = int(node_src[pos]), int(node_dst[pos])
        src_is_junction = node_counts.get(src_id, 0) >= 3
        junction_coord = tuple(
            int(round(col[pos]))
            for col in (src_coord_cols if src_is_junction else dst_coord_cols)
        )
        for coord in path_coords:
            pixel = tuple(int(round(c)) for c in coord)
            if pixel == junction_coord:
                continue
            remove_mask[pixel] = True

    pruned_skel = skel & ~remove_mask
    new_sk_obj, new_df = _run_skan(pruned_skel, spacing)
    return pruned_skel, new_sk_obj, new_df


def branch_lengths_by_type(df) -> Dict[str, np.ndarray]:
    """Branch-length arrays split by skan's branch_type, from a
    summarize() dataframe: type 1 = junction-to-endpoint (spur
    candidates), type 2 = junction-to-junction (bridge candidates)."""
    if df is None or len(df) == 0:
        return {"spur": np.array([]), "bridge": np.array([])}
    branch_dist = _col(df, "branch_distance", "branch-distance")
    branch_type = _col(df, "branch_type", "branch-type")
    return {
        "spur": branch_dist[branch_type == 1],
        "bridge": branch_dist[branch_type == 2],
    }


def suggest_cutoff(values: np.ndarray, min_samples: int = 10) -> Optional[float]:
    """Otsu's method applied to a 1D array of sizes/lengths instead of
    pixel intensities -- finds the value separating the distribution
    into a "low" and "high" population, i.e. the "noise-scale vs. real"
    split these three knobs are meant to sit at. Returns None if there
    isn't enough data (too few samples, or every value identical) for
    the suggestion to be meaningful."""
    if len(values) < min_samples or np.ptp(values) == 0:
        return None
    from skimage.filters import threshold_otsu

    try:
        return float(threshold_otsu(values))
    except Exception:
        return None


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------


def _print_stats(name: str, values: np.ndarray, cutoff: Optional[float]) -> None:
    print(f"\n{name}: n={len(values)}", end="")
    if len(values) == 0:
        print(" (no candidates found)")
        return
    print(
        f", min={values.min():.1f}, median={float(np.median(values)):.1f}, "
        f"max={values.max():.1f}"
    )
    if cutoff is not None:
        print(f"  -> suggested cutoff: {cutoff:.1f} px")
    else:
        print("  -> not enough data / no clear split for an automatic suggestion")


def load_mask(path: Path, auto_threshold: bool) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        raw = np.load(path)
    else:
        import imageio.v2 as imageio

        raw = imageio.imread(path)
    raw = np.asarray(raw)

    if auto_threshold:
        from skimage.filters import threshold_otsu

        t = threshold_otsu(raw)
        print(f"Auto-thresholded at {t:.2f} (Otsu) -- approximate; prefer a real exported mask.")
        return raw > t

    mask = raw.astype(bool)
    n_unique = len(np.unique(raw))
    if n_unique > 2:
        print(
            f"Warning: input has {n_unique} unique values, doesn't look "
            f"binary. Treating any nonzero pixel as signal. Pass "
            f"--auto-threshold if this is a raw image, not a mask."
        )
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Suggest Fill Holes / Prune Spurs / Collapse Bridges cutoffs "
            "for napari-organelle-contact-analyzer from a real mask."
        )
    )
    parser.add_argument(
        "mask_path", type=str,
        help="Path to a binary mask (.npy/.tif/.tiff/.png). See GETTING A MASK in the module docstring.",
    )
    parser.add_argument(
        "--auto-threshold", action="store_true",
        help="Treat the input as a raw (non-binary) image and Otsu-threshold it first. Approximate -- prefer exporting the plugin's real mask instead.",
    )
    parser.add_argument(
        "--fill-holes-px", type=float, default=None,
        help="Hole-fill amount to apply before the spur/bridge analysis. Default: use this run's own suggested cutoff. Pass 0 to disable filling entirely.",
    )
    parser.add_argument(
        "--prune-spurs-px", type=float, default=None,
        help="Spur-prune amount to apply before the bridge analysis. Default: use this run's own suggested cutoff. Pass 0 to disable pruning entirely.",
    )
    parser.add_argument(
        "--z-xy-ratio", type=float, default=1.0,
        help="Z-step / XY-pixel-size ratio for 3D stacks, matching the plugin's own voxel calibration convention. Ignored for 2D input. Default 1.0.",
    )
    parser.add_argument(
        "--out", type=str, default=".",
        help="Directory to save the histogram plot into (default: current directory).",
    )
    args = parser.parse_args()

    mask = load_mask(Path(args.mask_path), args.auto_threshold)
    print(f"Loaded mask: shape={mask.shape}, signal={int(mask.sum())} px/voxels")

    # 1) Hole sizes, always measured on the unfilled mask -- we want to
    # see what's actually there before any fill is applied.
    hole_sizes = hole_size_distribution(mask)
    hole_cutoff = suggest_cutoff(hole_sizes)
    _print_stats("Hole sizes (-> Fill Holes Up To)", hole_sizes, hole_cutoff)

    fill_amount = args.fill_holes_px
    if fill_amount is None:
        fill_amount = hole_cutoff if hole_cutoff is not None else 0.0
        print(f"  (using this run's suggested value: {fill_amount:.1f})")
    filled_mask = fill_small_holes(mask, fill_amount)

    # 2) Spur lengths, on the (now hole-filled) skeleton, unpruned.
    ndim = filled_mask.ndim
    spacing = (args.z_xy_ratio, 1.0, 1.0) if ndim == 3 else (1.0, 1.0)

    from skimage.morphology import skeletonize

    skel = skeletonize(filled_mask)
    sk_obj, df = _run_skan(skel, spacing)
    lengths = branch_lengths_by_type(df)
    spur_cutoff = suggest_cutoff(lengths["spur"])
    _print_stats("Spur lengths (-> Prune Spurs Under)", lengths["spur"], spur_cutoff)

    prune_amount = args.prune_spurs_px
    if prune_amount is None:
        prune_amount = spur_cutoff if spur_cutoff is not None else 0.0
        print(f"  (using this run's suggested value: {prune_amount:.1f})")

    # 3) Bridge lengths, on the *post-prune* skeleton -- pruning spurs
    # first can drop a junction's degree and reclassify what's touching
    # it, so this is the population the plugin will actually see once
    # both Fill Holes and Prune Spurs are set.
    pruned_skel, pruned_sk_obj, pruned_df = prune_spurs(
        skel, sk_obj, df, spacing, prune_amount
    )
    pruned_lengths = branch_lengths_by_type(pruned_df)
    bridge_cutoff = suggest_cutoff(pruned_lengths["bridge"])
    _print_stats(
        "Bridge lengths (-> Collapse Bridges Under)",
        pruned_lengths["bridge"],
        bridge_cutoff,
    )

    print("\n--- Suggested starting point ---")
    print(f"Fill Holes Up To (px):       {fill_amount:.1f}")
    print(f"Prune Spurs Under (px):      {prune_amount:.1f}")
    print(
        f"Collapse Bridges Under (px): "
        f"{bridge_cutoff:.1f}" if bridge_cutoff is not None else "n/a (no clear split found)"
    )
    print(
        "\nThese come from Otsu's method on each distribution and assume a "
        "roughly bimodal noise-vs-real split -- check the saved histogram "
        "plot for a visible valley before trusting a number outright."
    )

    # 4) Plots
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed -- skipping plots (pip install matplotlib to enable them).")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        ("Hole sizes\n(Fill Holes Up To)", hole_sizes, hole_cutoff, "px/voxels"),
        ("Spur lengths\n(Prune Spurs Under)", lengths["spur"], spur_cutoff, "px-equivalent"),
        (
            "Bridge lengths, post-prune\n(Collapse Bridges Under)",
            pruned_lengths["bridge"],
            bridge_cutoff,
            "px-equivalent",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (title, values, cutoff, unit) in zip(axes, panels):
        if len(values) == 0:
            ax.set_title(f"{title}\n(no candidates)")
            ax.set_xlabel(unit)
            continue
        bins = min(50, max(10, len(values) // 2))
        ax.hist(values, bins=bins, color="steelblue", edgecolor="white")
        if cutoff is not None:
            ax.axvline(
                cutoff, color="crimson", linestyle="--",
                label=f"suggested: {cutoff:.1f}",
            )
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel(unit)
        ax.set_ylabel("count")
    fig.tight_layout()
    out_path = out_dir / "network_threshold_diagnostics.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved histogram plot to {out_path}")


if __name__ == "__main__":
    main()
