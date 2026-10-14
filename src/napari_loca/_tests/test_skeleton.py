"""Skeleton / junction / branch tests on shapes with known topology.

Includes the regression test for the uint8 skeleton bug (Sept 2026):
on skimage versions where 3D ``skeletonize`` returns uint8 0/255, the
junction convolution overflowed and every skeleton pixel became a
junction. That bug produced wrong Branch/Junction numbers silently, so
it must never pass silently again.
"""

import numpy as np
import pytest

from napari_loca import _widget as W

SJ = W.OrganelleContactWidget._skeleton_and_junctions
GJ = W.OrganelleContactWidget._group_junctions


def topology(mask, merge_px=0.0, z_ratio=1.0, body_labels=None):
    skel, junc = SJ(mask)
    g = GJ(
        skel, junc, merge_px=merge_px, z_ratio=z_ratio, body_labels=body_labels
    )
    kept = g["n_branches"] - len(g["dropped_branches"])
    return skel, junc, g["n_groups"], kept, g


# ---------------------------------------------------------------- 2D
def line2d(n=30):
    m = np.zeros((11, n + 10), bool)
    m[5, 5 : 5 + n] = True
    return m


def t_shape():
    m = np.zeros((40, 41), bool)
    m[5, 5:36] = True  # horizontal bar
    m[5:35, 20] = True  # stem down from the bar's middle
    return m


def cross():
    m = np.zeros((41, 41), bool)
    m[20, 5:36] = True
    m[5:36, 20] = True
    return m


def ring(r=12):
    yy, xx = np.mgrid[:41, :41]
    d = np.hypot(yy - 20, xx - 20)
    return (d >= r - 0.5) & (d < r + 0.5)


def test_straight_line_has_no_junctions_and_one_branch():
    skel, junc, n_j, n_b, _ = topology(line2d(30))
    assert junc.sum() == 0
    assert n_j == 0
    assert n_b == 1
    assert skel.sum() == 30  # a 1-px line is its own skeleton


def test_t_shape_has_one_junction_three_branches():
    _, _, n_j, n_b, _ = topology(t_shape())
    assert n_j == 1
    assert n_b == 3


def test_cross_has_one_junction_four_branches():
    _, _, n_j, n_b, _ = topology(cross())
    assert n_j == 1
    assert n_b == 4


def test_closed_ring_has_no_junctions_one_branch():
    _, junc, n_j, n_b, _ = topology(ring())
    assert junc.sum() == 0
    assert n_j == 0
    assert n_b == 1


def test_thick_bar_skeleton_is_a_single_unbranched_line():
    # A filled 5x40 rectangle: medial axis should be one line, not a
    # ladder. Guards against the skeletonizer itself producing junctions
    # on a body that is simply wider than 1 px.
    m = np.zeros((20, 60), bool)
    m[8:13, 10:50] = True
    _, _, n_j, n_b, _ = topology(m)
    assert n_j == 0
    assert n_b == 1


def test_outputs_are_boolean():
    skel, junc = SJ(t_shape())
    assert skel.dtype == bool
    assert junc.dtype == bool


def test_empty_mask():
    skel, junc = SJ(np.zeros((10, 10), bool))
    assert not skel.any() and not junc.any()
    g = GJ(skel, junc)
    assert g["n_groups"] == 0 and g["n_branches"] == 0


# ---------------------------------------------------------------- 3D
def line3d():
    m = np.zeros((12, 12, 30), bool)
    m[6, 6, 4:26] = True
    return m


def t3d():
    m = np.zeros((30, 12, 31), bool)
    m[5, 6, 5:26] = True  # bar along X
    m[5:25, 6, 15] = True  # stem along Z
    return m


def test_3d_line_and_t():
    _, junc, n_j, n_b, _ = topology(line3d())
    assert junc.sum() == 0 and n_j == 0 and n_b == 1
    _, _, n_j, n_b, _ = topology(t3d())
    assert n_j == 1 and n_b == 3


# ---------------------------------------- uint8 regression (Sept 2026)
@pytest.fixture
def uint8_skeletonize(monkeypatch):
    """Make skeletonize behave like old skimage 3D: uint8 0/255."""
    real = W.skeletonize

    def fake(mask, *a, **k):
        return (np.asarray(real(mask, *a, **k)) > 0).astype(np.uint8) * 255

    monkeypatch.setattr(W, "skeletonize", fake)


@pytest.mark.parametrize(
    "shape_fn,n_j,n_b",
    [
        (line3d, 0, 1),
        (t3d, 1, 3),
        (line2d, 0, 1),
        (t_shape, 1, 3),
    ],
)
def test_uint8_skeleton_regression(uint8_skeletonize, shape_fn, n_j, n_b):
    skel, junc, got_j, got_b, _ = topology(shape_fn())
    assert skel.dtype == bool and junc.dtype == bool
    # The bug flagged every pixel of a straight line as a junction.
    assert junc.sum() < skel.sum() / 2
    assert (got_j, got_b) == (n_j, n_b)


def test_uint8_regression_through_metrics_bundle(
    uint8_skeletonize, make_harness
):
    """Same bug, checked at the level the user sees: Branch Length must
    be a pixel count (~ tens), not a sum of 255s (~ thousands)."""
    h = make_harness()
    m = line3d()
    out = h._compute_metrics_bundle(
        raw_signals=[m.astype(float)],
        masks=[m],
        contacts=np.zeros_like(m),
        ch_labels=["A"],
        roi_poly_data=None,
        roi_area=m.size,
    )
    assert out["Junction Count Mean (A)"] == 0
    assert out["Branch Count Mean (A)"] == 1
    assert out["Branch Length Mean (A)"] == m.sum()


# ---------------------------------------------------- branch length
BL = W.skeleton_branch_lengths


def _labels(mask):
    from scipy.ndimage import label

    lab, n = label(mask, structure=np.ones((3,) * mask.ndim))
    return lab, n


def test_branch_length_straight_line_equals_pixel_count():
    lab, n = _labels(line2d(30))
    assert BL(lab, n)[1] == pytest.approx(30.0)


def _diag_line(n):
    m = np.zeros((n + 10, n + 10), bool)
    idx = np.arange(5, 5 + n)
    m[idx, idx] = True
    return m


def test_branch_length_diagonal_is_euclidean():
    lab, n = _labels(_diag_line(40))
    assert BL(lab, n)[1] == pytest.approx(39 * np.sqrt(2) + 1)


def test_branch_length_corner_not_double_counted():
    """Three mutually adjacent pixels at a corner: the path is 1 + 1,
    not 1 + 1 + sqrt(2)."""
    m = np.zeros((20, 20), bool)
    m[5, 5:16] = True  # 11 px along the row, ends at (5, 15)
    m[5:16, 15] = True  # 10 more down the column
    lab, n = _labels(m)
    assert BL(lab, n)[1] == pytest.approx(20 + 1)


def test_branch_length_3d_scales_z():
    m = np.zeros((20, 5, 5), bool)
    m[2:12, 2, 2] = True  # 10 voxels along Z
    lab, n = _labels(m)
    assert BL(lab, n, 1.0)[1] == pytest.approx(10.0)
    assert BL(lab, n, 1.443)[1] == pytest.approx(9 * 1.443 + 1)
    d = np.zeros((12, 12, 12), bool)
    i = np.arange(1, 11)
    d[i, i, i] = True  # body diagonal, 10 voxels
    lab, n = _labels(d)
    assert BL(lab, n, 1.5)[1] == pytest.approx(9 * np.sqrt(1.5**2 + 2) + 1)


def test_branch_lengths_per_label():
    m = np.zeros((20, 40), bool)
    m[2, 2:12] = True  # 10 px
    m[10, 5:30] = True  # 25 px
    lab, n = _labels(m)
    got = BL(lab, n)
    assert got[0] == 0
    assert sorted(got[1:].tolist()) == pytest.approx([10.0, 25.0])
    assert BL(np.zeros((5, 5), int), 0).tolist() == [0.0]


def test_branch_length_nearly_orientation_independent(make_harness):
    """Same physical length at any angle, to within the ~10% that any
    path through 8-connected pixel centres overestimates oblique lines
    (worst near 22.5 deg; the same convention as Fiji AnalyzeSkeleton /
    MiNA). The old pixel count spread 1.0-1.41 over these angles. Bodies
    are 3-px-wide bars (a 1-px diagonal line is not one body under
    LocA's face connectivity)."""
    from scipy.ndimage import binary_dilation
    from skimage.draw import line

    h = make_harness()
    L = 60
    vals = []
    for ang in (0, 10, 22.5, 30, 45, 60, 80, 90):
        a = np.deg2rad(ang)
        m = np.zeros((90, 90), bool)
        r1 = int(round(10 + L * np.sin(a)))
        c1 = int(round(10 + L * np.cos(a)))
        m[line(10, 10, r1, c1)] = True
        m = binary_dilation(m, structure=np.ones((3, 3)))
        true = np.hypot(r1 - 10, c1 - 10) + 1
        got = h._compute_metrics_bundle(
            raw_signals=[m * 1.0],
            masks=[m],
            contacts=np.zeros_like(m),
            ch_labels=["A"],
            roi_poly_data=None,
            roi_area=m.size,
        )["Branch Length Mean (A)"]
        vals.append(got / true)
    assert max(vals) / min(vals) <= 1.11


# ------------------------------------- "Merge junctions within N px"
def two_ts_2d():
    """One bar with two stems 4 px apart: junction clusters centred at
    (10.25, 20) and (10.25, 24) -> centroid distance exactly 4 px."""
    m = np.zeros((25, 50), bool)
    m[10, 5:45] = True
    m[10:20, 20] = True
    m[10:20, 24] = True
    return m


def test_no_merge_by_default():
    _, _, n_j, n_b, g = topology(two_ts_2d())
    assert g["n_clusters"] == 2
    assert n_j == 2
    assert n_b == 5  # left, middle (1 px), right, two stems


@pytest.mark.parametrize(
    "merge_px,n_j,n_b",
    [
        (3.99, 2, 5),  # just below the 4 px centroid distance
        (4.0, 1, 4),  # merged; the 1-px middle rung stops counting
    ],
)
def test_merge_threshold_is_inclusive_and_drops_rung(merge_px, n_j, n_b):
    _, _, got_j, got_b, _ = topology(two_ts_2d(), merge_px=merge_px)
    assert (got_j, got_b) == (n_j, n_b)


def two_ts_3d():
    """Stem along Z with two X-stubs at z=10 and z=14: cluster
    centroids 4 planes apart."""
    m = np.zeros((30, 8, 30), bool)
    m[2:28, 4, 10] = True
    m[10, 4, 10:25] = True
    m[14, 4, 10:25] = True
    return m


@pytest.mark.parametrize(
    "z_ratio,merge_px,n_j",
    [
        (1.0, 4.0, 1),
        (1.5, 5.99, 2),  # 4 planes * 1.5 = 6.0 px apart
        (1.5, 6.0, 1),
    ],
)
def test_merge_distance_scales_z_by_voxel_ratio(z_ratio, merge_px, n_j):
    _, _, got_j, _, g = topology(
        two_ts_3d(), merge_px=merge_px, z_ratio=z_ratio
    )
    assert g["n_clusters"] == 2
    assert got_j == n_j


def test_merge_never_crosses_bodies():
    from scipy.ndimage import label

    m = np.zeros((25, 40), bool)
    m[10, 2:18] = True
    m[10:20, 14] = True  # T in body 1, junction near x=14
    m[10, 20:38] = True  # separate body (2 px gap)
    m[10:20, 26] = True  # T in body 2, junction near x=26
    lab, n = label(m)
    assert n == 2
    _, _, n_merged, _, _ = topology(m, merge_px=13.0)  # 12 px apart
    _, _, n_kept, _, _ = topology(m, merge_px=13.0, body_labels=lab)
    assert n_merged == 1  # without body labels they would merge
    assert n_kept == 2  # with them, never across bodies


def test_merge_through_metrics_bundle(make_harness):
    m = two_ts_2d()

    def out(px):
        return make_harness(junction_merge_px=px)._compute_metrics_bundle(
            raw_signals=[m * 1.0],
            masks=[m],
            contacts=np.zeros_like(m),
            ch_labels=["A"],
            roi_poly_data=None,
            roi_area=m.size,
        )

    assert out(0)["Junction Count Mean (A)"] == 2
    assert out(0)["Branch Count Mean (A)"] == 5
    assert out(4)["Junction Count Mean (A)"] == 1
    assert out(4)["Branch Count Mean (A)"] == 4


# ------------------------------------- holes filled before skeletonizing
FILL = W.fill_holes_per_body


def _lab(mask):
    from scipy.ndimage import label

    return label(mask)[0]


def test_fill_closes_small_hole_inside_one_body_2d():
    m = np.zeros((20, 20), bool)
    m[4:12, 4:12] = True
    m[7:9, 7:9] = False  # 4-px hole
    out = FILL(_lab(m))
    assert (out[4:12, 4:12] == 1).all()  # hole filled with the body label


def test_fill_respects_plane_hole_size_limit():
    """A hole bigger than SKELETON_HOLE_MAX_PX (a real ring/lumen) stays."""
    m = np.zeros((30, 30), bool)
    m[3:20, 3:20] = True
    m[8:13, 8:12] = False  # 20 px > 16
    out = FILL(_lab(m))
    assert not out[8:13, 8:12].any()
    m[8:13, 8:12] = True
    m[8:12, 8:12] = False  # exactly 16 px
    assert (FILL(_lab(m))[8:12, 8:12] == 1).all()


def test_fill_never_joins_two_bodies():
    """Two L-shaped bodies that meet only at their corners (diagonally,
    so they are separate bodies) and together enclose a 9-px gap: the
    gap stays open and both keep their own labels."""
    m = np.zeros((10, 10), bool)
    m[2, 2:6] = True  # body 1: top row ...
    m[2:6, 2] = True  # ... and left column
    m[6, 3:7] = True  # body 2: bottom row ...
    m[3:7, 6] = True  # ... and right column
    lab = _lab(m)
    assert lab.max() == 2
    assert not m[3:6, 3:6].any()  # the enclosed 3x3 gap
    assert np.array_equal(FILL(lab), lab)


def test_fill_3d_cavity_any_size_and_tunnel():
    """3D: an enclosed cavity is filled whatever its size; a 2x2 tunnel
    through a 3-plane-thick body is filled (a hole in every plane)."""
    m = np.zeros((9, 30, 30), bool)
    m[1:8, 2:14, 2:14] = True
    m[2:7, 4:12, 4:12] = False  # 320-voxel enclosed cavity
    m[3:6, 18:26, 18:26] = True
    m[3:6, 21:23, 21:23] = False  # tunnel through all 3 planes
    lab = _lab(m)
    out = FILL(lab)
    assert (out[2:7, 4:12, 4:12] == lab[1, 2, 2]).all()
    assert (out[3:6, 21:23, 21:23] == lab[4, 18, 18]).all()
    assert out.max() == lab.max() == 2  # bodies unchanged


def test_fill_hole_touching_another_body_in_z_is_left_open():
    """A plane hole in body 1 with body 2 directly above it: filling it
    would join the two through Z, so it stays open."""
    m = np.zeros((5, 12, 12), bool)
    m[2, 2:10, 2:10] = True
    m[2, 5:7, 5:7] = False  # hole in plane 2 (body 1)
    m[3, 5:7, 5:7] = True  # body 2 sits on the hole (face contact in Z)
    lab = _lab(m)
    assert lab.max() == 2
    assert np.array_equal(FILL(lab), lab)


def test_speckled_tube_skeleton_recovers_true_topology(make_harness):
    """The 'maze': a thick 3D T with 2% dark voxels inside. Unfilled, the
    skeleton is a shell of junction pixels; with the default fill the T
    comes back (1 junction, 3 branches) -- and Signal Area is the same
    either way, because only the skeleton uses the filled mask."""
    rng = np.random.default_rng(0)
    zz, yy, xx = np.mgrid[0:11, 0:60, 0:60]
    tube = ((zz - 5) ** 2 + (yy - 20) ** 2 <= 16) & (xx >= 5) & (xx < 55)
    tube |= ((zz - 5) ** 2 + (xx - 30) ** 2 <= 16) & (yy >= 20) & (yy < 55)
    m = tube & ~(rng.random(tube.shape) < 0.02)

    def run(fill):
        return make_harness(fill_skeleton_holes=fill)._compute_metrics_bundle(
            raw_signals=[m * 1.0],
            masks=[m],
            contacts=np.zeros_like(m),
            ch_labels=["A"],
            roi_poly_data=None,
            roi_area=m.size,
        )

    off, on = run(False), run(True)
    assert on["Junction Count Mean (A)"] == 1
    assert on["Branch Count Mean (A)"] == 3
    assert off["Branch Count Mean (A)"] != 3  # the maze is real without it
    assert on["Signal Area (A)"] == off["Signal Area (A)"] == int(m.sum())
    assert on["Body Count (A)"] == off["Body Count (A)"] == 1
