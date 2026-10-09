"""Core overlap and contact metrics on shapes with hand-computed answers.

Every expected number below can be checked with a pencil: the shapes are
axis-aligned boxes, so areas and distances are exact integers.
"""

import numpy as np
import pytest

from napari_loca import _widget as W
from napari_loca._tests.conftest import box


def contacts_for(masks, t, z_ratio=1.0, focus=None):
    dists = [W.contact_distance_map(m, z_ratio) for m in masks]
    return W.compute_contacts(masks, dists, t, focus=focus)


# ------------------------------------------------- overlap (2 channels)
def overlapping_boxes():
    # A: rows 10-19, cols 10-19 (100 px); B: rows 10-19, cols 15-24.
    a = box((40, 40), (10, 10), (20, 20))
    b = box((40, 40), (10, 15), (20, 25))
    return a, b


def test_intersection_union_jaccard_exact(make_harness):
    a, b = overlapping_boxes()
    h = make_harness()
    out = h._compute_metrics_bundle(
        raw_signals=[a * 100.0, b * 50.0],
        masks=[a, b],
        contacts=contacts_for([a, b], 0),
        ch_labels=["A", "B"],
        roi_poly_data=None,
        roi_area=a.size,
    )
    assert out["Intersection"] == 50
    assert out["Union"] == 150
    assert out["Intersection/Union (Contact Coefficient)"] == pytest.approx(
        1 / 3
    )
    assert out["Signal Area (A)"] == 100
    assert out["Signal Area (B)"] == 100
    assert out["Intersection/A Signal Area"] == pytest.approx(0.5)
    assert out["Intersection/B Signal Area"] == pytest.approx(0.5)
    assert out["ROI Area"] == 1600
    assert out["Signal Area/ROI Area (A)"] == pytest.approx(100 / 1600)
    # Mean Intensity is on RAW (un-normalized) signal inside own mask.
    assert out["Mean Intensity (A)"] == pytest.approx(100.0)
    assert out["Mean Intensity (B)"] == pytest.approx(50.0)


def test_identical_masks_give_jaccard_one(make_harness):
    a, _ = overlapping_boxes()
    h = make_harness()
    out = h._compute_metrics_bundle(
        raw_signals=[a * 1.0, a * 1.0],
        masks=[a, a.copy()],
        contacts=contacts_for([a, a.copy()], 0),
        ch_labels=["A", "B"],
        roi_poly_data=None,
        roi_area=a.size,
    )
    assert out["Intersection/Union (Contact Coefficient)"] == 1.0


def test_disjoint_masks_give_zero_overlap(make_harness):
    a = box((40, 40), (0, 0), (10, 10))
    b = box((40, 40), (30, 30), (40, 40))
    h = make_harness()
    out = h._compute_metrics_bundle(
        raw_signals=[a * 1.0, b * 1.0],
        masks=[a, b],
        contacts=contacts_for([a, b], 0),
        ch_labels=["A", "B"],
        roi_poly_data=None,
        roi_area=a.size,
    )
    assert out["Intersection"] == 0
    assert out["Contact Area"] == 0
    assert out["Intersection/Union (Contact Coefficient)"] == 0.0
    # Empty contact region -> NaN mean intensity, never a fake 0.
    assert np.isnan(out["Contact Mean Intensity (A)"])


def test_empty_channel_does_not_crash(make_harness):
    a, _ = overlapping_boxes()
    empty = np.zeros_like(a)
    h = make_harness()
    out = h._compute_metrics_bundle(
        raw_signals=[a * 1.0, empty * 1.0],
        masks=[a, empty],
        contacts=contacts_for([a, empty], 2),
        ch_labels=["A", "B"],
        roi_poly_data=None,
        roi_area=a.size,
    )
    assert out["Signal Area (B)"] == 0
    assert out["Body Count (B)"] == 0
    assert out["Intersection/B Signal Area"] == 0.0
    assert np.isnan(out["Mean Intensity (B)"])


# ------------------------------------------------------ contact area
def test_contact_threshold_zero_equals_intersection():
    a, b = overlapping_boxes()
    c = contacts_for([a, b], 0.0)
    assert np.array_equal(c, a & b)


def test_contact_area_grows_by_exact_columns():
    a, b = overlapping_boxes()
    # t=1: B's col 20 is 1 px from A; A's col 14 is 1 px from B.
    assert contacts_for([a, b], 1.0).sum() == 70
    assert contacts_for([a, b], 2.0).sum() == 90


@pytest.mark.parametrize(
    "t,expected",
    [
        (3.99, 0),  # nearest A-B pixel centres are exactly 4 px apart
        (4.0, 20),  # inclusive: A's col 19 + B's col 23, 10 rows each
        (5.0, 40),
    ],
)
def test_separated_boxes_contact_onset(t, expected):
    a = box((30, 50), (10, 10), (20, 20))  # cols 10-19
    b = box((30, 50), (10, 23), (20, 33))  # cols 23-32 (gap of 3)
    assert contacts_for([a, b], t).sum() == expected


def test_diagonal_distance_is_euclidean():
    a = np.zeros((10, 10), bool)
    b = np.zeros((10, 10), bool)
    a[2, 2] = True
    b[3, 3] = True  # sqrt(2) = 1.414 px away
    assert contacts_for([a, b], 1.41).sum() == 0
    assert contacts_for([a, b], 1.42).sum() == 2


def test_contacts_symmetric_in_channel_order():
    a, b = overlapping_boxes()
    assert np.array_equal(contacts_for([a, b], 2.5), contacts_for([b, a], 2.5))


# ------------------------------------------ Z / XY voxel anisotropy
@pytest.mark.parametrize(
    "z_ratio,t,expected_planes",
    [
        (1.0, 2.0, 2),  # 2 planes apart, isotropic: reached at t=2
        (1.5, 2.99, 0),  # 2 planes * 1.5 = 3.0 px: not yet
        (1.5, 3.0, 2),  # reached exactly at 3.0
        (1.443, 2.885, 0),  # his real acquisition: 2 * 1.443 = 2.886
        (1.443, 2.887, 2),
    ],
)
def test_z_steps_weighted_by_voxel_ratio(z_ratio, t, expected_planes):
    shape = (12, 8, 8)
    a = box(shape, (0, 2, 2), (5, 6, 6))  # planes 0-4
    b = box(shape, (6, 2, 2), (11, 6, 6))  # planes 6-10 (one empty plane)
    c = contacts_for([a, b], t, z_ratio)
    assert c.sum() == expected_planes * 16
    if expected_planes:
        assert c[4].sum() == 16 and c[6].sum() == 16


def test_xy_distance_unaffected_by_z_ratio():
    shape = (5, 10, 30)
    a = box(shape, (0, 0, 0), (5, 10, 10))
    b = box(shape, (0, 0, 13), (5, 10, 23))  # 3 px gap in X -> 4 px
    for r in (1.0, 2.0, 5.0):
        assert contacts_for([a, b], 3.99, r).sum() == 0
        assert contacts_for([a, b], 4.0, r).sum() == 2 * 5 * 10


# ------------------------------------------------------ 3+ channels
def test_three_channels_threshold_zero_is_triple_intersection():
    a = box((20, 20), (0, 0), (10, 10))
    b = box((20, 20), (5, 5), (15, 15))
    c = box((20, 20), (0, 5), (10, 15))
    got = contacts_for([a, b, c], 0.0)
    assert np.array_equal(got, a & b & c)


def sandwich():
    """A thin channel between two others that don't touch each other:
    A = 1-px column at x=9, B = x 0-6 (3 px from A), C = x 12+ (3 px
    from A, 6 px from B). The #6 case."""
    shape = (10, 20)
    a = box(shape, (0, 9), (10, 10))
    b = box(shape, (0, 0), (10, 7))
    c = box(shape, (0, 12), (10, 20))
    return a, b, c


def test_overlap_based_needs_all_but_one_channel_to_overlap():
    """Overlap-based (default) 3-channel definition, as in the glossary:
    all channels but one overlap exactly, the last is within reach. B
    and C never overlap and A overlaps neither, so nothing counts."""
    assert contacts_for(list(sandwich()), 3.0).sum() == 0


def test_overlap_based_counts_overlap_next_to_third_channel():
    """A (x 0-9) and B (x 5-14) overlap at x 5-9; C (x 11-14) sits inside
    B, 2 px from A. Contacts: the A&B overlap within 2 px of C (x 9),
    plus the B&C overlap within 2 px of A (x 11)."""
    shape = (4, 20)
    a = box(shape, (0, 0), (4, 10))
    b = box(shape, (0, 5), (4, 15))
    c = box(shape, (0, 11), (4, 15))
    got = contacts_for([a, b, c], 2.0)
    assert set(np.unique(np.nonzero(got)[1])) == {9, 11}
    assert got.sum() == 2 * 4


def test_focus_channel_counts_sandwiched_channel():
    """Focus channel = A: A's column is within 3 px of both B and C, so
    all of it counts -- the case the overlap-based method misses."""
    a, b, c = sandwich()
    assert np.array_equal(contacts_for([a, b, c], 3.0, focus=0), a)
    assert contacts_for([a, b, c], 2.99, focus=0).sum() == 0


def test_focus_channel_needs_every_other_channel_within_threshold():
    """Focus = B: B's edge (x=6) is 3 px from A but 6 px from C, so B has
    no tripartite contact at 3 px. Same for C (6 px from B). At 6 px,
    only B's edge column (x=6) is within reach of both A and C."""
    a, b, c = sandwich()
    assert contacts_for([a, b, c], 3.0, focus=1).sum() == 0
    assert contacts_for([a, b, c], 3.0, focus=2).sum() == 0
    got = contacts_for([a, b, c], 6.0, focus=1)
    assert np.array_equal(got, box(a.shape, (0, 6), (10, 7)))  # x=6


def test_focus_contacts_lie_on_focus_channel_and_grow_with_threshold():
    rng = np.random.default_rng(0)
    masks = [rng.random((6, 30, 30)) > 0.85 for _ in range(4)]
    for f in range(4):
        prev = None
        for t in (0.0, 1.0, 1.5, 2.5, 4.0):
            got = contacts_for(masks, t, z_ratio=1.443, focus=f)
            assert not (got & ~masks[f]).any()
            if prev is not None:
                assert not (prev & ~got).any()  # never shrinks
            prev = got


@pytest.mark.parametrize("n", [3, 4])
def test_threshold_zero_is_full_intersection_for_both_methods(n):
    rng = np.random.default_rng(n)
    masks = [rng.random((5, 25, 25)) > 0.4 for _ in range(n)]
    inter = np.logical_and.reduce(masks)
    assert inter.any()
    assert np.array_equal(contacts_for(masks, 0.0), inter)
    for f in range(n):
        assert np.array_equal(contacts_for(masks, 0.0, focus=f), inter)


def test_focus_channel_quadripartite():
    """Four channels: A's column with partners 1, 2 and 3 px away. A
    counts only once the threshold reaches the farthest partner."""
    shape = (5, 30)
    a = box(shape, (0, 10), (5, 11))
    b = box(shape, (0, 0), (5, 9))  # x 0-8: 2 px
    c = box(shape, (0, 13), (5, 20))  # 3 px
    d = box(shape, (0, 11), (5, 12))  # 1 px
    masks = [a, b, c, d]
    assert contacts_for(masks, 2.99, focus=0).sum() == 0
    assert np.array_equal(contacts_for(masks, 3.0, focus=0), a)


def test_focus_channel_weights_z_by_voxel_ratio():
    """Same Z rule as the overlap-based method: A's top plane is 2
    planes from B, i.e. 2 * 1.5 = 3.0 px; C overlaps A."""
    shape = (12, 8, 8)
    a = box(shape, (0, 2, 2), (5, 6, 6))  # planes 0-4
    b = box(shape, (6, 2, 2), (11, 6, 6))  # planes 6-10
    c = a.copy()
    assert contacts_for([a, b, c], 2.99, 1.5, focus=0).sum() == 0
    got = contacts_for([a, b, c], 3.0, 1.5, focus=0)
    assert got.sum() == 16 and got[4].sum() == 16


def test_focus_channel_out_of_range_rejected():
    a, b, c = sandwich()
    with pytest.raises(ValueError):
        contacts_for([a, b, c], 1.0, focus=3)


# ------------------------------------------------- contact intensity
def test_contact_mean_intensity_uses_raw_signal_in_contact_region(
    make_harness,
):
    a, b = overlapping_boxes()
    ra = np.where(a, 200.0, 7.0)  # background value must not leak in
    rb = np.where(b, 40.0, 3.0)
    c = contacts_for([a, b], 0)
    h = make_harness()
    out = h._compute_metrics_bundle(
        raw_signals=[ra, rb],
        masks=[a, b],
        contacts=c,
        ch_labels=["A", "B"],
        roi_poly_data=None,
        roi_area=a.size,
    )
    assert out["Contact Area"] == 50
    assert out["Contact Mean Intensity (A)"] == pytest.approx(200.0)
    assert out["Contact Mean Intensity (B)"] == pytest.approx(40.0)


def test_contact_mean_intensity_with_tolerance_includes_background(
    make_harness,
):
    """With t > 0, contact voxels of channel B include voxels outside
    A's mask, so A's Contact Mean Intensity averages in A's background.
    Pinned so this behaviour can't change unnoticed."""
    a, b = overlapping_boxes()
    ra = np.where(a, 200.0, 0.0)
    c = contacts_for([a, b], 1.0)  # 70 px: 60 inside A, 10 outside
    h = make_harness()
    out = h._compute_metrics_bundle(
        raw_signals=[ra, b * 1.0],
        masks=[a, b],
        contacts=c,
        ch_labels=["A", "B"],
        roi_poly_data=None,
        roi_area=a.size,
    )
    assert out["Contact Mean Intensity (A)"] == pytest.approx(200.0 * 60 / 70)


# ------------------------------------------------- contact sites
def test_contact_sites_distinguish_clustered_from_dispersed():
    """The old 'Avg Contact Dist' was 1.0 for both of these."""
    shape = (100, 100)
    clustered = np.zeros(shape, bool)
    for r, c in ((40, 40), (40, 50), (50, 40), (50, 50)):
        clustered[r : r + 5, c : c + 5] = True  # 4 sites, 10 px apart
    dispersed = np.zeros(shape, bool)
    for r, c in ((5, 5), (5, 85), (85, 5), (85, 85)):
        dispersed[r : r + 5, c : c + 5] = True  # 4 sites, 80 px apart
    n1, s1, d1 = W.compute_contact_sites(clustered)
    n2, s2, d2 = W.compute_contact_sites(dispersed)
    assert (n1, s1) == (4, 25.0) and (n2, s2) == (4, 25.0)
    assert d1 == pytest.approx(10.0)
    assert d2 == pytest.approx(80.0)


def test_contact_site_count_and_size():
    c = np.zeros((30, 30), bool)
    c[2:4, 2:4] = True  # 4 px
    c[10, 10:16] = True  # 6 px
    c[20, 20] = True  # 1 px
    c[21, 21] = True  # diagonal only -> separate site (face conn.)
    n, size, _ = W.compute_contact_sites(c)
    assert n == 4
    assert size == pytest.approx(12 / 4)


def test_contact_site_distance_scales_z():
    c = np.zeros((10, 5, 5), bool)
    c[2, 2, 2] = True
    c[6, 2, 2] = True  # 4 planes apart
    assert W.compute_contact_sites(c, 1.0)[2] == pytest.approx(4.0)
    assert W.compute_contact_sites(c, 1.443)[2] == pytest.approx(4 * 1.443)


def test_contact_sites_empty_and_single():
    n, size, d = W.compute_contact_sites(np.zeros((10, 10), bool))
    assert n == 0 and np.isnan(size) and np.isnan(d)
    c = np.zeros((10, 10), bool)
    c[3:5, 3:5] = True
    n, size, d = W.compute_contact_sites(c)
    assert n == 1 and size == 4.0 and np.isnan(d)


def test_contact_site_metrics_in_bundle(make_harness):
    a, b = overlapping_boxes()
    c = contacts_for([a, b], 0)
    out = make_harness()._compute_metrics_bundle(
        raw_signals=[a * 1.0, b * 1.0],
        masks=[a, b],
        contacts=c,
        ch_labels=["A", "B"],
        roi_poly_data=None,
        roi_area=a.size,
    )
    assert out["Contact Site Count"] == 1
    assert out["Mean Contact Site Size"] == 50.0
    assert np.isnan(out["Contact Site NN Distance"])
    assert "Avg Contact Dist" not in out
