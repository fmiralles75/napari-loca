"""Core overlap and contact metrics on shapes with hand-computed answers.

Every expected number below can be checked with a pencil: the shapes are
axis-aligned boxes, so areas and distances are exact integers.
"""

import numpy as np
import pytest

from napari_loca import _widget as W
from napari_loca._tests.conftest import box


def contacts_for(masks, t, z_ratio=1.0):
    dists = [W.contact_distance_map(m, z_ratio) for m in masks]
    return W.compute_contacts(masks, dists, t)


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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFINITION MISMATCH for >=3 channels: the glossary says a contact "
        "is a channel's signal within the threshold of EVERY other channel; "
        "the code requires the voxel to lie INSIDE all other channels and "
        "within the threshold of one. A thin channel sandwiched between two "
        "others counts under the glossary but not the code. Decide which "
        "is intended, then fix the code or the glossary and drop this."
    ),
)
def test_three_channel_contact_matches_glossary_definition():
    shape = (10, 20)
    a = box(shape, (0, 9), (10, 10))  # 1-px column at x=9
    b = box(shape, (0, 0), (10, 7))  # x 0-6  (2 px from A)
    c = box(shape, (0, 12), (10, 20))  # x 12+  (3 px from A)
    got = contacts_for([a, b, c], 3.0)
    # Glossary: A's column is within 3 px of both B and C.
    assert got[:, 9].all()


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
