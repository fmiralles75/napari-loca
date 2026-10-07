"""Body (connected-component) metrics and per-body morphology."""

import numpy as np
import pytest

from napari_loca import _widget as W
from napari_loca._tests.conftest import (
    box,
    disk,
    ellipsoid_physical,
)

FSB = W.OrganelleContactWidget._filter_small_bodies
MSW = W.OrganelleContactWidget._mean_sd_wmean


def bundle(h, masks, labels=None, raw=None):
    labels = labels or [f"C{i}" for i in range(len(masks))]
    raw = raw or [m.astype(float) for m in masks]
    return h._compute_metrics_bundle(
        raw_signals=raw,
        masks=masks,
        contacts=np.zeros_like(masks[0]),
        ch_labels=labels,
        roi_poly_data=None,
        roi_area=masks[0].size,
    )


def three_bodies():
    """Bodies of exactly 1, 2 and 6 px."""
    m = np.zeros((20, 20), bool)
    m[1, 1] = True
    m[5, 5:7] = True
    m[10:12, 10:13] = True
    return m


# ------------------------------------------------- small-body filter
@pytest.mark.parametrize(
    "min_size,n_kept,area",
    [
        (1, 3, 9),
        (2, 2, 8),
        (3, 1, 6),
        (6, 1, 6),
        (7, 0, 0),
    ],
)
def test_filter_small_bodies_counts(min_size, n_kept, area):
    binary, labels, n, total = FSB(three_bodies(), min_size)
    assert (n, total) == (n_kept, area)
    assert binary.sum() == area
    assert labels.max() == n_kept  # contiguous relabel 1..n
    assert set(np.unique(labels)) == set(range(n_kept + 1))


def test_bodies_use_face_connectivity_2d():
    """Documented: diagonally-touching pixels are separate bodies."""
    m = np.zeros((5, 5), bool)
    m[1, 1] = m[2, 2] = True
    _, _, n, _ = FSB(m, 1)
    assert n == 2


def test_bodies_use_face_connectivity_3d():
    m = np.zeros((4, 4, 4), bool)
    m[1, 1, 1] = m[2, 2, 2] = True  # corner-touching only
    m[1, 3, 3] = m[2, 3, 3] = True  # face-touching across Z
    _, _, n, _ = FSB(m, 1)
    assert n == 3


# ------------------------------------------------ fragmentation
def test_fragmentation_metrics_with_scoped_filter(make_harness):
    h = make_harness(min_body_size=2, filter_body_metrics=True)
    out = bundle(h, [three_bodies()])
    assert out["Signal Area (C0)"] == 9  # NOT filtered (scoped)
    assert out["Body Count (C0)"] == 2
    assert out["Average Area per Body (C0)"] == pytest.approx(4.0)
    assert out["Fragmentation Coefficient (C0)"] == pytest.approx(0.5)


def test_fragmentation_metrics_unfiltered(make_harness):
    h = make_harness(filter_body_metrics=False, filter_threshold_mask=False)
    out = bundle(h, [three_bodies()])
    assert out["Body Count (C0)"] == 3
    assert out["Average Area per Body (C0)"] == pytest.approx(3.0)
    assert out["Fragmentation Coefficient (C0)"] == pytest.approx(1 / 3)


def test_fragmentation_coefficient_is_one_over_body_count(make_harness):
    rng = np.random.default_rng(0)
    m = rng.random((64, 64)) > 0.7
    h = make_harness(filter_body_metrics=False)
    out = bundle(h, [m])
    assert out["Fragmentation Coefficient (C0)"] == pytest.approx(
        1 / out["Body Count (C0)"]
    )


# --------------------------------------------------- shape metrics
def test_mean_sd_weighted_mean():
    mean, sd, wmean = MSW([1.0, 3.0], [1.0, 3.0])
    assert mean == 2.0
    assert sd == 1.0  # population SD (ddof=0)
    assert wmean == pytest.approx(2.5)
    assert MSW([], []) == (0.0, 0.0, 0.0)


def test_aspect_ratio_of_rectangle(make_harness):
    # Discrete rectangle L x W: AR = sqrt((L^2-1)/(W^2-1)) exactly.
    m = box((40, 80), (16, 10), (24, 50))  # 8 rows x 40 cols
    out = bundle(make_harness(), [m])
    expected = np.sqrt((40**2 - 1) / (8**2 - 1))
    assert out["Aspect Ratio Mean (C0)"] == pytest.approx(expected, rel=1e-6)


def test_aspect_ratio_rotation_invariant(make_harness):
    from skimage.transform import rotate

    m = box((101, 101), (45, 20), (55, 80))
    r = rotate(m.astype(float), 37, order=0, preserve_range=True) > 0.5
    a0 = bundle(make_harness(), [m])["Aspect Ratio Mean (C0)"]
    a1 = bundle(make_harness(), [r])["Aspect Ratio Mean (C0)"]
    assert a1 == pytest.approx(a0, rel=0.03)


def test_disk_is_round(make_harness):
    m = disk((80, 80), (40, 40), 25)
    out = bundle(make_harness(), [m])
    assert out["Aspect Ratio Mean (C0)"] == pytest.approx(1.0, abs=0.01)
    # Form factor of a circle is 1 (Crofton perimeter).
    assert out["Form Factor Mean (C0)"] == pytest.approx(1.0, abs=0.04)


def _offset_disk(r, rng):
    n = int(r) + 4
    o = rng.random(2)
    yy, xx = np.mgrid[: 2 * n, : 2 * n]
    return np.hypot(yy - n + o[0], xx - n + o[1]) <= r


def test_form_factor_independent_of_object_size(make_harness):
    """A disk's FF must not drift with its size in pixels, or a
    condition with smaller bodies (e.g. fragmentation) would shift FF
    for that reason alone. Averaged over random sub-pixel positions, as
    bodies in an image are. The old 4-connected perimeter gave 0.67 at
    r=2 and 1.08 at r=25."""
    rng = np.random.default_rng(0)
    means = []
    for r in (2, 3, 5, 10, 25):
        ffs = [
            bundle(make_harness(), [_offset_disk(r, rng)])[
                "Form Factor Mean (C0)"
            ]
            for _ in range(8)
        ]
        means.append(np.mean(ffs))
    assert max(means) - min(means) < 0.06
    assert all(abs(m - 1) < 0.05 for m in means)


def test_form_factor_higher_for_irregular_shape(make_harness):
    star = np.zeros((80, 80), bool)
    star[38:42, 5:75] = True
    star[5:75, 38:42] = True
    ff_star = bundle(make_harness(), [star])["Form Factor Mean (C0)"]
    ff_disk = bundle(make_harness(), [disk((80, 80), (40, 40), 20)])[
        "Form Factor Mean (C0)"
    ]
    assert ff_star > 3 * ff_disk


def test_area_weighted_aspect_ratio(make_harness):
    m = np.zeros((60, 120), bool)
    m[5:25, 5:25] = True  # square: AR 1, area 400
    m[40:44, 30:110] = True  # 4 x 80 bar, area 320
    out = bundle(make_harness(), [m])
    ar_bar = np.sqrt((80**2 - 1) / (4**2 - 1))
    assert out["Aspect Ratio Mean (C0)"] == pytest.approx(
        (1 + ar_bar) / 2, rel=1e-6
    )
    assert out["Aspect Ratio Weighted Mean (C0)"] == pytest.approx(
        (400 * 1 + 320 * ar_bar) / 720, rel=1e-6
    )


def test_form_factor_nan_in_3d(make_harness):
    m = box((6, 20, 20), (1, 5, 5), (5, 15, 15))
    out = bundle(make_harness(), [m])
    assert np.isnan(out["Form Factor Mean (C0)"])
    assert out["Aspect Ratio Mean (C0)"] > 0


@pytest.mark.parametrize("z,xy", [(0.30, 0.10), (0.15, 0.10392), (0.10, 0.10)])
def test_3d_aspect_ratio_respects_voxel_size(make_harness, z, xy):
    """A physically round body must read AR ~1 however coarsely Z is
    sampled (regionprops without spacing gave ~z/xy)."""
    sphere = ellipsoid_physical(
        (25, 45, 45), (12, 22, 22), (1.0, 1.0, 1.0), (z, xy, xy)
    )
    h = make_harness(z_step=z, xy_pixel=xy)
    out = bundle(h, [sphere])
    assert out["Aspect Ratio Mean (C0)"] == pytest.approx(1.0, abs=0.05)


def test_3d_aspect_ratio_of_physically_elongated_body(make_harness):
    """2:1 ellipsoid lying along Z (the coarse axis) must read AR ~2."""
    z, xy = 0.30, 0.10
    ell = ellipsoid_physical(
        (35, 31, 31), (17, 15, 15), (2.0, 1.0, 1.0), (z, xy, xy)
    )
    out = bundle(make_harness(z_step=z, xy_pixel=xy), [ell])
    assert out["Aspect Ratio Mean (C0)"] == pytest.approx(2.0, rel=0.05)


def test_body_aspect_ratio_undefined_cases():
    assert W.body_aspect_ratio(np.array([[1, 1]]), (1, 1)) is None
    line = np.column_stack([np.zeros(10), np.arange(10)])
    assert W.body_aspect_ratio(line, (1, 1)) is None  # zero width
    flat = np.argwhere(np.ones((1, 5, 5)))  # one Z plane
    assert W.body_aspect_ratio(flat, (1.4, 1, 1)) is None


# ------------------------------------------------ network summary
def test_network_percentages(make_harness):
    m = np.zeros((40, 80), bool)
    m[5, 5:36] = True  # T: horizontal bar (31 px)
    m[5:30, 20] = True  #    stem (24 more px)
    m[20, 50:75] = True  # plain line, 25 px, no junction
    out = bundle(make_harness(), [m])
    t_area = 31 + 24
    assert out["Body Count (C0)"] == 2
    assert out["% Bodies with Junctions (C0)"] == pytest.approx(50.0)
    assert out["% Signal Area in Junction-Containing Bodies (C0)"] == (
        pytest.approx(100 * t_area / (t_area + 25))
    )
    assert out["Junction Count Mean (C0)"] == pytest.approx(0.5)
    assert out["Branch Count Mean (C0)"] == pytest.approx(2.0)  # (3+1)/2


def test_network_metrics_respect_body_filter(make_harness):
    m = np.zeros((40, 80), bool)
    m[20, 10:60] = True  # one long line
    m[2, 2] = True  # 1-px speck, filtered at min size 2
    out = bundle(make_harness(min_body_size=2), [m])
    assert out["Branch Count Mean (C0)"] == 1.0
    assert out["Branch Length Mean (C0)"] == 50.0


# ----------------------------------------- intensity comparisons
def test_region_mask_modes(make_harness):
    h = make_harness()
    a = box((10, 10), (0, 0), (5, 5))
    b = box((10, 10), (3, 3), (8, 8))
    c = np.zeros((10, 10), bool)
    c[0, 0] = True
    assert np.array_equal(
        h._region_mask_from_mode("Union", [0, 1], [a, b], c), a | b
    )
    assert np.array_equal(
        h._region_mask_from_mode("Intersection", [0, 1], [a, b], c), a & b
    )
    assert h._region_mask_from_mode("Contacts", [], [a, b], c) is c
    assert h._region_mask_from_mode("Union", [5], [a, b], c) is None
    assert h._region_mask_from_mode("bogus", [0], [a, b], c) is None


def test_intensity_comparison_with_subtraction(make_harness):
    a = box((10, 10), (0, 0), (5, 5))  # 25 px
    b = box((10, 10), (3, 3), (8, 8))  # overlap with a: 4 px
    raw_a = np.arange(100, dtype=float).reshape(10, 10)
    h = make_harness(
        enable_intensity_comparisons=True,
        intensity_comparisons=[
            {
                "enabled": True,
                "source_ch": 0,
                "mode": "Union",
                "region_channels": [0],
                "subtract_mode": "Intersection",
                "subtract_channels": [0, 1],
            }
        ],
    )
    out = h._compute_metrics_bundle(
        raw_signals=[raw_a, b * 1.0],
        masks=[a, b],
        contacts=np.zeros_like(a),
        ch_labels=["A", "B"],
        roi_poly_data=None,
        roi_area=a.size,
    )
    key = "Mean Intensity [A] in Union(A) - Intersection(A,B)"
    region = a & ~(a & b)
    assert region.sum() == 21
    assert out[key] == pytest.approx(raw_a[region].mean())
