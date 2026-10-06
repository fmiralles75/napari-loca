"""ROI geometry (Feret diameters) on polygons with hand-computed answers."""

import numpy as np
import pytest

from napari_organelle_contact_analyzer import _widget as W

GEOM = W.compute_roi_geometry
PERIM = W.polygon_perimeter


def rotate(pts, deg, center=(50.0, 50.0)):
    a = np.deg2rad(deg)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = np.asarray(center)
    return (pts - c) @ r.T + c


def ellipse_poly(a, b, n=720, center=(50.0, 50.0)):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack(
        (center[0] + a * np.cos(t), center[1] + b * np.sin(t))
    )


def rect(w, h, x0=40.0, y0=40.0):
    return np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]])


def ramanujan(a, b):
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))


@pytest.mark.parametrize("deg", [0, 17, 45, 90])
def test_circle(deg):
    mx, mn, ratio = GEOM(rotate(ellipse_poly(10, 10), deg))
    assert mx == pytest.approx(20.0, rel=1e-4)
    assert mn == pytest.approx(20.0, rel=1e-4)
    assert ratio == pytest.approx(1.0, rel=1e-4)


@pytest.mark.parametrize("deg", [0, 30, 45, 123])
def test_ellipse_ratio_is_axis_ratio(deg):
    poly = rotate(ellipse_poly(20, 10), deg)
    mx, mn, ratio = GEOM(poly)
    assert mx == pytest.approx(40.0, rel=1e-4)
    assert mn == pytest.approx(20.0, rel=1e-4)
    assert ratio == pytest.approx(2.0, rel=1e-4)


@pytest.mark.parametrize("deg", [0, 37, 90])
def test_square(deg):
    mx, mn, ratio = GEOM(rotate(rect(10, 10), deg))
    assert mx == pytest.approx(10 * np.sqrt(2))  # corner to corner
    assert mn == pytest.approx(10.0)
    assert ratio == pytest.approx(np.sqrt(2))


@pytest.mark.parametrize("deg", [0, 25, 90, 200])
def test_rectangle_2_to_1(deg):
    mx, mn, ratio = GEOM(rotate(rect(20, 10), deg))
    assert mx == pytest.approx(np.sqrt(500))
    assert mn == pytest.approx(10.0)
    assert ratio == pytest.approx(np.sqrt(5))


def test_l_shape_uses_convex_hull():
    # Hull (0,0) (20,0) (20,5) (5,20) (0,20). Max Feret (20,0)-(0,20) =
    # 20*sqrt(2); Min Feret is flush with the x+y=25 edge: 25/sqrt(2).
    poly = np.array(
        [[0, 0], [20, 0], [20, 5], [5, 5], [5, 20], [0, 20]], float
    )
    mx, mn, ratio = GEOM(poly)
    assert mx == pytest.approx(20 * np.sqrt(2))
    assert mn == pytest.approx(25 / np.sqrt(2))
    assert ratio == pytest.approx(1.6)
    assert PERIM(poly) == pytest.approx(80.0)  # the L, not its hull


def test_right_triangle():
    mx, mn, ratio = GEOM(np.array([[0, 0], [10, 0], [0, 10]], float))
    assert mx == pytest.approx(10 * np.sqrt(2))
    assert mn == pytest.approx(10 / np.sqrt(2))  # altitude to hypotenuse
    assert ratio == pytest.approx(2.0)


def test_polygon_perimeter():
    assert PERIM(rect(10, 10)) == pytest.approx(40.0)
    assert PERIM(rect(10, 10)[::-1]) == pytest.approx(40.0)
    assert np.isnan(PERIM(np.array([[1.0, 1.0]])))


def test_zyx_vertices_use_yx_columns():
    poly2 = rotate(rect(20, 10), 25)
    poly3 = np.column_stack((np.full(4, 7.0), poly2[:, 1], poly2[:, 0]))
    assert GEOM(poly3) == pytest.approx(GEOM(poly2))


def test_degenerate_polygons_give_nan():
    assert all(np.isnan(GEOM(np.array([[0, 0], [5, 5]], float))))
    line = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], float)
    assert all(np.isnan(GEOM(line)))


def test_roi_geometry_in_metrics_bundle(make_harness):
    m = np.zeros((100, 100), bool)
    poly = rotate(rect(20, 10), 30)  # (row, col) vertices
    out = make_harness()._compute_metrics_bundle(
        raw_signals=[m * 1.0],
        masks=[m],
        contacts=m.copy(),
        ch_labels=["A"],
        roi_poly_data=poly,
        roi_area=200,
    )
    assert out["Max Feret"] == pytest.approx(np.sqrt(500))
    assert out["Min Feret"] == pytest.approx(10.0)
    assert out["Shape Perimeter"] == pytest.approx(60.0)
    for old in (
        "Max Distance",
        "Max Perp Distance",
        "Distance Ratio",
        "Feret Ratio",
        "Ellipse Circumference",
        "Circumference/Perimeter Ratio",
    ):
        assert old not in out


# ------------------------------------- circularity / roundness / solidity
SHAPE = W.compute_roi_shape_descriptors


def descriptors(poly):
    return SHAPE(poly)


def _rect_moments(x0, x1, y0, y1):
    """Area, first and second moments of the rectangle [x0,x1]x[y0,y1]."""
    w, h = x1 - x0, y1 - y0
    return np.array(
        [
            w * h,
            (x1**2 - x0**2) / 2 * h,
            (y1**2 - y0**2) / 2 * w,
            (x1**3 - x0**3) / 3 * h,
            (y1**3 - y0**3) / 3 * w,
            (x1**2 - x0**2) / 2 * (y1**2 - y0**2) / 2,
        ]
    )


def _roundness_from_moments(m):
    a, sx, sy, sxx, syy, sxy = m
    cx, cy = sx / a, sy / a
    cov = np.array(
        [
            [sxx / a - cx**2, sxy / a - cx * cy],
            [sxy / a - cx * cy, syy / a - cy**2],
        ]
    )
    ev = np.linalg.eigvalsh(cov)
    return np.sqrt(ev[0] / ev[1])


@pytest.mark.parametrize("deg", [0, 33])
def test_circle_descriptors_are_one(deg):
    c, r, s = descriptors(rotate(ellipse_poly(10, 10), deg))
    assert c == pytest.approx(1.0, rel=1e-4)
    assert r == pytest.approx(1.0, rel=1e-4)
    assert s == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("deg", [0, 60])
def test_ellipse_descriptors(deg):
    c, r, s = descriptors(rotate(ellipse_poly(20, 10), deg))
    assert r == pytest.approx(0.5, rel=1e-4)  # = b / a
    expected_c = 4 * np.pi * (np.pi * 20 * 10) / ramanujan(20, 10) ** 2
    assert c == pytest.approx(expected_c, rel=1e-4)
    assert s == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("deg", [0, 45])
def test_square_descriptors(deg):
    c, r, s = descriptors(rotate(rect(10, 10), deg))
    assert c == pytest.approx(np.pi / 4)  # 4*pi*100/40^2
    assert r == pytest.approx(1.0)  # best-fit ellipse is a circle
    assert s == pytest.approx(1.0)


def test_l_shape_descriptors():
    # Area 175, perimeter 80, hull area 400 - 112.5 = 287.5.
    poly = np.array(
        [[0, 0], [20, 0], [20, 5], [5, 5], [5, 20], [0, 20]], float
    )
    c, r, s = descriptors(poly)
    assert c == pytest.approx(4 * np.pi * 175 / 80**2)
    # Independent check: L = 20x20 square minus the 15x15 square at
    # (5..20, 5..20); moments of rectangles are textbook integrals.
    m = _rect_moments(0, 20, 0, 20) - _rect_moments(5, 20, 5, 20)
    assert r == pytest.approx(_roundness_from_moments(m), rel=1e-9)
    assert s == pytest.approx(175 / 287.5)


def test_vertex_order_and_direction_do_not_matter():
    poly = np.array(
        [[0, 0], [20, 0], [20, 5], [5, 5], [5, 20], [0, 20]], float
    )
    assert descriptors(poly[::-1]) == pytest.approx(descriptors(poly))
    assert descriptors(np.roll(poly, 2, axis=0)) == pytest.approx(
        descriptors(poly)
    )


def test_bumpy_outline_lowers_circularity_not_roundness():
    t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    rad = 10 + 1.0 * np.sin(12 * t)  # 12 lobes
    bumpy = np.column_stack((50 + rad * np.cos(t), 50 + rad * np.sin(t)))
    c0, r0, s0 = descriptors(ellipse_poly(10, 10))
    c1, r1, s1 = descriptors(bumpy)
    assert c1 < c0 - 0.15
    assert abs(r1 - r0) < 0.2
    assert s1 < s0


def test_degenerate_descriptors_nan():
    line = np.array([[0, 0], [1, 1], [2, 2]], float)
    assert all(np.isnan(SHAPE(line)))
    assert all(np.isnan(SHAPE(line[:2])))


def test_descriptors_in_metrics_bundle(make_harness):
    m = np.zeros((100, 100), bool)
    out = make_harness()._compute_metrics_bundle(
        raw_signals=[m * 1.0],
        masks=[m],
        contacts=m.copy(),
        ch_labels=["A"],
        roi_poly_data=rect(10, 10),
        roi_area=100,
    )
    assert out["Circularity"] == pytest.approx(np.pi / 4)
    assert out["Roundness"] == pytest.approx(1.0)
    assert out["Solidity"] == pytest.approx(1.0)


@pytest.mark.parametrize("deg", [0, 30, 77])
def test_rectangle_roundness_is_inverse_side_ratio(deg):
    _, r, _ = descriptors(rotate(rect(20, 10), deg))
    assert r == pytest.approx(0.5)


def test_roundness_far_from_origin_is_stable():
    """Large coordinates (an ROI at the far corner of a 2048 px image)
    must not lose precision."""
    near = descriptors(rotate(ellipse_poly(20, 10), 40))
    far = descriptors(
        rotate(
            ellipse_poly(20, 10, center=(2000.0, 1900.0)),
            40,
            center=(2000.0, 1900.0),
        )
    )
    assert far == pytest.approx(near, rel=1e-9)


def test_roundness_matches_pixel_mask_moments():
    """Against the moments of the rasterized ROI (what ImageJ fits its
    ellipse to): agree to within pixelation error for a cell-sized ROI."""
    from skimage.draw import polygon
    from skimage.measure import label, regionprops

    rng = np.random.default_rng(3)
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    rad = 80 * (1 + 0.15 * rng.standard_normal(60).cumsum() / 8)
    poly = np.column_stack(
        (300 + 1.6 * rad * np.cos(t), 300 + rad * np.sin(t))
    )
    poly = rotate(poly, 25, center=(300.0, 300.0))
    mask = np.zeros((600, 600), bool)
    mask[polygon(poly[:, 1], poly[:, 0], mask.shape)] = True
    rp = regionprops(label(mask))[0]
    r_mask = rp.axis_minor_length / rp.axis_major_length
    _, r, _ = descriptors(poly)
    assert r == pytest.approx(r_mask, rel=0.01)


def _feret_brute_force(poly, n_angles=20000):
    ang = np.linspace(0, np.pi, n_angles, endpoint=False)
    dirs = np.column_stack((np.cos(ang), np.sin(ang)))
    proj = poly @ dirs.T
    widths = proj.max(0) - proj.min(0)
    return widths.max(), widths.min()


@pytest.mark.parametrize("seed", range(8))
def test_feret_matches_brute_force_on_random_polygons(seed):
    """Max/min Feret = max/min caliper width over all directions,
    checked against a dense sweep of 20,000 directions."""
    rng = np.random.default_rng(seed)
    n = rng.integers(5, 40)
    t = np.sort(rng.uniform(0, 2 * np.pi, n))
    rad = rng.uniform(20, 60, n)
    poly = np.column_stack((100 + rad * np.cos(t), 100 + rad * np.sin(t)))
    mx, mn, _ = GEOM(poly)
    bmx, bmn = _feret_brute_force(poly)
    assert mx == pytest.approx(bmx, rel=1e-6)
    # The sweep can only overestimate the minimum, and only slightly.
    assert mn <= bmn + 1e-9
    assert mn == pytest.approx(bmn, rel=1e-4)
