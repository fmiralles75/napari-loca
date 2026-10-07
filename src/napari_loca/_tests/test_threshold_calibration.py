"""Normalization, thresholding, voxel calibration and Z restriction."""

from types import SimpleNamespace

import numpy as np
import pytest

from napari_loca import _widget as W


# ---------------------------------------------------- normalization
def test_normalize_small_arrays_use_exact_min_max():
    """Under 10,000 values no voxel is ignored: plain min-max."""
    s = np.array([[100, 200], [300, 4095]], dtype=np.uint16)
    n = W.normalize_signal(s)
    assert n.min() == 0.0 and n.max() == 1.0
    assert n[0, 1] == pytest.approx(100 / 3995)


def test_normalize_ignores_extreme_hundredth_of_a_percent():
    """With N values, floor(N * 1e-4) are ignored at each end and the
    rest is mapped linearly onto 0-1, extremes clipped."""
    vals = np.arange(100_000, dtype=float)  # k = 10 at each end
    rng = np.random.default_rng(5)
    s = rng.permutation(vals).reshape(100, 1000)
    n = W.normalize_signal(s)
    lo, hi = 10.0, 99_989.0
    assert np.allclose(n, np.clip((s - lo) / (hi - lo), 0, 1))
    assert n.min() == 0.0 and n.max() == 1.0
    assert (n == 1.0).sum() == 11 and (n == 0.0).sum() == 11


def test_normalize_leaves_unit_range_and_constant_data_alone():
    f = np.array([0.0, 0.25, 0.9])
    assert np.array_equal(W.normalize_signal(f), f)
    c = np.full((4, 4), 7.0)
    assert np.array_equal(W.normalize_signal(c), c)
    big_c = np.full((200, 200), 7.0)
    assert np.array_equal(W.normalize_signal(big_c), big_c)


def test_normalize_ignores_nans():
    rng = np.random.default_rng(6)
    s = rng.gamma(2.0, 200.0, size=(120, 120))
    s2 = s.copy()
    s2[0, :5] = np.nan
    n, n2 = W.normalize_signal(s), W.normalize_signal(s2)
    assert np.isnan(n2[0, :5]).all()
    assert np.nanmax(np.abs(n2 - n)) < 0.01


@pytest.mark.parametrize(
    "gain,offset", [(1, 0), (3.7, 0), (1, 500), (0.2, 40)]
)
@pytest.mark.parametrize("shape", [(5, 32, 32), (8, 40, 40)])
def test_manual_threshold_invariant_to_gain_and_offset(gain, offset, shape):
    """Exposure/offset changes between images must not change the mask
    selected by the same manual threshold (both the exact min-max path
    and the percentile path)."""
    rng = np.random.default_rng(1)
    base = rng.gamma(2.0, 200.0, size=shape) + 10
    ref, _ = W.threshold_mask(W.normalize_signal(base), "Manual", "Otsu", 0.2)
    got, _ = W.threshold_mask(
        W.normalize_signal(base * gain + offset), "Manual", "Otsu", 0.2
    )
    assert np.array_equal(ref, got)


def _gamma_img(seed=2, shape=(128, 128)):
    return np.random.default_rng(seed).gamma(2.0, 200.0, size=shape) + 100


@pytest.mark.parametrize("n_hot", [1, 10, 50])
def test_manual_threshold_robust_to_hot_pixels(n_hot):
    """1M-pixel image (100 ignored at each end): up to 50 saturated
    pixels must not move the mask. Under min-max one pixel at 3x the
    max shrank the mask by ~25%."""
    img = _gamma_img(shape=(1000, 1000))
    ref, _ = W.threshold_mask(W.normalize_signal(img), "Manual", "", 0.3)
    hot = img.copy()
    hot.flat[::20_000][:n_hot] = img.max() * 3
    got, _ = W.threshold_mask(W.normalize_signal(hot), "Manual", "", 0.3)
    assert abs(int(got.sum()) - int(ref.sum())) <= 0.01 * ref.sum()


def test_manual_threshold_robust_to_dead_pixel():
    img = _gamma_img()
    ref, _ = W.threshold_mask(W.normalize_signal(img), "Manual", "", 0.3)
    dead = img.copy()
    dead[5, 5] = 0.0
    got, _ = W.threshold_mask(W.normalize_signal(dead), "Manual", "", 0.3)
    assert abs(int(got.sum()) - int(ref.sum())) <= 0.02 * ref.sum()


def test_otsu_robust_to_hot_pixel():
    img = np.full((128, 128), 200.0)
    img[30:90, 30:90] = 1000.0
    img += np.random.default_rng(7).normal(0, 20, img.shape)
    ref, _ = W.threshold_mask(W.normalize_signal(img), "Automatic", "Otsu", 0)
    img[0, 0] = 60_000.0
    got, _ = W.threshold_mask(W.normalize_signal(img), "Automatic", "Otsu", 0)
    assert abs(int(got.sum()) - int(ref.sum())) <= 2


# ------------------------------------------------------- thresholding
def test_manual_threshold_is_strictly_greater_than():
    s = np.array([0.1, 0.2, 0.3])
    m, t = W.threshold_mask(s, "Manual", "Otsu", 0.2)
    assert t == 0.2
    assert m.tolist() == [False, False, True]


def test_auto_threshold_uses_named_method():
    rng = np.random.default_rng(3)
    s = np.concatenate(
        [rng.normal(0.2, 0.02, 500), rng.normal(0.8, 0.02, 500)]
    )
    for name, fn in W.AUTO_METHODS.items():
        _, t = W.threshold_mask(s, "Automatic", name, 0.0)
        assert t == pytest.approx(float(fn(s)))


def test_auto_threshold_falls_back_to_mean(monkeypatch):
    def boom(_s):
        raise RuntimeError("unimodal histogram")

    monkeypatch.setitem(W.AUTO_METHODS, "Minimum", boom)
    s = np.array([0.0, 0.5, 1.0, 0.25])
    m, t = W.threshold_mask(s, "Automatic", "Minimum", 0.0)
    assert t == pytest.approx(s.mean())
    assert m.sum() == 2


def test_otsu_separates_bimodal_image():
    img = np.zeros((40, 40))
    img[10:30, 10:30] = 1000
    img += np.random.default_rng(4).normal(0, 20, img.shape)
    m, _ = W.threshold_mask(W.normalize_signal(img), "Automatic", "Otsu", 0)
    assert m.sum() == 400 and m[10:30, 10:30].all()


# ------------------------------------------------- voxel calibration
def layer(data_shape, scale):
    return SimpleNamespace(data=np.zeros(data_shape), scale=scale)


def test_z_ratio_from_layer_scale(make_harness):
    h = make_harness()
    r, cal, _ = h._get_z_xy_ratio(layer((17, 8, 8), (0.15, 0.10392, 0.10392)))
    assert cal and r == pytest.approx(0.15 / 0.10392)
    assert r == pytest.approx(1.443, abs=1e-3)


def test_z_ratio_uncalibrated_and_2d(make_harness):
    h = make_harness()
    assert h._get_z_xy_ratio(layer((5, 8, 8), (1.0, 1.0, 1.0)))[:2] == (
        1.0,
        False,
    )
    assert h._get_z_xy_ratio(layer((8, 8), (0.1, 0.1)))[:2] == (1.0, False)


def test_z_ratio_manual_override_wins(make_harness):
    h = make_harness(z_step=0.5, xy_pixel=0.1)
    r, cal, _ = h._get_z_xy_ratio(layer((5, 8, 8), (0.15, 0.1, 0.1)))
    assert cal and r == pytest.approx(5.0)


def test_z_ratio_ignores_leading_channel_axis(make_harness):
    h = make_harness()
    r, cal, _ = h._get_z_xy_ratio(layer((2, 17, 8, 8), (1.0, 0.3, 0.1, 0.1)))
    assert cal and r == pytest.approx(3.0)


# ---------------------------------------------- Restrict Z by Signal
def test_restrict_z_keeps_planes_with_signal(make_harness):
    h = make_harness()
    a = np.zeros((8, 4, 4), bool)
    b = np.zeros((8, 4, 4), bool)
    a[2:4, 1, 1] = True
    b[4, 2, 2] = True
    keep = h._compute_signal_restricted_z_indices([a, b], [0, 1])
    assert keep.tolist() == [2, 3, 4]
    keep_a = h._compute_signal_restricted_z_indices([a, b], [0])
    assert keep_a.tolist() == [2, 3]
    empty = np.zeros_like(a)
    assert (
        h._compute_signal_restricted_z_indices([empty, empty], [0, 1]) is None
    )


def test_restrict_z_keeps_a_contiguous_span(make_harness):
    """Empty planes between signal planes are kept, so planes that were
    several steps apart never become adjacent."""
    h = make_harness()
    a = np.zeros((10, 4, 4), bool)
    a[2, 1, 1] = True
    a[6, 1, 1] = True  # planes 3-5 empty in between
    keep = h._compute_signal_restricted_z_indices([a], [0])
    assert keep.tolist() == [2, 3, 4, 5, 6]


# --------------------------------------- raw-intensity mode / reporting
def _stack(seed=11, shape=(6, 64, 64)):
    rng = np.random.default_rng(seed)
    return rng.poisson(rng.gamma(2.0, 150.0, size=shape) + 100).astype(
        np.uint16
    )


def test_normalization_range_values():
    vals = np.arange(100_000, dtype=float)  # 10 ignored each end
    assert W.normalization_range(vals) == (10.0, 99_989.0)
    assert W.normalization_range(np.array([0.0, 0.5, 1.0])) is None
    assert W.normalization_range(np.full(50, 9.0)) is None


@pytest.mark.parametrize(
    "mode,manual", [("Manual", 0.08), ("Manual", 0.3), ("Automatic", 0.0)]
)
def test_scaled_threshold_reports_exact_raw_equivalent(mode, manual):
    raw = _stack()
    rng = W.normalization_range(raw)
    norm = W.normalize_signal(raw, rng)
    m, t_s, t_r = W.threshold_channel(
        raw, norm, rng, mode, "Otsu", manual, 0.0
    )
    lo, hi = rng
    assert t_r == pytest.approx(lo + t_s * (hi - lo))
    # The raw cutoff selects exactly the same voxels.
    assert np.array_equal(m, raw > t_r)


def test_raw_mode_masks_by_absolute_intensity():
    raw = _stack()
    rng = W.normalization_range(raw)
    norm = W.normalize_signal(raw, rng)
    m, t_s, t_r = W.threshold_channel(
        raw, norm, rng, W.THRESH_MODE_RAW, "Otsu", 0.0, 400.5
    )
    assert t_r == 400.5
    assert np.array_equal(m, raw > 400.5)
    lo, hi = rng
    assert t_s == pytest.approx((400.5 - lo) / (hi - lo))
    # Round trip: the reported scaled value reproduces the same mask.
    m2, _, _ = W.threshold_channel(raw, norm, rng, "Manual", "", t_s, 0.0)
    assert np.array_equal(m, m2)


def test_raw_mode_same_cutoff_in_every_image_scaled_mode_adapts():
    """Raw mode: same absolute cutoff whatever the exposure, so a 2x
    brighter image gives a bigger mask. Scaled mode adapts to it."""
    raw = _stack().astype(float)
    bright = raw * 2
    out = {}
    for name, im in (("dim", raw), ("bright", bright)):
        rng = W.normalization_range(im)
        norm = W.normalize_signal(im, rng)
        out[name, "raw"] = W.threshold_channel(
            im, norm, rng, W.THRESH_MODE_RAW, "", 0.0, 600.0
        )
        out[name, "scaled"] = W.threshold_channel(
            im, norm, rng, "Manual", "", 0.2, 0.0
        )
    assert out["dim", "raw"][2] == out["bright", "raw"][2] == 600.0
    assert out["bright", "raw"][0].sum() > out["dim", "raw"][0].sum()
    assert np.array_equal(out["dim", "scaled"][0], out["bright", "scaled"][0])
    assert out["bright", "scaled"][2] == pytest.approx(
        2 * out["dim", "scaled"][2]
    )


def test_unscaled_data_reports_same_value_on_both_scales():
    im = np.linspace(0, 1, 200).reshape(10, 20)
    m, t_s, t_r = W.threshold_channel(im, im, None, "Manual", "", 0.25, 0)
    assert t_s == t_r == 0.25
    m, t_s, t_r = W.threshold_channel(
        im, im, None, W.THRESH_MODE_RAW, "", 0, 0.5
    )
    assert t_s == t_r == 0.5


def test_threshold_columns():
    cols = W.threshold_columns(["Golgi", "Mito"], [(0.1, 500.0), (0.2, 900.0)])
    assert cols == {
        "Threshold Scaled (Golgi)": 0.1,
        "Threshold Raw (Golgi)": 500.0,
        "Threshold Scaled (Mito)": 0.2,
        "Threshold Raw (Mito)": 900.0,
    }


def test_pipeline_raw_mode_matches_scaled_mode(make_harness, pipeline):
    """Entering the raw equivalent of a scaled threshold reproduces
    every metric of that analysis."""
    a, b = _stack(1), _stack(2)
    out_s, _, _ = pipeline(
        [a, b], make_harness(), thresholds=[0.2, 0.2], contact_threshold=1.5
    )
    raws = [out_s["Threshold Raw (Ch1)"], out_s["Threshold Raw (Ch2)"]]
    out_r, _, _ = pipeline(
        [a, b],
        make_harness(),
        thresholds=[("raw", r) for r in raws],
        contact_threshold=1.5,
    )
    for k, v in out_s.items():
        if isinstance(v, float) and np.isnan(v):
            assert np.isnan(out_r[k]), k
        else:
            assert out_r[k] == pytest.approx(v, rel=1e-9), k
