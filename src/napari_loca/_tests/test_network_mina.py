"""MiNA-equivalent Network outputs, on shapes with hand-computed values.

MiNA (StuartLab/MiNA, ``MiNA_Analyze_Morphology.py``) reports, per image:
branch length mean/median over every branch pooled; "summed branch
lengths" (each network's total length) mean/median; "network branches"
(branches per network) mean/median; and donuts (networks with no free
end). LocA's unit is the body. The outputs tested here:

    Summed Branch Length Mean / Median  -- per-body total skeleton length
    Pooled Branch Length Mean / Median  -- every counted branch, pooled
    Branch Count Median                 -- branches per body
    Donut Count                         -- bodies with >=1 branch and no
                                           skeleton endpoint

How the hand values arise. Branch length (skeleton_branch_lengths) is
the path through the branch's pixel centres plus 1, so a straight N-px
branch is N. Junction pixels belong to no branch. A 1-px-wide T or cross
is already its own skeleton, and its junction cluster is the crossing
pixel plus the pixels right next to it (each of those sees 3+ skeleton
neighbours counting diagonals):
  * T  (bar row 5, cols 5-35; stem col 20, rows 5-34): cluster
    {(5,19),(5,20),(5,21),(6,20)} -> arms cols 5-18 (14), cols 22-35
    (14), stem rows 7-34 (28).
  * cross (row 20 cols 5-35, col 20 rows 5-35): cluster = centre + its
    4 face neighbours -> four arms of 14.
  * ring: a 1-px 21 x 21 square loop; skeletonize removes the 4 corner
    pixels, leaving a 76-px loop with cut corners. A closed
    loop has no endpoint, so the minimum spanning tree drops its longest
    step (one sqrt(2) corner): 4 sides x 18 unit steps + 3 x sqrt(2),
    + 1 = 73 + 3 sqrt(2).
"""

import math

import numpy as np
import pytest

from napari_loca import _widget as W

NEW = (
    "Summed Branch Length Mean",
    "Summed Branch Length Median",
    "Pooled Branch Length Mean",
    "Pooled Branch Length Median",
    "Branch Count Median",
    "Donut Count",
)
RING_LEN = 73 + 3 * math.sqrt(2)


def bundle(h, masks, roi_area=None):
    labels = [f"C{i}" for i in range(len(masks))]
    return h._compute_metrics_bundle(
        raw_signals=[m.astype(float) for m in masks],
        masks=masks,
        contacts=np.zeros_like(masks[0]),
        ch_labels=labels,
        roi_poly_data=None,
        roi_area=masks[0].size if roi_area is None else roi_area,
    )


def net(out, ch="C0"):
    return {k: out[f"{k} ({ch})"] for k in NEW}


# ------------------------------------------------------------ shapes
def put_line(m, r=5, c0=5, n=30):
    m[r, c0 : c0 + n] = True
    return m


def put_t(m, r=5, c=5):
    m[r, c : c + 31] = True  # bar, 31 px
    m[r : r + 30, c + 15] = True  # stem, 30 px from the bar down
    return m


def put_cross(m, r=20, c=20):
    m[r, c - 15 : c + 16] = True
    m[r - 15 : r + 16, c] = True
    return m


def put_ring(m, r0=10, c0=10, side=21):
    """1-px square loop (side x side, face-connected, so one body). The
    skeleton drops its 4 corner pixels, leaving the 76-px cut-corner
    loop described at the top."""
    r1, c1 = r0 + side - 1, c0 + side - 1
    m[r0, c0 : c1 + 1] = True
    m[r1, c0 : c1 + 1] = True
    m[r0 : r1 + 1, c0] = True
    m[r0 : r1 + 1, c1] = True
    return m


def blank(shape=(41, 41)):
    return np.zeros(shape, bool)


# ------------------------------------------------ single reference shapes
@pytest.mark.parametrize(
    "name,make,expected",
    [
        (
            "line",
            lambda: put_line(blank((11, 40))),
            {"lengths": [30], "branches": 1, "donut": 0},
        ),
        (
            "T",
            lambda: put_t(blank((40, 41))),
            {"lengths": [14, 14, 28], "branches": 3, "donut": 0},
        ),
        (
            "cross",
            lambda: put_cross(blank()),
            {"lengths": [14, 14, 14, 14], "branches": 4, "donut": 0},
        ),
        (
            "ring",
            lambda: put_ring(blank()),
            {"lengths": [RING_LEN], "branches": 1, "donut": 1},
        ),
    ],
)
def test_single_body_reference_shapes(make_harness, name, make, expected):
    out = bundle(make_harness(), [make()])
    got = net(out)
    lengths = expected["lengths"]
    total = sum(lengths)
    assert out["Body Count (C0)"] == 1
    assert got["Summed Branch Length Mean"] == pytest.approx(total)
    assert got["Summed Branch Length Median"] == pytest.approx(total)
    assert got["Pooled Branch Length Mean"] == pytest.approx(
        total / len(lengths)
    )
    assert got["Pooled Branch Length Median"] == pytest.approx(
        float(np.median(lengths))
    )
    assert got["Branch Count Median"] == expected["branches"]
    assert got["Donut Count"] == expected["donut"]
    # With one body, the per-body Branch Length Mean equals the pooled one.
    assert out["Branch Length Mean (C0)"] == pytest.approx(
        total / len(lengths)
    )


def test_two_bodies_pooled_vs_per_body_mean(make_harness):
    """Line (1 branch of 30) + T (14, 14, 28). Pooled: (30+14+14+28)/4 =
    21.5, median of [14, 14, 28, 30] = 21. Per-body Branch Length Mean
    (existing output) = mean(30, 56/3) = 24.33: the two definitions
    differ as soon as bodies have different branch counts."""
    m = blank((40, 80))
    put_t(m, r=5, c=5)
    put_line(m, r=30, c0=45, n=30)
    out = bundle(make_harness(), [m])
    got = net(out)
    assert out["Body Count (C0)"] == 2
    assert got["Pooled Branch Length Mean"] == pytest.approx(21.5)
    assert got["Pooled Branch Length Median"] == pytest.approx(21.0)
    assert got["Summed Branch Length Mean"] == pytest.approx(43.0)
    assert got["Summed Branch Length Median"] == pytest.approx(43.0)
    assert got["Branch Count Median"] == 2.0
    assert got["Donut Count"] == 0
    assert out["Branch Length Mean (C0)"] == pytest.approx((30 + 56 / 3) / 2)
    assert out["Branch Count Mean (C0)"] == 2.0


def test_three_bodies_odd_medians(make_harness):
    """Line + T + cross: pooled [30 | 14, 14, 28 | 14 x 4] -> mean
    142/8, median 14; summed [30, 56, 56] -> mean 142/3, median 56;
    branches [1, 3, 4] -> median 3."""
    m = blank((80, 80))
    put_line(m, r=2, c0=5, n=30)
    put_t(m, r=8, c=5)
    put_cross(m, r=55, c=55)
    got = net(bundle(make_harness(), [m]))
    assert got["Pooled Branch Length Mean"] == pytest.approx(142 / 8)
    assert got["Pooled Branch Length Median"] == pytest.approx(14.0)
    assert got["Summed Branch Length Mean"] == pytest.approx(142 / 3)
    assert got["Summed Branch Length Median"] == pytest.approx(56.0)
    assert got["Branch Count Median"] == 3.0


def test_empty_mask_gives_zeros(make_harness):
    out = bundle(make_harness(), [blank()])
    assert out["Body Count (C0)"] == 0
    for k, v in net(out).items():
        assert v == 0, k


def test_per_roi_uses_only_the_roi_part(make_harness):
    """Same masking as analyze_contacts' per-ROI loop (mask & ROI). ROI 1
    (rows 0-19) cuts the T's stem after row 19: stem branch rows 7-19 =
    13 -> [14, 14, 13]; the line (row 38) is outside. ROI 2 (rows 36-39)
    holds only the 25-px line (the stem ends at row 34)."""
    m = blank((40, 80))
    put_t(m, r=5, c=5)
    put_line(m, r=38, c0=50, n=25)
    h = make_harness()
    roi1 = blank((40, 80))
    roi1[:20, :] = True
    roi2 = blank((40, 80))
    roi2[36:, :] = True

    g1 = net(bundle(h, [m & roi1], roi_area=int(roi1.sum())))
    assert g1["Summed Branch Length Mean"] == pytest.approx(41.0)
    assert g1["Pooled Branch Length Mean"] == pytest.approx(41 / 3)
    assert g1["Pooled Branch Length Median"] == pytest.approx(14.0)
    assert g1["Branch Count Median"] == 3.0

    g2 = net(bundle(h, [m & roi2], roi_area=int(roi2.sum())))
    assert g2["Summed Branch Length Mean"] == pytest.approx(25.0)
    assert g2["Pooled Branch Length Median"] == pytest.approx(25.0)
    assert g2["Branch Count Median"] == 1.0


def test_dropped_rungs_leave_the_pooled_list(make_harness):
    """Bar with two stems 4 px apart (as test_skeleton.two_ts_2d).
    Branches: left 14, middle rung 1, right 19, stems 8 and 8. Merging at
    4 px drops the rung: pooled [8, 8, 14, 19] -> mean 12.25, median 11,
    summed 49; without merging [1, 8, 8, 14, 19] -> mean 10, median 8."""
    m = np.zeros((25, 50), bool)
    m[10, 5:45] = True
    m[10:20, 20] = True
    m[10:20, 24] = True
    off = net(bundle(make_harness(junction_merge_px=0.0), [m]))
    on = net(bundle(make_harness(junction_merge_px=4.0), [m]))
    assert off["Pooled Branch Length Mean"] == pytest.approx(10.0)
    assert off["Pooled Branch Length Median"] == pytest.approx(8.0)
    assert off["Summed Branch Length Mean"] == pytest.approx(50.0)
    assert on["Pooled Branch Length Mean"] == pytest.approx(12.25)
    assert on["Pooled Branch Length Median"] == pytest.approx(11.0)
    assert on["Summed Branch Length Mean"] == pytest.approx(49.0)


def test_pooled_mean_identity_on_phantom(make_harness):
    """Pooled mean = total length / total branches = Summed Branch Length
    Mean / Branch Count Mean, exactly, for any mask (also with merging)."""
    from pathlib import Path

    data = np.load(Path(__file__).parent / "golden" / "phantom.npz")
    mito = data["mito"]
    mask = W.normalize_signal(mito, W.normalization_range(mito)) > 0.3
    for px in (0.0, 3.0):
        out = bundle(make_harness(junction_merge_px=px), [mask])
        assert out["Branch Count Mean (C0)"] > 1
        assert out["Pooled Branch Length Mean (C0)"] == pytest.approx(
            out["Summed Branch Length Mean (C0)"]
            / out["Branch Count Mean (C0)"],
            rel=1e-12,
        )


def test_existing_network_outputs_unchanged_and_new_ones_last(make_harness):
    """New keys are appended after the existing Network outputs, so
    exported columns that existed before keep their positions."""
    m = put_t(blank((40, 41)))
    keys = list(bundle(make_harness(), [m]))
    i_last_old = keys.index("% Signal Area in Junction-Containing Bodies (C0)")
    new_pos = [keys.index(f"{k} (C0)") for k in NEW]
    assert new_pos == list(range(i_last_old + 1, i_last_old + 1 + len(NEW)))


def test_network_off_adds_nothing(make_harness):
    from napari_loca._tests.conftest import ALL_OUTPUTS

    sel = dict(ALL_OUTPUTS, **{"Morphology Network": False})
    out = bundle(make_harness(outputs=sel), [put_ring(blank())])
    assert not any(k.startswith(NEW) for k in out)


# ------------------------------------------------------------- donuts
def test_donut_rejects_free_ends_but_allows_junctions(make_harness):
    """MiNA counts a network as a donut when no vertex is a free end;
    junctions are allowed. Ring -> 1. Ring + spur -> 0 (endpoint).
    Ring + chord (theta, 2 junctions, no endpoint) -> 1."""
    ring = put_ring(blank())
    spur = ring.copy()
    spur[20, 31:38] = True  # spur off the right side
    theta = ring.copy()
    theta[20, 11:30] = True  # chord across the middle
    h = make_harness()
    assert net(bundle(h, [ring]))["Donut Count"] == 1
    assert net(bundle(h, [spur]))["Donut Count"] == 0
    out = bundle(h, [theta])
    assert out["Junction Count Mean (C0)"] == 2
    assert net(out)["Donut Count"] == 1


def test_donut_counts_bodies_not_images(make_harness):
    m = blank((41, 90))
    put_ring(m, r0=10, c0=5)
    put_ring(m, r0=10, c0=40)
    put_line(m, r=38, c0=5, n=60)
    out = bundle(make_harness(), [m])
    assert out["Body Count (C0)"] == 3
    assert net(out)["Donut Count"] == 2


def test_tips_touching_another_body_are_still_free_ends(make_harness):
    """Found on the real WT 004 crop (5 false donuts): bodies are
    face-connected, so a rod whose tips each touch another body only
    diagonally is its own body, but counted across bodies neither tip
    has fewer than 2 skeleton neighbours and the rod read as a closed
    ring. Endpoints are counted with same-body neighbours only."""
    m = blank((20, 20))
    m[10, 5:15] = True  # body A: 10-px rod, tips (10,5) and (10,14)
    m[8:10, 4] = True  # body B, (9,4) diagonal to A's left tip
    m[8:10, 15] = True  # body C, (9,15) diagonal to A's right tip
    # Raster labelling makes A the highest label, so the skeleton path
    # B-A-C (one branch under full connectivity) is credited to A.
    out = bundle(make_harness(), [m])
    assert out["Body Count (C0)"] == 3
    assert net(out)["Donut Count"] == 0


def test_skeleton_endpoints_respect_body_labels():
    from scipy.ndimage import label

    m = blank((10, 20))
    m[5, 2:8] = True
    m[4, 8] = True  # separate body, diagonal to (5, 7)
    lab, n = label(m)
    assert n == 2
    # Across bodies (5,7) has 2 neighbours and is not a tip.
    assert W.skeleton_endpoints(m).size == 2  # (5,2) and (4,8)
    assert W.skeleton_endpoints(m, lab).size == 3  # + (5,7)


def thick_square_ring(hole_h, hole_w, ndim=2):
    """Solid 14 x 14 square with a centred hole_h x hole_w hole."""
    m = blank((24, 24))
    m[5:19, 5:19] = True
    r0, c0 = 12 - hole_h // 2, 12 - hole_w // 2
    m[r0 : r0 + hole_h, c0 : c0 + hole_w] = False
    if ndim == 3:
        v = np.zeros((5, 24, 24), bool)
        v[1:4] = m
        return v
    return m


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize(
    "hole,fill,donut",
    [
        ((4, 4), True, 0),  # 16 px lumen: filled before skeletonizing
        ((4, 4), False, 1),
        ((4, 5), True, 1),  # 20 px lumen: above the cap, kept
        ((4, 5), False, 1),
    ],
)
def test_donut_survives_hole_fill_only_above_16_px(
    make_harness, ndim, hole, fill, donut
):
    m = thick_square_ring(*hole, ndim=ndim)
    out = bundle(make_harness(fill_skeleton_holes=fill), [m])
    assert out["Body Count (C0)"] == 1
    assert net(out)["Donut Count"] == donut


# ------------------------------------------------ skeleton_endpoints
@pytest.mark.parametrize(
    "make,n_end",
    [
        (lambda: put_line(blank((11, 40))), 2),
        (lambda: put_t(blank((40, 41))), 3),
        (lambda: put_cross(blank()), 4),
        (lambda: put_ring(blank()), 0),
        (lambda: np.pad(np.ones((1, 1), bool), 3), 1),  # isolated pixel
        (lambda: blank(), 0),
    ],
)
def test_skeleton_endpoints_2d(make, n_end):
    assert W.skeleton_endpoints(make()).size == n_end


def test_skeleton_endpoints_3d_and_location():
    m = np.zeros((12, 12, 30), bool)
    m[6, 6, 4:26] = True
    m[2:9, 3, 3] = True  # separate line along Z
    ends = W.skeleton_endpoints(m)
    got = {tuple(int(x) for x in np.unravel_index(i, m.shape)) for i in ends}
    assert got == {(6, 6, 4), (6, 6, 25), (2, 3, 3), (8, 3, 3)}
