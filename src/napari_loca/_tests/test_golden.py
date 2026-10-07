"""Golden-file regression test.

Runs the full analysis pipeline (normalize -> threshold -> contacts ->
every metric) on a fixed synthetic 2-channel Z-stack stored in
``golden/phantom.npz`` and compares every output with
``golden/expected.json``. Any change to any number fails the test.

That is the point: a refactor, a dependency upgrade (scikit-image,
scipy, numpy) or a "harmless" cleanup must not change results without
someone noticing. If a change is intended, regenerate and commit the
new expected values with a note on why:

    LOCA_UPDATE_GOLDEN=1 pytest -k golden

The phantom (not a random seed) is stored so the input is identical
on every machine and numpy version.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from napari_loca._tests.conftest import (
    _make_harness,
    run_pipeline,
)

HERE = Path(__file__).parent / "golden"
PHANTOM = HERE / "phantom.npz"
EXPECTED = HERE / "expected.json"

# His acquisition: XY 0.10392 um, Z 0.15 um.
Z_XY_RATIO = 0.15 / 0.10392

SCENARIOS = {
    # Calibrated like the real layers in test_widget.py, which checks the
    # widget reproduces this scenario exactly.
    "manual_t1.5": {
        "thresholds": [0.30, 0.30],
        "contact_threshold": 1.5,
        "harness": {"z_step": 0.15, "xy_pixel": 0.10392},
    },
    "manual_t0_merge3": {
        "thresholds": [0.30, 0.30],
        "contact_threshold": 0.0,
        "harness": {
            "junction_merge_px": 3.0,
            "z_step": 0.15,
            "xy_pixel": 0.10392,
        },
    },
    # Uncalibrated (Z/XY ratio 1.0 for shape/branch metrics). Min body
    # size 150 is large enough to actually remove Golgi bodies from the
    # mask (3397 -> 1378 px), so the mask filter path is exercised.
    "otsu_t2_minbody150_maskfilter": {
        "thresholds": None,
        "contact_threshold": 2.0,
        "harness": {"min_body_size": 150, "filter_threshold_mask": True},
    },
    # Raw-intensity mode at the raw equivalent of manual_t1.5's 0.30:
    # must reproduce manual_t1.5 (checked in test_raw_scenario_...).
    "raw_equiv_of_manual_t1.5": {
        "thresholds": "raw_from_manual_t1.5",
        "contact_threshold": 1.5,
        "harness": {"z_step": 0.15, "xy_pixel": 0.10392},
    },
}


def make_phantom(seed=7):
    """How phantom.npz was generated (kept for the record; the test
    reads the stored file, not this function)."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    shape = (12, 96, 96)
    mito = np.zeros(shape)
    for _ in range(30):  # curved tubules
        z0 = rng.uniform(3, 9)
        p = rng.uniform(10, 86, 2)
        ang = rng.uniform(0, 2 * np.pi)
        for _ in range(25):
            ang += rng.normal(0, 0.15)
            p = np.clip(p + [np.cos(ang), np.sin(ang)], 1, 94)
            mito[int(round(z0)), int(p[0]), int(p[1])] = 1
    mito = gaussian_filter(mito, (0.9, 1.2, 1.2))
    golgi = np.zeros(shape)
    for _ in range(25):  # blobs
        c = rng.uniform([2, 10, 10], [10, 86, 86]).astype(int)
        golgi[tuple(c)] = 1
    golgi = gaussian_filter(golgi, (1.0, 2.5, 2.5))

    def to_counts(x, peak):
        x = x / x.max() * peak + 100
        return rng.poisson(x).astype(np.uint16)

    return to_counts(golgi, 3000), to_counts(mito, 2500)


def run_all():
    data = np.load(PHANTOM)
    golgi, mito = data["golgi"], data["mito"]
    results = {}
    for name, sc in SCENARIOS.items():
        h = _make_harness(**sc["harness"])
        thresholds = sc["thresholds"]
        if thresholds == "raw_from_manual_t1.5":
            ref = results["manual_t1.5"]
            thresholds = [
                ("raw", ref["Threshold Raw (Golgi)"]),
                ("raw", ref["Threshold Raw (Mito)"]),
            ]
        out, _, _ = run_pipeline(
            [golgi, mito],
            h,
            thresholds=thresholds,
            contact_threshold=sc["contact_threshold"],
            z_xy_ratio=Z_XY_RATIO,
            labels=["Golgi", "Mito"],
        )
        results[name] = {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in out.items()
        }
    return results


def test_golden_pipeline_outputs():
    got = run_all()
    if os.environ.get("LOCA_UPDATE_GOLDEN"):
        EXPECTED.write_text(json.dumps(got, indent=1, sort_keys=True))
        pytest.skip("golden values regenerated")
    expected = json.loads(EXPECTED.read_text())
    problems = []
    for sc, exp in expected.items():
        g = got[sc]
        if set(g) != set(exp):
            problems.append(
                f"[{sc}] metric names changed: "
                f"added {sorted(set(g) - set(exp))}, "
                f"removed {sorted(set(exp) - set(g))}"
            )
        for k in sorted(set(g) & set(exp)):
            a, b = g[k], exp[k]
            same = (a is None and b is None) or (
                a is not None
                and b is not None
                and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
            )
            if not same:
                problems.append(f"[{sc}] {k}: expected {b}, got {a}")
    assert not problems, (
        "Pipeline outputs changed vs golden/expected.json:\n  "
        + "\n  ".join(problems)
        + "\nIf intended, regenerate with LOCA_UPDATE_GOLDEN=1."
    )


def test_golden_phantom_is_sane():
    """Guards against a broken phantom silently making the golden test
    trivial (e.g. empty masks give all-zero metrics that never change)."""
    exp = json.loads(EXPECTED.read_text())["manual_t1.5"]
    assert exp["Body Count (Mito)"] >= 5
    assert exp["Junction Count Mean (Mito)"] > 0
    assert 0 < exp["Contact Area"] < exp["Union"]


def test_raw_scenario_reproduces_scaled_scenario():
    got = run_all()
    a, b = got["manual_t1.5"], got["raw_equiv_of_manual_t1.5"]
    assert set(a) == set(b)
    for k in a:
        if a[k] is None:
            assert b[k] is None, k
        else:
            assert math.isclose(a[k], b[k], rel_tol=1e-9), k
