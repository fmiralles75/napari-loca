"""Shared fixtures for LocA's test suite.

Most tests exercise the widget's numerical code *without* building the
Qt widget or a napari viewer. ``OrganelleContactWidget`` keeps its
metric code in methods that only read a handful of attributes
(``output_selection``, a few checkboxes/spinboxes), so ``make_harness``
builds a tiny stand-in object that borrows those exact methods from the
real class. The code under test is therefore the code the plugin runs,
not a re-implementation of it.
"""

from typing import Any, Dict, Optional

import numpy as np
import pytest

from napari_loca import _widget as W

# Every metric family, switched on (Morphology is off by default in
# the plugin; the tests want all of it).
ALL_OUTPUTS: Dict[str, bool] = {
    "Intersection": True,
    "Union": True,
    "Intersection/Union (Contact Coefficient)": True,
    "Contact Area": True,
    "Signal Area": True,
    "Intersection/Ch Signal Area": True,
    "Fragmentation Metrics": True,
    "Morphology Shape": True,
    "Morphology Network": True,
    "ROI Area": True,
    "Signal Area/ROI Area": True,
    "Mean Intensity": True,
    "Contact Mean Intensity": True,
    "ROI Geometry": True,
    "Contact Spatial": True,
}


class FakeCheck:
    """Stand-in for QCheckBox."""

    def __init__(self, checked: bool = False):
        self._v = bool(checked)

    def isChecked(self) -> bool:  # noqa: N802 (Qt naming)
        return self._v


class FakeSpin:
    """Stand-in for QSpinBox / QDoubleSpinBox."""

    def __init__(self, value: float):
        self._v = value

    def value(self):
        return self._v


class FakeLineEdit:
    def __init__(self, text: str = ""):
        self._t = text

    def text(self) -> str:
        return self._t


class FakeLabel:
    def setText(self, _t):  # noqa: N802
        pass


_BORROWED = (
    "_compute_metrics_bundle",
    "_labeled_bodies_for_metrics",
    "_filter_small_bodies",
    "_skeleton_and_junctions",
    "_group_junctions",
    "_mean_sd_wmean",
    "_region_mask_from_mode",
    "_get_z_xy_ratio",
    "_get_voxel_sizes",
    "_compute_signal_restricted_z_indices",
    "_get_default_restrict_signal_z_channels",
    "_sanitize_excel_sheet_name",
    "_make_unique_sheet_name",
    "_write_graphpad_workbook_from_dataframe",
    "_collect_export_rows",
    "save_metrics",
    "export_graphpad_prism",
    "append_to_spreadsheet",
    "append_to_graphpad_prism",
    "add_analysis",
)


class Harness:
    """Minimal object carrying the real widget methods listed above."""


for _name in _BORROWED:
    # Copy from __dict__ so staticmethods stay staticmethods.
    setattr(Harness, _name, W.OrganelleContactWidget.__dict__[_name])


def _make_harness(
    outputs: Optional[Dict[str, bool]] = None,
    min_body_size: int = 2,
    filter_body_metrics: bool = True,
    filter_threshold_mask: bool = False,
    junction_merge_px: float = 0.0,
    z_step: Optional[float] = None,
    xy_pixel: Optional[float] = None,
    **extra: Any,
) -> Harness:
    h = Harness()
    h.output_selection = dict(ALL_OUTPUTS if outputs is None else outputs)
    h.min_body_size_spinbox = FakeSpin(min_body_size)
    h.filter_body_metrics_checkbox = FakeCheck(filter_body_metrics)
    h.filter_threshold_mask_checkbox = FakeCheck(filter_threshold_mask)
    h.junction_merge_spinbox = FakeSpin(junction_merge_px)
    manual = z_step is not None and xy_pixel is not None
    h.manual_voxel_calibration_checkbox = FakeCheck(manual)
    h.z_step_spinbox = FakeSpin(z_step if manual else 1.0)
    h.xy_pixel_spinbox = FakeSpin(xy_pixel if manual else 1.0)
    h._get_mapped_base_layer = lambda: None
    h.enable_intensity_comparisons = False
    h.intensity_comparisons = []
    h.restrict_signal_z_channels = []
    h.metrics_list = []
    h.last_metrics = None
    h.analysis_name_edit = FakeLineEdit("")
    h.sequential_label_checkbox = FakeCheck(False)
    h.analysis_count_label = FakeLabel()
    for k, v in extra.items():
        setattr(h, k, v)
    return h


@pytest.fixture
def make_harness():
    return _make_harness


def run_pipeline(
    images,
    harness: Harness,
    thresholds=None,
    contact_threshold: float = 0.0,
    z_xy_ratio: float = 1.0,
    labels=None,
    focus=None,
):
    """Same sequence of steps as OrganelleContactWidget.analyze_contacts
    (full-image mode, no ROI): normalize -> threshold -> optional
    small-body mask filter -> distance maps -> contacts -> metrics bundle
    + threshold columns. ``thresholds`` entries: a float (Manual, scaled
    0-1), None (Automatic, Otsu), or ("raw", value) for
    Manual (raw intensity). ``focus``: Focus channel contacts method
    (channel index), or None for Overlap-based."""
    n = len(images)
    labels = labels or [f"Ch{i + 1}" for i in range(n)]
    raws = [np.asarray(im) for im in images]
    ranges = [W.normalization_range(r) for r in raws]
    norm = [W.normalize_signal(r, rng) for r, rng in zip(raws, ranges)]
    masks, dists, used = [], [], []
    for i in range(n):
        t = None if thresholds is None else thresholds[i]
        if t is None:
            mode, manual, raw_v = "Automatic", 0.0, 0.0
        elif isinstance(t, tuple):
            mode, manual, raw_v = W.THRESH_MODE_RAW, 0.0, t[1]
        else:
            mode, manual, raw_v = "Manual", t, 0.0
        m, t_scaled, t_raw = W.threshold_channel(
            raws[i], norm[i], ranges[i], mode, "Otsu", manual, raw_v
        )
        if harness.filter_threshold_mask_checkbox.isChecked():
            m = harness._filter_small_bodies(
                m, harness.min_body_size_spinbox.value()
            )[0]
        used.append((t_scaled, t_raw))
        masks.append(m)
        dists.append(W.contact_distance_map(m, z_xy_ratio))
    contacts = W.compute_contacts(masks, dists, contact_threshold, focus=focus)
    metrics = harness._compute_metrics_bundle(
        raw_signals=raws,
        masks=masks,
        contacts=contacts,
        ch_labels=labels,
        roi_poly_data=None,
        roi_area=int(masks[0].size),
    )
    metrics.update(W.threshold_columns(labels, used))
    return metrics, masks, contacts


@pytest.fixture
def pipeline():
    return run_pipeline


# ---------------------------------------------------------------------
# Synthetic shapes
# ---------------------------------------------------------------------
def box(shape, lo, hi):
    """Boolean array, True on the half-open box [lo, hi) per axis."""
    m = np.zeros(shape, dtype=bool)
    m[tuple(slice(a, b) for a, b in zip(lo, hi))] = True
    return m


def disk(shape, center, radius):
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    d2 = sum((g - c) ** 2 for g, c in zip(grids, center))
    return d2 <= radius**2


def ellipsoid_physical(shape, center, radii_phys, spacing):
    """Ellipsoid defined in physical units, sampled on a grid with
    ``spacing`` (per-axis voxel size)."""
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    acc = 0
    for g, c, r, sp in zip(grids, center, radii_phys, spacing):
        acc = acc + ((g - c) * sp / r) ** 2
    return acc <= 1.0
