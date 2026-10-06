"""End-to-end: the real widget in a real napari viewer.

The other test files call the widget's numerical code directly. This
one checks the wiring: that clicking Analyze on real layers runs those
same steps with the settings shown in the UI (thresholds, contact
distance, Z/XY ratio read from the layer scale) and produces exactly
the golden numbers. Needs napari + a Qt binding (pytest-qt provides
the fixture); skipped otherwise.
"""

import json

import numpy as np
import pytest

pytest.importorskip("napari.layers")
pytest.importorskip("pytestqt")

from napari_organelle_contact_analyzer import _widget as W  # noqa: E402
from napari_organelle_contact_analyzer._tests.conftest import (  # noqa: E402
    ALL_OUTPUTS,
)
from napari_organelle_contact_analyzer._tests.test_golden import (  # noqa
    EXPECTED,
    PHANTOM,
)

SCALE = (0.15, 0.10392, 0.10392)


@pytest.fixture
def widget(make_napari_viewer, monkeypatch):
    # Never read or write the user's real saved settings.
    monkeypatch.setattr(
        W.OrganelleContactWidget, "_load_settings", lambda self: None
    )
    monkeypatch.setattr(
        W.OrganelleContactWidget, "_save_settings", lambda self: None
    )
    monkeypatch.setattr(W, "QMessageBox", _SilentBox)
    viewer = make_napari_viewer()
    data = np.load(PHANTOM)
    viewer.add_image(data["golgi"], name="Golgi", scale=SCALE)
    viewer.add_image(data["mito"], name="Mito", scale=SCALE)
    w = W.OrganelleContactWidget(viewer)
    w.output_selection.update(ALL_OUTPUTS)
    w.use_layer_names_checkbox.setChecked(True)
    for i in range(2):
        w.per_channel_mode[i].setCurrentText("Manual")
        w.per_channel_manual[i].setValue(0.30)
    w._set_contact_threshold(1.5, save=False)
    w._update_z_range_controls(force_full_reset=True)
    return w


class _SilentBox:
    @staticmethod
    def information(*a, **k):
        pass

    @staticmethod
    def warning(*a, **k):
        raise AssertionError(f"widget showed a warning: {a[1:]}")


def test_defaults_match_golden_scenario(widget):
    """The golden 'manual_t1.5' scenario assumes these UI defaults."""
    assert widget.min_body_size_spinbox.value() == 2
    assert widget.filter_body_metrics_checkbox.isChecked()
    assert not widget.filter_threshold_mask_checkbox.isChecked()
    assert widget.junction_merge_spinbox.value() == 0
    assert not widget.restrict_to_signal_z
    r, calibrated, _ = widget._get_z_xy_ratio()
    assert calibrated and r == pytest.approx(0.15 / 0.10392)


def test_analyze_reproduces_golden_numbers(widget):
    widget.analyze_contacts()
    got = widget.last_metrics
    exp = json.loads(EXPECTED.read_text())["manual_t1.5"]
    assert set(got) == set(exp)
    for k, v in exp.items():
        if v is None:
            assert np.isnan(got[k]), k
        else:
            assert got[k] == pytest.approx(v, rel=1e-9), k


def test_roi_rectangle_is_not_transposed(widget):
    # Wide, short ROI: rows 10-30, cols 5-85 (row, col) = (Y, X).
    rect = np.array([[10, 5], [10, 85], [30, 85], [30, 5]], float)
    widget.viewer.add_shapes([rect], shape_type="polygon", name="ROI")
    widget.per_shape_checkbox.setChecked(True)
    widget.analyze_contacts()
    [m] = widget.last_metrics
    nz = 12
    per_plane = m["ROI Area"] / nz
    assert 20 * 80 <= per_plane <= 21 * 81
    # Signal inside the ROI must be the mask restricted to rows 10-30,
    # cols 5-85; the transposed polygon selects different pixels.
    from skimage.draw import polygon

    mask = widget.last_masks[1]
    shape = mask.shape[1:]

    def roi_sum(rows, cols):
        roi = np.zeros(shape, bool)
        roi[polygon(rows, cols, shape=shape)] = True
        return int((mask & roi[None]).sum())

    right = roi_sum(rect[:, 0], rect[:, 1])
    transposed = roi_sum(rect[:, 1], rect[:, 0])
    assert right != transposed  # the check can discriminate
    assert m["Signal Area (Mito)"] == right


def test_raw_intensity_mode_in_widget(widget):
    """Raw mode at the reported raw equivalent reproduces the scaled-mode
    run; the 'Last run' labels show both values."""
    widget.analyze_contacts()
    scaled = dict(widget.last_metrics)
    assert "0.3000 scaled" in widget.per_channel_thresh_info[1].text()
    for i, name in enumerate(("Golgi", "Mito")):
        widget.per_channel_mode[i].setCurrentText(W.THRESH_MODE_RAW)
        assert widget.per_channel_raw[i].isEnabled()
        assert not widget.per_channel_manual[i].isEnabled()
        widget.per_channel_raw[i].setValue(scaled[f"Threshold Raw ({name})"])
    widget.analyze_contacts()
    raw = widget.last_metrics
    for k, v in scaled.items():
        if isinstance(v, float) and np.isnan(v):
            assert np.isnan(raw[k]), k
        elif k.startswith("Threshold Scaled"):
            # Raw spinbox keeps 2 decimals, so allow that rounding.
            assert raw[k] == pytest.approx(v, abs=1e-4), k
        else:
            assert raw[k] == pytest.approx(v, rel=1e-9), k
