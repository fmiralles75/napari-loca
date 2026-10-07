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

from napari_loca import _widget as W  # noqa: E402
from napari_loca._tests.conftest import (  # noqa: E402
    ALL_OUTPUTS,
)
from napari_loca._tests.test_golden import (  # noqa
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


# ---------------------------------------------------------------------
# Input controls: mouse wheel needs focus; spinboxes show full values
# ---------------------------------------------------------------------
def _send_wheel(w, notches=1):
    """Deliver one wheel event to ``w`` the way Qt would (unhandled
    events propagate to the parent, as during real scrolling)."""
    from qtpy.QtCore import QPoint, QPointF, Qt
    from qtpy.QtGui import QWheelEvent
    from qtpy.QtWidgets import QApplication

    pos = QPointF(w.width() / 2, w.height() / 2)
    glob = QPointF(w.mapToGlobal(pos.toPoint()))
    ev = QWheelEvent(
        pos,
        glob,
        QPoint(0, 0),
        QPoint(0, 120 * notches),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(w, ev)


def test_plain_qt_inputs_are_not_used():
    """Every spinbox/combo/slider must be a ScrollSafe* subclass, or
    scrolling the panel changes settings again. Checks the source so
    dialogs built later are covered too."""
    import inspect
    import re

    src = inspect.getsource(W)
    plain = re.findall(
        r"(?<![\w.])(QSpinBox|QDoubleSpinBox|QComboBox|QSlider)\(", src
    )
    assert plain == []


def test_wheel_ignored_until_control_is_focused(widget, monkeypatch):
    from qtpy.QtCore import Qt

    manual = widget.per_channel_manual[0]  # enabled: fixture sets Manual
    manual.setValue(0.08)
    body = widget.min_body_size_spinbox
    slider = widget.ct_slider
    combo = widget.channel_mode_combo
    for w in (manual, body, slider, combo):
        assert w.focusPolicy() == Qt.StrongFocus  # wheel can't focus it
        assert not w.hasFocus()

    def state():
        return (
            manual.value(),
            body.value(),
            slider.value(),
            combo.currentIndex(),
        )

    before = state()
    for w in (manual, body, slider, combo):
        _send_wheel(w)
        _send_wheel(w, -2)
    after = state()
    assert after == before

    # Once clicked into (focused), the wheel works as normal.
    for w in (manual, body, slider):
        monkeypatch.setattr(w, "hasFocus", lambda: True)
    _send_wheel(manual)
    assert manual.value() == pytest.approx(0.08 + manual.singleStep())
    _send_wheel(body)
    assert body.value() == before[1] + 1
    _send_wheel(slider)
    assert slider.value() != before[2]


def _hidden_text_px(spin):
    """How many pixels of the spinbox's text are cut off.

    Paints the inner line edit with the cursor at the end, then at the
    start: if the text fits, the line edit never scrolls and the two
    cursor positions are a full text-width apart; if it doesn't fit,
    they are only the visible width apart.
    """
    from qtpy.QtCore import Qt

    le = spin.lineEdit()
    full = le.fontMetrics().horizontalAdvance(le.text())
    if le.width() <= 0:  # collapsed: nothing is visible
        return full
    le.end(False)
    le.grab()
    x_end = le.inputMethodQuery(Qt.ImCursorRectangle).x()
    le.home(False)
    le.grab()
    x_home = le.inputMethodQuery(Qt.ImCursorRectangle).x()
    return full - (x_end - x_home)


def test_spinboxes_show_their_full_value(widget, qtbot):
    """Under napari's own stylesheet (as when docked), every spinbox in
    the panel shows its widest value in full even when a layout
    squeezes it to its minimum width -- in particular all 4 decimals of
    the manual threshold. Before the fix, napari's ``min-width`` let a
    crowded row shrink that box to 97 of the 115 px it needs (13 px of
    "0.0825" hidden), and long values like "10000.0000" lost 21 px even
    at full size."""
    from napari.qt import get_stylesheet
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QAbstractSpinBox, QApplication, QScrollArea

    widget.setStyleSheet(get_stylesheet("dark"))
    widget.setAttribute(Qt.WA_DontShowOnScreen, True)
    widget.resize(420, 900)
    widget.show()
    qtbot.wait(50)

    spins = widget.findChildren(QAbstractSpinBox)
    assert widget.per_channel_manual[0] in spins
    hidden = {}
    try:
        for n, spin in enumerate(spins):
            name = f"#{n} {type(spin).__name__} {spin.toolTip()[:30]!r}"
            need = W._spinbox_min_width(spin)
            assert need > 0, name
            # napari's fixed 90 px minimum replaced by the real need.
            assert spin.minimumWidth() == need, name
            old = spin.value()
            spin.blockSignals(True)
            spin.setFixedWidth(need)  # the narrowest a layout can make it
            QApplication.processEvents()
            for v in (spin.maximum(), spin.minimum()):
                spin.setValue(v)
                px = _hidden_text_px(spin)
                if px > 2:  # sub-pixel rounding between Qt and metrics
                    hidden[f"{name} {spin.text()!r} @{need}px"] = px
            spin.setValue(old)
            spin.setMaximumWidth(16777215)  # QWIDGETSIZE_MAX
            spin.setMinimumWidth(need)
            spin.blockSignals(False)

        manual = widget.per_channel_manual[0]
        manual.setFixedWidth(manual.minimumWidth())
        manual.setValue(0.0825)
        QApplication.processEvents()
        assert manual.text() == "0.0825"
        manual_hidden = _hidden_text_px(manual)

        # Report how wide a dock LocA needs (run with -s to see it).
        scroll = widget.findChild(QScrollArea)
        content = scroll.widget().minimumSizeHint().width()
        print(
            f"\nLocA panel content needs >= {content} px "
            f"(viewport at a 420-px panel: {scroll.viewport().width()})"
        )
    finally:
        widget.hide()
    assert hidden == {}, f"text cut off (px hidden): {hidden}"
    assert manual_hidden <= 2
