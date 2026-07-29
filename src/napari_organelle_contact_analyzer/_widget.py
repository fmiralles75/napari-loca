"""
napari-organelle-contact-analyzer widget.

Implements the main ``OrganelleContactWidget`` used to threshold organelle
signal channels, compute contact/overlap metrics between them, and export
results.

Note: this module previously monkey-patched ``tifffile.RESUNIT`` and
napari's ``_shapes_mouse_bindings.polygon_creating`` at import time to work
around bugs in those libraries. Both patches were removed in favor of
explicit version constraints in ``pyproject.toml``
(``tifffile<2025.2.18`` and ``napari>=0.6.1``) — see the comments there for
the upstream issues each constraint works around.
"""

import copy
import re
import warnings

from typing import TYPE_CHECKING, List, Dict, Any, Optional, Tuple

import napari
from magicgui import magic_factory
from magicgui.widgets import Container, create_widget
from qtpy.QtGui import QIntValidator
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QWidget,
    QLabel,
    QSlider,
    QLineEdit,
    QSpinBox,
    QComboBox,
    QFileDialog,
    QDoubleSpinBox,
    QCheckBox,
    QScrollArea,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QGridLayout,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSizePolicy,
)

from skimage import filters
from skimage.util import img_as_float
from scipy.ndimage import distance_transform_edt
from skimage.draw import polygon
import numpy as np
import pandas as pd
import imageio.v2 as imageio
from scipy.spatial import ConvexHull

if TYPE_CHECKING:
    import napari

AUTO_METHODS = {
    "Otsu": filters.threshold_otsu,
    "Li": filters.threshold_li,
    "Mean": filters.threshold_mean,
    "Minimum": filters.threshold_minimum,
    "Triangle": filters.threshold_triangle,
    "Yen": filters.threshold_yen,
    "Isodata": filters.threshold_isodata,
}


# -----------------------------
# Helpers
# -----------------------------
def compute_contact_density(contacts: np.ndarray, union_signal: np.ndarray):
    coords = np.column_stack(np.where(contacts))
    if coords.shape[0] < 2:
        avg_dist = np.nan
    else:
        from scipy.spatial import KDTree

        tree = KDTree(coords)
        dists, _ = tree.query(coords, k=2)
        avg_dist = np.mean(dists[:, 1])
    cell_area = np.sum(union_signal)
    density_ratio = avg_dist / cell_area if cell_area > 0 else np.nan
    return avg_dist, cell_area, density_ratio


def compute_roi_geometry(poly_data: np.ndarray):
    if poly_data.shape[1] > 2:
        points2d = np.column_stack((poly_data[:, 2], poly_data[:, 1]))
    else:
        points2d = poly_data
    if points2d.shape[0] < 3:
        return np.nan, np.nan, np.nan
    try:
        hull = ConvexHull(points2d)
        hull_points = points2d[hull.vertices]
        max_dist = 0
        p1 = None
        p2 = None
        for i in range(len(hull_points)):
            for j in range(i + 1, len(hull_points)):
                d = np.linalg.norm(hull_points[i] - hull_points[j])
                if d > max_dist:
                    max_dist = d
                    p1 = hull_points[i]
                    p2 = hull_points[j]
        max_perp = 0
        if p1 is not None and p2 is not None and np.linalg.norm(p2 - p1) != 0:
            for point in points2d:
                d_perp = np.abs(
                    np.cross(p2 - p1, p1 - point)
                ) / np.linalg.norm(p2 - p1)
                if d_perp > max_perp:
                    max_perp = d_perp
        dist_ratio = max_dist / max_perp if max_perp > 0 else np.nan
        return max_dist, max_perp, dist_ratio
    except Exception:
        return np.nan, np.nan, np.nan


def compute_additional_metrics(
    max_dist: float, max_perp: float, points2d: np.ndarray
):
    if max_dist > 0 and max_perp > 0:
        ellipse_circ = np.pi * (
            3 * (max_dist + max_perp)
            - np.sqrt((3 * max_dist + max_perp) * (max_dist + 3 * max_perp))
        )
    else:
        ellipse_circ = np.nan
    if points2d.shape[0] < 2:
        poly_perim = np.nan
    else:
        closed = np.vstack((points2d, points2d[0]))
        diffs = np.diff(closed, axis=0)
        poly_perim = np.sum(np.linalg.norm(diffs, axis=1))
    circ_ratio = (
        ellipse_circ / poly_perim if poly_perim and poly_perim != 0 else np.nan
    )
    return ellipse_circ, poly_perim, circ_ratio


def safe_mean_intensity(signal: np.ndarray, region_mask: np.ndarray) -> float:
    n = int(np.sum(region_mask))
    if n <= 0:
        return float("nan")
    return float(np.sum(signal[region_mask]) / n)


def parse_channel_list_text(txt: str, n_channels: int) -> List[int]:
    out: List[int] = []
    txt = (txt or "").strip()
    if not txt:
        return out
    parts = [p.strip() for p in txt.split(",") if p.strip()]
    for p in parts:
        try:
            v = int(p) - 1
            if 0 <= v < n_channels:
                out.append(v)
        except Exception:
            pass
    return out


def channels_to_text(channels_0_based: List[int]) -> str:
    if not channels_0_based:
        return ""
    chans = [int(c) for c in channels_0_based]
    chans = [c for c in chans if c >= 0]
    return ",".join(str(c + 1) for c in chans)


# -----------------------------
# Dialogs
# -----------------------------
class SignalIntensityComparisonsDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        n_channels: int,
        channel_labels: List[str],
        existing: List[Dict[str, Any]],
    ):
        super().__init__(parent)
        self.setWindowTitle("Signal Intensity Comparisons")

        self.n_channels = n_channels
        self.channel_labels = channel_labels[:]
        self._comparisons: List[Dict[str, Any]] = (
            copy.deepcopy(existing) if existing else []
        )

        layout = QVBoxLayout()

        info = QLabel(
            "Define mean-intensity comparisons.\n"
            "Each row measures the mean intensity of a Source Channel within a Base Region.\n"
            "Optionally subtract another region: Result Region = BaseRegion - SubtractRegion.\n\n"
            "Examples:\n"
            "• Mean intensity of Ch1 in Union(Ch2,3)\n"
            "• Mean intensity of Ch1 in Union(Ch2,3) minus Intersection(Ch1,2,3)\n\n"
            "Channel lists are comma-separated, 1-based (e.g., 2,3)."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            [
                "Enabled",
                "Source Channel",
                "Base Mode",
                "Base Channels",
                "Subtract Mode",
                "Subtract Channels",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.Stretch
        )

        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add Comparison")
        self.remove_btn = QPushButton("Remove Selected")
        self.add_btn.clicked.connect(self._add_row)
        self.remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.remove_btn)
        layout.addLayout(btn_row)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        self.setLayout(layout)

        for comp in self._comparisons:
            self._insert_row(comp)

        if self.table.rowCount() == 0:
            self._add_row()

    def _insert_row(self, comp: Dict[str, Any]):
        r = self.table.rowCount()
        self.table.insertRow(r)

        enabled_cb = QCheckBox()
        enabled_cb.setChecked(bool(comp.get("enabled", True)))
        self.table.setCellWidget(r, 0, enabled_cb)

        src_combo = QComboBox()
        src_combo.addItems(
            [
                f"{i+1}: {self.channel_labels[i]}"
                for i in range(self.n_channels)
            ]
        )
        src_combo.setCurrentIndex(int(comp.get("source_ch", 0)))
        self.table.setCellWidget(r, 1, src_combo)

        base_mode_combo = QComboBox()
        base_mode_combo.addItems(["Union", "Intersection", "Contacts"])
        base_mode = comp.get("mode", "Union")
        base_mode_combo.setCurrentText(
            base_mode
            if base_mode in ["Union", "Intersection", "Contacts"]
            else "Union"
        )
        self.table.setCellWidget(r, 2, base_mode_combo)

        base_channels = comp.get(
            "region_channels", [1] if self.n_channels >= 2 else []
        )
        base_item = QTableWidgetItem(channels_to_text(base_channels))
        base_item.setToolTip(
            "Enter channel numbers (1-based) separated by commas, e.g. 2,3"
        )
        self.table.setItem(r, 3, base_item)

        sub_mode_combo = QComboBox()
        sub_mode_combo.addItems(["None", "Union", "Intersection", "Contacts"])
        sub_mode = comp.get("subtract_mode", "None")
        sub_mode_combo.setCurrentText(
            sub_mode
            if sub_mode in ["None", "Union", "Intersection", "Contacts"]
            else "None"
        )
        self.table.setCellWidget(r, 4, sub_mode_combo)

        sub_channels = comp.get("subtract_channels", [])
        sub_item = QTableWidgetItem(channels_to_text(sub_channels))
        sub_item.setToolTip(
            "Enter channel numbers (1-based) separated by commas, e.g. 1,2,3"
        )
        self.table.setItem(r, 5, sub_item)

    def _add_row(self):
        default_region = [1] if self.n_channels >= 2 else []
        comp = {
            "enabled": True,
            "source_ch": 0,
            "mode": "Union",
            "region_channels": default_region,
            "subtract_mode": "None",
            "subtract_channels": [],
        }
        self._insert_row(comp)

    def _remove_selected(self):
        rows = sorted(
            set(idx.row() for idx in self.table.selectedIndexes()),
            reverse=True,
        )
        for r in rows:
            self.table.removeRow(r)

    def get_comparisons(self) -> List[Dict[str, Any]]:
        comps: List[Dict[str, Any]] = []
        for r in range(self.table.rowCount()):
            enabled_cb = self.table.cellWidget(r, 0)
            src_combo = self.table.cellWidget(r, 1)
            base_mode_combo = self.table.cellWidget(r, 2)
            base_item = self.table.item(r, 3)
            sub_mode_combo = self.table.cellWidget(r, 4)
            sub_item = self.table.item(r, 5)

            enabled = bool(enabled_cb.isChecked()) if enabled_cb else True
            source_ch = int(src_combo.currentIndex()) if src_combo else 0
            mode = (
                str(base_mode_combo.currentText())
                if base_mode_combo
                else "Union"
            )

            base_txt = base_item.text() if base_item else ""
            region_channels = parse_channel_list_text(
                base_txt, self.n_channels
            )

            subtract_mode = (
                str(sub_mode_combo.currentText()) if sub_mode_combo else "None"
            )
            subtract_txt = sub_item.text() if sub_item else ""
            subtract_channels = parse_channel_list_text(
                subtract_txt, self.n_channels
            )

            comps.append(
                {
                    "enabled": enabled,
                    "source_ch": source_ch,
                    "mode": mode,
                    "region_channels": region_channels,
                    "subtract_mode": subtract_mode,
                    "subtract_channels": subtract_channels,
                }
            )
        return comps


class OutputSelectionDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        current_selection: Dict[str, bool],
        enable_intensity_comparisons: bool,
        n_channels: int,
        channel_labels: List[str],
        comparisons: List[Dict[str, Any]],
    ):
        super().__init__(parent)
        self.setWindowTitle("Output Selection")

        self.n_channels = n_channels
        self.channel_labels = channel_labels[:]
        self._selection = current_selection.copy() if current_selection else {}
        self._enable_intensity_comparisons = bool(enable_intensity_comparisons)
        self._comparisons = copy.deepcopy(comparisons) if comparisons else []

        layout = QVBoxLayout()

        core_box = QGroupBox("Core Overlap Metrics")
        core_layout = QVBoxLayout()
        self.cb_intersection = QCheckBox("Intersection")
        self.cb_union = QCheckBox("Union")
        self.cb_jaccard = QCheckBox("Intersection/Union (Contact Coefficient)")
        self.cb_contact_area = QCheckBox("Contact Area")
        for cb, key in [
            (self.cb_intersection, "Intersection"),
            (self.cb_union, "Union"),
            (self.cb_jaccard, "Intersection/Union (Contact Coefficient)"),
            (self.cb_contact_area, "Contact Area"),
        ]:
            cb.setChecked(self._selection.get(key, True))
            core_layout.addWidget(cb)
        core_box.setLayout(core_layout)
        layout.addWidget(core_box)

        perch_box = QGroupBox("Per-Channel Area Metrics")
        perch_layout = QVBoxLayout()
        self.cb_signal_area = QCheckBox("Signal Area (per channel)")
        self.cb_intersection_over_ch = QCheckBox(
            "Intersection / Channel Signal Area (per channel)"
        )
        self.cb_signal_area.setChecked(
            self._selection.get("Signal Area", True)
        )
        self.cb_intersection_over_ch.setChecked(
            self._selection.get("Intersection/Ch Signal Area", True)
        )
        perch_layout.addWidget(self.cb_signal_area)
        perch_layout.addWidget(self.cb_intersection_over_ch)
        perch_box.setLayout(perch_layout)
        layout.addWidget(perch_box)

        roi_area_box = QGroupBox("Per-ROI Area Metrics")
        roi_area_layout = QVBoxLayout()
        self.cb_roi_area = QCheckBox("ROI Area (# pixels in ROI mask)")
        self.cb_roi_area_over_ch = QCheckBox(
            "Signal Area / ROI Area (per channel)"
        )
        self.cb_roi_area.setChecked(self._selection.get("ROI Area", True))
        self.cb_roi_area_over_ch.setChecked(
            self._selection.get("Signal Area/ROI Area", True)
        )
        roi_area_layout.addWidget(self.cb_roi_area)
        roi_area_layout.addWidget(self.cb_roi_area_over_ch)
        roi_area_box.setLayout(roi_area_layout)
        layout.addWidget(roi_area_box)

        intensity_box = QGroupBox("Intensity Metrics")
        intensity_layout = QVBoxLayout()
        self.cb_mean_intensity = QCheckBox(
            "Mean Intensity within each channel mask (per channel)"
        )
        self.cb_contact_mean_intensity = QCheckBox(
            "Mean Intensity within contact region (per channel)"
        )
        self.cb_mean_intensity.setChecked(
            self._selection.get("Mean Intensity", True)
        )
        self.cb_contact_mean_intensity.setChecked(
            self._selection.get("Contact Mean Intensity", True)
        )
        intensity_layout.addWidget(self.cb_mean_intensity)
        intensity_layout.addWidget(self.cb_contact_mean_intensity)

        self.cb_enable_intensity_comparisons = QCheckBox(
            "Compute Signal Intensity Comparisons (configured below)"
        )
        self.cb_enable_intensity_comparisons.setChecked(
            self._enable_intensity_comparisons
        )
        self.configure_intensity_comparisons_btn = QPushButton(
            "Configure Signal Intensity Comparisons…"
        )
        self.configure_intensity_comparisons_btn.clicked.connect(
            self._configure_intensity_comparisons
        )
        intensity_layout.addWidget(self.cb_enable_intensity_comparisons)
        intensity_layout.addWidget(self.configure_intensity_comparisons_btn)

        intensity_box.setLayout(intensity_layout)
        layout.addWidget(intensity_box)

        adv_box = QGroupBox("Advanced ROI/Spatial Metrics")
        adv_layout = QVBoxLayout()
        self.cb_roi_geometry = QCheckBox(
            "ROI geometry metrics (Max Distance, Perp, Ratio, Perimeter, etc.)"
        )
        self.cb_contact_density = QCheckBox(
            "Contact spatial metrics (Avg Contact Dist, etc.)"
        )
        self.cb_roi_geometry.setChecked(
            self._selection.get("ROI Geometry", True)
        )
        self.cb_contact_density.setChecked(
            self._selection.get("Contact Spatial", True)
        )
        adv_layout.addWidget(self.cb_roi_geometry)
        adv_layout.addWidget(self.cb_contact_density)
        adv_box.setLayout(adv_layout)
        layout.addWidget(adv_box)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        self.setLayout(layout)

    def _configure_intensity_comparisons(self):
        dlg = SignalIntensityComparisonsDialog(
            self,
            n_channels=self.n_channels,
            channel_labels=self.channel_labels,
            existing=self._comparisons,
        )
        if dlg.exec_() == QDialog.Accepted:
            self._comparisons = dlg.get_comparisons()

    def get_results(
        self,
    ) -> Tuple[Dict[str, bool], bool, List[Dict[str, Any]]]:
        sel: Dict[str, bool] = {}
        sel["Intersection"] = self.cb_intersection.isChecked()
        sel["Union"] = self.cb_union.isChecked()
        sel["Intersection/Union (Contact Coefficient)"] = (
            self.cb_jaccard.isChecked()
        )
        sel["Contact Area"] = self.cb_contact_area.isChecked()

        sel["Signal Area"] = self.cb_signal_area.isChecked()
        sel["Intersection/Ch Signal Area"] = (
            self.cb_intersection_over_ch.isChecked()
        )

        sel["ROI Area"] = self.cb_roi_area.isChecked()
        sel["Signal Area/ROI Area"] = self.cb_roi_area_over_ch.isChecked()

        sel["Mean Intensity"] = self.cb_mean_intensity.isChecked()
        sel["Contact Mean Intensity"] = (
            self.cb_contact_mean_intensity.isChecked()
        )

        sel["ROI Geometry"] = self.cb_roi_geometry.isChecked()
        sel["Contact Spatial"] = self.cb_contact_density.isChecked()

        enable_comp = self.cb_enable_intensity_comparisons.isChecked()
        return sel, enable_comp, self._comparisons


# -----------------------------
# Main Widget
# -----------------------------
class OrganelleContactWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        self.threshold = 0
        self._last_z_source_signature = None
        self.metrics_list: List[Dict[str, Any]] = []
        self.last_metrics: Any = {}
        self.last_masks: List[np.ndarray] = []
        self.current_roi_index = 0

        self._last_contacts_display: Optional[np.ndarray] = None
        self._last_base_layer: Optional["napari.layers.Image"] = None
        self._last_display_z_translate = None
        self._last_display_scale = None

        self.max_channels_supported = 4
        self.channel_layer_indices = [0, 1, 2, 3]

        self.use_layer_names_checkbox = QCheckBox(
            "Use layer names for channel labels"
        )
        self.use_layer_names_checkbox.setChecked(False)

        self.output_selection: Dict[str, bool] = {
            "Intersection": True,
            "Union": True,
            "Intersection/Union (Contact Coefficient)": True,
            "Contact Area": True,
            "Signal Area": True,
            "Intersection/Ch Signal Area": True,
            "ROI Area": True,
            "Signal Area/ROI Area": True,
            "Mean Intensity": True,
            "Contact Mean Intensity": True,
            "ROI Geometry": True,
            "Contact Spatial": True,
        }
        self.enable_intensity_comparisons: bool = False
        self.intensity_comparisons: List[Dict[str, Any]] = []

        # Saved-image scale bar settings
        self.include_scale_bar_in_saved_image = False
        self.saved_scale_bar_length = 10.0
        self.saved_scale_bar_unit = "px"
        self.saved_scale_bar_text_size = 20

        # Thresholded-signal Z restriction settings
        self.restrict_to_signal_z = False
        self.restrict_signal_z_channels: List[int] = []

        self.ct_label = QLabel(f"Contact Threshold (pixels): {self.threshold}")
        self.ct_label.setAlignment(Qt.AlignCenter)
        self.ct_slider = QSlider(Qt.Horizontal)
        self.ct_slider.setMinimum(0)
        self.ct_slider.setMaximum(100)
        self.ct_slider.setValue(self.threshold)
        self.ct_slider.valueChanged.connect(self.slider_changed)
        self.ct_text = QLineEdit(str(self.threshold))
        self.ct_text.setValidator(QIntValidator(0, 100))
        self.ct_text.editingFinished.connect(self.text_input_changed)

        self.channels_label = QLabel("Number of Channels:")
        self.channel_mode_combo = QComboBox()
        self.channel_mode_combo.addItems(["2", "3", "4"])
        self.channel_mode_combo.currentIndexChanged.connect(
            lambda _: self._on_analysis_source_changed()
        )

        self.z_range_label = QLabel("Z-stack Range:")
        self.z_min_spinbox = QSpinBox()
        self.z_max_spinbox = QSpinBox()
        self.z_min_spinbox.setMinimum(0)
        self.z_max_spinbox.setMinimum(0)

        self.auto_adjust_z_range_checkbox = QCheckBox(
            "Auto-adjust Z-stack range to current image"
        )
        self.auto_adjust_z_range_checkbox.setChecked(True)
        self.auto_adjust_z_range_checkbox.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed
        )
        self.auto_adjust_z_range_checkbox.stateChanged.connect(
            lambda _: self._update_z_range_controls(force_full_reset=False)
        )

        self.use_layer_names_checkbox.stateChanged.connect(
            lambda _: self._refresh_channel_labels()
        )

        try:
            self.viewer.layers.events.inserted.connect(
                lambda event: self._on_analysis_source_changed()
            )
            self.viewer.layers.events.removed.connect(
                lambda event: self._on_analysis_source_changed()
            )
            self.viewer.layers.events.reordered.connect(
                lambda event: self._on_analysis_source_changed()
            )
        except Exception:
            pass

        self.channel_thresh_container = QWidget()
        self.channel_thresh_layout = QVBoxLayout()
        self.channel_thresh_container.setLayout(self.channel_thresh_layout)
        self.per_channel_mode: List[QComboBox] = []
        self.per_channel_auto: List[QComboBox] = []
        self.per_channel_manual: List[QDoubleSpinBox] = []
        self.per_channel_label_widgets: List[QLabel] = []

        _thresh_rows: List[QHBoxLayout] = []
        for i in range(self.max_channels_supported):
            row = QHBoxLayout()

            label = QLabel(f"Channel {i+1}:")
            label.setWordWrap(True)
            label.setMaximumWidth(260)
            label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            self.per_channel_label_widgets.append(label)
            row.addWidget(label)

            mode_combo = QComboBox()
            mode_combo.addItems(["Automatic", "Manual"])
            self.per_channel_mode.append(mode_combo)
            row.addWidget(mode_combo)

            auto_combo = QComboBox()
            auto_combo.addItems(
                ["Otsu", "Li", "Mean", "Minimum", "Triangle", "Yen", "Isodata"]
            )
            self.per_channel_auto.append(auto_combo)
            row.addWidget(QLabel("Auto:"))
            row.addWidget(auto_combo)

            manual_spin = QDoubleSpinBox()
            manual_spin.setRange(0.0, 1.0)
            manual_spin.setSingleStep(0.01)
            manual_spin.setValue(0.5)
            manual_spin.setEnabled(False)
            self.per_channel_manual.append(manual_spin)
            row.addWidget(QLabel("Manual:"))
            row.addWidget(manual_spin)

            mode_combo.currentIndexChanged.connect(
                self._sync_thresh_mode_states
            )
            _thresh_rows.append(row)

        for row in reversed(_thresh_rows):
            self.channel_thresh_layout.addLayout(row)

        self.result_label = QLabel("Metrics: N/A")
        self.result_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.result_label.setWordWrap(True)

        self.analysis_name_label = QLabel("Analysis Name:")
        self.analysis_name_edit = QLineEdit()

        self.analysis_count_label = QLabel("Analyses Stored: 0")
        self.analysis_count_label.setAlignment(Qt.AlignCenter)

        self.per_shape_checkbox = QCheckBox("Calculate metrics per ROI shape")
        self.sequential_label_checkbox = QCheckBox(
            "Use sequential labeling for ROI shapes"
        )

        self.prev_roi_button = QPushButton("Previous ROI")
        self.next_roi_button = QPushButton("Next ROI")
        self.roi_nav_label = QLabel("ROI: N/A")
        self.prev_roi_button.clicked.connect(self.prev_roi)
        self.next_roi_button.clicked.connect(self.next_roi)

        self.show_thresh_after_checkbox = QCheckBox(
            "Show thresholded layers after Analyze"
        )
        self.show_thresh_after_checkbox.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed
        )
        self.show_thresh_after_checkbox.setChecked(False)

        self.show_contacts_after_checkbox = QCheckBox(
            "Show Contacts layer after Analyze"
        )
        self.show_contacts_after_checkbox.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed
        )
        self.show_contacts_after_checkbox.setChecked(True)

        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.analyze_contacts)

        self.output_selection_button = QPushButton("Output Selection")
        self.output_selection_button.clicked.connect(
            self.open_output_selection
        )

        self.channel_numbering_button = QPushButton("Channel Numbering")
        self.channel_numbering_button.clicked.connect(
            self.open_channel_numbering
        )

        self.metric_display_selection_button = QPushButton(
            "Metric Display Selection"
        )
        self.metric_display_selection_button.clicked.connect(
            self.open_metric_display_selection
        )

        self.scale_bar_settings_button = QPushButton("Scale Bar Settings")
        self.scale_bar_settings_button.clicked.connect(
            self.open_scale_bar_settings
        )

        self.scale_bar_status_label = QLabel("")
        self.scale_bar_status_label.setWordWrap(True)

        self.restrict_signal_z_checkbox = QCheckBox(
            "Restrict analyzed Z-stacks to slices with thresholded signal"
        )
        self.restrict_signal_z_checkbox.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed
        )
        self.restrict_signal_z_checkbox.setChecked(False)
        self.restrict_signal_z_checkbox.stateChanged.connect(
            self._on_restrict_signal_z_changed
        )

        self.restrict_signal_z_button = QPushButton("Signal Z Channels")
        self.restrict_signal_z_button.clicked.connect(
            self.open_signal_z_channel_selection
        )
        self.restrict_signal_z_button.setEnabled(False)

        self.add_analysis_button = QPushButton("Add Analysis")
        self.add_analysis_button.clicked.connect(self.add_analysis)

        self.clear_last_button = QPushButton("Clear Last Analysis")
        self.clear_last_button.clicked.connect(self.clear_last_analysis)

        self.clear_all_button = QPushButton("Clear All Analyses")
        self.clear_all_button.clicked.connect(self.clear_all_analyses)

        self.save_image_button = QPushButton("Save Image")
        self.save_image_button.clicked.connect(self.save_image)

        self.save_metrics_button = QPushButton("Export to Excel")
        self.save_metrics_button.clicked.connect(self.save_metrics)

        self.append_spreadsheet_button = QPushButton("Append to Excel Format")
        self.append_spreadsheet_button.clicked.connect(
            self.append_to_spreadsheet
        )

        self.export_graphpad_button = QPushButton("Export for GraphPad Prism")
        self.export_graphpad_button.clicked.connect(self.export_graphpad_prism)

        self.append_graphpad_button = QPushButton("Append to GraphPad")
        self.append_graphpad_button.clicked.connect(
            self.append_to_graphpad_prism
        )

        self.roi_button = QPushButton("Toggle ROI Selection")
        self.roi_button.clicked.connect(self.toggle_roi_selection)

        self.show_thresh_btns: List[QPushButton] = []
        for i in range(self.max_channels_supported):
            b = QPushButton(f"Show Thresholded Ch {i+1}")
            b.clicked.connect(
                lambda _, idx=i: self.show_thresholded_channel(idx)
            )
            self.show_thresh_btns.append(b)

        self.show_contacts_button = QPushButton("Show Contacts")
        self.show_contacts_button.clicked.connect(self.show_contacts)

        self._metric_display_keys: Optional[List[str]] = None

        self.init_ui()
        self._sync_thresh_mode_states()
        self._refresh_channel_labels()
        self._update_z_range_controls(force_full_reset=True)
        self._auto_set_scale_bar_unit_from_layer()
        self._update_scale_bar_status_label()

    # ---------------- UI ----------------
    def init_ui(self):
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        ct_layout = QHBoxLayout()
        ct_layout.addWidget(self.ct_label)
        ct_layout.addWidget(self.ct_slider)
        ct_layout.addWidget(self.ct_text)
        layout.addLayout(ct_layout)

        channel_layout = QHBoxLayout()
        channel_layout.addWidget(self.channels_label)
        channel_layout.addWidget(self.channel_mode_combo)
        channel_layout.addStretch(1)
        layout.addLayout(channel_layout)

        z_layout = QHBoxLayout()
        z_layout.addWidget(self.z_range_label)
        z_layout.addWidget(QLabel("Min:"))
        z_layout.addWidget(self.z_min_spinbox)
        z_layout.addWidget(QLabel("Max:"))
        z_layout.addWidget(self.z_max_spinbox)
        z_layout.addStretch(1)
        layout.addLayout(z_layout)

        layout.addWidget(self.auto_adjust_z_range_checkbox)
        layout.addWidget(self.use_layer_names_checkbox)
        layout.addWidget(self.channel_thresh_container)

        analyze_grid = QGridLayout()
        analyze_grid.addWidget(self.analyze_button, 0, 0)
        analyze_grid.addWidget(self.output_selection_button, 0, 1)
        analyze_grid.addWidget(self.channel_numbering_button, 0, 2)
        analyze_grid.addWidget(self.metric_display_selection_button, 1, 0)
        analyze_grid.addWidget(self.scale_bar_settings_button, 1, 1)
        analyze_grid.setColumnStretch(2, 1)
        layout.addLayout(analyze_grid)

        restrict_z_layout = QVBoxLayout()
        restrict_z_layout.addWidget(self.restrict_signal_z_checkbox)
        restrict_z_layout.addWidget(
            self.restrict_signal_z_button, alignment=Qt.AlignLeft
        )
        layout.addLayout(restrict_z_layout)

        layout.addWidget(self.scale_bar_status_label)

        toggle_layout = QVBoxLayout()
        toggle_layout.addWidget(self.show_thresh_after_checkbox)
        toggle_layout.addWidget(self.show_contacts_after_checkbox)
        layout.addLayout(toggle_layout)

        layout.addWidget(self.result_label)

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_roi_button)
        nav_layout.addWidget(self.roi_nav_label)
        nav_layout.addWidget(self.next_roi_button)
        nav_layout.addStretch(1)
        layout.addLayout(nav_layout)

        analysis_name_layout = QHBoxLayout()
        analysis_name_layout.addWidget(self.analysis_name_label)
        analysis_name_layout.addWidget(self.analysis_name_edit)
        layout.addLayout(analysis_name_layout)

        layout.addWidget(self.analysis_count_label)
        layout.addWidget(self.per_shape_checkbox)
        layout.addWidget(self.sequential_label_checkbox)

        manage_layout1 = QHBoxLayout()
        manage_layout1.addWidget(self.add_analysis_button)
        manage_layout1.addWidget(self.clear_last_button)
        manage_layout1.addWidget(self.clear_all_button)
        manage_layout1.addStretch(1)
        layout.addLayout(manage_layout1)

        manage_grid = QGridLayout()
        manage_grid.addWidget(self.save_image_button, 0, 0)
        manage_grid.addWidget(self.save_metrics_button, 0, 1)
        manage_grid.addWidget(self.append_spreadsheet_button, 1, 0)
        manage_grid.addWidget(self.export_graphpad_button, 1, 1)
        manage_grid.addWidget(self.append_graphpad_button, 2, 0)
        manage_grid.setColumnStretch(2, 1)
        layout.addLayout(manage_grid)

        layout.addWidget(self.roi_button)
        layout.addWidget(self.show_contacts_button)

        thresh_btn_layout = QGridLayout()
        for i, b in enumerate(self.show_thresh_btns):
            thresh_btn_layout.addWidget(b, i // 2, i % 2)
        layout.addLayout(thresh_btn_layout)

        layout.addStretch(1)
        container.setLayout(layout)

        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    # ---------------- Utility ----------------
    def _sync_thresh_mode_states(self):
        for i in range(self.max_channels_supported):
            mode = self.per_channel_mode[i].currentText()
            if mode == "Automatic":
                self.per_channel_auto[i].setEnabled(True)
                self.per_channel_manual[i].setEnabled(False)
            else:
                self.per_channel_auto[i].setEnabled(False)
                self.per_channel_manual[i].setEnabled(True)

    def slider_changed(self, value):
        self.threshold = value
        self.ct_label.setText(f"Contact Threshold (pixels): {self.threshold}")
        self.ct_text.setText(str(self.threshold))

    def text_input_changed(self):
        try:
            value = int(self.ct_text.text())
        except ValueError:
            return
        value = max(0, min(value, 100))
        self.threshold = value
        self.ct_label.setText(f"Contact Threshold (pixels): {self.threshold}")
        self.ct_slider.setValue(self.threshold)

    def _get_image_layers(self) -> List["napari.layers.Image"]:
        return [
            lyr
            for lyr in self.viewer.layers
            if getattr(lyr, "data", None) is not None
            and lyr.__class__.__name__ == "Image"
        ]

    def _get_active_channel_count(self) -> int:
        try:
            return int(self.channel_mode_combo.currentText())
        except Exception:
            return 2

    def _get_mapped_base_layer(self) -> Optional["napari.layers.Image"]:
        layers = self._get_image_layers()
        if not layers:
            return None
        idx = (
            self.channel_layer_indices[0] if self.channel_layer_indices else 0
        )
        idx = int(np.clip(idx, 0, len(layers) - 1))
        return layers[idx]

    def _on_analysis_source_changed(self):
        self._refresh_channel_labels()
        self._update_z_range_controls(force_full_reset=False)
        self._auto_set_scale_bar_unit_from_layer()

    def _update_z_range_controls(self, force_full_reset: bool = False):
        base_layer = self._get_mapped_base_layer()
        if base_layer is None:
            self.z_min_spinbox.setMaximum(0)
            self.z_max_spinbox.setMaximum(0)
            self.z_min_spinbox.setValue(0)
            self.z_max_spinbox.setValue(0)
            self._last_z_source_signature = None
            return

        data = np.asarray(base_layer.data)
        while data.ndim > 3:
            data = data[0]

        if data.ndim < 3:
            self.z_min_spinbox.setMaximum(0)
            self.z_max_spinbox.setMaximum(0)
            self.z_min_spinbox.setValue(0)
            self.z_max_spinbox.setValue(0)
            source_signature = (id(base_layer), tuple(data.shape))
            self._last_z_source_signature = source_signature
            return

        n_z = int(data.shape[0])
        max_z = max(0, n_z - 1)

        self.z_min_spinbox.setMaximum(max_z)
        self.z_max_spinbox.setMaximum(max_z)

        source_signature = (id(base_layer), tuple(data.shape))

        should_full_reset = force_full_reset or (
            self.auto_adjust_z_range_checkbox.isChecked()
            and source_signature != self._last_z_source_signature
        )

        if should_full_reset:
            self.z_min_spinbox.setValue(0)
            self.z_max_spinbox.setValue(max_z)
        else:
            current_min = min(self.z_min_spinbox.value(), max_z)
            current_max = min(self.z_max_spinbox.value(), max_z)
            if current_min > current_max:
                current_min = current_max
            self.z_min_spinbox.setValue(current_min)
            self.z_max_spinbox.setValue(current_max)

        self._last_z_source_signature = source_signature

    def _auto_set_scale_bar_unit_from_layer(self):
        base_layer = self._get_mapped_base_layer()
        default_unit = "px"

        try:
            if base_layer is not None and hasattr(base_layer, "units"):
                units = getattr(base_layer, "units", None)
                if units:
                    spatial_units = [
                        str(u).strip().lower() for u in units if u is not None
                    ]
                    spatial_units = [
                        u for u in spatial_units if u and u != "none"
                    ]

                    micrometer_tokens = {
                        "µm",
                        "um",
                        "micrometer",
                        "micrometers",
                        "micrometre",
                        "micrometres",
                        "micron",
                        "microns",
                    }

                    if any(u in micrometer_tokens for u in spatial_units):
                        default_unit = "µm"
        except Exception:
            pass

        self.saved_scale_bar_unit = default_unit
        self._update_scale_bar_status_label()

    def _update_scale_bar_status_label(self):
        if self.include_scale_bar_in_saved_image:
            txt = (
                f"Scale bar: ON, {self.saved_scale_bar_length:g} "
                f"{self.saved_scale_bar_unit}, text {self.saved_scale_bar_text_size}"
            )
        else:
            txt = "Scale bar: OFF"
        self.scale_bar_status_label.setText(txt)

    def _on_restrict_signal_z_changed(self, state):
        self.restrict_to_signal_z = bool(state)
        self.restrict_signal_z_button.setEnabled(bool(state))

    def _get_default_restrict_signal_z_channels(
        self, n_channels: int
    ) -> List[int]:
        valid = [
            int(c)
            for c in self.restrict_signal_z_channels
            if 0 <= int(c) < n_channels
        ]
        if valid:
            return valid
        return list(range(n_channels))

    def _compute_signal_restricted_z_indices(
        self, masks: List[np.ndarray], selected_channels: List[int]
    ) -> Optional[np.ndarray]:
        if not masks:
            return None
        if masks[0].ndim < 3:
            return None

        valid_channels = [
            int(c) for c in selected_channels if 0 <= int(c) < len(masks)
        ]
        if not valid_channels:
            valid_channels = list(range(len(masks)))

        z_keep = None
        reduce_axes = tuple(range(1, masks[0].ndim))

        for ch in valid_channels:
            has_signal = np.any(masks[ch], axis=reduce_axes)
            z_keep = has_signal if z_keep is None else (z_keep | has_signal)

        if z_keep is None:
            return None

        keep_idx = np.where(z_keep)[0]
        return keep_idx if keep_idx.size > 0 else None

    def _get_display_transform_for_current_analysis(
        self,
        base_layer: "napari.layers.Image",
        keep_idx: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Any]:
        scale = tuple(base_layer.scale)
        translate = list(base_layer.translate)

        base_data = np.asarray(base_layer.data)
        while base_data.ndim > 3:
            base_data = base_data[0]

        if base_data.ndim >= 3:
            z_min = int(self.z_min_spinbox.value())
            z_offset = z_min

            if keep_idx is not None and len(keep_idx) > 0:
                z_offset += int(keep_idx[0])

            if len(translate) >= 1 and len(scale) >= 1:
                translate[0] = translate[0] + z_offset * scale[0]

        return scale, tuple(translate)

    def _apply_label_qol(self, label_widget: QLabel, text: str):
        label_widget.setWordWrap(True)
        label_widget.setMaximumWidth(260)
        label_widget.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Preferred
        )
        if text and len(text) > 24:
            label_widget.setStyleSheet("font-size: 10px;")
        else:
            label_widget.setStyleSheet("")

    def get_channel_labels(self) -> List[str]:
        n = self._get_active_channel_count()
        layers = self._get_image_layers()
        labels = []
        use_names = self.use_layer_names_checkbox.isChecked()
        for ch in range(n):
            idx = (
                self.channel_layer_indices[ch]
                if ch < len(self.channel_layer_indices)
                else ch
            )
            idx = (
                int(np.clip(idx, 0, max(0, len(layers) - 1))) if layers else 0
            )
            if use_names and layers:
                labels.append(layers[idx].name)
            else:
                labels.append(f"Channel {ch+1}")
        return labels

    def _refresh_channel_labels(self):
        labels = self.get_channel_labels()
        for i in range(self.max_channels_supported):
            if i < len(labels):
                txt = f"{labels[i]}:"
                self.per_channel_label_widgets[i].setText(txt)
                self._apply_label_qol(
                    self.per_channel_label_widgets[i], labels[i]
                )
            else:
                txt = f"Channel {i+1}:"
                self.per_channel_label_widgets[i].setText(txt)
                self._apply_label_qol(self.per_channel_label_widgets[i], txt)

    def _get_signals_for_analysis(
        self,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], "napari.layers.Image"]:
        n = self._get_active_channel_count()
        layers = self._get_image_layers()
        if len(layers) < n:
            raise RuntimeError(
                f"Need at least {n} image layers to run {n}-channel analysis."
            )

        mapped_layers = []
        for ch in range(n):
            idx = (
                self.channel_layer_indices[ch]
                if ch < len(self.channel_layer_indices)
                else ch
            )
            idx = int(np.clip(idx, 0, len(layers) - 1))
            mapped_layers.append(layers[idx])

        base_layer = mapped_layers[0]
        raw = [np.asarray(lyr.data) for lyr in mapped_layers]

        def slice_to_current(sig: np.ndarray) -> np.ndarray:
            while sig.ndim > 3:
                sig = sig[0]
            return sig

        raw = [slice_to_current(s) for s in raw]

        current_shape = raw[0].shape
        if len(current_shape) >= 3:
            self._update_z_range_controls(force_full_reset=False)
            z_min = self.z_min_spinbox.value()
            z_max = self.z_max_spinbox.value()
            if z_min > z_max:
                z_min = z_max
                self.z_min_spinbox.setValue(z_min)
            raw = [s[z_min : z_max + 1, ...] for s in raw]
        else:
            self.z_min_spinbox.setMaximum(0)
            self.z_max_spinbox.setMaximum(0)
            self.z_min_spinbox.setValue(0)
            self.z_max_spinbox.setValue(0)

        norm = []
        for s in raw:
            s2 = s.astype(float, copy=False)
            if np.nanmax(s2) > 1.0 and np.nanmax(s2) > np.nanmin(s2):
                s2 = (s2 - np.nanmin(s2)) / (np.nanmax(s2) - np.nanmin(s2))
            norm.append(s2)

        return raw, norm, base_layer

    def _sanitize_excel_sheet_name(self, name: str) -> str:
        name = str(name) if name is not None else "Sheet"
        name = re.sub(r"[:\\/?*\[\]]", "_", name)
        name = name.strip("'")
        name = name.strip()
        if not name:
            name = "Sheet"
        return name[:31]

    def _make_unique_sheet_name(self, base_name: str, used_names: set) -> str:
        candidate = self._sanitize_excel_sheet_name(base_name)
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate

        i = 2
        while True:
            suffix = f"_{i}"
            trimmed = candidate[: max(0, 31 - len(suffix))]
            new_name = trimmed + suffix
            if new_name not in used_names:
                used_names.add(new_name)
                return new_name
            i += 1

    def _write_graphpad_workbook_from_dataframe(
        self, df_all: pd.DataFrame, file_path: str
    ):
        if df_all.empty:
            raise ValueError("No metrics available to export.")

        if "Analysis Name" not in df_all.columns:
            df_all["Analysis Name"] = "Analysis"

        metric_cols = [
            c
            for c in df_all.columns
            if c not in ["Analysis Name", "ROI Number"]
            and pd.api.types.is_numeric_dtype(df_all[c])
        ]

        if not metric_cols:
            raise ValueError("No numeric metric columns were found to export.")

        used_sheet_names = set()
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            analysis_labels = pd.unique(df_all["Analysis Name"])

            for metric in metric_cols:
                prism_dict = {}
                for label in analysis_labels:
                    vals = (
                        df_all.loc[df_all["Analysis Name"] == label, metric]
                        .dropna()
                        .tolist()
                    )
                    prism_dict[str(label)] = pd.Series(vals)

                df_metric = pd.DataFrame(prism_dict)
                sheet_name = self._make_unique_sheet_name(
                    metric, used_sheet_names
                )
                df_metric.to_excel(writer, sheet_name=sheet_name, index=False)

            df_all.to_excel(
                writer,
                sheet_name=self._make_unique_sheet_name(
                    "All_Metrics", used_sheet_names
                ),
                index=False,
            )

    # ---------------- Popups ----------------
    def open_output_selection(self):
        labels = self.get_channel_labels()
        dlg = OutputSelectionDialog(
            self,
            current_selection=self.output_selection,
            enable_intensity_comparisons=self.enable_intensity_comparisons,
            n_channels=self._get_active_channel_count(),
            channel_labels=labels,
            comparisons=self.intensity_comparisons,
        )
        if dlg.exec_() == QDialog.Accepted:
            sel, enable_comp, comps = dlg.get_results()
            self.output_selection = sel
            self.enable_intensity_comparisons = enable_comp
            self.intensity_comparisons = comps

    def open_channel_numbering(self):
        layers = self._get_image_layers()
        n = self._get_active_channel_count()

        if not layers:
            QMessageBox.information(
                self, "Channel Numbering", "No image layers found."
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Channel Numbering")
        layout = QVBoxLayout()

        info = QLabel(
            "Select which image layers correspond to Channel 1..N for analysis."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        combos: List[QComboBox] = []
        for ch in range(n):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"Channel {ch+1}:"))
            cb = QComboBox()
            cb.addItems([f"{i}: {lyr.name}" for i, lyr in enumerate(layers)])
            default_idx = int(
                np.clip(self.channel_layer_indices[ch], 0, len(layers) - 1)
            )
            cb.setCurrentIndex(default_idx)
            combos.append(cb)
            row.addWidget(cb)
            layout.addLayout(row)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        layout.addWidget(bb)

        dlg.setLayout(layout)
        if dlg.exec_() == QDialog.Accepted:
            for ch in range(n):
                self.channel_layer_indices[ch] = int(combos[ch].currentIndex())
            self._refresh_channel_labels()
            self._update_z_range_controls(force_full_reset=False)
            self._auto_set_scale_bar_unit_from_layer()

    def open_metric_display_selection(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Metric Display Selection")
        layout = QVBoxLayout()

        info = QLabel(
            "Choose how many metrics to display in the widget.\n"
            "- Default: Intersection, Union, Intersection/Union.\n"
            "- Full: display everything computed for the current ROI / full image.\n"
            "This only affects on-screen display, not what's saved."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        cb_full = QCheckBox(
            "Display full metrics (instead of default minimal set)"
        )
        cb_full.setChecked(self._metric_display_keys is not None)
        layout.addWidget(cb_full)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        layout.addWidget(bb)

        dlg.setLayout(layout)
        if dlg.exec_() == QDialog.Accepted:
            if cb_full.isChecked():
                self._metric_display_keys = []
            else:
                self._metric_display_keys = None
            if isinstance(self.last_metrics, list) and self.last_metrics:
                self.update_roi_display()
            elif isinstance(self.last_metrics, dict) and self.last_metrics:
                self._set_result_text_from_metrics(
                    self.last_metrics, prefix="Metrics:\n"
                )

    def open_scale_bar_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Scale Bar Settings")
        layout = QVBoxLayout()

        include_cb = QCheckBox("Include scale bar in saved image")
        include_cb.setChecked(self.include_scale_bar_in_saved_image)
        layout.addWidget(include_cb)

        form = QGridLayout()

        length_label = QLabel("Length:")
        length_spin = QDoubleSpinBox()
        length_spin.setRange(0.01, 1000000.0)
        length_spin.setDecimals(2)
        length_spin.setSingleStep(1.0)
        length_spin.setValue(float(self.saved_scale_bar_length))

        unit_label = QLabel("Unit:")
        unit_combo = QComboBox()
        unit_combo.addItems(["px", "µm"])
        unit_combo.setCurrentText(self.saved_scale_bar_unit)

        text_label = QLabel("Text Size:")
        text_spin = QSpinBox()
        text_spin.setRange(1, 200)
        text_spin.setValue(int(self.saved_scale_bar_text_size))

        enabled = include_cb.isChecked()
        length_label.setEnabled(enabled)
        length_spin.setEnabled(enabled)
        unit_label.setEnabled(enabled)
        unit_combo.setEnabled(enabled)
        text_label.setEnabled(enabled)
        text_spin.setEnabled(enabled)

        include_cb.stateChanged.connect(
            lambda state: (
                length_label.setEnabled(bool(state)),
                length_spin.setEnabled(bool(state)),
                unit_label.setEnabled(bool(state)),
                unit_combo.setEnabled(bool(state)),
                text_label.setEnabled(bool(state)),
                text_spin.setEnabled(bool(state)),
            )
        )

        form.addWidget(length_label, 0, 0)
        form.addWidget(length_spin, 0, 1)
        form.addWidget(unit_label, 1, 0)
        form.addWidget(unit_combo, 1, 1)
        form.addWidget(text_label, 2, 0)
        form.addWidget(text_spin, 2, 1)

        layout.addLayout(form)

        info = QLabel(
            "These settings affect only saved images, not the live viewer."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        layout.addWidget(bb)

        dlg.setLayout(layout)

        if dlg.exec_() == QDialog.Accepted:
            self.include_scale_bar_in_saved_image = include_cb.isChecked()
            self.saved_scale_bar_length = float(length_spin.value())
            self.saved_scale_bar_unit = str(unit_combo.currentText())
            self.saved_scale_bar_text_size = int(text_spin.value())
            self._update_scale_bar_status_label()

    def open_signal_z_channel_selection(self):
        n = self._get_active_channel_count()
        labels = self.get_channel_labels()
        current = set(self._get_default_restrict_signal_z_channels(n))

        dlg = QDialog(self)
        dlg.setWindowTitle("Signal Z Channels")
        layout = QVBoxLayout()

        info = QLabel(
            "Choose which analyzed channels determine which Z-stacks are kept.\n"
            "Only Z-slices containing thresholded signal in at least one selected channel will be analyzed.\n"
            "Default behavior is all analyzed channels."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        checkboxes: List[QCheckBox] = []
        for i in range(n):
            cb = QCheckBox(f"{i+1}: {labels[i]}")
            cb.setChecked(i in current)
            checkboxes.append(cb)
            layout.addWidget(cb)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        layout.addWidget(bb)

        dlg.setLayout(layout)

        if dlg.exec_() == QDialog.Accepted:
            selected = [i for i, cb in enumerate(checkboxes) if cb.isChecked()]
            self.restrict_signal_z_channels = (
                selected if selected else list(range(n))
            )

    # ---------------- Analysis ----------------
    def analyze_contacts(self):
        try:
            raw_signals, norm_signals, base_layer = (
                self._get_signals_for_analysis()
            )
        except Exception as e:
            QMessageBox.warning(self, "Analyze", str(e))
            return

        self._last_base_layer = base_layer

        n = self._get_active_channel_count()
        self._refresh_channel_labels()
        ch_labels = self.get_channel_labels()

        masks: List[np.ndarray] = []
        dists: List[np.ndarray] = []
        keep_idx = None

        for i in range(n):
            sig = norm_signals[i]
            mode = self.per_channel_mode[i].currentText()
            if mode == "Automatic":
                method = self.per_channel_auto[i].currentText()
                try:
                    thresh_val = AUTO_METHODS[method](sig)
                except Exception as e:
                    print(
                        f"Warning: Auto threshold '{method}' failed: {e}. Using mean fallback."
                    )
                    thresh_val = float(np.mean(sig))
            else:
                thresh_val = float(self.per_channel_manual[i].value())

            m = sig > thresh_val
            masks.append(m)
            dists.append(distance_transform_edt(~m))

        if self.restrict_to_signal_z and masks and masks[0].ndim >= 3:
            selected_channels = self._get_default_restrict_signal_z_channels(n)
            keep_idx = self._compute_signal_restricted_z_indices(
                masks, selected_channels
            )

            if keep_idx is not None:
                raw_signals = [s[keep_idx, ...] for s in raw_signals]
                norm_signals = [s[keep_idx, ...] for s in norm_signals]
                masks = [m[keep_idx, ...] for m in masks]
                dists = [d[keep_idx, ...] for d in dists]
            else:
                QMessageBox.information(
                    self,
                    "Restrict Z by Signal",
                    "No thresholded signal was found in the selected channels within the current Z range.\n"
                    "Using the full current Z range instead.",
                )
                keep_idx = None

        self._last_display_scale, self._last_display_z_translate = (
            self._get_display_transform_for_current_analysis(
                base_layer, keep_idx=keep_idx
            )
        )

        self.last_masks = masks

        union_signal = masks[0].copy()
        for m in masks[1:]:
            union_signal |= m

        contacts = np.zeros_like(masks[0], dtype=bool)
        for i in range(n):
            others = None
            for j in range(n):
                if j == i:
                    continue
                if others is None:
                    others = masks[j].copy()
                else:
                    others &= masks[j]
            if others is None:
                others = masks[i].copy()
            contacts |= (dists[i] <= self.threshold) & others

        roi_layer = None
        for layer in self.viewer.layers:
            if layer.name == "ROI":
                roi_layer = layer
                break

        ref_data = np.asarray(base_layer.data)
        while ref_data.ndim > 3:
            ref_data = ref_data[0]
        ref_xy = ref_data.shape[1:] if ref_data.ndim == 3 else ref_data.shape

        if (
            roi_layer is not None
            and roi_layer.data
            and self.per_shape_checkbox.isChecked()
        ):
            roi_polys = list(roi_layer.data)
            per_roi_metrics: List[Dict[str, Any]] = []
            union_contacts = np.zeros_like(masks[0], dtype=bool)

            for idx, poly in enumerate(roi_polys):
                poly_data = np.array(poly, dtype=float)

                # No world<->data conversion needed here: the ROI layer
                # is created (in toggle_roi_selection) with scale/translate
                # copied directly from base_layer's trailing (Y, X) axes,
                # so roi_layer.data is already expressed in base_layer's
                # own pixel-index coordinates. This used to be necessary
                # because the ROI layer defaulted to scale=1, making its
                # stored data equal to world coordinates -- that's no
                # longer the case, and re-applying this conversion now
                # would silently shift ROI masks on any image with
                # non-unit scale.

                if poly_data.shape[1] > 2:
                    poly_data[:, 1] = np.clip(
                        poly_data[:, 1], 0, ref_xy[0] - 1
                    )
                    poly_data[:, 2] = np.clip(
                        poly_data[:, 2], 0, ref_xy[1] - 1
                    )
                    rows = poly_data[:, 1]
                    cols = poly_data[:, 2]
                else:
                    # 2D roi polygon data is (row, col) = (Y, X), the
                    # same axis order as the trailing two columns in
                    # the >2 branch above (which is (Z, Y, X)) -- not
                    # (X, Y). This branch was effectively dead code
                    # before ROI layers were made 2D-only (see
                    # toggle_roi_selection), since roi_ndim used to
                    # always match the base image's 3 axes; fixed here
                    # to match that convention instead of the
                    # transposed one it previously had, since it's now
                    # the only path ROI masks take.
                    poly_data[:, 0] = np.clip(
                        poly_data[:, 0], 0, ref_xy[0] - 1
                    )
                    poly_data[:, 1] = np.clip(
                        poly_data[:, 1], 0, ref_xy[1] - 1
                    )
                    rows = poly_data[:, 0]
                    cols = poly_data[:, 1]

                rr, cc = polygon(rows, cols, shape=ref_xy)
                roi_mask_2d = np.zeros(ref_xy, dtype=bool)
                roi_mask_2d[rr, cc] = True
                roi_mask = roi_mask_2d
                if masks[0].ndim == 3:
                    roi_mask = np.tile(roi_mask_2d, (masks[0].shape[0], 1, 1))

                roi_area = int(np.sum(roi_mask))

                restricted_masks = [m & roi_mask for m in masks]
                restricted_union = restricted_masks[0].copy()
                for m in restricted_masks[1:]:
                    restricted_union |= m
                restricted_contacts = contacts & roi_mask

                metrics = self._compute_metrics_bundle(
                    raw_signals=raw_signals,
                    masks=restricted_masks,
                    contacts=restricted_contacts,
                    union_signal=restricted_union,
                    ch_labels=ch_labels,
                    roi_poly_data=poly_data,
                    roi_area=roi_area,
                )
                metrics["ROI Number"] = idx
                per_roi_metrics.append(metrics)
                union_contacts |= restricted_contacts

            self.last_metrics = per_roi_metrics
            self.current_roi_index = 0
            self.update_roi_navigation(len(per_roi_metrics))
            self.update_roi_display()
            contacts_display = union_contacts.astype(float)
        else:
            roi_area_full = int(masks[0].size)

            self.last_metrics = self._compute_metrics_bundle(
                raw_signals=raw_signals,
                masks=masks,
                contacts=contacts,
                union_signal=union_signal,
                ch_labels=ch_labels,
                roi_poly_data=None,
                roi_area=roi_area_full,
            )
            self._set_result_text_from_metrics(
                self.last_metrics, prefix="Full image metrics:\n"
            )
            contacts_display = contacts.astype(float)

        self._last_contacts_display = contacts_display

        if self.show_thresh_after_checkbox.isChecked():
            for i in range(n):
                self.show_thresholded_channel(i)

        if self.show_contacts_after_checkbox.isChecked():
            self._update_contacts_layer(contacts_display, base_layer)

    def _region_mask_from_mode(
        self,
        mode: str,
        region_channels: List[int],
        masks: List[np.ndarray],
        contacts: np.ndarray,
    ) -> Optional[np.ndarray]:
        n = len(masks)
        mode = (mode or "").strip()
        if mode == "Contacts":
            return contacts
        if mode not in ("Union", "Intersection"):
            return None
        chans = [c for c in (region_channels or []) if 0 <= int(c) < n]
        if not chans:
            return None
        reg = masks[chans[0]].copy()
        if mode == "Union":
            for j in chans[1:]:
                reg |= masks[j]
        else:
            for j in chans[1:]:
                reg &= masks[j]
        return reg

    def _compute_metrics_bundle(
        self,
        raw_signals: List[np.ndarray],
        masks: List[np.ndarray],
        contacts: np.ndarray,
        union_signal: np.ndarray,
        ch_labels: List[str],
        roi_poly_data: Optional[np.ndarray],
        roi_area: Optional[int],
    ) -> Dict[str, Any]:
        n = len(masks)
        out: Dict[str, Any] = {}

        intersection = masks[0].copy()
        for m in masks[1:]:
            intersection &= m
        union = masks[0].copy()
        for m in masks[1:]:
            union |= m

        if self.output_selection.get("Intersection", True):
            out["Intersection"] = int(np.sum(intersection))
        if self.output_selection.get("Union", True):
            out["Union"] = int(np.sum(union))
        if self.output_selection.get(
            "Intersection/Union (Contact Coefficient)", True
        ):
            u = int(np.sum(union))
            out["Intersection/Union (Contact Coefficient)"] = (
                float(np.sum(intersection) / u) if u > 0 else 0.0
            )
        if self.output_selection.get("Contact Area", True):
            out["Contact Area"] = int(np.sum(contacts))

        if self.output_selection.get("Signal Area", True):
            for i in range(n):
                out[f"Signal Area ({ch_labels[i]})"] = int(np.sum(masks[i]))

        if self.output_selection.get("Intersection/Ch Signal Area", True):
            inter_n = int(np.sum(intersection))
            for i in range(n):
                denom = int(np.sum(masks[i]))
                out[f"Intersection/{ch_labels[i]} Signal Area"] = (
                    float(inter_n / denom) if denom > 0 else 0.0
                )

        if roi_area is not None and self.output_selection.get(
            "ROI Area", True
        ):
            out["ROI Area"] = int(roi_area)

        if roi_area is not None and self.output_selection.get(
            "Signal Area/ROI Area", True
        ):
            ra = float(roi_area)
            for i in range(n):
                sa = float(np.sum(masks[i]))
                out[f"Signal Area/ROI Area ({ch_labels[i]})"] = (
                    (sa / ra) if ra > 0 else float("nan")
                )

        if self.output_selection.get("Mean Intensity", True):
            for i in range(n):
                out[f"Mean Intensity ({ch_labels[i]})"] = safe_mean_intensity(
                    raw_signals[i], masks[i]
                )

        if self.output_selection.get("Contact Mean Intensity", True):
            for i in range(n):
                out[f"Contact Mean Intensity ({ch_labels[i]})"] = (
                    safe_mean_intensity(raw_signals[i], contacts)
                )

        if roi_poly_data is not None and self.output_selection.get(
            "ROI Geometry", True
        ):
            geom = compute_roi_geometry(roi_poly_data)
            if roi_poly_data.shape[1] > 2:
                cols = roi_poly_data[:, 2]
                rows = roi_poly_data[:, 1]
            else:
                # (row, col) = (Y, X), matching the convention used
                # for mask rasterization above.
                rows = roi_poly_data[:, 0]
                cols = roi_poly_data[:, 1]
            pts2d = np.column_stack((cols, rows))
            ellipse_circ, poly_perim, circ_ratio = compute_additional_metrics(
                geom[0], geom[1], pts2d
            )
            out["Max Distance"] = float(geom[0])
            out["Max Perp Distance"] = float(geom[1])
            out["Distance Ratio"] = float(geom[2])
            out["Ellipse Circumference"] = float(ellipse_circ)
            out["Shape Perimeter"] = float(poly_perim)
            out["Circumference/Perimeter Ratio"] = float(circ_ratio)

        if self.output_selection.get("Contact Spatial", True):
            avg_dist, cell_area, density_ratio = compute_contact_density(
                contacts, union_signal
            )
            out["Avg Contact Dist"] = float(avg_dist)
            out["Union Signal Area"] = int(cell_area)
            out["Contact Density Ratio"] = float(density_ratio)

        if self.enable_intensity_comparisons:
            for comp in self.intensity_comparisons or []:
                if not comp.get("enabled", True):
                    continue

                source = int(comp.get("source_ch", 0))
                source = int(np.clip(source, 0, n - 1))

                base_mode = str(comp.get("mode", "Union"))
                base_channels = comp.get("region_channels", [])
                base_mask = self._region_mask_from_mode(
                    base_mode, base_channels, masks, contacts
                )
                if base_mask is None:
                    continue

                subtract_mode = str(comp.get("subtract_mode", "None"))
                subtract_channels = comp.get("subtract_channels", [])
                if subtract_mode and subtract_mode != "None":
                    sub_mask = self._region_mask_from_mode(
                        subtract_mode, subtract_channels, masks, contacts
                    )
                    if sub_mask is not None:
                        region_mask = base_mask & (~sub_mask)
                        sub_desc = (
                            "Contacts"
                            if subtract_mode == "Contacts"
                            else f"{subtract_mode}("
                            + ",".join(
                                ch_labels[j]
                                for j in (subtract_channels or [])
                                if 0 <= j < n
                            )
                            + ")"
                        )
                    else:
                        region_mask = base_mask
                        sub_desc = None
                else:
                    region_mask = base_mask
                    sub_desc = None

                base_desc = (
                    "Contacts"
                    if base_mode == "Contacts"
                    else f"{base_mode}("
                    + ",".join(
                        ch_labels[j]
                        for j in (base_channels or [])
                        if 0 <= j < n
                    )
                    + ")"
                )

                if sub_desc:
                    key = f"Mean Intensity [{ch_labels[source]}] in {base_desc} - {sub_desc}"
                else:
                    key = (
                        f"Mean Intensity [{ch_labels[source]}] in {base_desc}"
                    )

                out[key] = safe_mean_intensity(
                    raw_signals[source], region_mask
                )

        return out

    def _update_contacts_layer(
        self, contacts_display: np.ndarray, base_layer: "napari.layers.Image"
    ):
        base_data = np.asarray(base_layer.data)
        while base_data.ndim > 3:
            base_data = base_data[0]
        base_ndim = base_data.ndim

        cd = contacts_display
        if base_ndim == 3 and cd.ndim == 2:
            cd = np.tile(cd[np.newaxis, ...], (base_data.shape[0], 1, 1))

        layer_scale = (
            self._last_display_scale
            if self._last_display_scale is not None
            else base_layer.scale
        )
        layer_translate = (
            self._last_display_z_translate
            if self._last_display_z_translate is not None
            else base_layer.translate
        )

        if "Contacts" in self.viewer.layers:
            lyr = self.viewer.layers["Contacts"]
            lyr.data = cd
            lyr.scale = layer_scale
            lyr.translate = layer_translate
        else:
            self.viewer.add_image(
                cd,
                name="Contacts",
                colormap="white",
                opacity=0.5,
                scale=layer_scale,
                translate=layer_translate,
            )

    def show_contacts(self):
        if (
            self._last_contacts_display is None
            or self._last_base_layer is None
        ):
            QMessageBox.information(
                self,
                "Show Contacts",
                "No contacts available yet. Please run Analyze first.",
            )
            return
        self._update_contacts_layer(
            self._last_contacts_display, self._last_base_layer
        )
        print("Displayed Contacts layer.")

    # ---------------- ROI navigation/display ----------------
    def update_roi_navigation(self, total_rois: int):
        if total_rois > 0:
            self.roi_nav_label.setText(
                f"ROI: {self.current_roi_index + 1} of {total_rois}"
            )
        else:
            self.roi_nav_label.setText("ROI: N/A")
        self.prev_roi_button.setEnabled(total_rois > 1)
        self.next_roi_button.setEnabled(total_rois > 1)

    def _set_result_text_from_metrics(
        self, metrics: Dict[str, Any], prefix: str = ""
    ):
        if self._metric_display_keys is None:
            keys = [
                "Intersection",
                "Union",
                "Intersection/Union (Contact Coefficient)",
            ]
        elif self._metric_display_keys == []:
            keys = list(metrics.keys())
        else:
            keys = self._metric_display_keys

        lines = []
        for k in keys:
            if k in metrics:
                lines.append(f"{k}: {metrics[k]}")
        if not lines:
            for k, v in metrics.items():
                lines.append(f"{k}: {v}")

        self.result_label.setText(prefix + "\n".join(lines))

    def update_roi_display(self):
        if not isinstance(self.last_metrics, list) or not self.last_metrics:
            return
        current_metrics = self.last_metrics[self.current_roi_index]
        self._set_result_text_from_metrics(
            current_metrics, prefix="Metrics for current ROI:\n"
        )

        for layer in self.viewer.layers:
            if layer.name == "ROI":
                total = len(layer.data)
                new_colors = [
                    "yellow" if i == self.current_roi_index else "red"
                    for i in range(total)
                ]
                layer.edge_color = new_colors
                break

    def next_roi(self):
        if isinstance(self.last_metrics, list) and self.last_metrics:
            self.current_roi_index = (self.current_roi_index + 1) % len(
                self.last_metrics
            )
            self.update_roi_navigation(len(self.last_metrics))
            self.update_roi_display()

    def prev_roi(self):
        if isinstance(self.last_metrics, list) and self.last_metrics:
            self.current_roi_index = (self.current_roi_index - 1) % len(
                self.last_metrics
            )
            self.update_roi_navigation(len(self.last_metrics))
            self.update_roi_display()

    # ---------------- Saving / analyses management ----------------
    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)",
        )
        if not file_path:
            return

        scale_bar = getattr(self.viewer, "scale_bar", None)

        old_visible = None
        old_length = None
        old_font_size = None
        old_ticks = None
        old_unit = None

        try:
            if scale_bar is not None:
                old_visible = getattr(scale_bar, "visible", None)
                old_length = getattr(scale_bar, "length", None)
                old_font_size = getattr(scale_bar, "font_size", None)
                old_ticks = getattr(scale_bar, "ticks", None)
                old_unit = getattr(scale_bar, "unit", None)

                if self.include_scale_bar_in_saved_image:
                    if hasattr(scale_bar, "visible"):
                        scale_bar.visible = True
                    if hasattr(scale_bar, "length"):
                        scale_bar.length = float(self.saved_scale_bar_length)
                    if hasattr(scale_bar, "font_size"):
                        scale_bar.font_size = int(
                            self.saved_scale_bar_text_size
                        )
                    if hasattr(scale_bar, "ticks"):
                        scale_bar.ticks = False
                    if hasattr(scale_bar, "unit"):
                        unit_text = str(self.saved_scale_bar_unit).strip()
                        scale_bar.unit = unit_text if unit_text else None
                else:
                    if hasattr(scale_bar, "visible"):
                        scale_bar.visible = False

            screenshot = self.viewer.screenshot()
            imageio.imwrite(file_path, screenshot)
            print(f"Image saved to {file_path}")

        except Exception as e:
            print(f"Error saving image: {e}")

        finally:
            try:
                if scale_bar is not None:
                    if old_visible is not None and hasattr(
                        scale_bar, "visible"
                    ):
                        scale_bar.visible = old_visible
                    if old_length is not None and hasattr(scale_bar, "length"):
                        scale_bar.length = old_length
                    if old_font_size is not None and hasattr(
                        scale_bar, "font_size"
                    ):
                        scale_bar.font_size = old_font_size
                    if old_ticks is not None and hasattr(scale_bar, "ticks"):
                        scale_bar.ticks = old_ticks
                    if hasattr(scale_bar, "unit"):
                        scale_bar.unit = old_unit
            except Exception:
                pass

    def _collect_export_rows(self) -> List[Dict[str, Any]]:
        if self.metrics_list:
            return self.metrics_list
        if self.last_metrics:
            if isinstance(self.last_metrics, list):
                return self.last_metrics
            return [self.last_metrics]
        return []

    def save_metrics(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export to Excel",
            "Excel Export.xlsx",
            "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith((".xlsx", ".csv")):
            file_path = file_path + ".xlsx"

        rows = self._collect_export_rows()
        if not rows:
            print("No metrics available to save.")
            return

        df = pd.DataFrame(rows)
        if "ROI_index" in df.columns and "ROI Number" not in df.columns:
            df.rename(columns={"ROI_index": "ROI Number"}, inplace=True)

        if "Analysis Name" in df.columns:
            cols = df.columns.tolist()
            cols.insert(0, cols.pop(cols.index("Analysis Name")))
            df = df[cols]

        try:
            if file_path.lower().endswith(".csv"):
                df.to_csv(file_path, index=False)
            else:
                df.to_excel(file_path, index=False, engine="openpyxl")
            print(f"Metrics saved to {file_path}")
        except Exception as e:
            print(f"Error saving metrics: {e}")

    def export_graphpad_prism(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export for GraphPad Prism",
            "GraphPad Prism Export.xlsx",
            "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith((".xlsx", ".csv")):
            file_path = file_path + ".xlsx"

        rows = self._collect_export_rows()
        if not rows:
            QMessageBox.information(
                self, "GraphPad Export", "No metrics available to export."
            )
            return

        df_all = pd.DataFrame(rows)

        if df_all.empty:
            QMessageBox.information(
                self, "GraphPad Export", "No metrics available to export."
            )
            return

        try:
            if file_path.lower().endswith(".csv"):
                if "Analysis Name" not in df_all.columns:
                    df_all["Analysis Name"] = "Analysis"

                metric_cols = [
                    c
                    for c in df_all.columns
                    if c not in ["Analysis Name", "ROI Number"]
                    and pd.api.types.is_numeric_dtype(df_all[c])
                ]

                if not metric_cols:
                    QMessageBox.information(
                        self,
                        "GraphPad Export",
                        "No numeric metric columns were found to export.",
                    )
                    return

                first_metric = metric_cols[0]
                analysis_labels = pd.unique(df_all["Analysis Name"])
                prism_dict = {}
                for label in analysis_labels:
                    vals = (
                        df_all.loc[
                            df_all["Analysis Name"] == label, first_metric
                        ]
                        .dropna()
                        .tolist()
                    )
                    prism_dict[str(label)] = pd.Series(vals)
                df_first_metric = pd.DataFrame(prism_dict)
                df_first_metric.to_csv(file_path, index=False)

                QMessageBox.information(
                    self,
                    "GraphPad Export",
                    "CSV supports only one table, so only the first metric was exported.\n\n"
                    f"Exported metric: {first_metric}\n"
                    "Use .xlsx to export one sheet per metric.",
                )
                return

            self._write_graphpad_workbook_from_dataframe(df_all, file_path)

            QMessageBox.information(
                self,
                "GraphPad Export",
                "Export complete.\n\n"
                "Workbook format:\n"
                "- One sheet per numeric metric\n"
                "- Each column is one Analysis Name\n"
                "- Rows contain the measurements for that metric under that label\n"
                "- An 'All_Metrics' sheet is also included",
            )
        except Exception as e:
            QMessageBox.warning(
                self, "GraphPad Export", f"Failed to export:\n{e}"
            )

    def append_to_spreadsheet(self):
        in_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Excel File to Append",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)",
        )
        if not in_path:
            return

        rows = self._collect_export_rows()
        if not rows:
            QMessageBox.information(
                self, "Append", "No metrics available to append."
            )
            return

        new_df = pd.DataFrame(rows)

        try:
            if in_path.lower().endswith(".csv"):
                try:
                    old_df = pd.read_csv(in_path)
                    out_df = pd.concat([old_df, new_df], ignore_index=True)
                except Exception:
                    out_df = new_df
                out_df.to_csv(in_path, index=False)
            elif in_path.lower().endswith(".xlsx"):
                try:
                    old_df = pd.read_excel(in_path)
                    out_df = pd.concat([old_df, new_df], ignore_index=True)
                except Exception:
                    out_df = new_df
                out_df.to_excel(in_path, index=False, engine="openpyxl")
            else:
                QMessageBox.warning(
                    self, "Append", "Unsupported file type. Use CSV or XLSX."
                )
                return

            QMessageBox.information(
                self, "Append", f"Appended {len(new_df)} row(s) to:\n{in_path}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Append", f"Failed to append:\n{e}")

    def append_to_graphpad_prism(self):
        in_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GraphPad Workbook to Append",
            "",
            "Excel Files (*.xlsx);;All Files (*)",
        )
        if not in_path:
            return

        rows = self._collect_export_rows()
        if not rows:
            QMessageBox.information(
                self, "Append to GraphPad", "No metrics available to append."
            )
            return

        new_df = pd.DataFrame(rows)

        try:
            try:
                old_df = pd.read_excel(in_path, sheet_name="All_Metrics")
            except Exception:
                QMessageBox.warning(
                    self,
                    "Append to GraphPad",
                    "The selected workbook does not appear to be a GraphPad-formatted export with an 'All_Metrics' sheet.",
                )
                return

            out_df = pd.concat([old_df, new_df], ignore_index=True)
            self._write_graphpad_workbook_from_dataframe(out_df, in_path)

            QMessageBox.information(
                self,
                "Append to GraphPad",
                f"Appended {len(new_df)} row(s) to GraphPad-formatted workbook:\n{in_path}",
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Append to GraphPad", f"Failed to append:\n{e}"
            )

    def add_analysis(self):
        if not self.last_metrics:
            print("No analysis to add. Please run an analysis first.")
            return

        if isinstance(self.last_metrics, list):
            base_name = self.analysis_name_edit.text().strip() or "Analysis"
            if self.sequential_label_checkbox.isChecked():
                for idx, metrics in enumerate(self.last_metrics):
                    m = copy.deepcopy(metrics)
                    m["Analysis Name"] = f"{base_name} {idx+1}"
                    self.metrics_list.append(m)
            else:
                for metrics in self.last_metrics:
                    m = copy.deepcopy(metrics)
                    m["Analysis Name"] = base_name
                    self.metrics_list.append(m)
        else:
            name = self.analysis_name_edit.text().strip() or "Analysis"
            m = copy.deepcopy(self.last_metrics)
            m["Analysis Name"] = name
            self.metrics_list.append(m)

        self.analysis_count_label.setText(
            f"Analyses Stored: {len(self.metrics_list)}"
        )
        print("Analysis added. Total analyses stored:", len(self.metrics_list))

    def clear_last_analysis(self):
        if self.metrics_list:
            self.metrics_list.pop()
            self.analysis_count_label.setText(
                f"Analyses Stored: {len(self.metrics_list)}"
            )
            print(
                "Last analysis cleared. Total analyses stored:",
                len(self.metrics_list),
            )
        else:
            print("No analyses to clear.")

    def clear_all_analyses(self):
        self.metrics_list = []
        self.analysis_count_label.setText("Analyses Stored: 0")
        print("All analyses cleared.")

    # ---------------- Threshold display ----------------
    def show_thresholded_channel(self, ch_index: int):
        if not hasattr(self, "last_masks") or not self.last_masks:
            print(
                "No thresholded data available. Please run an analysis first."
            )
            return
        n = self._get_active_channel_count()
        if ch_index >= n:
            print(f"Channel {ch_index+1} is not active for current analysis.")
            return

        mask = self.last_masks[ch_index].astype(float)
        layer_name = (
            f"Thresholded ({self.get_channel_labels()[ch_index]})"
            if self.use_layer_names_checkbox.isChecked()
            else f"Thresholded Ch {ch_index+1}"
        )

        layers = self._get_image_layers()
        base_layer = (
            layers[self.channel_layer_indices[0]]
            if layers
            else self.viewer.layers[0]
        )

        layer_scale = (
            self._last_display_scale
            if self._last_display_scale is not None
            else base_layer.scale
        )
        layer_translate = (
            self._last_display_z_translate
            if self._last_display_z_translate is not None
            else base_layer.translate
        )

        if layer_name in self.viewer.layers:
            lyr = self.viewer.layers[layer_name]
            lyr.data = mask
            lyr.scale = layer_scale
            lyr.translate = layer_translate
        else:
            self.viewer.add_image(
                mask,
                name=layer_name,
                colormap="gray",
                opacity=0.8,
                scale=layer_scale,
                translate=layer_translate,
            )
        print(f"Displayed thresholded image for channel {ch_index + 1}.")

    # ---------------- ROI layer ----------------
    def toggle_roi_selection(self):
        roi_layer = None
        for layer in self.viewer.layers:
            if layer.name == "ROI":
                roi_layer = layer
                break

        if roi_layer is None:
            layers = self._get_image_layers()
            base_layer = (
                layers[self.channel_layer_indices[0]]
                if layers
                else self.viewer.layers[0]
            )
            # The ROI layer is deliberately created 2D (Y, X only), not
            # matching the base image's full ndim. napari shows a
            # lower-dimensional Shapes layer on *every* slice of a
            # higher-dimensional viewer, because a shape with no data
            # along an axis has nothing to slice-filter against --
            # see ShapeList._visible_shapes: when the layer's own
            # slice_key is empty, every shape is considered visible
            # regardless of which Z-slice you're on. That's what makes
            # ROIs drawn on one slice stay visible and selectable on
            # every other slice too.
            #
            # This is safe for the analysis below: compute_roi_geometry()
            # and the per-ROI mask builder already have working 2D
            # branches (poly_data.shape[1] == 2) alongside their 3D
            # ones, and the mask builder already tiles a 2D ROI mask
            # across every Z-plane for 3D images (np.tile(roi_mask_2d,
            # (masks[0].shape[0], 1, 1))) -- so a 2D-only ROI was
            # already the shape the analysis expected to apply
            # uniformly across the stack; only the drawing layer itself
            # was unnecessarily 3D.
            #
            # (Still rounds through PolygonBase.data's
            # `np.rint(bounding_box).astype(int)` per-shape slice_key
            # logic that caused the original reselection bug, but with
            # no non-displayed axes left to round, that logic never
            # runs against a mismatched value.)
            roi_ndim = 2
            base_scale = np.asarray(base_layer.scale)[-roi_ndim:]
            base_translate = np.asarray(base_layer.translate)[-roi_ndim:]

            roi_layer = self.viewer.add_shapes(
                name="ROI",
                shape_type="polygon",
                edge_color="red",
                edge_width=5,  # bumped 3->5: still looked thin at 3
                face_color="transparent",
                opacity=0.5,
                ndim=roi_ndim,
                scale=base_scale,
                translate=base_translate,
            )
            roi_layer.data = []
            roi_layer.mode = "select"
            print(
                "ROI layer created (2D, visible on every slice)."
                " Use the layer controls to add ROI(s).",
            )
        else:
            if len(list(roi_layer.data)) == 0:
                roi_layer.mode = "select"
                print(
                    "No ROI drawn yet. Please draw an ROI before toggling mode."
                )
            else:
                if roi_layer.mode == "select":
                    roi_layer.mode = "pan_zoom"
                    print("ROI layer mode set to 'pan_zoom'.")
                else:
                    roi_layer.mode = "select"
                    print("ROI layer mode set to 'select'.")

        if not hasattr(roi_layer, "_patched_get_value"):
            original_get_value = roi_layer.get_value

            def patched_get_value(*args, **kwargs):
                try:
                    return original_get_value(*args, **kwargs)
                except Exception as exc:
                    # napari's Shapes._get_value contract is to return a
                    # (shape_index, vertex_index) tuple. The mouse
                    # binding that calls get_value immediately does
                    # `shape_under_cursor, vertex_under_cursor = value`,
                    # which raises TypeError if value is bare None.
                    # Returning None here (as this used to) meant that
                    # the first time get_value raised anything, the very
                    # next click on the layer would crash silently
                    # inside napari's mouse-event handling -- which is
                    # indistinguishable, from the user's side, from
                    # "I can no longer reselect my ROI." Returning
                    # (None, None) keeps the contract so a single failed
                    # hit-test only misses that one click instead of
                    # looking like selection is broken.
                    warnings.warn(
                        "ROI layer get_value() raised "
                        f"{exc!r}; treating this click as "
                        "'no shape under cursor'. If this keeps "
                        "happening, that's the real bug to chase down.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return (None, None)

            roi_layer.get_value = patched_get_value
            roi_layer._patched_get_value = True
