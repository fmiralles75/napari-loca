"""
LocA (napari-loca) widget.

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
import json
import re
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import napari
import numpy as np
import pandas as pd
from qtpy.QtCore import QEvent, QRect, QSettings, QSize, Qt
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QStyleOptionSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import (
    center_of_mass as ndi_center_of_mass,
)
from scipy.ndimage import (
    convolve as ndi_convolve,
)
from scipy.ndimage import (
    distance_transform_edt,
)
from scipy.ndimage import (
    label as ndi_label,
)
from scipy.ndimage import (
    maximum as ndi_maximum,
)
from scipy.ndimage import (
    sum as ndi_sum,
)
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree
from skimage import filters
from skimage.draw import polygon
from skimage.measure import regionprops
from skimage.morphology import skeletonize

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

# Cross-session persistence (see OrganelleContactWidget._save_settings /
# _load_settings). Stored as one JSON blob under a single QSettings key
# rather than many individual QSettings keys, since QSettings' own
# type handling (particularly for bools) is inconsistent across
# platforms/backends -- JSON gives an exact, portable round-trip
# instead.
# Deliberately still the pre-rename package name: QSettings stores saved
# widget settings under this key, so changing it would silently reset
# every user's saved thresholds and options after the napari-loca rename.
SETTINGS_ORG = "napari-organelle-contact-analyzer"
SETTINGS_APP = "OrganelleContactWidget"
SETTINGS_KEY = "widget_state_json"
SETTINGS_SCHEMA_VERSION = 1


# -----------------------------
# Scroll-safe input widgets
# -----------------------------
# LocA's panel sits in a scroll area. By default Qt hands a mouse-wheel
# event (and keyboard focus) to whichever spinbox, combo box or slider
# happens to be under the pointer, so scrolling the panel silently
# changed settings the pointer passed over. These subclasses act on the
# wheel only once the user has clicked (or tabbed) into the control;
# otherwise they ignore the event, and Qt passes it on to the scroll
# area, which scrolls the panel instead. The spinbox classes also make
# sure every digit of their widest value is visible under napari's
# stylesheet (see _fit_spinbox_hint and _apply_spinbox_min_width).
# Every spinbox, combo box and slider in this module uses these
# classes (test_widget.py checks that).


def _require_click_focus(widget: QWidget) -> None:
    # StrongFocus = click or Tab. Qt's default for these controls is
    # WheelFocus, which lets the wheel itself grant focus and would
    # defeat the hasFocus() check below.
    widget.setFocusPolicy(Qt.StrongFocus)


def _guarded_wheel_event(widget: QWidget, event, base_cls) -> None:
    if widget.hasFocus():
        base_cls.wheelEvent(widget, event)
    else:
        event.ignore()  # propagate to the parent scroll area


# Room the spinbox's inner line edit needs beyond the text itself. Measured
# under napari's stylesheet: 7 px (2 px padding + 1 px margin per side,
# plus the cursor); 3 px extra for safety.
SPINBOX_TEXT_SLACK_PX = 10


def _spinbox_text_width(spin) -> int:
    """Pixel width of the widest text the spinbox can display."""
    fm = spin.fontMetrics()
    texts = [
        spin.prefix() + spin.textFromValue(v) + spin.suffix()
        for v in (spin.minimum(), spin.maximum())
    ]
    if spin.specialValueText():
        texts.append(spin.specialValueText())
    return max(fm.horizontalAdvance(t) for t in texts)


def _spinbox_text_room(spin, width: int, height: int) -> int:
    """Width (px) of the spinbox's text field at the given size, as the
    active style lays it out, excluding any part a button covers."""
    opt = QStyleOptionSpinBox()
    spin.initStyleOption(opt)
    opt.rect = QRect(0, 0, width, height)
    style = spin.style()

    def rect(sc):
        return style.subControlRect(QStyle.CC_SpinBox, opt, sc, spin)

    field = rect(QStyle.SC_SpinBoxEditField)
    left, right = field.left(), field.right()
    for btn in (rect(QStyle.SC_SpinBoxUp), rect(QStyle.SC_SpinBoxDown)):
        if btn.width() <= 0 or not btn.intersects(field):
            continue
        if btn.center().x() < field.center().x():
            left = max(left, btn.right() + 1)
        else:
            right = min(right, btn.left() - 1)
    return right - left + 1


def _fit_spinbox_hint(spin, hint: QSize) -> QSize:
    """Widen a spinbox size hint until its widest value fits.

    Qt's stylesheet engine sizes a spinbox for ONE button column, but
    napari's stylesheet puts - and + on opposite sides, so at its own
    size hint the text field is one button (~20 px) too narrow
    (measured: "10000.0000" and "1000000000.00" each lost 21 px).
    Short values hid this, because napari's ``min-width: 70px`` pads
    their hint. This asks the active style how wide the text field
    really is at the hinted size and adds whatever is missing.
    """
    try:
        room = _spinbox_text_room(spin, hint.width(), hint.height())
        need = _spinbox_text_width(spin) + SPINBOX_TEXT_SLACK_PX
        missing = need - room
    except Exception:  # sizing must never break the panel
        return hint
    if missing > 0:
        return QSize(hint.width() + missing, hint.height())
    return hint


def _spinbox_min_width(spin) -> int:
    """Narrowest width (px) at which the spinbox's widest value still
    shows in full: text + slack + whatever the style spends on padding
    and buttons (that overhead doesn't depend on the width)."""
    try:
        hint = spin.sizeHint()
        chrome = hint.width() - _spinbox_text_room(
            spin, hint.width(), hint.height()
        )
        return _spinbox_text_width(spin) + SPINBOX_TEXT_SLACK_PX + chrome
    except Exception:  # never break the panel; just don't shrink it
        return spin.sizeHint().width()


def _apply_spinbox_min_width(spin) -> None:
    """Set the spinbox's minimum width to what its widest value needs.

    napari's stylesheet gives every spinbox ``min-width: 70px`` (90 px
    with padding), and Qt lets a layout squeeze a widget down to an
    explicit minimum even below its size hint. In a crowded row that
    cut digits off (measured: the manual threshold box got 97 of the
    115 px it needs, hiding 13 px of "0.0825"). Replacing that fixed
    90 px with each box's real requirement still lets small boxes (Z
    range: "16") shrink, but never far enough to hide a digit. The
    stylesheet re-applies its value whenever it re-polishes the widget,
    and the requirement changes with the range, so this re-runs after
    those events and setters.
    """
    need = _spinbox_min_width(spin)
    if spin.minimumWidth() != need:
        spin.setMinimumWidth(need)


_SPIN_REPOLISH_EVENTS = (
    QEvent.Polish,
    QEvent.StyleChange,
    QEvent.FontChange,
    QEvent.EnabledChange,
    QEvent.Show,
)


class _FullValueSpinMixin:
    """Spinbox sizing shared by ScrollSafeSpinBox and
    ScrollSafeDoubleSpinBox: the size hint fits the widest value, and
    no layout can squeeze the box below what that value needs."""

    def sizeHint(self):  # noqa: N802
        return _fit_spinbox_hint(self, super().sizeHint())

    def minimumSizeHint(self):  # noqa: N802
        return _fit_spinbox_hint(self, super().minimumSizeHint())

    def event(self, event):
        handled = super().event(event)
        if event.type() in _SPIN_REPOLISH_EVENTS:
            _apply_spinbox_min_width(self)
        return handled

    # Anything that changes the widest displayable text.
    def setRange(self, *args):  # noqa: N802
        super().setRange(*args)
        _apply_spinbox_min_width(self)

    def setMinimum(self, *args):  # noqa: N802
        super().setMinimum(*args)
        _apply_spinbox_min_width(self)

    def setMaximum(self, *args):  # noqa: N802
        super().setMaximum(*args)
        _apply_spinbox_min_width(self)

    def setDecimals(self, *args):  # noqa: N802 (QDoubleSpinBox only)
        super().setDecimals(*args)
        _apply_spinbox_min_width(self)

    def setPrefix(self, *args):  # noqa: N802
        super().setPrefix(*args)
        _apply_spinbox_min_width(self)

    def setSuffix(self, *args):  # noqa: N802
        super().setSuffix(*args)
        _apply_spinbox_min_width(self)

    def setSpecialValueText(self, *args):  # noqa: N802
        super().setSpecialValueText(*args)
        _apply_spinbox_min_width(self)


class ScrollSafeSpinBox(_FullValueSpinMixin, QSpinBox):
    """QSpinBox: wheel only when focused; every digit always visible."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _require_click_focus(self)

    def wheelEvent(self, event):  # noqa: N802 (Qt naming)
        _guarded_wheel_event(self, event, QSpinBox)


class ScrollSafeDoubleSpinBox(_FullValueSpinMixin, QDoubleSpinBox):
    """QDoubleSpinBox: wheel only when focused; every digit (all
    decimals) always visible."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _require_click_focus(self)

    def wheelEvent(self, event):  # noqa: N802
        _guarded_wheel_event(self, event, QDoubleSpinBox)


class ScrollSafeComboBox(QComboBox):
    """QComboBox that only reacts to the mouse wheel when focused."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _require_click_focus(self)

    def wheelEvent(self, event):  # noqa: N802
        _guarded_wheel_event(self, event, QComboBox)


class ScrollSafeSlider(QSlider):
    """QSlider that only reacts to the mouse wheel when focused."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _require_click_focus(self)

    def wheelEvent(self, event):  # noqa: N802
        _guarded_wheel_event(self, event, QSlider)


# -----------------------------
# Helpers
# -----------------------------
def compute_contact_sites(
    contacts: np.ndarray, z_xy_ratio: float = 1.0
) -> Tuple[int, float, float]:
    """Describe the contact mask as discrete contact *sites*.

    A site is one connected component of ``contacts`` (face
    connectivity, the same rule used for bodies). Returns
    (site_count, mean_site_size, nn_distance):

    * ``mean_site_size`` -- mean pixels/voxels per site (NaN if none);
    * ``nn_distance`` -- mean distance from each site's centroid to the
      nearest other site's centroid, in XY pixels with Z weighted by
      ``z_xy_ratio`` (NaN with fewer than two sites).

    Replaces the old "Avg Contact Dist", which measured the distance
    between neighbouring contact *pixels* and was therefore 1.0 for
    every contiguous contact, however the contacts were arranged."""
    labels, n = ndi_label(contacts)
    if n == 0:
        return 0, float("nan"), float("nan")
    ids = np.arange(1, n + 1)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    mean_size = float(np.mean(sizes))
    if n < 2:
        return int(n), mean_size, float("nan")
    cents = np.asarray(
        ndi_center_of_mass(contacts, labels, ids), dtype=float
    ).reshape(n, contacts.ndim)
    if contacts.ndim == 3:
        cents[:, 0] *= float(z_xy_ratio)
    d, _ = cKDTree(cents).query(cents, k=2)
    return int(n), mean_size, float(np.mean(d[:, 1]))


def body_aspect_ratio(coords: np.ndarray, spacing) -> Optional[float]:
    """Aspect ratio (major / minor axis of the best-fit ellipse or
    ellipsoid) of one body from its pixel/voxel coordinates, measured in
    physical proportions: ``coords`` are scaled by ``spacing`` (per-axis
    voxel size, or the Z/XY ratio for Z) before fitting. Equals
    regionprops' axis_major_length / axis_minor_length when spacing is
    isotropic (both reduce to sqrt(largest / smallest eigenvalue of the
    coordinate covariance)), but -- unlike regionprops without spacing
    -- does not read a round body as elongated just because Z is
    sampled more coarsely than XY. Returns None for bodies with no
    extent along some axis (a single pixel, a 1-px line, a body
    confined to one Z plane), where the ratio is undefined."""
    if coords.shape[0] < 2:
        return None
    c = coords.astype(float) * np.asarray(spacing, dtype=float)
    ev = np.linalg.eigvalsh(np.cov(c.T, bias=True))
    lo, hi = float(ev[0]), float(ev[-1])
    if hi <= 0 or lo <= 1e-12 * hi:
        return None
    return float(np.sqrt(hi / lo))


def skeleton_branch_lengths(
    branch_labels: np.ndarray, n_branches: int, z_xy_ratio: float = 1.0
) -> np.ndarray:
    """Euclidean length of every labelled skeleton branch, in XY pixels
    with Z steps weighted by ``z_xy_ratio``. Returns an array of length
    ``n_branches + 1`` (index 0 unused, = 0).

    Length = the path through the branch's pixel centres (1 per face
    step, sqrt(2) per in-plane diagonal, etc.) plus one pixel for the
    two half-pixel end caps, so a straight N-pixel line still measures
    N. The path is the minimum spanning tree of the branch's pixel
    adjacency graph, which avoids double counting where a thinned
    skeleton turns a corner (three mutually-adjacent pixels). Unlike a
    pixel count, this is the same for a branch in any orientation."""
    out = np.zeros(n_branches + 1, dtype=float)
    if n_branches == 0:
        return out
    from scipy.sparse.csgraph import minimum_spanning_tree

    shape = branch_labels.shape
    ndim = branch_labels.ndim
    lin = np.flatnonzero(branch_labels)  # sorted
    coords = np.column_stack(np.unravel_index(lin, shape))
    lab = branch_labels.ravel()[lin]
    spacing = np.ones(ndim)
    if ndim == 3:
        spacing[0] = float(z_xy_ratio)
    # Half of the neighbourhood: each adjacent pair is visited once.
    offsets = [
        o
        for o in (np.argwhere(np.ones((3,) * ndim)) - 1)
        if tuple(o) > (0,) * ndim
    ]
    rows, cols, wts = [], [], []
    for o in offsets:
        nb = coords + o
        ok = np.all((nb >= 0) & (nb < shape), axis=1)
        if not ok.any():
            continue
        src = np.nonzero(ok)[0]
        nlin = np.ravel_multi_index(tuple(nb[ok].T), shape)
        pos = np.searchsorted(lin, nlin)
        pos = np.minimum(pos, lin.size - 1)
        hit = lin[pos] == nlin
        rows.append(src[hit])
        cols.append(pos[hit])
        wts.append(
            np.full(int(hit.sum()), float(np.sqrt(np.sum((o * spacing) ** 2))))
        )
    if rows:
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        w = np.concatenate(wts)
        g = coo_matrix((w, (r, c)), shape=(lin.size, lin.size)).tocsr()
        mst = minimum_spanning_tree(g).tocoo()
        np.add.at(out, lab[mst.row], mst.data)
    out[1:] += 1.0
    return out


def compute_roi_geometry(poly_data: np.ndarray):
    """Feret (caliper) diameters of an ROI polygon, in XY pixels.

    Returns (max_feret, min_feret, feret_ratio):

    * ``max_feret`` -- the longest distance between any two points of
      the ROI (its largest caliper width);
    * ``min_feret`` -- the narrowest caliper width: the smallest
      distance between two parallel lines that enclose the ROI;
    * ``feret_ratio`` = max / min -- orientation-independent; 1.0 for a
      circle, a/b for an a:b ellipse. Shapes with corners score higher
      than their side ratio because the max Feret runs corner to
      corner: a square is sqrt(2) = 1.41, a 2:1 rectangle sqrt(5) = 2.24.

    These match Fiji/ImageJ's "Feret" and "MinFeret" for polygon ROIs.
    The min Feret is exact: for a convex polygon (the ROI's convex hull)
    the narrowest caliper always lies flush with one hull edge, so it is
    the minimum over hull edges of the farthest hull vertex from that
    edge's line.

    (Replaces "Max Perp Distance", which measured only from the longest
    chord to the farthest point on ONE side -- so a circle or square
    scored a ratio of 2.)"""
    if poly_data.shape[1] > 2:
        points2d = np.column_stack((poly_data[:, 2], poly_data[:, 1]))
    else:
        points2d = poly_data
    points2d = np.asarray(points2d, dtype=float)
    if points2d.shape[0] < 3:
        return np.nan, np.nan, np.nan
    try:
        hull = ConvexHull(points2d)
    except Exception:  # noqa: BLE001 -- collinear/degenerate polygon
        return np.nan, np.nan, np.nan
    hp = points2d[hull.vertices]  # counter-clockwise
    # One hull vertex at a time keeps memory O(n) even for freehand ROIs
    # with thousands of hull vertices.
    max_feret = 0.0
    for p in hp:
        max_feret = max(max_feret, float(np.max(np.hypot(*(hp - p).T))))
    min_feret = np.inf
    for p, q in zip(hp, np.roll(hp, -1, axis=0)):
        edge = q - p
        length = float(np.hypot(*edge))
        if length == 0:
            continue
        normal = np.array([-edge[1], edge[0]]) / length
        # Width of the hull measured perpendicular to this edge.
        min_feret = min(min_feret, float(np.max(np.abs((hp - p) @ normal))))
    min_feret = float(min_feret)
    ratio = max_feret / min_feret if min_feret > 0 else np.nan
    return max_feret, min_feret, ratio


def compute_roi_shape_descriptors(
    points2d: np.ndarray,
) -> Tuple[float, float, float]:
    """(circularity, roundness, solidity) of an ROI polygon, computed
    exactly from its vertices -- the same definitions as Fiji/ImageJ's
    Shape Descriptors:

    * Circularity = 4*pi*Area / Perimeter^2 -- 1.0 for a circle; lower
      for elongated AND for irregular outlines (the usual single
      "roundness" number);
    * Roundness = 4*Area / (pi * MajorAxis^2) = minor / major axis of
      the best-fit ellipse (the ellipse with the polygon's area and
      second moments, as in ImageJ) = 1 / aspect ratio. 1.0 for a
      circle or a square, 0.5 for a 2:1 ellipse or rectangle; lower
      only as the shape elongates, barely affected by outline bumps;
    * Solidity = Area / ConvexHullArea -- 1.0 when the outline has no
      indentations; lower for lobed or concave shapes.

    Area is the polygon's own (shoelace) area, not a pixel count, so
    there is no pixelation bias. NaN for degenerate polygons."""
    pts = np.asarray(points2d, dtype=float)
    nan3 = (float("nan"),) * 3
    if pts.shape[0] < 3:
        return nan3
    # Centre first: the moment formulas below subtract large, nearly
    # equal terms when the ROI sits far from the image origin.
    x = pts[:, 0] - pts[:, 0].mean()
    y = pts[:, 1] - pts[:, 1].mean()
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    signed_area = 0.5 * cross.sum()
    area = abs(signed_area)
    perim = polygon_perimeter(pts)
    try:
        hull_area = float(ConvexHull(pts).volume)  # 2D "volume" = area
    except Exception:  # noqa: BLE001 -- collinear/degenerate polygon
        return nan3
    if area <= 0 or perim <= 0 or hull_area <= 0:
        return nan3
    circularity = 4 * np.pi * area / perim**2
    # Exact second moments of the polygon (Green's theorem); dividing by
    # the signed area makes the vertex direction irrelevant.
    cx = ((x + x1) * cross).sum() / (6 * signed_area)
    cy = ((y + y1) * cross).sum() / (6 * signed_area)
    sxx = ((x * x + x * x1 + x1 * x1) * cross).sum() / (12 * signed_area)
    syy = ((y * y + y * y1 + y1 * y1) * cross).sum() / (12 * signed_area)
    sxy = ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * cross).sum() / (
        24 * signed_area
    )
    cov = np.array(
        [[sxx - cx * cx, sxy - cx * cy], [sxy - cx * cy, syy - cy * cy]]
    )
    ev = np.linalg.eigvalsh(cov)
    roundness = (
        float(np.sqrt(max(ev[0], 0.0) / ev[1])) if ev[1] > 0 else float("nan")
    )
    solidity = area / hull_area
    return float(circularity), float(roundness), float(solidity)


def polygon_perimeter(points2d: np.ndarray) -> float:
    """Perimeter of a closed polygon (last vertex joins the first);
    NaN with fewer than two vertices."""
    pts = np.asarray(points2d, dtype=float)
    if pts.shape[0] < 2:
        return float("nan")
    return float(
        np.sum(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1))
    )


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
# Core analysis steps
# -----------------------------
# These are the numerical steps of an analysis run, kept as plain
# functions (no Qt, no viewer) so the exact code the widget runs can be
# unit-tested against known answers. OrganelleContactWidget calls them;
# do not re-implement any of this inline in the widget.
NORMALIZE_TAIL_FRACTION = 1e-4  # 0.01% of voxels ignored at each end


def normalize_signal(
    sig: np.ndarray, rng: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """Robust-normalize one channel to 0-1, as done before every
    threshold.

    The 0 and 1 reference points are the intensities with 0.01% of the
    voxels below / above them (computed over the whole array passed in
    -- for the widget, the Z-cropped volume), and values outside are
    clipped to 0-1. Plain min-max scaling made a manual threshold
    relative to the single brightest voxel, so one hot pixel, cosmic
    ray or small saturated spot rescaled the whole image and changed
    the mask everywhere; likewise one dead (zero) pixel moved the
    bottom of the range. Ignoring the extreme 0.01% removes that
    without touching real signal (on a 17 x 2048 x 2048 stack that is
    ~7,000 voxels at each end). Arrays under 10,000 values fall back to
    exact min-max. Already-0-1 data (max <= 1) and constant images are
    returned unchanged (as float).

    ``rng`` is the (lo, hi) pair from ``normalization_range`` when the
    caller already has it (avoids recomputing it on large stacks)."""
    s2 = np.asarray(sig).astype(float, copy=False)
    if rng is None:
        rng = normalization_range(s2)
    if rng is None:
        return s2
    lo, hi = rng
    return np.clip((s2 - lo) / (hi - lo), 0.0, 1.0)


def normalization_range(sig: np.ndarray) -> Optional[Tuple[float, float]]:
    """The raw intensities that ``normalize_signal`` maps to 0 and 1, or
    None when the data is left unscaled (already 0-1, or constant). A
    scaled threshold t corresponds to the raw intensity lo + t*(hi-lo)."""
    s2 = np.asarray(sig).astype(float, copy=False)
    vmax = np.nanmax(s2)
    vmin = np.nanmin(s2)
    if not (vmax > 1.0 and vmax > vmin):
        return None
    flat = s2.ravel()
    if np.isnan(vmax) or np.isnan(flat).any():
        flat = flat[~np.isnan(flat)]
    k = int(np.floor(NORMALIZE_TAIL_FRACTION * flat.size))
    if k > 0:
        kth = (k, flat.size - 1 - k)
        part = np.partition(flat, kth)
        lo, hi = float(part[kth[0]]), float(part[kth[1]])
        del part
        if not hi > lo:
            lo, hi = float(vmin), float(vmax)
    else:
        lo, hi = float(vmin), float(vmax)
    return lo, hi


def threshold_mask(
    sig: np.ndarray, mode: str, method: str, manual_value: float
) -> Tuple[np.ndarray, float]:
    """Threshold one normalized channel. ``mode`` is "Automatic" (use
    ``AUTO_METHODS[method]``, falling back to the mean if that method
    fails on this image) or anything else for manual (``manual_value``).
    Strictly greater-than. Returns (mask, threshold_value_used)."""
    if mode == "Automatic":
        try:
            thresh_val = AUTO_METHODS[method](sig)
        except Exception as e:  # noqa: BLE001
            print(
                f"Warning: Auto threshold '{method}' failed: {e}. "
                "Using mean fallback."
            )
            thresh_val = float(np.mean(sig))
    else:
        thresh_val = float(manual_value)
    return sig > thresh_val, float(thresh_val)


THRESH_MODE_RAW = "Manual (raw intensity)"


def threshold_channel(
    raw: np.ndarray,
    norm: np.ndarray,
    rng: Optional[Tuple[float, float]],
    mode: str,
    method: str,
    manual_value: float,
    raw_value: float,
) -> Tuple[np.ndarray, float, float]:
    """Threshold one channel in any of the three modes, and report the
    cutoff on BOTH scales so it can be recorded whichever mode was used.

    * "Automatic" / "Manual": threshold the scaled (0-1) signal ``norm``
      (see normalize_signal) with the method / ``manual_value``.
    * THRESH_MODE_RAW: keep voxels whose raw intensity is above
      ``raw_value`` -- the same absolute cutoff in every image.

    ``rng`` is that channel's ``normalization_range`` (None if the data
    wasn't rescaled). Returns (mask, scaled_cutoff, raw_cutoff). The two
    cutoffs describe the same mask: raw = lo + scaled * (hi - lo). In raw
    mode the scaled equivalent can fall outside 0-1 (cutoff below the
    0.01th or above the 99.99th percentile)."""
    lo, hi = rng if rng is not None else (0.0, 1.0)
    if mode == THRESH_MODE_RAW:
        raw_cut = float(raw_value)
        mask = np.asarray(raw) > raw_cut
        scaled_cut = (raw_cut - lo) / (hi - lo)
        return mask, float(scaled_cut), raw_cut
    mask, scaled_cut = threshold_mask(norm, mode, method, manual_value)
    return mask, float(scaled_cut), float(lo + scaled_cut * (hi - lo))


def threshold_columns(
    ch_labels: List[str], thresholds_used: List[Tuple[float, float]]
) -> Dict[str, float]:
    """Export columns recording each channel's applied cutoff:
    "Threshold Scaled (<ch>)" (0-1 scale) and "Threshold Raw (<ch>)"
    (raw intensity units)."""
    out: Dict[str, float] = {}
    for label, (t_scaled, t_raw) in zip(ch_labels, thresholds_used):
        out[f"Threshold Scaled ({label})"] = float(t_scaled)
        out[f"Threshold Raw ({label})"] = float(t_raw)
    return out


def contact_distance_map(mask: np.ndarray, z_xy_ratio: float) -> np.ndarray:
    """Distance (in XY pixels) from every voxel to the nearest voxel of
    ``mask``; 0 inside the mask. On 3D data the Z axis is weighted by
    ``z_xy_ratio`` (Z step / XY pixel size)."""
    sampling = (z_xy_ratio, 1.0, 1.0) if mask.ndim == 3 else None
    return distance_transform_edt(~mask, sampling=sampling)


# How contacts are defined when 3-4 channels are analyzed (see
# compute_contacts and the "Contact Area" glossary entry). With 2 channels
# both reduce to the same symmetric definition, so the choice only
# applies to 3-4 channels.
CONTACT_METHOD_OVERLAP = "Overlap-based"
CONTACT_METHOD_FOCUS = "Focus channel"
CONTACT_METHODS = (CONTACT_METHOD_OVERLAP, CONTACT_METHOD_FOCUS)


def compute_contacts(
    masks: List[np.ndarray],
    dists: List[np.ndarray],
    threshold: float,
    focus: Optional[int] = None,
) -> np.ndarray:
    """Contact voxels between thresholded channel masks.

    ``dists[i]`` is channel i's distance map (contact_distance_map).

    Overlap-based (``focus`` None, the default): for each channel i,
    the voxels inside *every other* channel's mask that lie within
    ``threshold`` of channel i; the union over i. So all channels but
    one must overlap exactly, and only the remaining one gets the
    distance tolerance. With two channels this is "voxels of either
    channel within ``threshold`` of the other".

    Focus channel (``focus`` = a channel index): the voxels of the focus
    channel's mask that lie within ``threshold`` of *every* other
    channel. Contacts therefore always sit on the focus organelle, and
    the other channels need not overlap each other.

    With ``threshold`` = 0 both reduce exactly to the intersection of
    all channels."""
    n = len(masks)
    if focus is not None:
        focus = int(focus)
        if not 0 <= focus < n:
            raise ValueError(
                f"focus channel {focus} out of range for {n} channels"
            )
        contacts = masks[focus].copy()
        for j in range(n):
            if j != focus:
                contacts &= dists[j] <= threshold
        return contacts

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
        contacts |= (dists[i] <= threshold) & others
    return contacts


# Kept in one place, and in the same grouping as the Output Selection
# dialog, so this stays easy to keep in sync as metrics are added,
# renamed, or removed.
METRIC_GLOSSARY_HTML = """
<h3>Threshold Scale</h3>
<p>Each channel is rescaled to 0&ndash;1 before thresholding, so a Manual
threshold is a fraction of that channel's intensity range in the analyzed
Z range. The range runs from the intensity with 0.01% of voxels below it
to the intensity with 0.01% of voxels above it (values outside are
clipped), rather than from the single dimmest to the single brightest
voxel &mdash; so one hot pixel, cosmic ray or small saturated spot no
longer rescales the image and changes the mask.</p>
<p><b>Manual (raw intensity)</b> mode instead keeps every voxel brighter
than a fixed raw intensity (detector counts), identically in every image.
That removes per-image scaling, but any brightness difference not caused
by biology (laser or detector drift, bleaching, staining or expression
level) then changes the mask. Use it only when all compared images were
acquired and labeled identically, and check that background and bright
structures have similar intensities across conditions.</p>
<p><b>Threshold Scaled / Threshold Raw (per channel)</b> &mdash; the cutoff
actually applied in each analysis, on both scales (raw = the scaled value
converted back to intensity units), whatever mode was used. Also shown
under each channel's controls as "Last run". Comparing these across
images shows how much the effective cutoff varied, and gives the values to
report in methods. Keep the same threshold policy across every condition
you compare.</p>

<h3>Core Overlap Metrics</h3>
<p><b>Intersection</b> &mdash; pixel count where all channels are
simultaneously thresholded-positive (logical AND across channels).</p>
<p><b>Union</b> &mdash; pixel count where at least one channel is
thresholded-positive (logical OR across channels).</p>
<p><b>Intersection/Union (Contact Coefficient)</b> &mdash; the Jaccard
index of the two above, a normalized 0-1 overlap score.</p>
<p><b>Contact Area</b> &mdash; pixel count of the contact region: a
distance-tolerant version of Intersection. Distances are measured
between thresholded masks, in XY pixels (see the Z/XY note below). How
the region is defined depends on the number of channels:</p>
<ul>
<li><i>2 channels:</i> the voxels of either channel that lie within the
Contact Threshold of the other channel. The region therefore covers both
organelles' signal on either side of a gap.</li>
<li><i>3&ndash;4 channels, Contacts method "Overlap-based"</i> (default):
a voxel counts when every channel but one is present there (their masks
overlap exactly at that voxel) and the remaining channel lies within the
Contact Threshold. Each channel is tried as the remaining one and the
results are combined. At least all-but-one channels must truly overlap,
so organelles that sit close together without overlapping each other
register no contact.</li>
<li><i>3&ndash;4 channels, Contacts method "Focus channel":</i> a voxel of
the chosen focus channel counts when <b>every</b> other channel lies
within the Contact Threshold of it. The other channels need not overlap
each other or the focus channel, so this captures three or four
organelles clustered within the set distance. The region always lies on
the focus channel's own signal, so Contact Area can be read against that
channel's Signal Area.</li>
</ul>
<p>With a Contact Threshold of 0 px, every definition above reduces to
the Intersection (voxels where all channels overlap). Contact Site
Count, Mean Contact Site Size, Contact Site NN Distance, Contact Mean
Intensity, the "Contacts" region in Signal Intensity Comparisons and the
Contacts layer all use the same contact region.</p>
<p><b>Contact Method</b> (3&ndash;4 channels only) &mdash; text column
recording which method produced each row ("Overlap-based" or "Focus
channel (<i>name</i>)"). Exported to Excel; not exported to Prism,
which takes numbers only. Use one method across every condition you
compare.</p>
<p><i>Z/XY calibration:</i> on a Z-stack, "Threshold (px)" still means
pixels of lateral (XY) distance, but the Z axis is weighted relative
to XY using the image's calibration (Z step / XY pixel size), so a
step in Z isn't silently treated as the same physical distance as an
XY pixel when the two differ -- very common in fluorescence
microscopy. This is auto-detected from the image's own metadata when
available (shown in the Contact Analysis section); if no calibration
is detected, or it looks wrong, or the image wasn't opened through
this plugin's own reader, use "Manually specify Z/XY calibration" to
override it. With no calibration at all (detected or manual), this
behaves exactly as before (Z and XY pixels treated as equal
distance).</p>

<h3>Per-Channel Area Metrics</h3>
<p><b>Signal Area (per channel)</b> &mdash; pixel count of that
channel's thresholded signal alone.</p>
<p><b>Intersection/Ch Signal Area (per channel)</b> &mdash; what
fraction of that channel's own signal area falls inside the full
intersection.</p>
<p><b>Body Count (per channel)</b> &mdash; number of separate connected
components ("bodies") in the channel's thresholded mask.</p>
<p><b>Average Area per Body (per channel)</b> &mdash; the average size
of those connected components (Signal Area / Body Count).</p>
<p><b>Fragmentation Coefficient (per channel)</b> &mdash; Average Area
per Body divided by Signal Area (equivalent to 1/Body Count). Near 1
means the channel's signal is essentially one contiguous body; near 0
means it's spread across many bodies.</p>
<p><i>Tip:</i> in the Thresholding section, "Bodies Ch N" (or Body Labels in
Auto-Display Setup, under Metrics &amp; Display Settings) displays exactly the
connected-component groupings these three metrics are computed from,
as a color-coded Labels layer &mdash; each body gets its own color.</p>
<p><i>Minimum Body Size:</i> thresholded masks often leave behind
single-pixel noise, which would otherwise be counted as its own
"body." The Minimum Body Size spinbox (default 2 px/voxels) excludes
any connected component smaller than that. "Apply to Body analyses"
(on by default) scopes this to Body Count / Average Area per Body /
Fragmentation Coefficient and the Body Labels layer only &mdash;
Signal Area, Intersection, Union, and Contact Area are unaffected.
"Also apply to thresholded mask" (off by default) instead removes
small bodies from each channel's mask before anything is computed,
so it changes every downstream metric. When only the first toggle is
on, Average Area per Body and Fragmentation Coefficient are computed
from the filtered signal area (the surviving bodies' combined size),
not the full Signal Area, so the two stay internally consistent.</p>

<h3>Morphology Metrics (per channel, opt-in)</h3>
<p>Both Shape and Network below use the same body definition (and
Minimum Body Size filtering, if enabled) as Fragmentation Metrics.
Each reports two flavors: an <b>unweighted mean/SD</b> (every body
counts once &mdash; "what does a typical object look like") and an
<b>area-weighted mean</b> (each body counted in proportion to its own
size &mdash; "where does most of the signal mass sit"). These can
meaningfully disagree; that disagreement is informative, not a
contradiction to resolve.</p>
<p><b>Aspect Ratio</b> &mdash; major/minor axis length of each body's
best-fit ellipse (ellipsoid in 3D). Near 1 is circular; higher is more
elongated. On Z-stacks the fit uses physical proportions (Z weighted by the
Z/XY calibration), so a round body isn't read as elongated just because Z
is sampled more coarsely than XY.</p>
<p><b>Form Factor</b> &mdash; perimeter&sup2; / (4&pi;&times;area), the
inverse of circularity. More sensitive to branching/irregular outlines
than Aspect Ratio alone. The perimeter is the Crofton estimate, which
stays close to the true value for small and large bodies alike (the
simple pixel-edge perimeter makes small round bodies look rounder than
large ones). Not defined for 3D (Z-stack) bodies, where it
reports as blank/NaN; Aspect Ratio still works in 3D.</p>
<p><b>Branch Count / Junction Count / Branch Length (per body)</b>
&mdash; from skeletonizing the mask down to a 1-pixel-wide medial axis
and classifying each skeleton pixel by neighbor count: a junction is a
skeleton pixel with 3 or more neighbors, and touching junction pixels
count as one junction. Branch Length is the length of the path through
the branch's pixel centers (1 per straight step, &radic;2 per diagonal,
Z steps weighted by the Z/XY calibration) plus one pixel for the end caps,
in XY pixels &mdash; multiply by the pixel size for &micro;m. It does not
depend on which way a branch is oriented.</p>
<p><b>Merge junctions within</b> (Morphology section, off by default)
&mdash; junctions whose centers lie within this many XY pixels (Z scaled
by the voxel ratio) are counted as one, and the short skeleton segments
joining them stop counting as branches. Use it when wide bodies produce
ladder-like skeletons whose rungs add junctions that are not real branch
points. Merging chains transitively, so large values can fold a whole
dense region into one junction: keep the value small (about a body's
width) and identical across every condition you compare.</p>
<p><b>% Bodies with Junctions</b> &mdash; fraction of bodies (by count)
with at least one junction, i.e. showing any branching.</p>
<p><b>% Signal Area in Junction-Containing Bodies</b> &mdash; fraction
of total signal area (by mass) sitting inside branched bodies, rather
than unbranched puncta/rods.</p>
<p><i>Caution:</i> don't treat the two percentages above as a hard
reticular-vs-fragmented classification. MiNA's own developers
originally reported an equivalent "number of individuals vs. number of
networks" metric and later removed it: as a network fragments, the
resulting pieces often each retain a junction point, so a naive
per-object junction-presence count can rise even as the structure is
clearly becoming more fragmented. Read these as trend indicators
alongside Fragmentation Coefficient, not in isolation.</p>
<p><i>Tip:</i> in the Morphology section, "Skeleton Ch N" and
"Junction Ch N" (or Skeleton/Junctions in Auto-Display Setup) display the
skeleton (green) and junctions (blue) these metrics are computed from &mdash; the same
color convention used by MiNA/Fiji's Analyze Skeleton.</p>
<p><i>Performance:</i> both Shape and Network are off by default and
add real computation time, especially Network on large 3D stacks,
since skeletonization runs on the full volume.</p>

<h3>Per-ROI Area Metrics</h3>
<p><b>ROI Area</b> &mdash; pixel count of the ROI (or the full image,
when not using per-ROI analysis).</p>
<p><b>Signal Area/ROI Area (per channel)</b> &mdash; that channel's
signal area as a fraction of the ROI/image.</p>

<h3>Intensity Metrics</h3>
<p><b>Mean Intensity (per channel)</b> &mdash; average raw intensity
within that channel's own thresholded mask.</p>
<p><b>Contact Mean Intensity (per channel)</b> &mdash; average raw
intensity within the Contact Area region. With a Contact Threshold above
0, the region includes voxels that belong to one channel but lie just
outside another (e.g. one organelle's edge facing a neighbour across a
gap), so that other channel's average includes some of its background.
In Focus channel mode the focus channel's own average never does,
because the region lies on its signal.</p>
<p><b>Signal Intensity Comparisons</b> &mdash; your own configured
comparisons (mean intensity of a source channel within a
Union/Intersection/Contacts region, optionally minus another region).
Off by default; only appears if you've set one up via Output
Selection.</p>

<h3>Advanced ROI/Spatial Metrics</h3>
<p>ROI geometry describes the shape of each drawn ROI itself, in XY
pixels (multiply by the pixel size for &micro;m).</p>
<p><b>Max Feret</b> &mdash; the ROI's longest caliper width: the
greatest distance between any two of its points.</p>
<p><b>Min Feret</b> &mdash; the ROI's narrowest caliper width: the
smallest gap between two parallel lines that enclose it. Max and Min
Feret match Fiji/ImageJ's Feret and MinFeret.</p>
<p><b>Shape Perimeter</b> &mdash; the ROI polygon's actual measured
perimeter.</p>
<p><b>Circularity</b> &mdash; 4&pi;&times;Area / Perimeter&sup2;: 1.0 for a
circle, lower for elongated <i>and</i> for irregular outlines. The
standard single measure of how round a cell is.</p>
<p><b>Roundness</b> &mdash; 4&times;Area / (&pi;&times;Major axis&sup2;),
using the ROI's best-fit ellipse; equals 1 / aspect ratio. 1.0 for a
circle or square, 0.5 for a 2:1 oval or rectangle; lower only as the cell
elongates, and barely affected by outline bumpiness.</p>
<p><b>Solidity</b> &mdash; Area / convex hull area: 1.0 when the outline
has no indentations, lower for lobed or concave cells. Circularity,
Roundness and Solidity match Fiji/ImageJ's Shape Descriptors. All are
computed from the drawn outline, so draw ROIs the same way across
conditions (jagged hand-drawn edges add perimeter and lower
Circularity).</p>
<p><b>Contact Site Count</b> &mdash; number of separate contact sites:
connected pieces of the Contact Area (face connectivity, the same rule
used for bodies).</p>
<p><b>Mean Contact Site Size</b> &mdash; average pixels/voxels per
contact site (Contact Area / Contact Site Count).</p>
<p><b>Contact Site NN Distance</b> &mdash; average distance from each
contact site's center to the nearest other site's center, in XY pixels
with Z weighted by the Z/XY calibration. Low = contacts clustered;
high = contacts spread out. Blank with fewer than two sites.</p>
"""


class MetricsGlossaryDialog(QDialog):
    """A read-only reference window describing every output metric the
    plugin can compute, grouped the same way as the Output Selection
    dialog so the two stay easy to cross-reference."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setWindowTitle("Output Metric Descriptions")
        self.resize(480, 560)

        layout = QVBoxLayout()

        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml(METRIC_GLOSSARY_HTML)
        layout.addWidget(text)

        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(self.accept)
        layout.addWidget(bb)

        self.setLayout(layout)


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

        src_combo = ScrollSafeComboBox()
        src_combo.addItems(
            [
                f"{i+1}: {self.channel_labels[i]}"
                for i in range(self.n_channels)
            ]
        )
        src_combo.setCurrentIndex(int(comp.get("source_ch", 0)))
        self.table.setCellWidget(r, 1, src_combo)

        base_mode_combo = ScrollSafeComboBox()
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

        sub_mode_combo = ScrollSafeComboBox()
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
            {idx.row() for idx in self.table.selectedIndexes()},
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
        self.cb_fragmentation = QCheckBox(
            "Body Count / Average Area per Body / Fragmentation "
            "Coefficient (per channel)"
        )
        self.cb_signal_area.setChecked(
            self._selection.get("Signal Area", True)
        )
        self.cb_intersection_over_ch.setChecked(
            self._selection.get("Intersection/Ch Signal Area", True)
        )
        self.cb_fragmentation.setChecked(
            self._selection.get("Fragmentation Metrics", True)
        )
        perch_layout.addWidget(self.cb_signal_area)
        perch_layout.addWidget(self.cb_intersection_over_ch)
        perch_layout.addWidget(self.cb_fragmentation)
        perch_box.setLayout(perch_layout)
        layout.addWidget(perch_box)

        morph_box = QGroupBox("Morphology Metrics (per channel)")
        morph_layout = QVBoxLayout()
        self.cb_morph_shape = QCheckBox(
            "Shape: Aspect Ratio / Form Factor (mean, SD, area-weighted "
            "mean per body)"
        )
        self.cb_morph_network = QCheckBox(
            "Network: Branch Count / Junction Count / Branch Length "
            "(skeleton-based)"
        )
        morph_note = QLabel(
            "Both require the body labeling used by Fragmentation "
            "Metrics; enabling them adds noticeable computation time, "
            "especially on large 3D stacks."
        )
        morph_note.setWordWrap(True)
        self.cb_morph_shape.setChecked(
            self._selection.get("Morphology Shape", False)
        )
        self.cb_morph_network.setChecked(
            self._selection.get("Morphology Network", False)
        )
        morph_layout.addWidget(self.cb_morph_shape)
        morph_layout.addWidget(self.cb_morph_network)
        morph_layout.addWidget(morph_note)
        morph_box.setLayout(morph_layout)
        layout.addWidget(morph_box)

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
            "ROI geometry (Feret diameters, Perimeter, Circularity, "
            "Roundness, Solidity)"
        )
        self.cb_contact_density = QCheckBox(
            "Contact site metrics (Site Count, Size, NN Distance)"
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
        sel["Fragmentation Metrics"] = self.cb_fragmentation.isChecked()

        sel["Morphology Shape"] = self.cb_morph_shape.isChecked()
        sel["Morphology Network"] = self.cb_morph_network.isChecked()

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


class DisplayLayersDialog(QDialog):
    """Choose, per channel, which layers are displayed automatically
    after Analyze runs: Thresholded, Body Labels, Skeleton and
    Junctions for each channel, plus the combined Contacts layer.
    Replaces the four all-channels-or-none "Auto-show ..." checkboxes,
    so a mixed selection like "Thresholded Ch 1 + Ch 2, Skeleton and
    Junctions Ch 2 only" can be saved as one setting. Skeleton and
    Junctions are separate rows, mirroring the separate Skeleton /
    Junction buttons in the Morphology section.

    Ported from the ``wip/morphology-network-controls`` branch, minus
    its "Collapsed" row (that layer only exists with the collapse
    knobs, which were not carried over)."""

    ROWS = [
        ("thresholded", "Thresholded"),
        ("body_labels", "Body Labels"),
        ("skeleton", "Skeleton"),
        ("junctions", "Junctions"),
    ]

    def __init__(
        self,
        parent: QWidget,
        current_selection: Dict[str, Any],
        n_channels_max: int,
        channel_labels: List[str],
    ):
        super().__init__(parent)
        self.setWindowTitle("Auto-Display Setup")
        self.n_channels_max = n_channels_max
        self.channel_labels = channel_labels[:]
        sel = current_selection or {}

        layout = QVBoxLayout()
        info = QLabel(
            "Choose which layers are shown automatically after "
            "clicking Analyze. Each cell below is independent -- mix "
            "and match any combination of channels and layer types."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        grid_box = QGroupBox("Per-Channel Layers")
        grid = QGridLayout()
        for col in range(self.n_channels_max):
            label = (
                self.channel_labels[col]
                if col < len(self.channel_labels)
                else f"Ch {col + 1}"
            )
            grid.addWidget(
                QLabel(f"<b>{label}</b>"),
                0,
                col + 1,
                alignment=Qt.AlignCenter,
            )

        self.checkboxes: Dict[str, List[QCheckBox]] = {}
        for row_idx, (key, row_label) in enumerate(self.ROWS, start=1):
            grid.addWidget(QLabel(row_label), row_idx, 0)
            saved_row = sel.get(key, [])
            boxes = []
            for col in range(self.n_channels_max):
                cb = QCheckBox()
                cb.setChecked(
                    bool(saved_row[col]) if col < len(saved_row) else False
                )
                grid.addWidget(cb, row_idx, col + 1, alignment=Qt.AlignCenter)
                boxes.append(cb)
            self.checkboxes[key] = boxes
        grid_box.setLayout(grid)
        layout.addWidget(grid_box)

        select_btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_none_btn = QPushButton("Select None")
        select_all_btn.clicked.connect(lambda: self._set_all(True))
        select_none_btn.clicked.connect(lambda: self._set_all(False))
        select_btn_layout.addWidget(select_all_btn)
        select_btn_layout.addWidget(select_none_btn)
        select_btn_layout.addStretch(1)
        layout.addLayout(select_btn_layout)

        other_box = QGroupBox("Other")
        other_layout = QVBoxLayout()
        self.cb_contacts = QCheckBox("Contacts")
        self.cb_contacts.setToolTip(
            "The combined Contacts overlap layer (not per-channel)."
        )
        self.cb_contacts.setChecked(bool(sel.get("contacts", True)))
        other_layout.addWidget(self.cb_contacts)
        other_box.setLayout(other_layout)
        layout.addWidget(other_box)

        note = QLabel(
            "Only channels active in the current analysis are actually "
            "shown; toggles for inactive channels are simply ignored. "
            "Skeleton/Junctions are computed on demand from the "
            "thresholded bodies, so they work whether or not Morphology "
            "Network metrics are enabled in Output Selection."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        self.setLayout(layout)

    def _set_all(self, checked: bool) -> None:
        for boxes in self.checkboxes.values():
            for cb in boxes:
                cb.setChecked(checked)

    def get_results(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            key: [cb.isChecked() for cb in boxes]
            for key, boxes in self.checkboxes.items()
        }
        result["contacts"] = self.cb_contacts.isChecked()
        return result


# -----------------------------
# Main Widget
# -----------------------------
class OrganelleContactWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        self.threshold = 0.0
        self._last_z_source_signature = None
        self.metrics_list: List[Dict[str, Any]] = []
        self.last_metrics: Any = {}
        self.last_masks: List[np.ndarray] = []
        self.current_roi_index = 0

        self._last_contacts_display: Optional[np.ndarray] = None
        self._last_base_layer: Optional[napari.layers.Image] = None
        self._last_display_z_translate = None
        self._last_display_scale = None

        self.max_channels_supported = 4
        self.channel_layer_indices = [0, 1, 2, 3]

        # Which layers auto-display after Analyze, set in the
        # "Auto-Display Setup" dialog (DisplayLayersDialog): one bool
        # per channel for each per-channel layer type, plus one flag
        # for the combined Contacts layer. Defaults match the old
        # checkboxes: everything off except Contacts.
        self.display_layers_selection: Dict[str, Any] = {
            "thresholded": [False] * self.max_channels_supported,
            "body_labels": [False] * self.max_channels_supported,
            "skeleton": [False] * self.max_channels_supported,
            "junctions": [False] * self.max_channels_supported,
            "contacts": True,
        }

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
            "Fragmentation Metrics": True,
            "Morphology Shape": False,
            "Morphology Network": False,
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

        # Contact threshold in (decimal) XY pixels. Distances are
        # measured with Z steps weighted by the Z/XY voxel ratio, so a
        # voxel directly above/below is <ratio> px away -- e.g. 1.443
        # px here -- and a decimal threshold can include it without
        # also reaching 2 px sideways. The slider moves in 0.1 px
        # steps (its integer value is px x CT_SLIDER_STEPS_PER_PX);
        # the text box accepts up to two decimals.
        self.CT_SLIDER_STEPS_PER_PX = 10
        self.ct_label = QLabel(f"Threshold (px): {self.threshold:g}")
        self.ct_label.setAlignment(Qt.AlignCenter)
        self.ct_label.setToolTip(
            "Contact threshold, in XY pixels (decimals allowed): the "
            "maximum distance between two channels' thresholded signal "
            "for them to be counted as 'in contact'. Z steps count as "
            "the Z/XY ratio shown below, so a voxel directly above or "
            "below is only included once the threshold reaches that "
            "ratio. With 3-4 channels, 'Contacts method' below sets "
            "which channels the distance applies to."
        )
        self.ct_slider = ScrollSafeSlider(Qt.Horizontal)
        self.ct_slider.setMinimum(0)
        self.ct_slider.setMaximum(100 * self.CT_SLIDER_STEPS_PER_PX)
        self.ct_slider.setValue(
            int(round(self.threshold * self.CT_SLIDER_STEPS_PER_PX))
        )
        self.ct_slider.valueChanged.connect(self.slider_changed)
        self.ct_text = QLineEdit(f"{self.threshold:g}")
        ct_validator = QDoubleValidator(0.0, 100.0, 2)
        ct_validator.setNotation(QDoubleValidator.StandardNotation)
        self.ct_text.setValidator(ct_validator)
        self.ct_text.setMaximumWidth(60)
        self.ct_text.editingFinished.connect(self.text_input_changed)

        # How contacts are defined with 3-4 channels (see
        # compute_contacts). Grayed out with 2 channels, where both
        # methods are the same definition.
        self.contact_method_label = QLabel("Contacts method:")
        self.contact_method_combo = ScrollSafeComboBox()
        self.contact_method_combo.addItems(list(CONTACT_METHODS))
        self.contact_method_combo.setToolTip(
            "Only used with 3-4 channels.\n"
            "Overlap-based: a voxel is a contact when all channels but "
            "one overlap there exactly and the remaining channel is "
            "within the contact threshold (each channel is tried as the "
            "remaining one).\n"
            "Focus channel: a voxel of the focus channel is a contact "
            "when every other channel is within the contact threshold "
            "of it; the other channels need not overlap each other.\n"
            "At a threshold of 0 px both count only voxels where all "
            "channels overlap."
        )
        self.contact_method_combo.currentIndexChanged.connect(
            lambda _: self._on_contact_method_changed()
        )
        self.contact_focus_label = QLabel("Focus channel:")
        self.contact_focus_combo = ScrollSafeComboBox()
        # Channel names can be long layer names: cap the width so the
        # panel doesn't grow wider than the dock.
        self.contact_focus_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.contact_focus_combo.setMinimumContentsLength(14)
        self.contact_focus_combo.setToolTip(
            "Focus channel method: contacts are this channel's voxels "
            "lying within the contact threshold of every other channel."
        )
        # The focus channel the user picked, kept separately from the
        # list's current index: dropping to fewer channels shrinks the
        # list, and the choice must come back when the channel does
        # rather than silently move to another channel.
        self._contact_focus_wanted = 0
        self.contact_focus_combo.currentIndexChanged.connect(
            self._on_contact_focus_changed
        )

        # Z/XY voxel calibration -- corrects the Contact Threshold's
        # proximity search so a step in Z isn't silently treated as the
        # same physical distance as an XY pixel when the two differ
        # (very common in Z-stack microscopy). "Threshold (px)" keeps
        # its existing meaning for lateral distance; only the Z axis's
        # relative weight in the 3D distance transform changes.
        # Multi-line status (ratio / what a Z step equals / threshold in
        # physical units). Built from short explicit lines with word
        # wrap OFF: a word-wrapped QLabel inside this scroll-area panel
        # gets a one-line height from the layout and its extra lines
        # were being clipped by the rows above and below. The minimum
        # height is set from the line count every time the text
        # changes (see _update_voxel_calibration_status_label).
        self.voxel_calibration_status_label = QLabel("")
        self.voxel_calibration_status_label.setWordWrap(False)
        self.voxel_calibration_status_label.setTextFormat(Qt.PlainText)
        self.voxel_calibration_status_label.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Minimum
        )
        self.manual_voxel_calibration_checkbox = QCheckBox(
            "Manually specify Z/XY calibration"
        )
        self.manual_voxel_calibration_checkbox.setToolTip(
            "Override the Z/XY pixel-size ratio auto-detected from the "
            "image's metadata (napari layer scale). Use this if your "
            "image has no calibration metadata, the detected values "
            "look wrong, or the image wasn't opened through this "
            "plugin's own reader."
        )
        self.manual_voxel_calibration_checkbox.setChecked(False)
        self.manual_voxel_calibration_checkbox.stateChanged.connect(
            self._on_voxel_calibration_changed
        )

        self.z_step_spinbox = ScrollSafeDoubleSpinBox()
        self.z_step_spinbox.setDecimals(4)
        self.z_step_spinbox.setMinimum(0.0001)
        self.z_step_spinbox.setMaximum(10000.0)
        self.z_step_spinbox.setValue(1.0)
        self.z_step_spinbox.setToolTip(
            "Z step size (distance between slices), any consistent "
            "physical unit -- only the ratio to XY pixel size matters."
        )
        self.z_step_spinbox.setEnabled(False)
        self.z_step_spinbox.valueChanged.connect(
            lambda _: self._on_voxel_calibration_changed()
        )

        self.xy_pixel_spinbox = ScrollSafeDoubleSpinBox()
        self.xy_pixel_spinbox.setDecimals(4)
        self.xy_pixel_spinbox.setMinimum(0.0001)
        self.xy_pixel_spinbox.setMaximum(10000.0)
        self.xy_pixel_spinbox.setValue(1.0)
        self.xy_pixel_spinbox.setToolTip(
            "XY pixel size, in the same physical unit as the Z step "
            "size above."
        )
        self.xy_pixel_spinbox.setEnabled(False)
        self.xy_pixel_spinbox.valueChanged.connect(
            lambda _: self._on_voxel_calibration_changed()
        )

        self.channels_label = QLabel("Channels:")
        self.channel_mode_combo = ScrollSafeComboBox()
        self.channel_mode_combo.addItems(["2", "3", "4"])
        self.channel_mode_combo.setToolTip(
            "How many image layers to treat as channels for analysis."
        )
        self.channel_mode_combo.currentIndexChanged.connect(
            lambda _: self._on_analysis_source_changed()
        )
        self.channel_mode_combo.currentIndexChanged.connect(
            lambda _: self._save_settings()
        )

        self.z_range_label = QLabel("Range:")
        self.z_min_spinbox = ScrollSafeSpinBox()
        self.z_max_spinbox = ScrollSafeSpinBox()
        self.z_min_spinbox.setMinimum(0)
        self.z_max_spinbox.setMinimum(0)

        self.auto_adjust_z_range_checkbox = QCheckBox(
            "Auto-adjust Z range to image"
        )
        self.auto_adjust_z_range_checkbox.setToolTip(
            "Automatically reset the Z min/max range whenever a new image "
            "or channel is selected."
        )
        self.auto_adjust_z_range_checkbox.setChecked(True)
        self.auto_adjust_z_range_checkbox.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed
        )
        self.auto_adjust_z_range_checkbox.stateChanged.connect(
            lambda _: self._update_z_range_controls(force_full_reset=False)
        )
        self.auto_adjust_z_range_checkbox.stateChanged.connect(
            lambda _: self._save_settings()
        )

        self.use_layer_names_checkbox.stateChanged.connect(
            lambda _: self._refresh_channel_labels()
        )
        self.use_layer_names_checkbox.stateChanged.connect(
            lambda _: self._save_settings()
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
        self.channel_thresh_layout.setSpacing(10)
        self.channel_thresh_container.setLayout(self.channel_thresh_layout)
        self.per_channel_mode: List[QComboBox] = []
        self.per_channel_auto: List[QComboBox] = []
        self.per_channel_manual: List[QDoubleSpinBox] = []
        self.per_channel_raw: List[QDoubleSpinBox] = []
        self.per_channel_thresh_info: List[QLabel] = []
        self.per_channel_label_widgets: List[QLabel] = []
        # One QWidget per channel (rather than a bare layout) so the whole
        # row -- label, mode combo, auto combo, manual spinbox -- can be
        # grayed out together via a single setEnabled(False) call when a
        # channel isn't part of the active channel count (see
        # _sync_active_channel_row_states).
        self.per_channel_row_widgets: List[QWidget] = []

        _thresh_rows: List[QWidget] = []
        for i in range(self.max_channels_supported):
            row_widget = QWidget()
            row_outer = QVBoxLayout(row_widget)
            row_outer.setContentsMargins(0, 0, 0, 0)
            row_outer.setSpacing(2)

            label = QLabel(f"Channel {i+1}:")
            label.setWordWrap(True)
            label.setMaximumWidth(150)
            label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            self.per_channel_label_widgets.append(label)

            mode_combo = ScrollSafeComboBox()
            mode_combo.addItems(["Automatic", "Manual", THRESH_MODE_RAW])
            mode_combo.setToolTip(
                "Automatic: a method picks the cutoff per image.\n"
                "Manual: a fraction (0-1) of each image's intensity range.\n"
                "Manual (raw intensity): the same absolute intensity in "
                "every image -- use only when all images were acquired "
                "and labeled identically."
            )
            self.per_channel_mode.append(mode_combo)

            # Line 1: which channel, and Automatic vs. Manual thresholding.
            line1 = QHBoxLayout()
            line1.addWidget(label)
            line1.addWidget(mode_combo)
            line1.addStretch(1)
            row_outer.addLayout(line1)

            auto_combo = ScrollSafeComboBox()
            auto_combo.addItems(
                ["Otsu", "Li", "Mean", "Minimum", "Triangle", "Yen", "Isodata"]
            )
            self.per_channel_auto.append(auto_combo)
            auto_combo.currentIndexChanged.connect(
                lambda _: self._save_settings()
            )

            manual_spin = ScrollSafeDoubleSpinBox()
            manual_spin.setRange(0.0, 1.0)
            # Qt's default is 2 decimals, which snapped the threshold to
            # 1% steps of the channel's range -- too coarse, since useful
            # fluorescence thresholds typically sit around 0.03-0.15.
            manual_spin.setDecimals(4)
            manual_spin.setSingleStep(0.001)
            manual_spin.setValue(0.5)
            manual_spin.setToolTip(
                "Fraction of this channel's intensity range (0.01th to "
                "99.99th percentile of the analyzed Z range). See "
                "Metric Descriptions > Threshold Scale."
            )
            manual_spin.setEnabled(False)
            self.per_channel_manual.append(manual_spin)
            manual_spin.valueChanged.connect(lambda _: self._save_settings())

            raw_spin = ScrollSafeDoubleSpinBox()
            raw_spin.setRange(0.0, 1.0e9)
            raw_spin.setDecimals(2)
            raw_spin.setSingleStep(1.0)
            raw_spin.setValue(0.0)
            raw_spin.setToolTip(
                "Raw intensity cutoff (camera/detector counts): voxels "
                "brighter than this are signal, identically in every "
                "image."
            )
            raw_spin.setEnabled(False)
            self.per_channel_raw.append(raw_spin)
            raw_spin.valueChanged.connect(lambda _: self._save_settings())

            # Auto method, Manual value and Raw cutoff, one per line.
            # Side by side they were wider than a napari dock, and
            # since the panel never scrolls sideways the layout
            # squeezed the spinboxes and cut off their digits.
            values = QGridLayout()
            for r, (text, control) in enumerate(
                (
                    ("Auto:", auto_combo),
                    ("Manual:", manual_spin),
                    ("Raw:", raw_spin),
                )
            ):
                values.addWidget(QLabel(text), r, 0)
                values.addWidget(control, r, 1, alignment=Qt.AlignLeft)
            values.setColumnStretch(2, 1)
            row_outer.addLayout(values)

            info = QLabel("Last run: -")
            info.setWordWrap(True)
            info.setToolTip(
                "Cutoff applied to this channel in the last analysis, as a "
                "scaled (0-1) value and the equivalent raw intensity. Also "
                "exported as 'Threshold Scaled' / 'Threshold Raw'."
            )
            self.per_channel_thresh_info.append(info)
            row_outer.addWidget(info)

            mode_combo.currentIndexChanged.connect(
                self._sync_thresh_mode_states
            )
            self.per_channel_row_widgets.append(row_widget)
            _thresh_rows.append(row_widget)

        for row_widget in reversed(_thresh_rows):
            self.channel_thresh_layout.addWidget(row_widget)

        self.result_label = QLabel("Metrics: N/A")
        self.result_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.result_label.setWordWrap(True)

        self.analysis_name_label = QLabel("Analysis Name:")
        self.analysis_name_edit = QLineEdit()

        self.analysis_count_label = QLabel("Analyses Stored: 0")
        self.analysis_count_label.setAlignment(Qt.AlignCenter)

        self.per_shape_checkbox = QCheckBox("Calculate metrics per ROI")
        self.per_shape_checkbox.setToolTip(
            "Calculate metrics separately for each drawn ROI shape, "
            "instead of only for the full image."
        )
        self.per_shape_checkbox.stateChanged.connect(
            lambda _: self._save_settings()
        )
        self.sequential_label_checkbox = QCheckBox("Sequential ROI labeling")
        self.sequential_label_checkbox.setToolTip(
            "Use sequential labeling for ROI shapes."
        )
        self.sequential_label_checkbox.stateChanged.connect(
            lambda _: self._save_settings()
        )

        self.prev_roi_button = QPushButton("Previous")
        self.next_roi_button = QPushButton("Next")
        self.prev_roi_button.setToolTip("Previous ROI")
        self.next_roi_button.setToolTip("Next ROI")
        self.roi_nav_label = QLabel("ROI: N/A")
        self.prev_roi_button.clicked.connect(self.prev_roi)
        self.next_roi_button.clicked.connect(self.next_roi)

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

        self.display_layers_button = QPushButton("Auto-Display Setup")
        self.display_layers_button.setToolTip(
            "Choose exactly which layers (per channel: Thresholded, "
            "Body Labels, Skeleton, Junctions; plus Contacts) are shown "
            "automatically after Analyze runs, in any combination."
        )
        self.display_layers_button.clicked.connect(
            self.open_display_layers_setup
        )

        self.metrics_glossary_button = QPushButton("Metric Descriptions")
        self.metrics_glossary_button.setToolTip(
            "Show a description of every output metric this plugin can "
            "compute."
        )
        self.metrics_glossary_button.clicked.connect(
            self.open_metrics_glossary
        )

        self.scale_bar_status_label = QLabel("")
        self.scale_bar_status_label.setWordWrap(True)

        self.restrict_signal_z_checkbox = QCheckBox(
            "Restrict Z range to signal"
        )
        self.restrict_signal_z_checkbox.setToolTip(
            "Restrict analyzed Z-stacks to the span from the first to the "
            "last slice with thresholded signal (empty slices in between "
            "are kept so Z spacing is preserved)."
        )
        self.restrict_signal_z_checkbox.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed
        )
        self.restrict_signal_z_checkbox.setChecked(False)
        self.restrict_signal_z_checkbox.stateChanged.connect(
            self._on_restrict_signal_z_changed
        )

        self.restrict_signal_z_button = QPushButton("Signal Z Channels")
        self.restrict_signal_z_button.setToolTip(
            "Choose which channels' thresholded signal determines the "
            "Z-restriction above."
        )
        self.restrict_signal_z_button.clicked.connect(
            self.open_signal_z_channel_selection
        )
        self.restrict_signal_z_button.setEnabled(False)

        self.add_analysis_button = QPushButton("Add Analysis")
        self.add_analysis_button.clicked.connect(self.add_analysis)

        self.clear_last_button = QPushButton("Clear Last")
        self.clear_last_button.setToolTip("Clear Last Analysis")
        self.clear_last_button.clicked.connect(self.clear_last_analysis)

        self.clear_all_button = QPushButton("Clear All")
        self.clear_all_button.setToolTip("Clear All Analyses")
        self.clear_all_button.clicked.connect(self.clear_all_analyses)

        self.save_image_button = QPushButton("Save Image")
        self.save_image_button.clicked.connect(self.save_image)

        self.save_metrics_button = QPushButton("Export to Excel")
        self.save_metrics_button.clicked.connect(self.save_metrics)

        self.append_spreadsheet_button = QPushButton("Append to Excel")
        self.append_spreadsheet_button.setToolTip("Append to Excel Format")
        self.append_spreadsheet_button.clicked.connect(
            self.append_to_spreadsheet
        )

        self.export_graphpad_button = QPushButton("Export to GraphPad")
        self.export_graphpad_button.setToolTip("Export for GraphPad Prism")
        self.export_graphpad_button.clicked.connect(self.export_graphpad_prism)

        self.append_graphpad_button = QPushButton("Append to GraphPad")
        self.append_graphpad_button.clicked.connect(
            self.append_to_graphpad_prism
        )

        self.roi_button = QPushButton("Toggle ROI Selection")
        self.roi_button.clicked.connect(self.toggle_roi_selection)

        self.show_thresh_btns: List[QPushButton] = []
        for i in range(self.max_channels_supported):
            b = QPushButton(f"Show Ch {i+1}")
            b.setToolTip(f"Show Thresholded Channel {i+1}")
            b.clicked.connect(
                lambda _, idx=i: self.show_thresholded_channel(idx)
            )
            self.show_thresh_btns.append(b)

        self.show_body_labels_btns: List[QPushButton] = []
        for i in range(self.max_channels_supported):
            b = QPushButton(f"Bodies Ch {i+1}")
            b.setToolTip(
                f'Show the Body Count/Fragmentation "bodies" for '
                f"Channel {i+1} as a color-coded Labels layer"
            )
            b.clicked.connect(
                lambda _, idx=i: self.show_body_labels_channel(idx)
            )
            self.show_body_labels_btns.append(b)

        self.min_body_size_label = QLabel("Minimum Body Size (px):")
        self.min_body_size_spinbox = ScrollSafeSpinBox()
        self.min_body_size_spinbox.setMinimum(1)
        self.min_body_size_spinbox.setMaximum(100000)
        self.min_body_size_spinbox.setValue(2)
        self.min_body_size_spinbox.setToolTip(
            "Connected components smaller than this many pixels/voxels "
            "are treated as noise (e.g. leftover single-pixel islands) "
            "and excluded from body counting, per the toggles below."
        )
        self.min_body_size_spinbox.valueChanged.connect(
            lambda _: self._save_settings()
        )

        self.filter_body_metrics_checkbox = QCheckBox("Apply to Body analyses")
        self.filter_body_metrics_checkbox.setToolTip(
            "Exclude bodies smaller than the Minimum Body Size from Body "
            "Count, Average Area per Body, and Fragmentation "
            "Coefficient (and from the Body Labels layer), without "
            "changing Signal Area, Intersection, Union, Contact Area, "
            "or any other metric."
        )
        self.filter_body_metrics_checkbox.setChecked(True)
        self.filter_body_metrics_checkbox.stateChanged.connect(
            lambda _: self._save_settings()
        )

        self.filter_threshold_mask_checkbox = QCheckBox(
            "Also apply to thresholded mask"
        )
        self.filter_threshold_mask_checkbox.setToolTip(
            "Remove bodies smaller than the Minimum Body Size from each "
            "channel's thresholded mask itself, before any metric is "
            "computed. This changes Signal Area, Intersection, Union, "
            "Contact Area, Mean Intensity, and every other downstream "
            "metric -- not just the Body analyses."
        )
        self.filter_threshold_mask_checkbox.setChecked(False)
        self.filter_threshold_mask_checkbox.stateChanged.connect(
            self._on_filter_threshold_mask_changed
        )

        self.junction_merge_spinbox = ScrollSafeDoubleSpinBox()
        self.junction_merge_spinbox.setRange(0.0, 50.0)
        self.junction_merge_spinbox.setSingleStep(0.5)
        self.junction_merge_spinbox.setDecimals(1)
        self.junction_merge_spinbox.setValue(0.0)
        self.junction_merge_spinbox.setSpecialValueText("Off")
        self.junction_merge_spinbox.setSuffix(" px")
        self.junction_merge_spinbox.setToolTip(
            "Merge junction clusters whose centers lie within this "
            "distance (XY pixels; Z is scaled by the Z/XY voxel ratio) "
            "into one junction, and stop counting the short skeleton "
            "segments that only connect junctions inside one merged "
            "group (e.g. the 'rungs' a medial axis forms inside a wide "
            "body). Merging is transitive (chains of nearby junctions "
            "become one), only happens within a body, and applies to "
            "both the Morphology Network metrics and the Junctions "
            "layer. Off = the original per-cluster counting. Keep it "
            "the same across every condition you compare."
        )
        self.junction_merge_spinbox.valueChanged.connect(
            lambda _: self._save_settings()
        )

        # Separate Skeleton and Junction buttons per channel, so each
        # layer can be shown on its own.
        self.show_skeleton_btns: List[QPushButton] = []
        self.show_junction_btns: List[QPushButton] = []
        for i in range(self.max_channels_supported):
            b = QPushButton(f"Skeleton Ch {i+1}")
            b.setToolTip(
                f"Show the Morphology Network skeleton (green) for "
                f"Channel {i+1}."
            )
            b.clicked.connect(
                lambda _, idx=i: self.show_skeleton_channel(
                    idx, show_skeleton=True, show_junctions=False
                )
            )
            self.show_skeleton_btns.append(b)

            jb = QPushButton(f"Junction Ch {i+1}")
            jb.setToolTip(
                f"Show the Morphology Network junction points (blue) "
                f"for Channel {i+1}, after any 'Merge junctions within' "
                f"merging -- the same junctions Junction Count counts."
            )
            jb.clicked.connect(
                lambda _, idx=i: self.show_skeleton_channel(
                    idx, show_skeleton=False, show_junctions=True
                )
            )
            self.show_junction_btns.append(jb)

        self.show_contacts_button = QPushButton("Show Contacts")
        self.show_contacts_button.clicked.connect(self.show_contacts)

        self._metric_display_keys: Optional[List[str]] = None

        self.init_ui()
        self._load_settings()
        self._sync_thresh_mode_states()
        self._on_filter_threshold_mask_changed()
        self._refresh_channel_labels()
        self._update_z_range_controls(force_full_reset=True)
        self._auto_set_scale_bar_unit_from_layer()
        self._update_scale_bar_status_label()
        self._on_voxel_calibration_changed()

    # ---------------- UI ----------------
    def _group_box(self, title: str, inner_layout) -> QGroupBox:
        """Build a titled QGroupBox around ``inner_layout``. A shared
        helper so every section of the widget gets the same visual
        treatment (title styling, margins) with one place to adjust it."""
        box = QGroupBox(title)
        # Nudge the title a few pixels above the box's top border (via a
        # negative "top" offset) and add matching top padding inside the
        # box, so the title has clear space of its own instead of
        # crowding the first row of controls beneath it.
        box.setStyleSheet(
            "QGroupBox {"
            "  font-weight: bold;"
            "  margin-top: 10px;"
            "  padding-top: 12px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  subcontrol-position: top left;"
            "  left: 6px;"
            "  top: -4px;"
            "  padding: 0 3px;"
            "}"
        )
        inner_layout.setContentsMargins(6, 4, 6, 6)
        inner_layout.setSpacing(4)
        box.setLayout(inner_layout)
        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        return box

    def init_ui(self):
        # The widget is organized into labeled sections that follow the
        # order a user works through the plugin: pick channels, set the
        # Z range, threshold each channel, optionally draw ROIs, run the
        # contact analysis, configure what's measured/displayed, review
        # and manage results, then export.
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # --- Channel & Image Setup ---
        setup_layout = QVBoxLayout()
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(self.channels_label)
        channel_layout.addWidget(self.channel_mode_combo)
        channel_layout.addStretch(1)
        setup_layout.addLayout(channel_layout)
        setup_layout.addWidget(self.use_layer_names_checkbox)
        setup_layout.addWidget(self.channel_numbering_button)
        layout.addWidget(
            self._group_box("Channel && Image Setup", setup_layout)
        )

        # --- Z-Stack Range ---
        z_group_layout = QVBoxLayout()
        z_layout = QHBoxLayout()
        z_layout.addWidget(self.z_range_label)
        z_layout.addWidget(QLabel("Min:"))
        z_layout.addWidget(self.z_min_spinbox)
        z_layout.addWidget(QLabel("Max:"))
        z_layout.addWidget(self.z_max_spinbox)
        z_layout.addStretch(1)
        z_group_layout.addLayout(z_layout)
        z_group_layout.addWidget(self.auto_adjust_z_range_checkbox)
        z_group_layout.addWidget(self.restrict_signal_z_checkbox)
        z_group_layout.addWidget(
            self.restrict_signal_z_button, alignment=Qt.AlignLeft
        )
        layout.addWidget(self._group_box("Z-Stack Range", z_group_layout))

        # --- Thresholding ---
        thresh_group_layout = QVBoxLayout()
        thresh_group_layout.addWidget(self.channel_thresh_container)
        thresh_btn_layout = QGridLayout()
        for i, b in enumerate(self.show_thresh_btns):
            thresh_btn_layout.addWidget(b, i // 2, i % 2)
        thresh_group_layout.addLayout(thresh_btn_layout)
        body_btn_layout = QGridLayout()
        for i, b in enumerate(self.show_body_labels_btns):
            body_btn_layout.addWidget(b, i // 2, i % 2)
        thresh_group_layout.addLayout(body_btn_layout)

        min_size_layout = QHBoxLayout()
        min_size_layout.addWidget(self.min_body_size_label)
        min_size_layout.addWidget(self.min_body_size_spinbox)
        min_size_layout.addStretch(1)
        thresh_group_layout.addLayout(min_size_layout)
        thresh_group_layout.addWidget(self.filter_body_metrics_checkbox)
        thresh_group_layout.addWidget(self.filter_threshold_mask_checkbox)

        layout.addWidget(self._group_box("Thresholding", thresh_group_layout))

        # --- Morphology ---
        # Optional, more expensive shape/network analysis (Aspect
        # Ratio, Form Factor, Branch/Junction counts). Sits right after
        # Thresholding since it reads the same body labeling, and
        # before ROI Tools since it's an enrichment on the base
        # analysis rather than a required step. Enable the actual
        # metrics via Output Selection -> Morphology Metrics; the
        # controls here are just for visualizing the skeleton/junctions
        # once computed.
        morph_group_layout = QVBoxLayout()
        morph_note = QLabel(
            "Enable Morphology Shape/Network metrics in Output "
            "Selection first."
        )
        morph_note.setWordWrap(True)
        morph_group_layout.addWidget(morph_note)
        junction_merge_layout = QHBoxLayout()
        junction_merge_layout.addWidget(QLabel("Merge junctions within:"))
        junction_merge_layout.addWidget(self.junction_merge_spinbox)
        morph_group_layout.addLayout(junction_merge_layout)
        skel_btn_layout = QGridLayout()
        for i in range(self.max_channels_supported):
            skel_btn_layout.addWidget(self.show_skeleton_btns[i], i, 0)
            skel_btn_layout.addWidget(self.show_junction_btns[i], i, 1)
        morph_group_layout.addLayout(skel_btn_layout)
        layout.addWidget(self._group_box("Morphology", morph_group_layout))

        # --- ROI Tools ---
        roi_group_layout = QVBoxLayout()
        roi_group_layout.addWidget(self.roi_button)
        roi_group_layout.addWidget(self.per_shape_checkbox)
        roi_group_layout.addWidget(self.sequential_label_checkbox)
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_roi_button)
        nav_layout.addWidget(self.roi_nav_label)
        nav_layout.addWidget(self.next_roi_button)
        nav_layout.addStretch(1)
        roi_group_layout.addLayout(nav_layout)
        layout.addWidget(self._group_box("ROI Tools", roi_group_layout))

        # --- Contact Analysis ---
        contact_group_layout = QVBoxLayout()
        ct_layout = QHBoxLayout()
        ct_layout.addWidget(self.ct_label)
        ct_layout.addWidget(self.ct_slider)
        ct_layout.addWidget(self.ct_text)
        contact_group_layout.addLayout(ct_layout)
        contact_method_layout = QGridLayout()
        contact_method_layout.addWidget(self.contact_method_label, 0, 0)
        contact_method_layout.addWidget(
            self.contact_method_combo, 0, 1, alignment=Qt.AlignLeft
        )
        contact_method_layout.addWidget(self.contact_focus_label, 1, 0)
        contact_method_layout.addWidget(
            self.contact_focus_combo, 1, 1, alignment=Qt.AlignLeft
        )
        contact_method_layout.setColumnStretch(2, 1)
        contact_group_layout.addLayout(contact_method_layout)

        contact_group_layout.addWidget(self.voxel_calibration_status_label)
        contact_group_layout.addWidget(self.manual_voxel_calibration_checkbox)
        # One per line: side by side the two spinboxes were wider than
        # a napari dock (see the threshold rows above).
        voxel_cal_layout = QGridLayout()
        voxel_cal_layout.addWidget(QLabel("Z step:"), 0, 0)
        voxel_cal_layout.addWidget(
            self.z_step_spinbox, 0, 1, alignment=Qt.AlignLeft
        )
        voxel_cal_layout.addWidget(QLabel("XY pixel:"), 1, 0)
        voxel_cal_layout.addWidget(
            self.xy_pixel_spinbox, 1, 1, alignment=Qt.AlignLeft
        )
        voxel_cal_layout.setColumnStretch(2, 1)
        contact_group_layout.addLayout(voxel_cal_layout)

        contact_group_layout.addWidget(self.analyze_button)
        contact_group_layout.addWidget(self.show_contacts_button)
        layout.addWidget(
            self._group_box("Contact Analysis", contact_group_layout)
        )

        # --- Metrics & Display Settings ---
        metrics_group_layout = QVBoxLayout()
        metrics_grid = QGridLayout()
        metrics_grid.addWidget(self.output_selection_button, 0, 0)
        metrics_grid.addWidget(self.metric_display_selection_button, 0, 1)
        metrics_grid.addWidget(self.scale_bar_settings_button, 1, 0)
        metrics_grid.addWidget(self.metrics_glossary_button, 1, 1)
        metrics_grid.addWidget(self.display_layers_button, 2, 0)
        metrics_grid.setColumnStretch(2, 1)
        metrics_group_layout.addLayout(metrics_grid)
        metrics_group_layout.addWidget(self.scale_bar_status_label)
        layout.addWidget(
            self._group_box(
                "Metrics && Display Settings", metrics_group_layout
            )
        )

        # --- Results & Analysis Management ---
        results_group_layout = QVBoxLayout()
        results_group_layout.addWidget(self.result_label)
        analysis_name_layout = QHBoxLayout()
        analysis_name_layout.addWidget(self.analysis_name_label)
        analysis_name_layout.addWidget(self.analysis_name_edit)
        results_group_layout.addLayout(analysis_name_layout)
        results_group_layout.addWidget(self.analysis_count_label)
        manage_grid1 = QGridLayout()
        manage_grid1.addWidget(self.add_analysis_button, 0, 0)
        manage_grid1.addWidget(self.clear_last_button, 0, 1)
        manage_grid1.addWidget(self.clear_all_button, 1, 0)
        manage_grid1.setColumnStretch(2, 1)
        results_group_layout.addLayout(manage_grid1)
        layout.addWidget(
            self._group_box(
                "Results && Analysis Management", results_group_layout
            )
        )

        # --- Saving & Export ---
        save_group_layout = QVBoxLayout()
        manage_grid = QGridLayout()
        manage_grid.addWidget(self.save_image_button, 0, 0, 1, 2)
        manage_grid.addWidget(self.save_metrics_button, 1, 0)
        manage_grid.addWidget(self.export_graphpad_button, 1, 1)
        manage_grid.addWidget(self.append_spreadsheet_button, 2, 0)
        manage_grid.addWidget(self.append_graphpad_button, 2, 1)
        manage_grid.setColumnStretch(2, 1)
        save_group_layout.addLayout(manage_grid)
        layout.addWidget(
            self._group_box("Saving && Export", save_group_layout)
        )

        layout.addStretch(1)
        container.setLayout(layout)

        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        # The sections above are laid out narrow enough (2-column button
        # grids, wrapped per-channel threshold rows, shortened labels) to
        # fit napari's default dock width without needing to scroll
        # sideways -- so the horizontal scrollbar is disabled outright
        # rather than left on "as needed".
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    # ---------------- Utility ----------------
    def _sync_thresh_mode_states(self):
        for i in range(self.max_channels_supported):
            mode = self.per_channel_mode[i].currentText()
            self.per_channel_auto[i].setEnabled(mode == "Automatic")
            self.per_channel_manual[i].setEnabled(mode == "Manual")
            self.per_channel_raw[i].setEnabled(mode == THRESH_MODE_RAW)
        self._save_settings()

    def _sync_active_channel_row_states(self):
        """Gray out per-channel threshold rows beyond the active channel
        count (set via the "Channels" combo), so it's obvious at a
        glance which rows actually affect the analysis."""
        n = self._get_active_channel_count()
        for i in range(self.max_channels_supported):
            self.per_channel_row_widgets[i].setEnabled(i < n)

    def _on_filter_threshold_mask_changed(self, _state=None):
        """When the thresholded mask itself is already being filtered
        (the more aggressive toggle), the scoped "Body analyses only"
        toggle has no additional effect -- the mask body-count would
        read from is already clean. Gray it out rather than leave a
        checkbox on screen that silently does nothing, matching how
        inactive channel rows are grayed out elsewhere in this UI."""
        global_on = self.filter_threshold_mask_checkbox.isChecked()
        self.filter_body_metrics_checkbox.setEnabled(not global_on)
        self._save_settings()

    # ---------------- Settings persistence ----------------
    def _settings_snapshot(self) -> Dict[str, Any]:
        """Everything about the current configuration that's safe to
        remember across napari sessions: thresholding choices, which
        metrics to compute/display, and cosmetic preferences.
        Deliberately excludes anything tied to *this* viewer session --
        which image layers map to which channel, the current Z range,
        drawn ROI shapes, stored analyses -- since restoring those
        against a different image next time could silently point the
        analysis at the wrong data."""
        return {
            "version": SETTINGS_SCHEMA_VERSION,
            "output_selection": self.output_selection,
            "enable_intensity_comparisons": self.enable_intensity_comparisons,
            "intensity_comparisons": self.intensity_comparisons,
            "channel_mode_index": self.channel_mode_combo.currentIndex(),
            "use_layer_names": self.use_layer_names_checkbox.isChecked(),
            "per_channel_mode": [
                cb.currentIndex() for cb in self.per_channel_mode
            ],
            "per_channel_auto": [
                cb.currentIndex() for cb in self.per_channel_auto
            ],
            "per_channel_manual": [
                sp.value() for sp in self.per_channel_manual
            ],
            "per_channel_raw": [sp.value() for sp in self.per_channel_raw],
            "contact_threshold": self.threshold,
            "contact_method_index": self.contact_method_combo.currentIndex(),
            "contact_focus_index": self._contact_focus_wanted,
            "auto_adjust_z_range": (
                self.auto_adjust_z_range_checkbox.isChecked()
            ),
            "restrict_signal_z": self.restrict_signal_z_checkbox.isChecked(),
            "restrict_signal_z_channels": self.restrict_signal_z_channels,
            "display_layers_selection": self.display_layers_selection,
            "min_body_size": self.min_body_size_spinbox.value(),
            "junction_merge_px": self.junction_merge_spinbox.value(),
            "filter_body_metrics": (
                self.filter_body_metrics_checkbox.isChecked()
            ),
            "filter_threshold_mask": (
                self.filter_threshold_mask_checkbox.isChecked()
            ),
            "manual_voxel_calibration": (
                self.manual_voxel_calibration_checkbox.isChecked()
            ),
            "z_step_size": self.z_step_spinbox.value(),
            "xy_pixel_size": self.xy_pixel_spinbox.value(),
            "per_shape": self.per_shape_checkbox.isChecked(),
            "sequential_label": self.sequential_label_checkbox.isChecked(),
            "include_scale_bar_in_saved_image": (
                self.include_scale_bar_in_saved_image
            ),
            "saved_scale_bar_length": self.saved_scale_bar_length,
            "saved_scale_bar_unit": self.saved_scale_bar_unit,
            "saved_scale_bar_text_size": self.saved_scale_bar_text_size,
            "metric_display_keys": self._metric_display_keys,
        }

    def _save_settings(self):
        # Called from many small change handlers (checkboxes, combos,
        # dialog "OK" buttons) rather than only on close -- napari
        # doesn't reliably call closeEvent on this widget when the
        # whole application quits, so writing through on every change
        # is the only way this doesn't silently fail to persist.
        try:
            settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
            settings.setValue(
                SETTINGS_KEY, json.dumps(self._settings_snapshot())
            )
        except Exception as e:
            print(f"Warning: could not save widget settings: {e}")

    def _load_settings(self):
        try:
            settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
            raw = settings.value(SETTINGS_KEY, None)
            if not raw:
                return
            data = json.loads(raw)
        except Exception as e:
            print(f"Warning: could not load saved widget settings: {e}")
            return

        try:
            if isinstance(data.get("output_selection"), dict):
                self.output_selection.update(data["output_selection"])

            self.enable_intensity_comparisons = bool(
                data.get(
                    "enable_intensity_comparisons",
                    self.enable_intensity_comparisons,
                )
            )
            if isinstance(data.get("intensity_comparisons"), list):
                self.intensity_comparisons = data["intensity_comparisons"]

            if isinstance(data.get("restrict_signal_z_channels"), list):
                self.restrict_signal_z_channels = [
                    int(c) for c in data["restrict_signal_z_channels"]
                ]

            self.include_scale_bar_in_saved_image = bool(
                data.get(
                    "include_scale_bar_in_saved_image",
                    self.include_scale_bar_in_saved_image,
                )
            )
            self.saved_scale_bar_length = float(
                data.get("saved_scale_bar_length", self.saved_scale_bar_length)
            )
            self.saved_scale_bar_unit = str(
                data.get("saved_scale_bar_unit", self.saved_scale_bar_unit)
            )
            self.saved_scale_bar_text_size = int(
                data.get(
                    "saved_scale_bar_text_size",
                    self.saved_scale_bar_text_size,
                )
            )
            mdk = data.get("metric_display_keys", None)
            self._metric_display_keys = (
                list(mdk) if isinstance(mdk, list) else None
            )

            # Widget updates last: several of these are wired to also
            # call _save_settings() on change, so by the time any of
            # those cascades fire, every plain attribute above is
            # already in its final restored state.
            if "channel_mode_index" in data:
                idx = int(data["channel_mode_index"])
                if 0 <= idx < self.channel_mode_combo.count():
                    self.channel_mode_combo.setCurrentIndex(idx)

            self.use_layer_names_checkbox.setChecked(
                bool(data.get("use_layer_names", False))
            )

            for i, v in enumerate(data.get("per_channel_mode", [])):
                if i < len(self.per_channel_mode):
                    combo = self.per_channel_mode[i]
                    if 0 <= int(v) < combo.count():
                        combo.setCurrentIndex(int(v))

            for i, v in enumerate(data.get("per_channel_auto", [])):
                if i < len(self.per_channel_auto):
                    combo = self.per_channel_auto[i]
                    if 0 <= int(v) < combo.count():
                        combo.setCurrentIndex(int(v))

            for i, v in enumerate(data.get("per_channel_manual", [])):
                if i < len(self.per_channel_manual):
                    self.per_channel_manual[i].setValue(float(v))

            for i, v in enumerate(data.get("per_channel_raw", [])):
                if i < len(self.per_channel_raw):
                    self.per_channel_raw[i].setValue(float(v))

            if "contact_method_index" in data:
                idx = int(data["contact_method_index"])
                if 0 <= idx < self.contact_method_combo.count():
                    self.contact_method_combo.setCurrentIndex(idx)
            if "contact_focus_index" in data:
                idx = int(data["contact_focus_index"])
                if 0 <= idx < self.max_channels_supported:
                    self._contact_focus_wanted = idx
            self._refresh_contact_method_controls()

            if "contact_threshold" in data:
                t = float(
                    np.clip(float(data["contact_threshold"]), 0.0, 100.0)
                )
                self._set_contact_threshold(t, save=False)

            self.auto_adjust_z_range_checkbox.setChecked(
                bool(data.get("auto_adjust_z_range", True))
            )
            self.restrict_signal_z_checkbox.setChecked(
                bool(data.get("restrict_signal_z", False))
            )
            if isinstance(data.get("display_layers_selection"), dict):
                saved = data["display_layers_selection"]
                nmax = self.max_channels_supported
                for key in (
                    "thresholded",
                    "body_labels",
                    "skeleton",
                    "junctions",
                ):
                    row = saved.get(key, [])
                    if isinstance(row, list):
                        padded = [bool(v) for v in row[:nmax]]
                        padded += [False] * (nmax - len(padded))
                        self.display_layers_selection[key] = padded
                self.display_layers_selection["contacts"] = bool(
                    saved.get("contacts", True)
                )
            elif any(
                k in data
                for k in (
                    "show_thresh_after",
                    "show_body_labels_after",
                    "show_skeleton_after",
                    "show_contacts_after",
                )
            ):
                # Migrate the old all-channels "Auto-show ..."
                # checkboxes: each one that was on becomes "all
                # channels on" for that layer type.
                nmax = self.max_channels_supported
                if bool(data.get("show_thresh_after", False)):
                    self.display_layers_selection["thresholded"] = [
                        True
                    ] * nmax
                if bool(data.get("show_body_labels_after", False)):
                    self.display_layers_selection["body_labels"] = [
                        True
                    ] * nmax
                if bool(data.get("show_skeleton_after", False)):
                    self.display_layers_selection["skeleton"] = [True] * nmax
                    self.display_layers_selection["junctions"] = [True] * nmax
                self.display_layers_selection["contacts"] = bool(
                    data.get("show_contacts_after", True)
                )
            if "min_body_size" in data:
                self.min_body_size_spinbox.setValue(
                    int(np.clip(int(data["min_body_size"]), 1, 100000))
                )
            if "junction_merge_px" in data:
                self.junction_merge_spinbox.setValue(
                    float(np.clip(float(data["junction_merge_px"]), 0.0, 50.0))
                )
            self.filter_body_metrics_checkbox.setChecked(
                bool(data.get("filter_body_metrics", True))
            )
            self.filter_threshold_mask_checkbox.setChecked(
                bool(data.get("filter_threshold_mask", False))
            )
            if "z_step_size" in data:
                self.z_step_spinbox.setValue(
                    float(np.clip(float(data["z_step_size"]), 0.0001, 10000.0))
                )
            if "xy_pixel_size" in data:
                self.xy_pixel_spinbox.setValue(
                    float(
                        np.clip(float(data["xy_pixel_size"]), 0.0001, 10000.0)
                    )
                )
            self.manual_voxel_calibration_checkbox.setChecked(
                bool(data.get("manual_voxel_calibration", False))
            )
            self.per_shape_checkbox.setChecked(
                bool(data.get("per_shape", False))
            )
            self.sequential_label_checkbox.setChecked(
                bool(data.get("sequential_label", False))
            )
        except Exception as e:
            print(f"Warning: error applying saved widget settings: {e}")

        # Belt-and-suspenders: guarantees the persisted file reflects
        # everything just restored, regardless of exactly which widget
        # changes above did or didn't trigger their own save via a
        # connected signal.
        self._save_settings()

    def _set_contact_threshold(self, value: float, save: bool = True):
        """Single place that sets the (decimal, px) contact threshold
        and keeps the label, text box and slider in sync. The slider is
        updated with its signals blocked so a typed value such as 1.45
        isn't rounded to the slider's 0.1 px grid by its own echo."""
        value = round(float(np.clip(value, 0.0, 100.0)), 2)
        self.threshold = value
        self.ct_label.setText(f"Threshold (px): {value:g}")
        self.ct_text.setText(f"{value:g}")
        self.ct_slider.blockSignals(True)
        self.ct_slider.setValue(
            int(round(value * self.CT_SLIDER_STEPS_PER_PX))
        )
        self.ct_slider.blockSignals(False)
        self._update_voxel_calibration_status_label()
        if save:
            self._save_settings()

    def slider_changed(self, value):
        self._set_contact_threshold(value / self.CT_SLIDER_STEPS_PER_PX)

    def text_input_changed(self):
        try:
            value = float(self.ct_text.text().replace(",", "."))
        except ValueError:
            return
        self._set_contact_threshold(value)

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
        self._update_voxel_calibration_status_label()

    def _get_z_xy_ratio(
        self, base_layer: Optional["napari.layers.Image"] = None
    ) -> Tuple[float, bool, str]:
        """Determine the Z-step / XY-pixel-size ratio to use when
        weighting the Z axis in the Contact Threshold's distance
        transform. Returns (ratio, is_calibrated, description):
        ``ratio`` is 1.0 (no correction) whenever no real calibration
        is available, so this safely degrades to today's behavior for
        uncalibrated images. ``is_calibrated`` is False in that case,
        so callers/labels can say so plainly rather than implying a
        detected value of "exactly 1.0"."""
        if self.manual_voxel_calibration_checkbox.isChecked():
            z = float(self.z_step_spinbox.value())
            xy = float(self.xy_pixel_spinbox.value())
            ratio = (z / xy) if xy > 0 else 1.0
            return (
                ratio,
                True,
                f"Manual: Z={z:g} / XY={xy:g} (ratio {ratio:.3f})",
            )

        if base_layer is None:
            base_layer = self._get_mapped_base_layer()
        if base_layer is None:
            return 1.0, False, "No image selected."

        try:
            data = np.asarray(base_layer.data)
            while data.ndim > 3:
                data = data[0]
            if data.ndim < 3:
                return 1.0, False, "2D image -- no Z axis to correct."

            scale = np.asarray(base_layer.scale, dtype=float)
            if scale.shape[0] < 3:
                return 1.0, False, "No calibration detected on this layer."
            z, y, x = scale[-3], scale[-2], scale[-1]
            xy = float(np.mean([y, x]))
            if z <= 0 or xy <= 0:
                return 1.0, False, "No calibration detected on this layer."
            if z == 1.0 and xy == 1.0:
                # napari's default, uncalibrated scale -- not a real
                # 1:1 physical measurement, just the absence of one.
                return (
                    1.0,
                    False,
                    "No calibration detected -- treating as isotropic "
                    "(ratio 1.0).",
                )
            ratio = z / xy
            return (
                ratio,
                True,
                f"Detected from image: Z={z:g}, XY={xy:g} "
                f"(ratio {ratio:.3f})",
            )
        except Exception as e:
            return 1.0, False, f"Could not read calibration ({e})."

    def _on_voxel_calibration_changed(self, _state=None):
        manual = self.manual_voxel_calibration_checkbox.isChecked()
        self.z_step_spinbox.setEnabled(manual)
        self.xy_pixel_spinbox.setEnabled(manual)
        self._update_voxel_calibration_status_label()
        self._save_settings()

    def _get_voxel_sizes(self) -> Optional[Tuple[float, float]]:
        """(z_step, xy_pixel) in physical units, from the manual
        calibration or the base layer's scale -- the same sources
        _get_z_xy_ratio uses -- or None when uncalibrated."""
        try:
            if self.manual_voxel_calibration_checkbox.isChecked():
                z = float(self.z_step_spinbox.value())
                xy = float(self.xy_pixel_spinbox.value())
                return (z, xy) if z > 0 and xy > 0 else None
            base_layer = self._get_mapped_base_layer()
            if base_layer is None:
                return None
            scale = np.asarray(base_layer.scale, dtype=float)
            if scale.shape[0] < 3:
                return None
            z = float(scale[-3])
            xy = float(np.mean(scale[-2:]))
            if z <= 0 or xy <= 0 or (z == 1.0 and xy == 1.0):
                return None
            return z, xy
        except Exception:
            return None

    def _update_voxel_calibration_status_label(self):
        """Short multi-line calibration readout under the Contact
        Threshold: the Z/XY ratio and where it came from, what a step
        in Z equals in threshold pixels, and the current threshold in
        physical units. The full description stays in the tooltip."""
        if not hasattr(self, "voxel_calibration_status_label"):
            return
        ratio, calibrated, desc = self._get_z_xy_ratio()
        manual = self.manual_voxel_calibration_checkbox.isChecked()
        lines = []
        if calibrated:
            source = "manual" if manual else "from image"
            lines.append(f"Z/XY ratio: {ratio:.3f} ({source})")
            lines.append(f"Voxel directly above/below = {ratio:.3f} px away")
            sizes = self._get_voxel_sizes()
            if sizes is not None:
                z, xy = sizes
                t = float(getattr(self, "threshold", 0.0))
                lines.append(
                    f"XY px {xy:g} \u00b7 Z step {z:g} \u00b7 "
                    f"threshold \u2248 {t * xy:.3g} \u00b5m"
                )
        else:
            lines.append("Z/XY ratio: 1.000 (no calibration)")
            lines.append("Z steps treated as 1 px")
        lbl = self.voxel_calibration_status_label
        lbl.setText("\n".join(lines))
        lbl.setToolTip(desc)
        lbl.setMinimumHeight(lbl.fontMetrics().lineSpacing() * len(lines) + 4)

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
        self._save_settings()

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
        if keep_idx.size == 0:
            return None
        # Keep the contiguous span from the first to the last plane with
        # signal, including any empty planes in between. Dropping those
        # would make planes that were 2+ steps apart adjacent, joining
        # bodies across the gap and shifting the displayed Z offset.
        return np.arange(keep_idx[0], keep_idx[-1] + 1)

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
        # Kept in sync with the per-channel row label's own
        # setMaximumWidth(150) at creation time (see __init__) -- the
        # row now shares its line with the mode combo, so it needs to
        # stay narrower than it used to when it had the line to itself.
        label_widget.setMaximumWidth(150)
        label_widget.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Preferred
        )
        if text and len(text) > 14:
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
        self._sync_active_channel_row_states()
        self._refresh_contact_method_controls()

    # ---------------- Contacts method (3-4 channels) ----------------
    def _refresh_contact_method_controls(self):
        """Fill the Focus channel list with the active channels' names
        (keeping the current choice where possible) and gray out what
        doesn't apply: both controls with 2 channels, the focus list
        unless the Focus channel method is selected."""
        n = self._get_active_channel_count()
        combo = self.contact_focus_combo
        labels = self.get_channel_labels()
        target = min(max(self._contact_focus_wanted, 0), len(labels) - 1)
        items = [combo.itemText(i) for i in range(combo.count())]
        if items != labels or combo.currentIndex() != target:
            combo.blockSignals(True)
            if items != labels:
                combo.clear()
                combo.addItems(labels)
            combo.setCurrentIndex(target)
            combo.blockSignals(False)
        multi = n >= 3
        focus_on = (
            self.contact_method_combo.currentText() == CONTACT_METHOD_FOCUS
        )
        self.contact_method_label.setEnabled(multi)
        self.contact_method_combo.setEnabled(multi)
        self.contact_focus_label.setEnabled(multi and focus_on)
        self.contact_focus_combo.setEnabled(multi and focus_on)

    def _on_contact_method_changed(self):
        self._refresh_contact_method_controls()
        self._save_settings()

    def _on_contact_focus_changed(self, idx: int):
        if idx >= 0:
            self._contact_focus_wanted = int(idx)
        self._save_settings()

    def _contact_focus_index(self, n: int) -> Optional[int]:
        """Focus channel index for compute_contacts, or None for the
        overlap-based method. Always None with fewer than 3 channels."""
        if n < 3:
            return None
        if self.contact_method_combo.currentText() != CONTACT_METHOD_FOCUS:
            return None
        idx = self.contact_focus_combo.currentIndex()
        return idx if 0 <= idx < n else 0

    def _contact_method_columns(
        self, n: int, ch_labels: List[str]
    ) -> Dict[str, Any]:
        """'Contact Method' column for 3-4 channel analyses, so every
        exported row says which contact definition produced it. Not
        added with 2 channels, where there is only one definition."""
        if n < 3:
            return {}
        focus = self._contact_focus_index(n)
        if focus is None:
            return {"Contact Method": CONTACT_METHOD_OVERLAP}
        return {
            "Contact Method": f"{CONTACT_METHOD_FOCUS} ({ch_labels[focus]})"
        }

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

        # Keep each channel's scaling range so thresholds can be reported
        # in raw intensity too (see threshold_channel).
        self._last_norm_ranges = [normalization_range(s) for s in raw]
        norm = [
            normalize_signal(s, rng)
            for s, rng in zip(raw, self._last_norm_ranges)
        ]

        return raw, norm, base_layer

    def _sanitize_excel_sheet_name(self, name: str) -> str:
        name = str(name) if name is not None else "Sheet"
        name = re.sub(r"[:\\/?*\[\]]", "_", name)
        name = name.strip("'")
        name = name.strip()
        if not name:
            name = "Sheet"
        if len(name) <= 31:
            return name
        # Excel caps sheet names at 31 characters. Per-channel metrics
        # end in "(<channel>)", which plain truncation cut off -- e.g.
        # "Fragmentation Coefficient (Channel 1)" and "(Channel 2)" both
        # became "Fragmentation Coefficient (Chan". Keep the
        # parenthesised suffix and shorten the part before it instead.
        m = re.match(r"^(.*\S)\s*(\([^()]*\))$", name)
        if m and len(m.group(2)) <= 20:
            prefix, suffix = m.group(1), m.group(2)
            budget = 31 - len(suffix) - 1  # 1 for the "~" marker
            return prefix[:budget].rstrip() + "~" + suffix
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

        # Sheet names are capped at 31 characters, so a sheet's name
        # alone can't always carry its full metric name. A "Sheet Index"
        # sheet (written first) maps every sheet to the exact metric it
        # holds, so no sheet is ever ambiguous.
        used_sheet_names = {"Sheet Index", "All_Metrics"}
        sheet_for_metric = {
            metric: self._make_unique_sheet_name(metric, used_sheet_names)
            for metric in metric_cols
        }
        index_df = pd.DataFrame(
            {
                "Sheet": list(sheet_for_metric.values()),
                "Metric": list(sheet_for_metric.keys()),
            }
        )
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            index_df.to_excel(writer, sheet_name="Sheet Index", index=False)
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
                df_metric.to_excel(
                    writer, sheet_name=sheet_for_metric[metric], index=False
                )

            df_all.to_excel(writer, sheet_name="All_Metrics", index=False)

    # ---------------- Popups ----------------
    def open_display_layers_setup(self):
        dlg = DisplayLayersDialog(
            self,
            current_selection=self.display_layers_selection,
            n_channels_max=self.max_channels_supported,
            channel_labels=self.get_channel_labels(),
        )
        if dlg.exec_() == QDialog.Accepted:
            self.display_layers_selection = dlg.get_results()
            self._save_settings()

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
            self._save_settings()

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
            cb = ScrollSafeComboBox()
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
            self._save_settings()
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
        length_spin = ScrollSafeDoubleSpinBox()
        length_spin.setRange(0.01, 1000000.0)
        length_spin.setDecimals(2)
        length_spin.setSingleStep(1.0)
        length_spin.setValue(float(self.saved_scale_bar_length))

        unit_label = QLabel("Unit:")
        unit_combo = ScrollSafeComboBox()
        unit_combo.addItems(["px", "µm"])
        unit_combo.setCurrentText(self.saved_scale_bar_unit)

        text_label = QLabel("Text Size:")
        text_spin = ScrollSafeSpinBox()
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
            self._save_settings()

    def open_metrics_glossary(self):
        dlg = MetricsGlossaryDialog(self)
        dlg.exec_()

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
            self._save_settings()

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

        # Weight the Z axis in the Contact Threshold's distance search
        # relative to XY, so a step in Z isn't silently treated as the
        # same physical distance as an XY pixel when the two differ
        # (see _get_z_xy_ratio). Computed once -- it's a property of
        # the image's calibration, not of any one channel -- and
        # reused for every channel's distance transform below.
        z_xy_ratio, _, _ = self._get_z_xy_ratio(base_layer)

        ranges = getattr(self, "_last_norm_ranges", None) or [None] * n
        thresholds_used: List[Tuple[float, float]] = []
        for i in range(n):
            m, t_scaled, t_raw = threshold_channel(
                raw_signals[i],
                norm_signals[i],
                ranges[i] if i < len(ranges) else None,
                self.per_channel_mode[i].currentText(),
                self.per_channel_auto[i].currentText(),
                self.per_channel_manual[i].value(),
                self.per_channel_raw[i].value(),
            )
            thresholds_used.append((t_scaled, t_raw))
            if self.filter_threshold_mask_checkbox.isChecked():
                # The more aggressive toggle: strip small/noise bodies
                # out of the mask itself, before anything (Signal Area,
                # Intersection, Union, Contact Area, Mean Intensity,
                # Body Count, ...) is computed from it.
                m, _, _, _ = self._filter_small_bodies(
                    m, self.min_body_size_spinbox.value()
                )
            masks.append(m)
            dists.append(contact_distance_map(m, z_xy_ratio))

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

        contacts = compute_contacts(
            masks, dists, self.threshold, focus=self._contact_focus_index(n)
        )
        method_cols = self._contact_method_columns(n, ch_labels)

        # The cutoff actually applied to each channel, on both scales,
        # shown next to the threshold controls and exported with every
        # row -- so it can be compared across images and reported in
        # methods whichever threshold mode was used.
        thr_cols = threshold_columns(ch_labels, thresholds_used)
        for i, (t_scaled, t_raw) in enumerate(thresholds_used):
            if i < len(self.per_channel_thresh_info):
                self.per_channel_thresh_info[i].setText(
                    f"Last run: {t_scaled:.4f} scaled = {t_raw:,.1f} raw"
                )

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
                restricted_contacts = contacts & roi_mask

                metrics = self._compute_metrics_bundle(
                    raw_signals=raw_signals,
                    masks=restricted_masks,
                    contacts=restricted_contacts,
                    ch_labels=ch_labels,
                    roi_poly_data=poly_data,
                    roi_area=roi_area,
                )
                metrics.update(thr_cols)
                metrics.update(method_cols)
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
                ch_labels=ch_labels,
                roi_poly_data=None,
                roi_area=roi_area_full,
            )
            self.last_metrics.update(thr_cols)
            self.last_metrics.update(method_cols)
            self._set_result_text_from_metrics(
                self.last_metrics, prefix="Full image metrics:\n"
            )
            contacts_display = contacts.astype(float)

        self._last_contacts_display = contacts_display

        # Which layers to auto-display, per channel and layer type --
        # set in the "Auto-Display Setup" dialog (DisplayLayersDialog).
        sel = self.display_layers_selection
        thresh_sel = sel.get("thresholded", [])
        body_sel = sel.get("body_labels", [])
        skel_sel = sel.get("skeleton", [])
        junc_sel = sel.get("junctions", [])
        for i in range(n):
            if i < len(thresh_sel) and thresh_sel[i]:
                self.show_thresholded_channel(i)
        for i in range(n):
            if i < len(body_sel) and body_sel[i]:
                self.show_body_labels_channel(i)
        for i in range(n):
            show_skel = i < len(skel_sel) and bool(skel_sel[i])
            show_junc = i < len(junc_sel) and bool(junc_sel[i])
            if show_skel or show_junc:
                self.show_skeleton_channel(
                    i, show_skeleton=show_skel, show_junctions=show_junc
                )

        if sel.get("contacts", True):
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

    @staticmethod
    def _filter_small_bodies(
        mask: np.ndarray, min_size: int
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Label ``mask``'s connected components (bodies) and drop any
        component smaller than ``min_size`` pixels/voxels -- typically
        single-pixel islands left over after thresholding, which would
        otherwise be counted as their own "body" and skew Body Count /
        Average Area per Body / Fragmentation Coefficient.

        Returns (filtered_binary_mask, filtered_labels, body_count,
        total_area), where filtered_labels is relabeled contiguously
        (1..body_count) over the surviving bodies only, and total_area
        is their combined pixel/voxel count (i.e. the "filtered signal
        area" used to keep Average Area per Body internally consistent
        when only the scoped body-metrics filter is active)."""
        labeled, n = ndi_label(mask)
        if n == 0 or min_size <= 1:
            # Nothing to filter: a size-1 (or looser) cutoff keeps every
            # non-empty body, including single pixels, so skip the
            # extra relabeling pass.
            return (
                mask.astype(bool),
                labeled,
                int(n),
                int(np.sum(mask)),
            )
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # background is never a "body"
        keep = sizes >= min_size
        filtered_binary = keep[labeled] & mask.astype(bool)
        if np.any(filtered_binary):
            relabeled, n_kept = ndi_label(filtered_binary)
        else:
            relabeled, n_kept = np.zeros_like(labeled), 0
        return (
            filtered_binary,
            relabeled,
            int(n_kept),
            int(np.sum(filtered_binary)),
        )

    def _labeled_bodies_for_metrics(
        self, mask: np.ndarray
    ) -> Tuple[np.ndarray, int, int, np.ndarray]:
        """Label ``mask`` into bodies, applying the Minimum Body Size
        filter if either the "Apply to Body analyses" or "Also apply to
        thresholded mask" toggle is on -- the same logic Fragmentation
        Metrics uses, factored out here so every body-derived metric
        (Fragmentation, Morphology Shape, Morphology Network) shares one
        consistent definition of "a body". Returns (labels, body_count,
        signal_area, binary_mask): ``labels`` is contiguous (1..body_count)
        and ``binary_mask``/``signal_area`` reflect the surviving bodies
        only."""
        apply_body_filter = (
            self.filter_body_metrics_checkbox.isChecked()
            or self.filter_threshold_mask_checkbox.isChecked()
        )
        if apply_body_filter:
            binary, labels, n_bodies, signal_area = self._filter_small_bodies(
                mask, self.min_body_size_spinbox.value()
            )
        else:
            binary = mask.astype(bool)
            labels, n_bodies = ndi_label(mask)
            signal_area = int(np.sum(mask))
        return labels, int(n_bodies), int(signal_area), binary

    @staticmethod
    def _skeleton_and_junctions(
        binary_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Skeletonize ``binary_mask`` to a 1-pixel-wide medial axis,
        then classify skeleton pixels by neighbor count *within the
        skeleton*, using full/diagonal connectivity (deliberately
        denser than the face-only connectivity used for body labeling,
        since a thinned skeleton can zigzag diagonally and a junction
        can be missed under face-only connectivity): >=3 neighbors
        marks a junction/branch point. Returns (skeleton, junction_mask)
        -- both boolean arrays the same shape as ``binary_mask``. Shared
        by the Morphology Network metrics and the Skeleton layer
        visualization, so what's counted and what's displayed always
        match."""
        # `> 0` is load-bearing. On skimage versions where 3D
        # skeletonize returns uint8 0/255 (the old skeletonize_3d
        # behaviour) rather than bool, leaving it un-coerced breaks
        # everything below, silently:
        #   * convolving 255-valued pixels with a 26-neighbour kernel
        #     overflows uint8 (2*255 = 510 -> 254), so
        #     `neighbor_count >= 3` is true for essentially every
        #     skeleton pixel -- a straight line with no junctions has
        #     all of its pixels flagged as junctions;
        #   * junction_mask then comes out uint8 0/1 rather than bool,
        #     so `skel & ~junction_mask` downstream evaluates
        #     255 & 254 = 254, which is nonzero -- junctions are never
        #     removed and "branches" become whole connected components;
        #   * summing that array inflates every branch length 254x.
        # Verified against real data: fixing this changed Branch Count
        # from 378 to 3529 and Junction Count from 323 to 1482 on an
        # unchanged 36,110-pixel skeleton.
        skel = np.asarray(skeletonize(binary_mask)) > 0
        if not np.any(skel):
            return skel, np.zeros_like(skel, dtype=bool)
        ndim = skel.ndim
        struct = np.ones((3,) * ndim, dtype=int)
        struct[tuple(1 for _ in range(ndim))] = 0
        # int32 accumulator: a 0/1 skeleton cannot overflow it.
        neighbor_count = ndi_convolve(
            skel.astype(np.int32),
            struct.astype(np.int32),
            mode="constant",
            cval=0,
        )
        junction_mask = skel & (neighbor_count >= 3)
        return skel, junction_mask

    @staticmethod
    def _group_junctions(
        skel: np.ndarray,
        junction_mask: np.ndarray,
        merge_px: float = 0.0,
        z_ratio: float = 1.0,
        body_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Group junction pixels into junctions, optionally merging
        nearby ones.

        Step 1 (always): touching junction pixels form one cluster
        (full/diagonal connectivity) -- a single branch point often
        flags several adjacent pixels.

        Step 2 (only if ``merge_px`` > 0): clusters whose centroids lie
        within ``merge_px`` of each other are merged into one junction
        group (single linkage, so chains merge transitively). Distance
        is in XY pixels with Z multiplied by ``z_ratio``. Clusters in
        different bodies are never merged. This is the step Nellie
        performs in ``_clean_junctions``: a medial axis through a body
        wider than a tubule forms ladders and small loops whose rungs
        each add two junctions that are not real branch points.

        Branches are the connected pieces of the skeleton left after
        removing junction pixels. A branch is marked *internal* (and
        dropped from counts) when it touches at least two clusters, all
        of them in the same merged group, and is no longer than
        ``merge_px`` -- i.e. it is a rung inside a merged node.

        Returns a dict: ``n_groups``; ``group_centroids`` (n_groups x
        ndim, pixel coords, size-weighted); ``group_body`` (body label
        per group, 0 if ``body_labels`` is None); ``n_clusters``;
        ``branch_labels`` / ``n_branches`` (all branches, before
        dropping); ``dropped_branches`` (branch labels to exclude)."""
        ndim = junction_mask.ndim
        full = np.ones((3,) * ndim, dtype=int)
        branch_only = skel & ~junction_mask
        branch_labels, n_branches = ndi_label(branch_only, structure=full)
        clusters, n_clusters = ndi_label(junction_mask, structure=full)
        out: Dict[str, Any] = {
            "n_groups": 0,
            "group_centroids": np.zeros((0, ndim), dtype=float),
            "group_body": np.zeros(0, dtype=int),
            "n_clusters": int(n_clusters),
            "branch_labels": branch_labels,
            "n_branches": int(n_branches),
            "dropped_branches": np.zeros(0, dtype=int),
        }
        if n_clusters == 0:
            return out

        ids = np.arange(1, n_clusters + 1)
        centroids = np.atleast_2d(
            np.asarray(
                ndi_center_of_mass(junction_mask, clusters, ids), dtype=float
            )
        ).reshape(n_clusters, ndim)
        sizes = np.atleast_1d(ndi_sum(junction_mask, clusters, ids)).astype(
            float
        )
        if body_labels is not None:
            cbody = np.atleast_1d(
                ndi_maximum(body_labels, clusters, ids)
            ).astype(int)
        else:
            cbody = np.zeros(n_clusters, dtype=int)

        group = np.arange(n_clusters)
        if merge_px > 0 and n_clusters > 1:
            pts = centroids.copy()
            if ndim == 3:
                pts[:, 0] *= float(z_ratio)
            pairs = cKDTree(pts).query_pairs(
                r=float(merge_px), output_type="ndarray"
            )
            if len(pairs):
                pairs = pairs[cbody[pairs[:, 0]] == cbody[pairs[:, 1]]]
            if len(pairs):
                graph = coo_matrix(
                    (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                    shape=(n_clusters, n_clusters),
                )
                _, group = connected_components(graph, directed=False)
        _, group = np.unique(group, return_inverse=True)
        group = group.ravel()
        n_groups = int(group.max()) + 1

        gsum = np.zeros((n_groups, ndim), dtype=float)
        np.add.at(gsum, group, centroids * sizes[:, None])
        gw = np.bincount(group, weights=sizes, minlength=n_groups)
        gbody = np.zeros(n_groups, dtype=int)
        gbody[group] = cbody
        out["n_groups"] = n_groups
        out["group_centroids"] = gsum / gw[:, None]
        out["group_body"] = gbody

        if merge_px > 0 and n_groups < n_clusters and n_branches > 0:
            cluster_group = np.full(n_clusters + 1, -1, dtype=int)
            cluster_group[1:] = group
            jc = np.argwhere(junction_mask)
            jl = clusters[tuple(jc.T)]
            shape = np.asarray(junction_mask.shape)
            b_hits, c_hits = [], []
            for off in np.argwhere(full) - 1:
                if not off.any():
                    continue
                nb = jc + off
                ok = np.all((nb >= 0) & (nb < shape), axis=1)
                bl = branch_labels[tuple(nb[ok].T)]
                hit = bl > 0
                b_hits.append(bl[hit])
                c_hits.append(jl[ok][hit])
            b_hits = np.concatenate(b_hits)
            c_hits = np.concatenate(c_hits)
            if b_hits.size:
                bc = np.unique(np.stack([b_hits, c_hits], axis=1), axis=0)
                n_cl = np.bincount(bc[:, 0], minlength=n_branches + 1)
                bg = np.unique(
                    np.stack([bc[:, 0], cluster_group[bc[:, 1]]], axis=1),
                    axis=0,
                )
                n_gr = np.bincount(bg[:, 0], minlength=n_branches + 1)
                blen = skeleton_branch_lengths(
                    branch_labels, n_branches, z_ratio
                )
                drop = (n_cl >= 2) & (n_gr == 1) & (blen <= merge_px)
                drop[0] = False
                out["dropped_branches"] = np.nonzero(drop)[0]
        return out

    @staticmethod
    def _mean_sd_wmean(
        values: List[float], weights: List[float]
    ) -> Tuple[float, float, float]:
        """Given a list of per-body values (e.g. Aspect Ratio, Branch
        Count) and matching per-body weights (area/volume), return
        (unweighted mean, unweighted SD, area-weighted mean). Every body
        counts once for the first two; the third scales each body's
        contribution by its own size. Returns (0.0, 0.0, 0.0) if
        ``values`` is empty (e.g. no surviving bodies)."""
        if len(values) == 0:
            return 0.0, 0.0, 0.0
        arr = np.asarray(values, dtype=float)
        w = np.asarray(weights, dtype=float)
        mean = float(np.mean(arr))
        sd = float(np.std(arr))
        wsum = float(np.sum(w))
        wmean = float(np.sum(arr * w) / wsum) if wsum > 0 else mean
        return mean, sd, wmean

    def _compute_metrics_bundle(
        self,
        raw_signals: List[np.ndarray],
        masks: List[np.ndarray],
        contacts: np.ndarray,
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

        if self.output_selection.get("Fragmentation Metrics", True):
            # Connected-component count of each channel's thresholded
            # mask ("bodies" of signal). Uses face-connectivity only
            # (scipy.ndimage.label's default structure), i.e.
            # 4-connected in 2D / 6-connected in 3D, so
            # diagonally-touching pixels/voxels count as separate
            # bodies.
            #
            # Average Area per Body = Signal Area / Body Count -- an
            # estimate of average body size (units: pixels/voxels per
            # body).
            #
            # Fragmentation Coefficient = Average Area per Body /
            # Signal Area, which is algebraically just 1 / Body Count,
            # but computed from the two metrics above per the intended
            # definition. Units are 1/Body (unitless). A value near 1
            # means nearly all signal sits in a single body (not
            # fragmented); a value near 0 means the signal is spread
            # across many bodies (highly fragmented).
            #
            # If either the "Apply to Body analyses" or "Also apply to
            # thresholded mask" toggle is on, bodies smaller than the
            # Minimum Body Size (default 2 px/voxels) are excluded from
            # Body Count -- and, so Average Area per Body / Fragmentation
            # Coefficient stay internally consistent with that count,
            # ``signal_area`` here is the *filtered* signal area (the
            # combined size of the surviving bodies only), not the raw
            # Signal Area reported elsewhere. When the mask itself was
            # already filtered upstream (the "thresholded mask" toggle),
            # this is a no-op: masks[i] is already clean.
            for i in range(n):
                _, n_bodies, signal_area, _ = self._labeled_bodies_for_metrics(
                    masks[i]
                )
                avg_area_per_body = (
                    float(signal_area / n_bodies) if n_bodies > 0 else 0.0
                )
                frag_coef = (
                    float(avg_area_per_body / signal_area)
                    if signal_area > 0
                    else 0.0
                )
                out[f"Body Count ({ch_labels[i]})"] = int(n_bodies)
                out[f"Average Area per Body ({ch_labels[i]})"] = (
                    avg_area_per_body
                )
                out[f"Fragmentation Coefficient ({ch_labels[i]})"] = frag_coef

        if self.output_selection.get("Morphology Shape", False):
            # Per-body shape descriptors via skimage.measure.regionprops,
            # computed once per channel (vectorized across every body,
            # not a per-body Python loop). Aspect Ratio = major/minor
            # axis length of each body's best-fit ellipse (the
            # Koopman et al. 2006 convention); Form Factor =
            # perimeter^2 / (4*pi*area), the inverse of circularity --
            # more sensitive to branching/irregularity than AR alone.
            # Bodies use the same body definition (and Minimum Body
            # Size filtering, if enabled) as Fragmentation Metrics.
            #
            # Both an unweighted mean/SD (every body counts once --
            # "what does a typical object look like") and an
            # area-weighted mean (each body counted in proportion to
            # its own pixel/voxel count -- "where does most of the
            # signal mass sit, shape-wise") are reported, since the two
            # can meaningfully disagree; see the Metric Descriptions
            # dialog for why that's informative rather than a
            # contradiction.
            #
            # Form Factor's perimeter term is only defined by
            # regionprops for 2D regions; on a 3D (Z-stack) analysis,
            # Form Factor is reported as NaN rather than a misleading
            # 0.0, while Aspect Ratio (based on the 3D inertia tensor)
            # still works normally.
            #
            # Aspect Ratio is fitted in physical proportions (Z scaled
            # by the Z/XY voxel ratio -- see body_aspect_ratio), so a
            # round body in a Z-stack doesn't read as elongated.
            # Form Factor uses the Crofton perimeter estimate: the
            # default 4-connected perimeter is size-biased (a perfect
            # disk scores 0.82 at r=3 px and 1.08 at r=25 px), so FF
            # would shift whenever bodies get smaller, e.g. on
            # fragmentation. Crofton stays within ~4% of 1 for disks
            # from r=2 px up.
            shape_z_ratio = (
                self._get_z_xy_ratio()[0]
                if masks and masks[0].ndim == 3
                else 1.0
            )
            for i in range(n):
                labels_i, n_bodies_i, _, _ = self._labeled_bodies_for_metrics(
                    masks[i]
                )
                spacing = (
                    (shape_z_ratio, 1.0, 1.0)
                    if labels_i.ndim == 3
                    else (1.0,) * labels_i.ndim
                )
                ar_vals: List[float] = []
                ff_vals: List[float] = []
                ar_areas: List[float] = []
                ff_areas: List[float] = []
                if n_bodies_i > 0:
                    try:
                        for rp in regionprops(labels_i):
                            area_i = float(rp.area)
                            try:
                                ar = body_aspect_ratio(rp.coords, spacing)
                                if ar is not None:
                                    ar_vals.append(ar)
                                    ar_areas.append(area_i)
                            except Exception:
                                pass
                            try:
                                perim = rp.perimeter_crofton
                                if perim is not None and area_i > 0:
                                    ff_vals.append(
                                        float(
                                            (perim**2) / (4 * np.pi * area_i)
                                        )
                                    )
                                    ff_areas.append(area_i)
                            except Exception:
                                pass
                    except Exception as e:
                        print(
                            f"Warning: Morphology Shape computation "
                            f"failed for {ch_labels[i]}: {e}"
                        )

                ar_mean, ar_sd, ar_wmean = self._mean_sd_wmean(
                    ar_vals, ar_areas
                )
                if ff_vals:
                    ff_mean, ff_sd, ff_wmean = self._mean_sd_wmean(
                        ff_vals, ff_areas
                    )
                else:
                    ff_mean = ff_sd = ff_wmean = float("nan")

                out[f"Aspect Ratio Mean ({ch_labels[i]})"] = ar_mean
                out[f"Aspect Ratio SD ({ch_labels[i]})"] = ar_sd
                out[f"Aspect Ratio Weighted Mean ({ch_labels[i]})"] = ar_wmean
                out[f"Form Factor Mean ({ch_labels[i]})"] = ff_mean
                out[f"Form Factor SD ({ch_labels[i]})"] = ff_sd
                out[f"Form Factor Weighted Mean ({ch_labels[i]})"] = ff_wmean

        if self.output_selection.get("Morphology Network", False):
            # Skeleton/graph-based network descriptors. The mask is
            # skeletonized once per channel (not per body) down to a
            # 1-pixel-wide medial axis, then every skeleton pixel is
            # classified by its neighbor count *within the skeleton*,
            # using full/diagonal connectivity (deliberately denser than
            # the face-only connectivity used for body labeling, since a
            # thinned skeleton can zigzag diagonally and a junction can
            # be missed under face-only connectivity): 1 neighbor =
            # endpoint, 2 = mid-branch, >=3 = a junction/branch point.
            #
            # Branches are found by removing junction pixels and
            # relabeling what's left; each surviving fragment is one
            # branch. Branch length is the Euclidean path length through
            # the branch's pixel centres, in XY pixels with Z steps
            # weighted by the Z/XY voxel ratio (skeleton_branch_lengths)
            # -- so it no longer depends on how a branch is oriented.
            #
            # As with Morphology Shape, both an unweighted mean (per
            # body) and an area-weighted mean are reported, plus two
            # reticular-fraction summaries: % of bodies with at least
            # one junction (by object count) and % of signal area
            # sitting in junction-containing bodies (by mass). Treat
            # these as trend indicators, not a hard reticular/fragmented
            # classification -- see the Metric Descriptions dialog for
            # why a naive per-object junction-presence count can be
            # misleading during fragmentation.
            junction_merge_px = float(self.junction_merge_spinbox.value())
            # Used for both junction merging and branch length, so read
            # it whenever the data is 3D (it is ignored in 2D).
            junction_z_ratio = (
                self._get_z_xy_ratio()[0]
                if masks and masks[0].ndim == 3
                else 1.0
            )
            for i in range(n):
                labels_i, n_bodies_i, _, binary_i = (
                    self._labeled_bodies_for_metrics(masks[i])
                )
                branch_counts = np.zeros(n_bodies_i + 1, dtype=np.float64)
                junction_counts = np.zeros(n_bodies_i + 1, dtype=np.float64)
                branch_len_totals = np.zeros(n_bodies_i + 1, dtype=np.float64)
                body_areas = np.zeros(n_bodies_i + 1, dtype=np.float64)

                if n_bodies_i > 0:
                    try:
                        skel, junction_mask = self._skeleton_and_junctions(
                            binary_i
                        )
                        # Junction clusters (touching junction pixels =
                        # one junction), optionally merged by distance
                        # -- see _group_junctions and the "Merge
                        # junctions within" control.
                        jg = self._group_junctions(
                            skel,
                            junction_mask,
                            merge_px=junction_merge_px,
                            z_ratio=junction_z_ratio,
                            body_labels=labels_i,
                        )
                        branch_labels = jg["branch_labels"]
                        n_frags = jg["n_branches"]
                        if n_frags > 0:
                            frag_ids = np.arange(1, n_frags + 1)
                            frag_body = ndi_maximum(
                                labels_i,
                                labels=branch_labels,
                                index=frag_ids,
                            )
                            frag_len = skeleton_branch_lengths(
                                branch_labels, n_frags, junction_z_ratio
                            )[1:]
                            frag_body = np.atleast_1d(frag_body).astype(int)
                            valid = (
                                (frag_body >= 1)
                                & (frag_body <= n_bodies_i)
                                & ~np.isin(frag_ids, jg["dropped_branches"])
                            )
                            np.add.at(branch_counts, frag_body[valid], 1.0)
                            np.add.at(
                                branch_len_totals,
                                frag_body[valid],
                                frag_len[valid],
                            )

                        if jg["n_groups"] > 0:
                            junc_body = jg["group_body"]
                            jvalid = (junc_body >= 1) & (
                                junc_body <= n_bodies_i
                            )
                            np.add.at(
                                junction_counts,
                                junc_body[jvalid],
                                1.0,
                            )

                        areas_full = np.bincount(
                            labels_i.ravel(), minlength=n_bodies_i + 1
                        )
                        body_areas[: len(areas_full)] = areas_full
                    except Exception as e:
                        print(
                            f"Warning: Morphology Network computation "
                            f"failed for {ch_labels[i]}: {e}"
                        )

                b_counts = branch_counts[1 : n_bodies_i + 1]
                j_counts = junction_counts[1 : n_bodies_i + 1]
                b_len_totals = branch_len_totals[1 : n_bodies_i + 1]
                b_areas = body_areas[1 : n_bodies_i + 1]

                with np.errstate(divide="ignore", invalid="ignore"):
                    mean_branch_len_per_body = np.where(
                        b_counts > 0, b_len_totals / b_counts, 0.0
                    )

                bc_mean, _, bc_wmean = self._mean_sd_wmean(
                    list(b_counts), list(b_areas)
                )
                jc_mean, _, jc_wmean = self._mean_sd_wmean(
                    list(j_counts), list(b_areas)
                )
                bl_mean, _, bl_wmean = self._mean_sd_wmean(
                    list(mean_branch_len_per_body), list(b_areas)
                )

                pct_bodies_with_junction = (
                    float(np.sum(j_counts > 0) / n_bodies_i * 100.0)
                    if n_bodies_i > 0
                    else 0.0
                )
                total_area = float(np.sum(b_areas))
                pct_area_with_junction = (
                    float(np.sum(b_areas[j_counts > 0]) / total_area * 100.0)
                    if total_area > 0
                    else 0.0
                )

                out[f"Branch Count Mean ({ch_labels[i]})"] = bc_mean
                out[f"Branch Count Weighted Mean ({ch_labels[i]})"] = bc_wmean
                out[f"Junction Count Mean ({ch_labels[i]})"] = jc_mean
                out[f"Junction Count Weighted Mean ({ch_labels[i]})"] = (
                    jc_wmean
                )
                out[f"Branch Length Mean ({ch_labels[i]})"] = bl_mean
                out[f"Branch Length Weighted Mean ({ch_labels[i]})"] = bl_wmean
                out[f"% Bodies with Junctions ({ch_labels[i]})"] = (
                    pct_bodies_with_junction
                )
                out[
                    f"% Signal Area in Junction-Containing Bodies "
                    f"({ch_labels[i]})"
                ] = pct_area_with_junction

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
            out["Max Feret"] = float(geom[0])
            out["Min Feret"] = float(geom[1])
            out["Shape Perimeter"] = polygon_perimeter(pts2d)
            circ, rnd, sol = compute_roi_shape_descriptors(pts2d)
            out["Circularity"] = circ
            out["Roundness"] = rnd
            out["Solidity"] = sol

        if self.output_selection.get("Contact Spatial", True):
            # Contacts as discrete sites (connected components of the
            # contact mask) -- see compute_contact_sites. Replaces the
            # old "Avg Contact Dist" (pixel-to-pixel spacing, 1.0 for
            # any contiguous contact) and its ratio to Union.
            site_z_ratio = (
                self._get_z_xy_ratio()[0] if contacts.ndim == 3 else 1.0
            )
            n_sites, site_size, site_nn = compute_contact_sites(
                contacts, site_z_ratio
            )
            out["Contact Site Count"] = int(n_sites)
            out["Mean Contact Site Size"] = float(site_size)
            out["Contact Site NN Distance"] = float(site_nn)

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

    def show_body_labels_channel(self, ch_index: int):
        """Show a Labels layer where every separate connected component
        ("body") of the channel's thresholded mask gets its own integer
        label -- and, via napari's default Labels colormap, its own
        distinct color. This is exactly the grouping the Body Count /
        Average Area per Body / Fragmentation Coefficient metrics are
        computed from, made visible so it can be visually spot-checked."""
        if not hasattr(self, "last_masks") or not self.last_masks:
            print(
                "No thresholded data available. Please run an analysis first."
            )
            return
        n = self._get_active_channel_count()
        if ch_index >= n:
            print(f"Channel {ch_index+1} is not active for current analysis.")
            return

        # Match whatever's actually being counted: exclude the same
        # small/noise bodies from the layer that the Minimum Body Size
        # filter excludes from Body Count.
        labeled, n_bodies, _, _ = self._labeled_bodies_for_metrics(
            self.last_masks[ch_index]
        )
        layer_name = (
            f"Body Labels ({self.get_channel_labels()[ch_index]})"
            if self.use_layer_names_checkbox.isChecked()
            else f"Body Labels Ch {ch_index+1}"
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
            lyr.data = labeled
            lyr.scale = layer_scale
            lyr.translate = layer_translate
        else:
            self.viewer.add_labels(
                labeled,
                name=layer_name,
                opacity=0.75,
                scale=layer_scale,
                translate=layer_translate,
            )
        print(
            f"Displayed {n_bodies} body label(s) for channel "
            f"{ch_index + 1}."
        )

    def show_skeleton_channel(
        self,
        ch_index: int,
        show_skeleton: bool = True,
        show_junctions: bool = True,
    ):
        """Show the skeleton/junction layers the Morphology Network
        metrics (Branch Count, Junction Count, Branch Length, %
        reticular fractions) are computed from: a green skeleton
        overlay and a blue junction-point layer, matching the color
        convention used by MiNA/Fiji's Analyze Skeleton so the overlay
        reads intuitively for anyone used to that tool.

        ``show_skeleton`` / ``show_junctions`` independently choose
        which of the two layers is created/updated (the Skeleton Ch N
        and Junction Ch N buttons each request one; Auto-Display Setup
        can request either or both). Skeletonization runs either way."""
        if not show_skeleton and not show_junctions:
            return
        if not hasattr(self, "last_masks") or not self.last_masks:
            print(
                "No thresholded data available. Please run an analysis first."
            )
            return
        n = self._get_active_channel_count()
        if ch_index >= n:
            print(f"Channel {ch_index+1} is not active for current analysis.")
            return

        body_labels_disp, n_bodies, _, binary = (
            self._labeled_bodies_for_metrics(self.last_masks[ch_index])
        )
        if n_bodies == 0:
            print(f"No bodies to skeletonize for channel {ch_index + 1}.")
            return

        try:
            skel, junction_mask = self._skeleton_and_junctions(binary)
        except Exception as e:
            print(f"Warning: skeletonization failed: {e}")
            return

        ch_label = self.get_channel_labels()[ch_index]
        skel_name = (
            f"Skeleton ({ch_label})"
            if self.use_layer_names_checkbox.isChecked()
            else f"Skeleton Ch {ch_index+1}"
        )
        junction_name = (
            f"Junctions ({ch_label})"
            if self.use_layer_names_checkbox.isChecked()
            else f"Junctions Ch {ch_index+1}"
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

        if show_skeleton:
            skel_data = skel.astype(float)
            if skel_name in self.viewer.layers:
                lyr = self.viewer.layers[skel_name]
                lyr.data = skel_data
                lyr.scale = layer_scale
                lyr.translate = layer_translate
            else:
                self.viewer.add_image(
                    skel_data,
                    name=skel_name,
                    colormap="green",
                    blending="additive",
                    opacity=0.9,
                    scale=layer_scale,
                    translate=layer_translate,
                )

        junction_count_shown = None
        if show_junctions:
            # One point per junction (cluster centroid, or merged-group
            # centroid when "Merge junctions within" is set), not per raw
            # flagged pixel -- exactly what the Junction Count metric counts.
            merge_px = float(self.junction_merge_spinbox.value())
            jg = self._group_junctions(
                skel,
                junction_mask,
                merge_px=merge_px,
                z_ratio=self._get_z_xy_ratio()[0] if merge_px > 0 else 1.0,
                body_labels=body_labels_disp,
            )
            junction_coords = jg["group_centroids"]
            if merge_px > 0:
                print(
                    f"Junction merging ({merge_px:g} px): "
                    f"{jg['n_clusters']} clusters -> {jg['n_groups']} "
                    f"junctions; {len(jg['dropped_branches'])} internal "
                    f"branch(es) no longer counted."
                )

            if junction_coords.size and layer_scale is not None:
                scale_arr = np.asarray(layer_scale)
                if scale_arr.shape[0] == junction_coords.shape[1]:
                    junction_coords_world = junction_coords * scale_arr
                    if layer_translate is not None:
                        junction_coords_world = (
                            junction_coords_world + np.asarray(layer_translate)
                        )
                else:
                    junction_coords_world = junction_coords
            else:
                junction_coords_world = junction_coords

            if junction_name in self.viewer.layers:
                lyr = self.viewer.layers[junction_name]
                lyr.data = junction_coords_world
            else:
                self.viewer.add_points(
                    junction_coords_world,
                    name=junction_name,
                    face_color="blue",
                    size=6,
                    opacity=0.9,
                )
            junction_count_shown = int(junction_coords.shape[0])

        shown = []
        if show_skeleton:
            shown.append(f"skeleton ({int(skel.sum())} px)")
        if junction_count_shown is not None:
            shown.append(f"{junction_count_shown} junction(s)")
        print(f"Displayed {' and '.join(shown)} for channel {ch_index + 1}.")

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
