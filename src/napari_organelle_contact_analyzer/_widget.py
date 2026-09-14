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
import json
import re
import warnings

from typing import TYPE_CHECKING, List, Dict, Any, Optional, Tuple, Callable

import napari
from magicgui import magic_factory
from magicgui.widgets import Container, create_widget
from qtpy.QtGui import QIntValidator
from qtpy.QtCore import Qt, QSettings, QEvent, QObject
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
    QAbstractSpinBox,
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
    QTextEdit,
    QProgressBar,
    QApplication,
)

from skimage import filters
from skimage.util import img_as_float
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from scipy.ndimage import (
    distance_transform_edt,
    label as ndi_label,
)
from scipy.spatial import cKDTree
from skimage.draw import polygon
import numpy as np
import pandas as pd
import imageio.v2 as imageio
from scipy.spatial import ConvexHull

# skan builds a proper graph from a skeletonized mask -- one node per
# true junction, correctly separated even when several true crossings
# sit close together -- for the Morphology Network metrics. See
# _skan_network_analysis()'s docstring for why this replaced an
# earlier pixel-proximity clustering approach that could under-count
# junctions in dense, tangled networks.
from skan import Skeleton as SkanSkeleton, summarize as skan_summarize

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
SETTINGS_ORG = "napari-organelle-contact-analyzer"
SETTINGS_APP = "OrganelleContactWidget"
SETTINGS_KEY = "widget_state_json"
SETTINGS_SCHEMA_VERSION = 1


# -----------------------------
# Helpers
# -----------------------------
def compute_contact_density(contacts: np.ndarray, union_mask: np.ndarray):
    """``union_mask`` should be the same union-of-all-channel-masks used
    for the "Union" core overlap metric -- passed in directly rather than
    recomputed here, so this and "Union" always agree and there's only
    one place that mask gets built."""
    coords = np.column_stack(np.where(contacts))
    if coords.shape[0] < 2:
        avg_dist = np.nan
    else:
        from scipy.spatial import KDTree

        tree = KDTree(coords)
        dists, _ = tree.query(coords, k=2)
        avg_dist = np.mean(dists[:, 1])
    cell_area = np.sum(union_mask)
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


# Kept in one place, and in the same grouping as the Output Selection
# dialog, so this stays easy to keep in sync as metrics are added,
# renamed, or removed.
METRIC_GLOSSARY_HTML = """
<h3>Core Overlap Metrics</h3>
<p><b>Intersection</b> &mdash; pixel count where all channels are
simultaneously thresholded-positive (logical AND across channels).</p>
<p><b>Union</b> &mdash; pixel count where at least one channel is
thresholded-positive (logical OR across channels).</p>
<p><b>Intersection/Union (Contact Coefficient)</b> &mdash; the Jaccard
index of the two above, a normalized 0-1 overlap score.</p>
<p><b>Contact Area</b> &mdash; pixel count where a channel's signal sits
within the Contact Threshold distance of every other channel's signal
(a distance-tolerant version of intersection, not a strict AND).</p>
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
<p><i>Tip:</i> in the Thresholding section, "Bodies Ch N" (or "Body
Labels" in Auto-Display Setup, under Metrics && Display Settings)
displays exactly the connected-component groupings these three
metrics are computed from, as a color-coded Labels layer &mdash; each
body gets its own color.</p>
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
<p><i>Fill Holes Up To (px):</i> before either Shape or Network runs,
small fully-enclosed background holes (e.g. thresholding noise inside
an otherwise-solid body) up to this many pixels/voxels are filled in.
Without this, a hole -- even a single noisy pixel -- gets its own ring
in the skeleton and distorts Aspect Ratio/Form Factor. Larger,
presumably real gaps are left untouched. Default is 4; set to 0 to
disable. Body Count, Fragmentation Coefficient, and the Body Labels
layer are never affected by this -- only Shape/Network and the
Skeleton/Junctions visualization use the filled version.</p>
<p><i>Prune Spurs Under (px):</i> Network only. A jagged, pixel-noisy
mask <b>boundary</b> (as opposed to an interior hole) makes skeletonize
trace a short spurious stub off the network for every small bump/notch
along that edge, inflating Branch/Junction Count -- filling holes
doesn't address this, since it's a boundary artifact, not an interior
one. This removes any branch shorter than the given length that skan
classifies as running from a junction to a free end (the classic spur
shape). Branches connecting two real junctions, or a body's entire
skeleton if it has no junctions at all, are left alone regardless of
length. Chosen over eroding/smoothing the mask itself specifically
because erosion risks deleting real thin branches (mitochondrial
tubules can be only a few pixels wide); pruning only ever touches the
skeleton, never the underlying mask, so Body Area/Aspect Ratio/Form
Factor/Body Count are unaffected. Same units as Branch Length
(XY-pixel-equivalent, calibration-aware in 3D). Default is 3; set to 0
to disable.</p>
<p><i>Collapse Bridges Under (px):</i> Network only, off by default.
Prune Spurs removes dangling dead-end stubs; this instead handles a
different artifact shape -- two junctions connected *to each other* by
a very short branch, which a single irregular/wide spot on a jagged
mask boundary can produce (skeletonize resolves it as two closely-spaced
junctions joined by a short bridge, rather than one clean junction).
Any junction-to-junction branch shorter than this length has its two
junctions reported as a single merged junction instead, placed at the
average position of the originals. Unlike Prune Spurs, this never
touches skeleton pixels -- only what's counted: the connecting branch
is dropped from Branch Count/Branch Length, and every other branch
touching either junction is otherwise unaffected. Same units as Branch
Length. 0 disables it (default).</p>
<p><i>Collapse Wide Regions Over (px):</i> Network only, off by
default. Fill Holes, Prune Spurs, and Collapse Bridges all correct
skeletonize *artifacts* on a body that's genuinely filament-shaped
throughout. This instead handles a body that's tubular in most places
but has a locally wide, swollen/globular stretch woven into it --
still part of the same reticular network, not a separate blob --
where skeletonizing that stretch produces a dense maze/cross-hatch
pattern that no amount of hole-filling, pruning, or bridge-collapsing
can clean up, because the pattern isn't a skeleton artifact; it's what
you get from applying a 1-pixel-wide medial-axis to something that
isn't 1 pixel wide anywhere. Local width is measured pixel-by-pixel as
twice the distance-transform value there (the diameter of the biggest
circle/sphere that fits at that point) -- independent of the
skeleton's own topology, and independent of the rest of the body, so a
thin tubule and a swollen stretch on the very same connected body are
judged separately. Every skeleton node inside a patch that exceeds
this threshold is merged into one junction, and every branch entirely
inside that patch is dropped from Branch Count/Branch Length --
but a real branch connecting the patch to the rest of the network is
kept and counted normally, so the network stays connected and the
swollen stretch reads as a single node feeding into it, not a helix of
spurious branches (a patch with only 1-2 real connections is left as
an ordinary point along a path, not counted as a junction -- only 3+
connections make it one). Skeleton pixels are never altered -- see the
"Skeleton (Collapsed)" overlay note below. One caveat: a kept
branch's length can include some of the path it
traced inside the patch before reaching its node there, so Branch
Length may slightly overstate the real external tubule length. Same
units as Branch Length. 0 disables it (default).</p>
<p><i>Collapse Junction Clusters Within (px):</i> Network only, off by
default. A different tool from Collapse Wide Regions Over: that one
flags a patch by how <i>thick</i> the mask is there. This instead
flags a patch by how <i>densely packed</i> real junctions are,
regardless of mask width -- for a convoluted tangle that isn't
actually a wide/swollen blob (normal tubule width throughout), but
still skeletonizes into a maze because many genuinely thin strands are
crammed into a small physical area. If raising Collapse Wide Regions
catches real, healthy tubules right alongside the tangle without ever
isolating just the tangle, that's a sign this control is the better
fit for what you're looking at. Any two skan-identified true junctions
(degree &ge;3) within this distance of each other are merged into a
single junction, regardless of whether a branch directly connects them
or how many hops apart they are in the graph -- so a dense pileup of,
say, 10 junctions within a small radius of one another all collapse
into one. As with Collapse Wide Regions, branches entirely inside a
collapsed cluster are dropped from Branch Count/Branch Length, while a
branch reaching a junction outside the cluster is kept and counted
normally, and skeleton pixels are never touched. Worth knowing: two
real, distinct branch points that just happen to sit physically near
each other -- not because they're part of the same tangle, but because
two separate strands of the network cross nearby -- could get
incorrectly merged; pick a radius small enough that only a genuinely
dense pileup of junctions falls within it, not two isolated crossing
branches. Same units as Branch Length. 0 disables it (default).</p>
<p><i>Collapse Maze Regions Within (px):</i> Network only, off by
default. Neither Collapse Wide Regions (mask thickness) nor Collapse
Junction Clusters (raw spacing between true junctions) is guaranteed
to cleanly separate a maze/crosshatch artifact from real branching --
both properties can overlap between the two in a given image. This
measures something genuinely different: local <i>loop</i> density. A
maze is characterized by many small closed loops packed into a small
area (like a woven mesh); real branching, even where dense, tends to
stay much more tree-like, with far fewer nearby loops. Every
independent loop in the skeleton graph is found (any branch connecting
two nodes already reachable from each other via some other path closes
one); loop endpoints from 2 or more <i>distinct</i> loops that sit
within this radius of each other are merged into one group, the same
way Collapse Junction Clusters merges nearby junctions. A single loop
found on its own -- e.g. one real, biologically meaningful closed
ring-shaped structure -- is deliberately left alone, since the goal is
catching a pileup of loops, not any one real loop. As with the other
three controls, branches entirely inside a merged group are dropped
from Branch Count/Branch Length, branches reaching outside it are kept
and counted normally, and skeleton pixels are never touched. Same
units as Branch Length. 0 disables it (default).</p>
<p><i>"Skeleton (Collapsed)" overlay:</i> a second Skeleton
visualization layer, in magenta, showing what the skeleton looks like
<i>after</i> Collapse Bridges/Collapse Wide Regions/Collapse Junction
Clusters/Collapse Maze Regions are applied -- every branch absorbed
into a merged junction by any of the four is left out, so only the
surviving branches remain (the "Collapse Ch N" button shows this
together with the Junctions layer, so the merged-junction markers sit
alongside the simplified network). The normal green Skeleton layer is
completely unaffected and always shows every pixel regardless. Purely
a display convenience for seeing/tuning what each threshold is doing
to the network's shape -- it has no effect on Branch/Junction Count
itself, which is computed the same way whether or not you're looking
at this layer.</p>
<p><i>Diagnostic tip:</i> if a skeleton still looks tangled after
raising both Fill Holes and Prune Spurs substantially, try raising
Collapse Bridges too. If that meaningfully simplifies it, the tangle
was mostly this artifact; if it barely changes, the network is likely
just genuinely that complex, and the skeleton is representing it
accurately.</p>
<p><b>Aspect Ratio</b> &mdash; major/minor axis length of each body's
best-fit ellipse. Near 1 is circular; higher is more elongated. On a 3D
(Z-stack) analysis, this now accounts for the same Z/XY voxel
calibration ratio used by Contact Threshold (see above), so an object
that's genuinely round in physical space isn't reported as artificially
elongated just because Z and XY pixel sizes differ.</p>
<p><b>Form Factor</b> &mdash; perimeter&sup2; / (4&pi;&times;area), the
inverse of circularity. More sensitive to branching/irregular outlines
than Aspect Ratio alone. Not defined for 3D (Z-stack) bodies, where it
reports as blank/NaN; Aspect Ratio still works in 3D.</p>
<p><b>Branch Count / Junction Count / Branch Length (per body)</b>
&mdash; the mask is skeletonized down to a 1-pixel-wide medial axis,
then analyzed with <a href="https://skeleton-analysis.org">skan</a>,
which builds a proper graph over the skeleton's own pixel adjacency
(one node per true junction, one edge per branch) rather than
classifying pixels individually and clustering nearby ones. This
matters most on dense, tangled networks: two genuinely distinct
junctions sitting close together can have their neighbor-flagged
pixels touch, and a pixel-clustering approach can merge them into one
reported junction -- skan's graph-based approach keeps them correctly
distinct. A junction is a graph node touched by 3 or more branches.
Branch Length is each branch's true physical length as skan computes
it (following the skeleton's actual path, not a straight line), using
the same Z/XY calibration ratio as Contact Threshold and Aspect Ratio
-- not a raw pixel count. Branch Count and Junction Count are
topological (they count objects, not distances), so they were never
affected by voxel calibration in the first place.</p>
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
<p><i>Tip:</i> in the Morphology section, "Skeleton Ch N" (or
"Skeleton"/"Junctions" in Auto-Display Setup, under Metrics &&
Display Settings -- independently toggleable per channel) displays
the skeleton (green) and junctions (blue) these metrics are computed
from &mdash; the same
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
intensity within the Contact Area region.</p>
<p><b>Signal Intensity Comparisons</b> &mdash; your own configured
comparisons (mean intensity of a source channel within a
Union/Intersection/Contacts region, optionally minus another region).
Off by default; only appears if you've set one up via Output
Selection.</p>

<h3>Advanced ROI/Spatial Metrics</h3>
<p><b>Max Distance</b> &mdash; longest pairwise distance between points
on the ROI's convex hull (its Feret diameter).</p>
<p><b>Max Perp Distance</b> &mdash; the widest perpendicular spread
relative to that long axis.</p>
<p><b>Distance Ratio</b> &mdash; Max Distance / Max Perp Distance, an
elongation score.</p>
<p><b>Ellipse Circumference</b> &mdash; the circumference of a
theoretical ellipse with those two distances as its axes.</p>
<p><b>Shape Perimeter</b> &mdash; the ROI polygon's actual measured
perimeter.</p>
<p><b>Circumference/Perimeter Ratio</b> &mdash; the theoretical ellipse
circumference over the real perimeter, a rough roundness/complexity
score.</p>
<p><b>Avg Contact Dist</b> &mdash; average nearest-neighbor distance
between contact-region pixels (a spacing/clustering measure).</p>
<p><b>Avg Contact Dist / Union Signal Area</b> &mdash; Avg Contact Dist
divided by the union-of-all-channels signal area (the same area
reported as "Union" above).</p>
"""


class _NoWheelFilter(QObject):
    """Blocks mouse-wheel events on whatever widget it's installed on,
    so scrolling the panel (e.g. in the containing QScrollArea) doesn't
    also nudge a spin box's value whenever the cursor happens to be
    sitting over one -- a well-known Qt annoyance, since QAbstractSpinBox
    (and QComboBox/QSlider) respond to wheel events by default even
    without focus. Installed once, after init_ui() has built every spin
    box, via self.findChildren(QAbstractSpinBox) -- far less invasive
    than touching each individual QSpinBox(...)/QDoubleSpinBox(...)
    call site across the file, and automatically covers any added
    later."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            event.ignore()
            return True
        return super().eventFilter(obj, event)


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
    """Lets the user configure, in one place, exactly which layers get
    displayed automatically after Analyze runs -- replacing four
    separate "Auto-show ..." checkboxes that were scattered across the
    Thresholding/Morphology/Contact Analysis sections and only offered
    an all-channels-or-none toggle per layer type. This dialog exposes
    the same underlying per-channel display calls
    (show_thresholded_channel / show_body_labels_channel /
    show_skeleton_channel, plus the Contacts layer) as a single grid,
    so e.g. "Ch 1 + Ch 2 thresholded, Bodies Ch 1, Skeleton Ch 3,
    Junctions Ch 3" can be set up as one saved combination -- exactly
    the kind of mixed, per-channel selection the old checkboxes
    couldn't express. Skeleton, Junctions, and Collapsed are
    intentionally three separate columns (not one combined "skeleton
    layers" toggle like the checkbox this dialog replaces), mirroring
    the three independent "Skeleton Ch N"/"Junction Ch N"/
    "Collapse Ch N" buttons in the main UI -- show_skeleton_channel
    accepts independent show_skeleton/show_junctions/show_collapsed
    flags for exactly this reason."""

    ROWS = [
        ("thresholded", "Thresholded"),
        ("body_labels", "Body Labels"),
        ("skeleton", "Skeleton"),
        ("junctions", "Junctions"),
        ("collapsed", "Collapsed"),
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
                QLabel(f"<b>{label}</b>"), 0, col + 1,
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
            "Body Labels/Skeleton/Junctions/Collapsed additionally require "
            "Morphology Shape/Network to be enabled in Output "
            "Selection for that channel's data to exist yet."
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

        self.threshold = 0
        self._last_z_source_signature = None
        self.metrics_list: List[Dict[str, Any]] = []
        self.last_metrics: Any = {}
        self.last_masks: List[np.ndarray] = []
        self.current_roi_index = 0

        # Per-body records (Aspect Ratio, Form Factor, Branch Count,
        # Junction Count, Branch Length -- one row per surviving body
        # per channel), for the "Export Per-Body Data" button. Mirrors
        # metrics_list/last_metrics: last_per_body_rows holds the most
        # recent Analyze run (tagged with an Analysis Name only once
        # "Add Analysis" is clicked), per_body_records accumulates
        # across every "Add Analysis" click, same as metrics_list does
        # for the aggregated metrics.
        self._last_bundle_per_body_rows: List[Dict[str, Any]] = []
        self.last_per_body_rows: List[Dict[str, Any]] = []
        self.per_body_records: List[Dict[str, Any]] = []
        self._last_added_per_body_count = 0

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

        # Which layers auto-display after Analyze, configured via the
        # "Auto-Display Setup" dialog (DisplayLayersDialog) -- one
        # bool per channel (index 0..max_channels_supported-1) for
        # each per-channel layer type, plus a single flag for the
        # combined Contacts layer. Defaults mirror the four checkboxes
        # this replaced: everything off except Contacts.
        self.display_layers_selection: Dict[str, Any] = {
            "thresholded": [False] * self.max_channels_supported,
            "body_labels": [False] * self.max_channels_supported,
            "skeleton": [False] * self.max_channels_supported,
            "junctions": [False] * self.max_channels_supported,
            "collapsed": [False] * self.max_channels_supported,
            "contacts": True,
        }

        # Saved-image scale bar settings
        self.include_scale_bar_in_saved_image = False
        self.saved_scale_bar_length = 10.0
        self.saved_scale_bar_unit = "px"
        self.saved_scale_bar_text_size = 20

        # Thresholded-signal Z restriction settings
        self.restrict_to_signal_z = False
        self.restrict_signal_z_channels: List[int] = []
        # Default (False) trims only the empty Z-slices before the
        # first and after the last signal-containing slice, so the
        # kept range always stays contiguous -- downstream
        # connected-component labeling, regionprops, and
        # skeletonization keep seeing uniformly-spaced data. Opt-in
        # (True) restores the older, more compact behavior that also
        # drops empty slices *between* signal-containing ones, which
        # can weld together structures that were never physically
        # touching. See the Limitations & Caveats reference.
        self.restrict_signal_z_drop_interior = False

        self.ct_label = QLabel(f"Threshold (px): {self.threshold}")
        self.ct_label.setAlignment(Qt.AlignCenter)
        self.ct_label.setToolTip(
            "Contact threshold, in pixels: the maximum distance between "
            "two channels' thresholded signal for them to be counted as "
            "'in contact'."
        )
        self.ct_slider = QSlider(Qt.Horizontal)
        self.ct_slider.setMinimum(0)
        self.ct_slider.setMaximum(100)
        self.ct_slider.setValue(self.threshold)
        self.ct_slider.valueChanged.connect(self.slider_changed)
        self.ct_text = QLineEdit(str(self.threshold))
        self.ct_text.setValidator(QIntValidator(0, 100))
        self.ct_text.editingFinished.connect(self.text_input_changed)

        # Z/XY voxel calibration -- corrects the Contact Threshold's
        # proximity search so a step in Z isn't silently treated as the
        # same physical distance as an XY pixel when the two differ
        # (very common in Z-stack microscopy). "Threshold (px)" keeps
        # its existing meaning for lateral distance; only the Z axis's
        # relative weight in the 3D distance transform changes.
        self.voxel_calibration_status_label = QLabel("")
        self.voxel_calibration_status_label.setWordWrap(True)
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

        self.z_step_spinbox = QDoubleSpinBox()
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

        self.xy_pixel_spinbox = QDoubleSpinBox()
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
        self.channel_mode_combo = QComboBox()
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
        self.z_min_spinbox = QSpinBox()
        self.z_max_spinbox = QSpinBox()
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

            mode_combo = QComboBox()
            mode_combo.addItems(["Automatic", "Manual"])
            self.per_channel_mode.append(mode_combo)

            # Line 1: which channel, and Automatic vs. Manual thresholding.
            line1 = QHBoxLayout()
            line1.addWidget(label)
            line1.addWidget(mode_combo)
            line1.addStretch(1)
            row_outer.addLayout(line1)

            auto_combo = QComboBox()
            auto_combo.addItems(
                ["Otsu", "Li", "Mean", "Minimum", "Triangle", "Yen", "Isodata"]
            )
            self.per_channel_auto.append(auto_combo)
            auto_combo.currentIndexChanged.connect(
                lambda _: self._save_settings()
            )

            manual_spin = QDoubleSpinBox()
            manual_spin.setRange(0.0, 1.0)
            manual_spin.setSingleStep(0.01)
            manual_spin.setValue(0.5)
            manual_spin.setEnabled(False)
            self.per_channel_manual.append(manual_spin)
            manual_spin.valueChanged.connect(lambda _: self._save_settings())

            # Line 2: the Automatic method and Manual value, split onto
            # their own line (rather than crammed alongside line 1) so
            # this fits in a narrow napari dock without a horizontal
            # scrollbar.
            line2 = QHBoxLayout()
            line2.addWidget(QLabel("Auto:"))
            line2.addWidget(auto_combo)
            line2.addWidget(QLabel("Manual:"))
            line2.addWidget(manual_spin)
            row_outer.addLayout(line2)

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

        self.display_layers_button = QPushButton("Auto-Display Setup")
        self.display_layers_button.setToolTip(
            "Choose exactly which layers (per channel: Thresholded, "
            "Body Labels, Skeleton, Junctions; plus Contacts) are shown "
            "automatically after Analyze runs, in any combination."
        )
        self.display_layers_button.clicked.connect(
            self.open_display_layers_setup
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
            "Restrict analyzed Z-stacks to slices with thresholded signal. "
            "By default this only trims empty slices off the top/bottom of "
            "the range, keeping any empty slices in the middle intact -- "
            "see the checkbox below to change that."
        )
        self.restrict_signal_z_checkbox.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed
        )
        self.restrict_signal_z_checkbox.setChecked(False)
        self.restrict_signal_z_checkbox.stateChanged.connect(
            self._on_restrict_signal_z_changed
        )

        self.restrict_signal_z_drop_interior_checkbox = QCheckBox(
            "Also drop interior Z gaps"
        )
        self.restrict_signal_z_drop_interior_checkbox.setToolTip(
            "Off (default): only the empty slices before the first and "
            "after the last signal-containing slice are trimmed, so the "
            "kept Z range always stays one contiguous block. On: every "
            "empty slice is dropped, including gaps between "
            "signal-containing slices -- more compact, but the removed "
            "gap no longer exists as far as Body Count, Morphology Shape, "
            "and Morphology Network are concerned, which can weld "
            "together structures that were never actually touching. "
            "Contact Threshold / Contact Area are unaffected either way."
        )
        self.restrict_signal_z_drop_interior_checkbox.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed
        )
        self.restrict_signal_z_drop_interior_checkbox.setChecked(False)
        self.restrict_signal_z_drop_interior_checkbox.setEnabled(False)
        self.restrict_signal_z_drop_interior_checkbox.stateChanged.connect(
            self._on_restrict_signal_z_drop_interior_changed
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

        self.export_per_body_button = QPushButton("Export Per-Body Data")
        self.export_per_body_button.setToolTip(
            "Export one row per surviving body per channel (Aspect Ratio, "
            "Form Factor, Branch Count, Junction Count, Branch Length, "
            "Area), tagged by Analysis Name and Channel -- raw material "
            "for building your own histograms. Requires Morphology Shape "
            "and/or Morphology Network enabled in Output Selection."
        )
        self.export_per_body_button.clicked.connect(
            self.export_per_body_data
        )

        self.append_spreadsheet_button = QPushButton("Append to Excel")
        self.append_spreadsheet_button.setToolTip(
            "Append to Excel Format"
        )
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
                f"Show the Body Count/Fragmentation \"bodies\" for "
                f"Channel {i+1} as a color-coded Labels layer"
            )
            b.clicked.connect(
                lambda _, idx=i: self.show_body_labels_channel(idx)
            )
            self.show_body_labels_btns.append(b)

        self.min_body_size_label = QLabel("Minimum Body Size (px):")
        self.min_body_size_spinbox = QSpinBox()
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

        self.filter_body_metrics_checkbox = QCheckBox(
            "Apply to Body analyses"
        )
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

        # Three independent buttons per channel rather than one combined
        # "Skeleton Ch N" button -- each toggles exactly one of the
        # three Skeleton visualization layers (raw green skeleton, blue
        # junction points, magenta collapsed-branches overlay) on its
        # own, matching the independent show_skeleton/show_junctions/
        # show_collapsed flags show_skeleton_channel already supports.
        # Button text is deliberately just the layer name ("Skeleton" /
        # "Junction" / "Collapse"), not "<Layer> Ch N" -- the row it
        # sits in (see the skel_btn_layout grid below) already has a
        # "Ch N:" label, and three short single-word buttons side by
        # side stay within the panel's width budget where three
        # "<Layer> Ch N"-length buttons would not (see the QScrollArea
        # note in init_ui's tail for why that budget matters).
        self.show_skeleton_btns: List[QPushButton] = []
        self.show_junction_btns: List[QPushButton] = []
        self.show_collapse_btns: List[QPushButton] = []
        for i in range(self.max_channels_supported):
            skel_b = QPushButton("Skeleton")
            skel_b.setToolTip(
                f"Show only the raw Morphology Network skeleton (green) "
                f"for Channel {i+1}, unmodified -- every skeleton pixel, "
                f"regardless of any Collapse setting."
            )
            skel_b.clicked.connect(
                lambda _, idx=i: self.show_skeleton_channel(
                    idx, show_skeleton=True, show_junctions=False,
                    show_collapsed=False,
                )
            )
            self.show_skeleton_btns.append(skel_b)

            junc_b = QPushButton("Junction")
            junc_b.setToolTip(
                f"Show only the Morphology Network junction points "
                f"(blue) for Channel {i+1}, after any Collapse Bridges/"
                f"Wide Regions/Junction Clusters merging is applied."
            )
            junc_b.clicked.connect(
                lambda _, idx=i: self.show_skeleton_channel(
                    idx, show_skeleton=False, show_junctions=True,
                    show_collapsed=False,
                )
            )
            self.show_junction_btns.append(junc_b)

            collapse_b = QPushButton("Collapse")
            collapse_b.setToolTip(
                f"Show the 'Skeleton (Collapsed)' network for Channel "
                f"{i+1} (magenta lines + the same blue junction points "
                f"as Junction) -- what the skeleton looks like AFTER "
                f"Collapse Bridges/Wide Regions/Junction Clusters/Maze "
                f"Regions are applied: every branch absorbed into a "
                f"merged junction is left out, so only the surviving "
                f"branches and merged junction markers remain. Identical "
                f"to Skeleton if all four collapse settings are 0."
            )
            collapse_b.clicked.connect(
                lambda _, idx=i: self.show_skeleton_channel(
                    idx, show_skeleton=False, show_junctions=True,
                    show_collapsed=True,
                )
            )
            self.show_collapse_btns.append(collapse_b)

        self.max_hole_size_label = QLabel("Fill Holes Up To (px):")
        self.max_hole_size_label.setWordWrap(True)
        self.max_hole_size_label.setMaximumWidth(150)
        self.max_hole_size_label.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Preferred
        )
        self.max_hole_size_spinbox = QSpinBox()
        self.max_hole_size_spinbox.setMinimum(0)
        self.max_hole_size_spinbox.setMaximum(100000)
        self.max_hole_size_spinbox.setValue(4)
        self.max_hole_size_spinbox.setToolTip(
            "Before computing Morphology Shape and Morphology Network "
            "(and before displaying the Skeleton/Junctions layers), fill "
            "any fully-enclosed background hole up to this many "
            "pixels/voxels -- e.g. thresholding noise inside an "
            "otherwise-solid body. Without this, skeletonize traces a "
            "small ring around every such hole, which both looks noisy "
            "and inflates Branch/Junction Count; small holes also distort "
            "Aspect Ratio/Form Factor. Larger, presumably real gaps are "
            "left alone. 0 disables filling. Does not affect Body Count, "
            "Fragmentation Coefficient, or the Body Labels layer, which "
            "all keep using the unfilled mask."
        )
        self.max_hole_size_spinbox.valueChanged.connect(
            lambda _: self._save_settings()
        )

        self.prune_branch_length_label = QLabel("Prune Spurs Under (px):")
        self.prune_branch_length_label.setWordWrap(True)
        self.prune_branch_length_label.setMaximumWidth(150)
        self.prune_branch_length_label.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Preferred
        )
        self.prune_branch_length_spinbox = QDoubleSpinBox()
        self.prune_branch_length_spinbox.setDecimals(1)
        self.prune_branch_length_spinbox.setMinimum(0.0)
        self.prune_branch_length_spinbox.setMaximum(100000.0)
        self.prune_branch_length_spinbox.setValue(3.0)
        self.prune_branch_length_spinbox.setToolTip(
            "Morphology Network only (Shape is unaffected -- it doesn't "
            "use a skeleton). A jagged, pixel-noisy mask boundary makes "
            "skeletonize trace a short spurious stub ('spur') off the "
            "network for every small bump/notch on that boundary, "
            "inflating Branch/Junction Count. This removes any branch "
            "shorter than this length that has exactly one free end "
            "(the classic spur shape: attached to the network at one "
            "end, dangling at the other). Branches connecting two real "
            "junctions, or a body's entire skeleton if it has no "
            "junctions at all, are left alone regardless of length -- "
            "only true dead-end stubs are pruned. Same units as Branch "
            "Length (XY-pixel-equivalent, calibration-aware in 3D). "
            "0 disables pruning. Applied consistently to the Network "
            "metrics and the Skeleton/Junctions visualization."
        )
        self.prune_branch_length_spinbox.valueChanged.connect(
            lambda _: self._save_settings()
        )

        self.collapse_bridge_length_label = QLabel(
            "Collapse Bridges Under (px):"
        )
        self.collapse_bridge_length_label.setWordWrap(True)
        self.collapse_bridge_length_label.setMaximumWidth(150)
        self.collapse_bridge_length_label.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Preferred
        )
        self.collapse_bridge_length_spinbox = QDoubleSpinBox()
        self.collapse_bridge_length_spinbox.setDecimals(1)
        self.collapse_bridge_length_spinbox.setMinimum(0.0)
        self.collapse_bridge_length_spinbox.setMaximum(100000.0)
        self.collapse_bridge_length_spinbox.setValue(0.0)
        self.collapse_bridge_length_spinbox.setToolTip(
            "Morphology Network only. Distinct from Prune Spurs Under: "
            "that removes dangling dead-end stubs, this instead merges "
            "two junctions that are connected to *each other* by a very "
            "short branch into a single junction, whenever that "
            "connecting branch is shorter than this length. Skeletonize "
            "can produce this pattern -- two closely-spaced junctions "
            "joined by a short bridge -- at a single irregular/wide spot "
            "on a jagged mask boundary, which otherwise inflates "
            "Junction Count and Branch Count without being a real "
            "second branch point. Unlike Prune Spurs (which deletes "
            "pixels), this only affects what's counted: the connecting "
            "branch's length is dropped from Branch Length and the two "
            "junctions are reported as one, but the skeleton pixels "
            "themselves -- and every other branch touching either "
            "junction -- are left as-is. If your skeleton stays just as "
            "tangled after raising this a lot, that's an indication the "
            "network's real complexity, not this artifact, is behind "
            "it. Same units as Branch Length (XY-pixel-equivalent, "
            "calibration-aware in 3D). 0 disables collapsing (default)."
        )
        self.collapse_bridge_length_spinbox.valueChanged.connect(
            lambda _: self._save_settings()
        )

        self.max_local_width_label = QLabel(
            "Collapse Wide Regions Over (px):"
        )
        self.max_local_width_label.setWordWrap(True)
        self.max_local_width_label.setMaximumWidth(150)
        self.max_local_width_label.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Preferred
        )
        self.max_local_width_spinbox = QDoubleSpinBox()
        self.max_local_width_spinbox.setDecimals(1)
        self.max_local_width_spinbox.setMinimum(0.0)
        self.max_local_width_spinbox.setMaximum(100000.0)
        self.max_local_width_spinbox.setValue(0.0)
        self.max_local_width_spinbox.setToolTip(
            "Morphology Network only. A fundamentally different problem "
            "from Fill Holes/Prune Spurs/Collapse Bridges: those three "
            "clean up thin-skeleton-level noise (holes, dangling stubs, "
            "kinked junctions) on a body that's genuinely tubular "
            "throughout. This instead handles a body that's tubular in "
            "most places but has a locally wide, swollen/globular "
            "stretch woven into it -- a real reticular network that "
            "just happens to bulge in one spot, not a separate blob. "
            "Skeletonizing that wide stretch produces a dense, "
            "maze-like tangle of short ridges (many interior pixels sit "
            "roughly equidistant from the boundary in several "
            "directions at once), which inflates Branch/Junction Count "
            "as if the network branched dozens of times right there.\n\n"
            "Any patch whose local width (2x the largest inscribed-"
            "circle radius at that point, via a distance transform) "
            "exceeds this value gets every skeleton node inside it "
            "merged into a single junction, and every branch entirely "
            "inside it dropped from Branch Count/Branch Length -- but "
            "real branches connecting that patch to the rest of the "
            "network are kept and counted normally, so the network "
            "stays connected and the swollen stretch reads as one node, "
            "not a helix of spurious branches. (A patch with only 1-2 "
            "real connections is correctly left as an ordinary point "
            "along a path, not counted as a junction at all -- only "
            "3+ connections make it one.) Skeleton pixels themselves "
            "are never altered; the Skeleton visualization's magenta "
            "'Skeleton (Collapsed)' layer (shared with Collapse "
            "Bridges/Collapse Junction Clusters/Collapse Maze Regions) "
            "shows what the skeleton looks like with all of this "
            "collapsing already applied. "
            "One caveat: a kept branch's length "
            "can include a bit of the path it traced inside the patch "
            "before reaching its node there, so Branch Length may "
            "slightly overstate the real external tubule length. Same "
            "units as Branch Length (XY-pixel-equivalent, calibration-"
            "aware in 3D). 0 disables this entirely (default)."
        )
        self.max_local_width_spinbox.valueChanged.connect(
            lambda _: self._save_settings()
        )

        self.collapse_cluster_radius_label = QLabel(
            "Collapse Junction Clusters Within (px):"
        )
        self.collapse_cluster_radius_label.setWordWrap(True)
        self.collapse_cluster_radius_label.setMaximumWidth(150)
        self.collapse_cluster_radius_label.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Preferred
        )
        self.collapse_cluster_radius_spinbox = QDoubleSpinBox()
        self.collapse_cluster_radius_spinbox.setDecimals(1)
        self.collapse_cluster_radius_spinbox.setMinimum(0.0)
        self.collapse_cluster_radius_spinbox.setMaximum(100000.0)
        self.collapse_cluster_radius_spinbox.setValue(0.0)
        self.collapse_cluster_radius_spinbox.setToolTip(
            "Morphology Network only, off by default. A different tool "
            "from Collapse Wide Regions Over: that one flags a patch by "
            "how THICK the mask is there. This instead flags a patch by "
            "how DENSELY PACKED real junctions are, regardless of mask "
            "width -- for a convoluted tangle that isn't actually a "
            "wide/swollen blob (normal tubule width throughout), but "
            "still skeletonizes into a maze because many genuinely thin "
            "strands are crammed into a small physical area. If raising "
            "Collapse Wide Regions catches real, healthy tubules right "
            "alongside the tangle without ever isolating just the "
            "tangle, that's a sign this control is the better fit.\n\n"
            "Any two skan-identified true junctions (degree >=3) within "
            "this distance of each other are merged into a single "
            "junction, regardless of whether a branch directly connects "
            "them or how many hops apart they are in the graph -- so a "
            "dense pileup of, say, 10 junctions within a small radius "
            "of one another all collapse into one. Branches entirely "
            "inside a collapsed cluster are dropped from Branch Count/"
            "Branch Length; branches reaching a junction outside the "
            "cluster are kept and counted normally, same as Collapse "
            "Wide Regions. Risk to be aware of: two real, distinct "
            "branch points that just happen to sit physically near each "
            "other (e.g. two separate strands crossing nearby) -- not "
            "because they're part of the same tangle -- could get "
            "incorrectly merged. Pick a radius small enough that only a "
            "genuinely dense pileup of many junctions falls within it, "
            "not two isolated crossing branches. Skeleton pixels are "
            "never touched. Same units as Branch Length (XY-pixel-"
            "equivalent, calibration-aware in 3D). 0 disables this "
            "entirely (default)."
        )
        self.collapse_cluster_radius_spinbox.valueChanged.connect(
            lambda _: self._save_settings()
        )

        self.collapse_maze_radius_label = QLabel(
            "Collapse Maze Regions Within (px):"
        )
        self.collapse_maze_radius_label.setWordWrap(True)
        self.collapse_maze_radius_label.setMaximumWidth(150)
        self.collapse_maze_radius_label.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Preferred
        )
        self.collapse_maze_radius_spinbox = QDoubleSpinBox()
        self.collapse_maze_radius_spinbox.setDecimals(1)
        self.collapse_maze_radius_spinbox.setMinimum(0.0)
        self.collapse_maze_radius_spinbox.setMaximum(100000.0)
        self.collapse_maze_radius_spinbox.setValue(0.0)
        self.collapse_maze_radius_spinbox.setToolTip(
            "Morphology Network only, off by default. Neither Collapse "
            "Wide Regions (mask thickness) nor Collapse Junction "
            "Clusters (raw spacing between true junctions) reliably "
            "separated a maze/crosshatch artifact from real branching "
            "in testing -- both properties can overlap between the "
            "two. This instead measures something genuinely different: "
            "local LOOP density. A maze pattern is characterized by "
            "many small closed loops packed into a small area (like a "
            "woven mesh); real branching, even when dense, tends to be "
            "much more tree-like with far fewer nearby loops.\n\n"
            "Every independent loop in the skeleton graph is found "
            "(any branch connecting two nodes already connected via "
            "another path closes one). Only where 2 or more distinct "
            "loops sit within this radius of each other -- a genuinely "
            "dense pileup, not a single isolated one -- are all their "
            "nodes merged into one group, the same way Collapse "
            "Junction Clusters merges nearby junctions. A single real "
            "closed loop on its own (e.g. one biologically real ring-"
            "shaped structure) is deliberately left untouched, since "
            "only 1 loop is present there, not a pileup. Branches "
            "entirely inside a merged group are dropped from Branch "
            "Count/Branch Length; branches reaching outside it are "
            "kept and counted normally, same as the other Collapse "
            "controls. Skeleton pixels are never touched. Same units "
            "as Branch Length (XY-pixel-equivalent, calibration-aware "
            "in 3D). 0 disables this entirely (default)."
        )
        self.collapse_maze_radius_spinbox.valueChanged.connect(
            lambda _: self._save_settings()
        )

        self.show_contacts_button = QPushButton("Show Contacts")
        self.show_contacts_button.clicked.connect(self.show_contacts)

        # Analyze runs synchronously on the GUI thread (see
        # analyze_contacts's docstring for why), so this progress bar
        # is driven by periodic QApplication.processEvents() pumps at
        # natural checkpoints (per-channel, per-ROI, per-channel within
        # Morphology Shape/Network) rather than a background worker
        # signal -- those same pumps are also what keeps napari's
        # window repainting and able to accept window-manager focus
        # (e.g. alt-tab) during a long Analyze run, instead of the
        # whole application appearing frozen for its entire duration.
        self.analysis_progress_bar = QProgressBar()
        self.analysis_progress_bar.setVisible(False)
        self.analysis_progress_bar.setTextVisible(True)

        self._metric_display_keys: Optional[List[str]] = None

        self.init_ui()

        # Stop accidental value changes when the user scrolls the
        # panel with the cursor sitting over a spin box -- see
        # _NoWheelFilter. Installed after init_ui() so this picks up
        # every spin box the UI actually built, without needing to
        # touch each individual creation call above.
        self._no_wheel_filter = _NoWheelFilter(self)
        for spinbox in self.findChildren(QAbstractSpinBox):
            spinbox.installEventFilter(self._no_wheel_filter)

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
        # margin-top reserves space above the box's border for the
        # title to sit in; padding-top then keeps the first row of
        # controls from crowding the border beneath it. The title
        # previously used a negative "top" offset to nudge it upward
        # within that margin band -- on top of a bold font's height,
        # that pushed the title's ascenders right to (or past) the very
        # top edge of the box's own allocated space, clipping it,
        # especially once the layout's inter-section spacing was
        # tightened. Left at its default (centered) position within a
        # slightly taller margin-top band instead, which comfortably
        # contains the bold title text without needing a negative
        # offset at all.
        box.setStyleSheet(
            "QGroupBox {"
            "  font-weight: bold;"
            "  margin-top: 14px;"
            "  padding-top: 10px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  subcontrol-position: top left;"
            "  left: 6px;"
            "  padding: 0 3px;"
            "}"
        )
        inner_layout.setContentsMargins(5, 3, 5, 5)
        inner_layout.setSpacing(3)
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
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

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
            self.restrict_signal_z_drop_interior_checkbox
        )
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
        hole_size_layout = QHBoxLayout()
        hole_size_layout.addWidget(self.max_hole_size_label)
        hole_size_layout.addWidget(self.max_hole_size_spinbox)
        hole_size_layout.addStretch(1)
        morph_group_layout.addLayout(hole_size_layout)
        prune_layout = QHBoxLayout()
        prune_layout.addWidget(self.prune_branch_length_label)
        prune_layout.addWidget(self.prune_branch_length_spinbox)
        prune_layout.addStretch(1)
        morph_group_layout.addLayout(prune_layout)
        collapse_layout = QHBoxLayout()
        collapse_layout.addWidget(self.collapse_bridge_length_label)
        collapse_layout.addWidget(self.collapse_bridge_length_spinbox)
        collapse_layout.addStretch(1)
        morph_group_layout.addLayout(collapse_layout)
        max_width_layout = QHBoxLayout()
        max_width_layout.addWidget(self.max_local_width_label)
        max_width_layout.addWidget(self.max_local_width_spinbox)
        max_width_layout.addStretch(1)
        morph_group_layout.addLayout(max_width_layout)
        cluster_radius_layout = QHBoxLayout()
        cluster_radius_layout.addWidget(self.collapse_cluster_radius_label)
        cluster_radius_layout.addWidget(
            self.collapse_cluster_radius_spinbox
        )
        cluster_radius_layout.addStretch(1)
        morph_group_layout.addLayout(cluster_radius_layout)
        maze_radius_layout = QHBoxLayout()
        maze_radius_layout.addWidget(self.collapse_maze_radius_label)
        maze_radius_layout.addWidget(self.collapse_maze_radius_spinbox)
        maze_radius_layout.addStretch(1)
        morph_group_layout.addLayout(maze_radius_layout)
        # A "Ch N:" row label plus one row per channel, 3 columns:
        # Skeleton / Junction / Collapse -- column headers name each
        # layer once instead of repeating it in every button (which is
        # why the buttons themselves just say "Skeleton"/"Junction"/
        # "Collapse", not "Skeleton Ch N" etc. -- see the note where
        # they're created), keeping the row narrow enough to stay
        # within the panel's width budget (see the QScrollArea note in
        # init_ui's tail for why that budget matters).
        skel_btn_layout = QGridLayout()
        for col, text in enumerate(("Skeleton", "Junction", "Collapse")):
            header = QLabel(f"<b>{text}</b>")
            header.setAlignment(Qt.AlignCenter)
            skel_btn_layout.addWidget(header, 0, col + 1)
        for i in range(self.max_channels_supported):
            skel_btn_layout.addWidget(QLabel(f"Ch {i+1}:"), i + 1, 0)
            skel_btn_layout.addWidget(self.show_skeleton_btns[i], i + 1, 1)
            skel_btn_layout.addWidget(self.show_junction_btns[i], i + 1, 2)
            skel_btn_layout.addWidget(self.show_collapse_btns[i], i + 1, 3)
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

        # --- Metrics & Display Settings ---
        # Deliberately sits before Contact Analysis/Analyze: Output
        # Selection determines *what gets computed* when Analyze runs
        # (including whether Morphology Shape/Network run at all, since
        # both default off), so it belongs with the rest of the setup
        # steps, not after the button that consumes it -- otherwise the
        # natural first-time path is "click Analyze, then discover you
        # needed to configure this first, then re-run."
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

        # --- Contact Analysis ---
        contact_group_layout = QVBoxLayout()
        ct_layout = QHBoxLayout()
        ct_layout.addWidget(self.ct_label)
        ct_layout.addWidget(self.ct_slider)
        ct_layout.addWidget(self.ct_text)
        contact_group_layout.addLayout(ct_layout)

        contact_group_layout.addWidget(self.voxel_calibration_status_label)
        contact_group_layout.addWidget(
            self.manual_voxel_calibration_checkbox
        )
        voxel_cal_layout = QHBoxLayout()
        voxel_cal_layout.addWidget(QLabel("Z step:"))
        voxel_cal_layout.addWidget(self.z_step_spinbox)
        voxel_cal_layout.addWidget(QLabel("XY pixel:"))
        voxel_cal_layout.addWidget(self.xy_pixel_spinbox)
        contact_group_layout.addLayout(voxel_cal_layout)

        contact_group_layout.addWidget(self.analyze_button)
        contact_group_layout.addWidget(self.analysis_progress_bar)
        contact_group_layout.addWidget(self.show_contacts_button)
        layout.addWidget(
            self._group_box("Contact Analysis", contact_group_layout)
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
        manage_grid.addWidget(self.export_per_body_button, 3, 0, 1, 2)
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
            if mode == "Automatic":
                self.per_channel_auto[i].setEnabled(True)
                self.per_channel_manual[i].setEnabled(False)
            else:
                self.per_channel_auto[i].setEnabled(False)
                self.per_channel_manual[i].setEnabled(True)
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
            "contact_threshold": self.threshold,
            "auto_adjust_z_range": (
                self.auto_adjust_z_range_checkbox.isChecked()
            ),
            "restrict_signal_z": self.restrict_signal_z_checkbox.isChecked(),
            "restrict_signal_z_channels": self.restrict_signal_z_channels,
            "restrict_signal_z_drop_interior": (
                self.restrict_signal_z_drop_interior_checkbox.isChecked()
            ),
            "display_layers_selection": self.display_layers_selection,
            "max_hole_size": self.max_hole_size_spinbox.value(),
            "prune_branch_length": self.prune_branch_length_spinbox.value(),
            "collapse_bridge_length": (
                self.collapse_bridge_length_spinbox.value()
            ),
            "max_local_width": self.max_local_width_spinbox.value(),
            "collapse_cluster_radius": (
                self.collapse_cluster_radius_spinbox.value()
            ),
            "collapse_maze_radius": (
                self.collapse_maze_radius_spinbox.value()
            ),
            "min_body_size": self.min_body_size_spinbox.value(),
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
                data.get(
                    "saved_scale_bar_length", self.saved_scale_bar_length
                )
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
                if i < len(self.per_channel_mode) and 0 <= int(v) < 2:
                    self.per_channel_mode[i].setCurrentIndex(int(v))

            for i, v in enumerate(data.get("per_channel_auto", [])):
                if i < len(self.per_channel_auto):
                    combo = self.per_channel_auto[i]
                    if 0 <= int(v) < combo.count():
                        combo.setCurrentIndex(int(v))

            for i, v in enumerate(data.get("per_channel_manual", [])):
                if i < len(self.per_channel_manual):
                    self.per_channel_manual[i].setValue(float(v))

            if "contact_threshold" in data:
                t = int(np.clip(int(data["contact_threshold"]), 0, 100))
                self.threshold = t
                self.ct_slider.setValue(t)
                self.ct_text.setText(str(t))
                self.ct_label.setText(f"Threshold (px): {t}")

            self.auto_adjust_z_range_checkbox.setChecked(
                bool(data.get("auto_adjust_z_range", True))
            )
            self.restrict_signal_z_checkbox.setChecked(
                bool(data.get("restrict_signal_z", False))
            )
            self.restrict_signal_z_drop_interior_checkbox.setChecked(
                bool(data.get("restrict_signal_z_drop_interior", False))
            )
            if isinstance(data.get("display_layers_selection"), dict):
                saved = data["display_layers_selection"]
                n = self.max_channels_supported
                for key in (
                    "thresholded", "body_labels", "skeleton", "junctions",
                    "collapsed",
                ):
                    row = saved.get(key, [])
                    if isinstance(row, list):
                        padded = [bool(v) for v in row[:n]]
                        padded += [False] * (n - len(padded))
                        self.display_layers_selection[key] = padded
                self.display_layers_selection["contacts"] = bool(
                    saved.get("contacts", True)
                )
            elif (
                "show_thresh_after" in data
                or "show_body_labels_after" in data
                or "show_skeleton_after" in data
            ):
                # Migrate pre-"Auto-Display Setup" settings: the old
                # checkboxes applied to every channel at once, so carry
                # that forward as "all channels on" for whichever
                # layer types were previously enabled, rather than
                # silently resetting everything to off.
                n = self.max_channels_supported
                if bool(data.get("show_thresh_after", False)):
                    self.display_layers_selection["thresholded"] = [True] * n
                if bool(data.get("show_body_labels_after", False)):
                    self.display_layers_selection["body_labels"] = [True] * n
                if bool(data.get("show_skeleton_after", False)):
                    self.display_layers_selection["skeleton"] = [True] * n
                    self.display_layers_selection["junctions"] = [True] * n
                self.display_layers_selection["contacts"] = bool(
                    data.get("show_contacts_after", True)
                )
            if "max_hole_size" in data:
                self.max_hole_size_spinbox.setValue(
                    int(np.clip(int(data["max_hole_size"]), 0, 100000))
                )
            if "prune_branch_length" in data:
                self.prune_branch_length_spinbox.setValue(
                    float(
                        np.clip(
                            float(data["prune_branch_length"]),
                            0.0,
                            100000.0,
                        )
                    )
                )
            if "collapse_bridge_length" in data:
                self.collapse_bridge_length_spinbox.setValue(
                    float(
                        np.clip(
                            float(data["collapse_bridge_length"]),
                            0.0,
                            100000.0,
                        )
                    )
                )
            if "max_local_width" in data:
                self.max_local_width_spinbox.setValue(
                    float(
                        np.clip(
                            float(data["max_local_width"]), 0.0, 100000.0
                        )
                    )
                )
            if "collapse_cluster_radius" in data:
                self.collapse_cluster_radius_spinbox.setValue(
                    float(
                        np.clip(
                            float(data["collapse_cluster_radius"]),
                            0.0,
                            100000.0,
                        )
                    )
                )
            if "collapse_maze_radius" in data:
                self.collapse_maze_radius_spinbox.setValue(
                    float(
                        np.clip(
                            float(data["collapse_maze_radius"]),
                            0.0,
                            100000.0,
                        )
                    )
                )
            if "min_body_size" in data:
                self.min_body_size_spinbox.setValue(
                    int(np.clip(int(data["min_body_size"]), 1, 100000))
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

    def slider_changed(self, value):
        self.threshold = value
        self.ct_label.setText(f"Threshold (px): {self.threshold}")
        self.ct_text.setText(str(self.threshold))
        self._save_settings()

    def text_input_changed(self):
        try:
            value = int(self.ct_text.text())
        except ValueError:
            return
        value = max(0, min(value, 100))
        self.threshold = value
        self.ct_label.setText(f"Threshold (px): {self.threshold}")
        self.ct_slider.setValue(self.threshold)
        self._save_settings()

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

    def _update_voxel_calibration_status_label(self):
        _, _, desc = self._get_z_xy_ratio()
        self.voxel_calibration_status_label.setText(desc)

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
        self.restrict_signal_z_drop_interior_checkbox.setEnabled(bool(state))
        self._save_settings()

    def _on_restrict_signal_z_drop_interior_changed(self, state):
        self.restrict_signal_z_drop_interior = bool(state)
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
        self,
        masks: List[np.ndarray],
        selected_channels: List[int],
        drop_interior: bool = False,
    ) -> Optional[np.ndarray]:
        """Return the Z indices to keep for "Restrict Z range to
        signal". When ``drop_interior`` is False (the default), only
        the empty slices before the first and after the last
        signal-containing slice are trimmed, so the result is always
        one contiguous range -- downstream connected-component
        labeling, regionprops, and skeletonization keep seeing
        uniformly-spaced data, matching the sampling/spacing those
        now assume. When True, every empty slice is dropped, including
        ones between two signal-containing slices, which is more
        compact but can weld together structures that were never
        physically adjacent (see the Limitations & Caveats reference).
        Contact Threshold / Contact Area are unaffected by this choice
        either way, since their distance transform already runs on the
        full, correctly-spaced volume before this crop is applied."""
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

        if z_keep is None or not np.any(z_keep):
            return None

        nz = np.where(z_keep)[0]
        if drop_interior:
            keep_idx = nz
        else:
            keep_idx = np.arange(int(nz[0]), int(nz[-1]) + 1)

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
            self._save_settings()

    def open_display_layers_setup(self):
        labels = self.get_channel_labels()
        dlg = DisplayLayersDialog(
            self,
            current_selection=self.display_layers_selection,
            n_channels_max=self.max_channels_supported,
            channel_labels=labels,
        )
        if dlg.exec_() == QDialog.Accepted:
            self.display_layers_selection = dlg.get_results()
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
    def _pump_events(self):
        """Let Qt process pending events -- repaints, and critically,
        window-manager events like alt-tab/focus changes -- during a
        long synchronous computation. Analyze runs entirely on the GUI
        thread; without periodic pumps like this one, napari's window
        can't repaint or even accept being switched into for the
        entire duration of a long analysis, which is what "napari
        freezes and is unable to be tabbed into" actually is. Called
        at natural checkpoints (per channel, per ROI, per channel
        within Morphology Shape/Network -- the slowest part on large
        3D stacks) frequently enough that no single gap between calls
        should run more than a couple of seconds on typical data.
        Wrapped defensively since this is cosmetic -- a failure here
        should never break the analysis itself."""
        try:
            QApplication.processEvents()
        except Exception:
            pass

    def _set_analysis_progress(
        self, done: int, total: int, message: str
    ) -> None:
        """Move the progress bar to ``done``/``total`` with ``message``
        as its label, and pump events so the change is actually
        visible immediately rather than queued behind whatever
        computation runs next."""
        total = max(total, 1)
        self.analysis_progress_bar.setMaximum(total)
        self.analysis_progress_bar.setValue(min(done, total))
        self.analysis_progress_bar.setFormat(f"{message} (%p%)")
        self._pump_events()

    def _set_analysis_progress_message(self, message: str) -> None:
        """Update the progress bar's label without moving its value --
        for sub-steps (e.g. "Morphology Network: channel 2/3") nested
        inside a step that's already been given its own tick, where we
        don't have a precise enough total to award partial credit."""
        self.analysis_progress_bar.setFormat(f"{message} (%p%)")
        self._pump_events()

    def analyze_contacts(self):
        """Threshold every active channel, compute contacts, and
        compute every enabled metric (per ROI, if any are drawn and
        "Per Shape" is on; otherwise for the full image).

        This entire method runs on napari's GUI thread rather than a
        background worker. A background-thread version was considered
        (and would be the more thorough fix for the UI freezing during
        a long run), but this codebase's metrics computation reads
        many small pieces of widget state directly (checkboxes,
        spinboxes) throughout -- moving that safely off the GUI thread
        would mean either duplicating all of it into a settings
        snapshot passed into a worker, or accepting the same
        widget-reads-from-a-background-thread risk anyway, and neither
        could be verified against a real napari/Qt runtime in the
        environment this was written in. Instead, self._pump_events()
        is called at every natural checkpoint below (and inside
        _compute_metrics_bundle's per-channel loops), which keeps
        napari's window repainting and able to accept window-manager
        focus throughout -- addressing the actual "frozen, can't tab
        into it" symptom -- while leaving the simpler, already-verified
        single-threaded structure in place. See self.analysis_progress_bar
        for the visible progress this same instrumentation drives."""
        self.analyze_button.setEnabled(False)
        self.analysis_progress_bar.setVisible(True)
        self.analysis_progress_bar.setValue(0)
        try:
            self._analyze_contacts_impl()
        finally:
            self.analyze_button.setEnabled(True)
            self.analysis_progress_bar.setVisible(False)

    def _analyze_contacts_impl(self):
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
            if self.filter_threshold_mask_checkbox.isChecked():
                # The more aggressive toggle: strip small/noise bodies
                # out of the mask itself, before anything (Signal Area,
                # Intersection, Union, Contact Area, Mean Intensity,
                # Body Count, ...) is computed from it.
                m, _, _, _ = self._filter_small_bodies(
                    m, self.min_body_size_spinbox.value()
                )
            masks.append(m)
            sampling = (
                (z_xy_ratio, 1.0, 1.0) if m.ndim == 3 else None
            )
            dists.append(distance_transform_edt(~m, sampling=sampling))
            self._set_analysis_progress(
                i + 1, n, f"Thresholding channel {i + 1}/{n}"
            )

        # Contact Threshold's ``dists`` were already computed above on
        # the full, correctly Z/XY-spaced volume, so this crop can't
        # affect Contact Threshold/Contact Area correctness. By default
        # (restrict_signal_z_drop_interior=False) the kept range stays
        # one contiguous block, so everything computed downstream of
        # this crop -- Body Count, Morphology Shape, Morphology Network
        # -- keeps seeing uniformly-spaced data too. See
        # _compute_signal_restricted_z_indices for the opt-in
        # non-contiguous mode and its tradeoffs.
        if self.restrict_to_signal_z and masks and masks[0].ndim >= 3:
            selected_channels = self._get_default_restrict_signal_z_channels(n)
            keep_idx = self._compute_signal_restricted_z_indices(
                masks, selected_channels,
                drop_interior=self.restrict_signal_z_drop_interior,
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
            all_per_body_rows: List[Dict[str, Any]] = []
            union_contacts = np.zeros_like(masks[0], dtype=bool)
            n_rois = len(roi_polys)

            for idx, poly in enumerate(roi_polys):
                self._set_analysis_progress(
                    idx, n_rois, f"Computing metrics: ROI {idx + 1}/{n_rois}"
                )
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
                    z_xy_ratio=z_xy_ratio,
                    progress_callback=(
                        lambda msg, _idx=idx: self._set_analysis_progress_message(
                            f"ROI {_idx + 1}/{n_rois}: {msg}"
                        )
                    ),
                )
                metrics["ROI Number"] = idx
                per_roi_metrics.append(metrics)
                for row in self._last_bundle_per_body_rows:
                    r = dict(row)
                    r["ROI Number"] = idx
                    all_per_body_rows.append(r)
                union_contacts |= restricted_contacts

            self._set_analysis_progress(
                n_rois, n_rois, f"Computing metrics: {n_rois}/{n_rois} ROIs"
            )
            self.last_metrics = per_roi_metrics
            self.last_per_body_rows = all_per_body_rows
            self.current_roi_index = 0
            self.update_roi_navigation(len(per_roi_metrics))
            self.update_roi_display()
            contacts_display = union_contacts.astype(float)
        else:
            roi_area_full = int(masks[0].size)

            self._set_analysis_progress(0, 1, "Computing metrics")
            self.last_metrics = self._compute_metrics_bundle(
                raw_signals=raw_signals,
                masks=masks,
                contacts=contacts,
                ch_labels=ch_labels,
                roi_poly_data=None,
                roi_area=roi_area_full,
                z_xy_ratio=z_xy_ratio,
                progress_callback=self._set_analysis_progress_message,
            )
            self._set_analysis_progress(1, 1, "Computing metrics")
            self.last_per_body_rows = list(self._last_bundle_per_body_rows)
            self._set_result_text_from_metrics(
                self.last_metrics, prefix="Full image metrics:\n"
            )
            contacts_display = contacts.astype(float)

        self._last_contacts_display = contacts_display

        # Which layers to auto-display, per channel and layer type,
        # configured via the "Auto-Display Setup" dialog
        # (DisplayLayersDialog) -- see self.display_layers_selection.
        sel = self.display_layers_selection
        thresh_sel = sel.get("thresholded", [])
        body_sel = sel.get("body_labels", [])
        skel_sel = sel.get("skeleton", [])
        junc_sel = sel.get("junctions", [])
        collapsed_sel = sel.get("collapsed", [])
        for i in range(n):
            if i < len(thresh_sel) and thresh_sel[i]:
                self.show_thresholded_channel(i)
        for i in range(n):
            if i < len(body_sel) and body_sel[i]:
                self.show_body_labels_channel(i)
        for i in range(n):
            show_skel = i < len(skel_sel) and skel_sel[i]
            show_junc = i < len(junc_sel) and junc_sel[i]
            show_collapsed = i < len(collapsed_sel) and collapsed_sel[i]
            if show_skel or show_junc or show_collapsed:
                self._set_analysis_progress_message(
                    f"Rendering skeleton: channel {i + 1}/{n}"
                )
                self.show_skeleton_channel(
                    i,
                    show_skeleton=show_skel,
                    show_junctions=show_junc,
                    show_collapsed=show_collapsed,
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

    @staticmethod
    def _fill_small_holes(
        binary_mask: np.ndarray, max_hole_size: int
    ) -> np.ndarray:
        """Fill background holes (fully-enclosed pockets of False
        inside a body) up to ``max_hole_size`` pixels/voxels, leaving
        larger holes untouched. ``max_hole_size`` <= 0 is a no-op.

        Motivation: ``skimage.morphology.skeletonize`` computes a true
        topological medial axis, and the medial axis of a shape with a
        hole in it is a closed loop running around that hole. A mask
        riddled with tiny thresholding-noise holes therefore produces
        a skeleton that's mostly small spurious rings around that
        noise rather than the network's real branch structure -- and
        each ring merging into the surrounding skeleton is a genuine
        (if spurious) graph junction as far as any topology-based
        analysis is concerned, inflating Branch/Junction Count. The
        same tiny holes also distort
        Aspect Ratio/Form Factor (they add to a body's perimeter and
        shift its inertia tensor) without representing anything
        biologically meaningful at that scale. Filling them first
        fixes both.

        Background is labeled with full/diagonal connectivity
        (deliberately denser than the face-only connectivity used for
        body labeling) so a hole connected to the array border only
        through a diagonal gap isn't mistaken for a fully-enclosed
        hole. Filling holes never changes which foreground pixels are
        connected to which -- it only adds pixels strictly inside an
        already-connected body -- so body count/labeling/order is
        unaffected; this is safe to apply before Morphology Shape/
        Network without disturbing Body Count or Fragmentation
        Coefficient (which deliberately keep using the unfilled mask).
        Returns a new array; ``binary_mask`` is not modified."""
        if max_hole_size <= 0 or not np.any(binary_mask):
            return binary_mask
        ndim = binary_mask.ndim
        struct = np.ones((3,) * ndim, dtype=int)
        background = ~binary_mask
        bg_labels, n_bg = ndi_label(background, structure=struct)
        if n_bg == 0:
            return binary_mask

        border_labels: set = set()
        for axis in range(ndim):
            for edge in (0, -1):
                slicer = [slice(None)] * ndim
                slicer[axis] = edge
                border_labels.update(
                    np.unique(bg_labels[tuple(slicer)]).tolist()
                )
        border_labels.discard(0)

        sizes = np.bincount(bg_labels.ravel(), minlength=n_bg + 1)
        fill_ids = [
            lbl
            for lbl in range(1, n_bg + 1)
            if lbl not in border_labels and sizes[lbl] <= max_hole_size
        ]
        if not fill_ids:
            return binary_mask

        fill_mask = np.isin(bg_labels, fill_ids)
        filled = binary_mask.copy()
        filled[fill_mask] = True
        return filled

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
    def _wide_region_labels(
        binary_mask: np.ndarray,
        spacing: Tuple[float, ...],
        max_width: float,
    ) -> Optional[np.ndarray]:
        """Label connected patches of ``binary_mask`` that are locally
        too wide to be a filament -- e.g. a genuinely swollen/globular
        stretch of an otherwise-reticular mitochondrial network. Unlike
        an earlier whole-body version of this idea, this operates at
        SUB-body granularity: a single connected body can contain both
        thin tubules (never flagged) and a locally wide, blob-like
        stretch (flagged), because real reticular networks often mix
        both in one connected structure -- flagging the whole body
        would incorrectly discard the thin tubule parts along with it.

        "Local width" at a pixel is 2x the distance-transform value
        there -- the diameter of the largest circle (or sphere, in 3D)
        that fits at that point, i.e. how thick the mask is right there,
        not the body's overall length. ``spacing`` uses the same
        relative Z/XY calibration ratio as everywhere else in
        Morphology Network.

        Returns an int array the same shape as ``binary_mask`` (0 =
        not part of any wide patch, 1..k = which of the k connected
        wide patches a pixel belongs to), or None if ``max_width`` <= 0
        (feature off) or nothing in the mask exceeds it. Consumed by
        ``_skan_network_analysis`` (to union-find-merge every skeleton
        node inside one patch into a single junction, and drop branches
        entirely internal to a patch from Branch Count/Length) and by
        the Skeleton visualization (to highlight which pixels fall
        inside an active collapse patch)."""
        if max_width <= 0 or not np.any(binary_mask):
            return None
        dist = distance_transform_edt(binary_mask, sampling=spacing)
        wide_pixels = (2.0 * dist) > max_width
        if not np.any(wide_pixels):
            return None
        region_labels, n_regions = ndi_label(wide_pixels)
        if n_regions == 0:
            return None
        return region_labels

    def _skan_network_analysis(
        self,
        binary_mask: np.ndarray,
        labels_i: np.ndarray,
        n_bodies_i: int,
        spacing: Tuple[float, ...],
        prune_length: float,
        collapse_length: float = 0.0,
        max_width: float = 0.0,
        cluster_radius: float = 0.0,
        loop_radius: float = 0.0,
    ) -> Dict[str, Any]:
        """Skeletonize ``binary_mask`` and analyze its branch/junction
        network with skan (https://skeleton-analysis.org), replacing
        an earlier hand-rolled approach that classified skeleton
        pixels by neighbor count and then merged nearby
        junction-flagged *pixels* via connected-component clustering.
        That pixel-clustering step could under-count Junction Count on
        dense, tangled networks: two genuinely distinct branch points
        sitting close together (common in a complex reticular network)
        could have their few-pixel "junction blobs" touch and get
        merged into a single reported junction. skan instead builds a
        proper graph over the skeleton's own pixel adjacency
        (``skan.Skeleton``), and ``skan.summarize()`` traces that graph
        into one row per branch with its endpoints, type, and true
        physical length (via ``spacing``) -- so junction identity
        comes from real skeleton topology, not from how close two
        pixel blobs happen to sit.

        A branch "node" here is anything skan's summarize() table
        references as a branch endpoint (``node_id_src``/
        ``node_id_dst``); a node referenced by only 1 branch-endpoint
        slot across the whole table is a free endpoint (degree 1), one
        referenced >=3 times is a true junction (degree >=3). This
        mirrors each node's real degree in the skeleton graph without
        depending on any skan-version-specific "is this a junction"
        column -- only on the src/dst node-id and coordinate columns,
        whose names are looked up defensively across the naming
        conventions skan has used (hyphen-separated pre-0.13,
        underscore-separated 0.13+).

        Spur pruning (``prune_length`` > 0): branches skan classifies
        as branch_type==1 ("junction-to-endpoint", i.e. a spur)
        shorter than ``prune_length`` are removed from the skeleton
        *pixel array* -- except the junction-end pixel itself, which
        other branches may still need -- and skan is re-run once on
        the pruned array so surviving junctions are correctly
        reclassified (a junction that only had that one spur attached,
        plus two real branches, is no longer degree >=3 once the
        spur's connection is gone). Single pass, mirroring the design
        of the pruning step this replaced -- see the Limitations
        reference. ``prune_length`` <= 0 is a no-op.

        Bridge collapsing (``collapse_length`` > 0, applied after any
        spur pruning above): branches skan classifies as
        branch_type==2 ("junction-to-junction") shorter than
        ``collapse_length`` are treated as two junctions that are
        really one -- the classic pattern from a single irregular/wide
        spot on a jagged mask boundary, where skeletonize resolves it
        as two closely-spaced junctions joined by a short bridge
        instead of one clean junction. Unlike spur pruning, this is
        done purely at the graph level via union-find over node ids,
        *not* by touching skeleton pixels: a collapsed bridge is
        excluded from Branch Count/Branch Length (it's absorbed into
        the merged junction, not a real branch), and every junction
        node unioned together by one or more collapsed bridges is
        reported as a single junction, placed at the average position
        of the distinct original nodes in that group -- degree is
        recomputed per group from surviving (non-collapsed) branches
        only, so a group's junction status correctly accounts for the
        bridge(s) it absorbed. The skeleton pixel array itself, and
        every other branch touching either original junction, are left
        completely unchanged -- only what's counted and where the
        merged marker is placed changes. ``collapse_length`` <= 0 is a
        no-op and reproduces the pre-collapse result exactly (every
        node is its own singleton group).

        Wide-region collapsing (``max_width`` > 0, applied alongside
        bridge collapsing above -- both feed the same union-find):
        handles a genuinely reticular network that also contains a
        locally wide, swollen/globular stretch (e.g. a real swollen
        mitochondrion embedded in an otherwise thin, branching network
        -- NOT a separate blob body). Skeletonizing a wide region
        produces a dense maze of many closely-spaced junctions/branches
        that don't correspond to real distinct branch points; via
        ``_wide_region_labels``, every skeleton node whose coordinate
        falls inside the same connected wide patch is union-find-merged
        into one group (regardless of branch length/type, unlike
        bridge collapsing above), and any branch whose *both* endpoints
        fall inside that same patch is absorbed (excluded from Branch
        Count/Branch Length) the same way a collapsed bridge is. A
        branch with only one endpoint inside the patch -- a real
        tubule connecting the swollen region to the rest of the
        network -- is left as a normal, fully-counted branch, so the
        network stays connected and the swollen region is represented
        as a single junction (only counted as one if 3+ such real
        branches survive touching it -- with just 1-2, it's correctly
        treated as an ordinary point along a path, not a branch point).
        As with bridge collapsing, skeleton pixels are never touched --
        only what's counted and where merged junction markers are
        placed. One caveat worth knowing: a surviving branch's length
        still includes whatever distance the raw skeleton traced
        *inside* the patch before reaching its node there, so Branch
        Length for branches touching a collapsed region can slightly
        overstate the real external tubule length. ``max_width`` <= 0
        is a no-op.

        Junction-cluster collapsing (``cluster_radius`` > 0, applied
        alongside both mechanisms above -- all three feed the same
        union-find): handles a convoluted patch that ISN'T locally
        wide by the mask-thickness measure above -- e.g. a dense tangle
        of many genuinely thin strands crammed into a small physical
        area, which skeletonizes into a maze of many true (degree>=3)
        junction nodes sitting close together, without the underlying
        mask ever being especially thick anywhere. Unlike wide-region
        collapsing, this doesn't look at mask width at all: every node
        skan classifies as a real junction (degree >=3 in the raw,
        pre-collapse graph) is found, and any two such junction nodes
        within ``cluster_radius`` of each other (straight-line
        distance, spacing-aware, via a KD-tree over just the junction
        nodes -- not every skeleton pixel) are union-find-merged into
        one group, regardless of whether a branch directly connects
        them or how many hops apart they are in the graph. As with the
        other two mechanisms, any branch whose two endpoints end up in
        the same merged group afterward is absorbed (dropped from
        Branch Count/Branch Length); a branch reaching a node outside
        the group survives normally. This is the right tool when a
        convoluted region and the network's genuinely thin, healthy
        tubules don't separate on width at all (raising ``max_width``
        catches real tubules right along with the tangle) -- density of
        real junction points is a different, often better-separated
        signal in that case. The obvious risk: two real, distinct
        branch points that just happen to sit near each other in
        physical space (not because they're part of the same tangle,
        but because two separate strands of the network cross nearby)
        could get incorrectly merged -- pick a radius small enough that
        only a genuinely dense pileup of many junctions falls within it
        of each other, not two isolated crossing branches. Skeleton
        pixels are never touched here either. ``cluster_radius`` <= 0
        is a no-op.

        Maze/loop-density collapsing (``loop_radius`` > 0, applied
        alongside all three mechanisms above -- all four feed the same
        union-find): handles the case where neither mask width nor raw
        junction spacing reliably separates a maze/crosshatch artifact
        from real branching -- both properties can overlap between the
        two in a given image. This instead measures local LOOP
        density, a different topological signal: a maze is
        characterized by many small closed loops packed into a small
        area (like a woven mesh), whereas real branching -- even where
        it's dense -- tends to stay much more tree-like, with far
        fewer nearby loops. Every independent loop in the (already
        collapsed-so-far) graph is found via a Kruskal-style pass: a
        FRESH union-find processes every branch in order, and any
        branch whose two endpoints are already connected (via some
        other path) closes a loop -- its two endpoints are that loop's
        "loop nodes". Loop nodes from 2 or more DISTINCT loops that lie
        within ``loop_radius`` of each other (spacing-aware, via a
        KD-tree, mirroring cluster_radius's approach) are union-find-
        merged into one group; a single loop found in isolation (no
        other loop nearby -- e.g. one real, biologically meaningful
        closed ring-shaped structure) is deliberately left alone, since
        the point is to catch a *pileup* of loops, not any one loop on
        its own. As with the other mechanisms, a branch is absorbed if
        both endpoints land in the same merged group; skeleton pixels
        are never touched. ``loop_radius`` <= 0 is a no-op.

        All four collapsing mechanisms above feed one shared
        union-find structure, and a branch is absorbed from Branch
        Count/Branch Length if its two endpoints end up in the same
        group by ANY combination of bridge/region/cluster/loop merges
        (checked once, after all unions are applied, rather than
        tracking a separate flag per mechanism) -- so e.g. a chain
        A-collapsed-into-B via a short bridge, B-collapsed-into-C via a
        shared wide region, correctly absorbs a hypothetical direct
        branch from A to C too, not just the original A-B/B-C branches.

        Returns a dict with:
        - "skeleton": the (possibly pruned) skeleton, boolean array
          the same shape as ``binary_mask`` -- for the Skeleton
          visualization layer.
        - "junction_coords": (n_junctions, ndim) pixel-index
          coordinates of each junction node -- for the Junctions
          visualization layer.
        - "collapsed_pixel_mask": boolean array the same shape as
          ``binary_mask`` -- every skeleton pixel belonging to a branch
          absorbed by ANY of the three collapsing mechanisms above
          (bridge, wide-region, or junction-cluster), for the Skeleton
          visualization's diagnostic collapsed-branches overlay.
        - "branch_counts", "junction_counts", "branch_len_totals",
          "body_areas": float arrays of length ``n_bodies_i + 1``
          (index 0 unused, index ``bid`` = body ``bid``), matching
          what _compute_metrics_bundle's Morphology Network block
          expects.

        Any exception here (including a skan API mismatch on an
        unexpectedly old/new installed version) is left to propagate
        to the caller, which already wraps this in a
        try/except-and-warn -- consistent with the rest of Morphology
        Network's error handling."""
        ndim = binary_mask.ndim

        def _col(df, *names):
            for name in names:
                if name in df.columns:
                    return df[name].to_numpy()
            raise KeyError(
                f"None of {names} found in skan summarize() output "
                f"(columns: {list(df.columns)}); skan's column naming "
                f"may have changed -- check the installed skan version."
            )

        def _run_skan(sk_img):
            if not np.any(sk_img):
                return None, None
            sk_obj = SkanSkeleton(sk_img, spacing=spacing)
            if sk_obj.n_paths == 0:
                return sk_obj, None
            try:
                df = skan_summarize(sk_obj, separator="_")
            except TypeError:
                df = skan_summarize(sk_obj)
            return sk_obj, df.reset_index(drop=True)

        branch_counts = np.zeros(n_bodies_i + 1, dtype=np.float64)
        junction_counts = np.zeros(n_bodies_i + 1, dtype=np.float64)
        branch_len_totals = np.zeros(n_bodies_i + 1, dtype=np.float64)
        body_areas = np.zeros(n_bodies_i + 1, dtype=np.float64)
        if n_bodies_i > 0:
            areas_full = np.bincount(
                labels_i.ravel(), minlength=n_bodies_i + 1
            )
            body_areas[: len(areas_full)] = areas_full

        # Computed from the mask, not the skeleton, so it doesn't
        # depend on spur pruning/collapsing below -- a locally wide
        # patch is a property of the underlying shape.
        wide_region_labels = self._wide_region_labels(
            binary_mask, spacing, max_width
        )

        def _zero_result(skel_arr):
            return {
                "skeleton": skel_arr,
                "junction_coords": np.zeros((0, ndim), dtype=float),
                "collapsed_pixel_mask": np.zeros_like(skel_arr, dtype=bool),
                "branch_counts": branch_counts,
                "junction_counts": junction_counts,
                "branch_len_totals": branch_len_totals,
                "body_areas": body_areas,
            }

        if n_bodies_i == 0:
            return _zero_result(np.zeros_like(binary_mask, dtype=bool))

        skel = skeletonize(binary_mask)
        sk_obj, df = _run_skan(skel)
        if df is None:
            return _zero_result(skel)

        # --- optional spur pruning ---
        if prune_length > 0:
            branch_dist = _col(df, "branch_distance", "branch-distance")
            branch_type = _col(df, "branch_type", "branch-type")
            node_src = _col(df, "node_id_src", "node-id-src").astype(int)
            node_dst = _col(df, "node_id_dst", "node-id-dst").astype(int)
            # Junction/endpoint coordinates read straight from the
            # dataframe's own image-coordinate columns, deliberately
            # *not* via sk_obj.coordinates[node_id] -- that array's
            # indexing convention relative to node ids isn't
            # documented clearly enough to trust blindly (skan's own
            # docs note some of its entries "are non-sensical" outside
            # specific access patterns), whereas these columns are
            # exactly what the getting-started tutorial demonstrates.
            src_coord_cols = [
                _col(df, f"image_coord_src_{d}", f"image-coord-src-{d}")
                for d in range(ndim)
            ]
            dst_coord_cols = [
                _col(df, f"image_coord_dst_{d}", f"image-coord-dst-{d}")
                for d in range(ndim)
            ]
            spur_positions = np.where(
                (branch_type == 1) & (branch_dist < prune_length)
            )[0]
            if len(spur_positions) > 0:
                node_counts: Dict[int, int] = {}
                for nid in np.concatenate([node_src, node_dst]):
                    nid = int(nid)
                    node_counts[nid] = node_counts.get(nid, 0) + 1

                remove_mask = np.zeros_like(skel, dtype=bool)
                for pos in spur_positions:
                    path_coords = sk_obj.path_coordinates(int(pos))
                    src_id, dst_id = int(node_src[pos]), int(node_dst[pos])
                    src_is_junction = node_counts.get(src_id, 0) >= 3
                    junction_coord = tuple(
                        int(round(col[pos]))
                        for col in (
                            src_coord_cols if src_is_junction
                            else dst_coord_cols
                        )
                    )
                    for coord in path_coords:
                        pixel = tuple(int(round(c)) for c in coord)
                        if pixel == junction_coord:
                            continue
                        remove_mask[pixel] = True
                skel = skel & ~remove_mask
                sk_obj, df = _run_skan(skel)
                if df is None:
                    return _zero_result(skel)

        # --- extract final branch/junction structure ---
        branch_dist = _col(df, "branch_distance", "branch-distance")
        branch_type = _col(df, "branch_type", "branch-type")
        node_src = _col(df, "node_id_src", "node-id-src").astype(int)
        node_dst = _col(df, "node_id_dst", "node-id-dst").astype(int)
        coord_src = np.stack(
            [
                _col(df, f"image_coord_src_{d}", f"image-coord-src-{d}")
                for d in range(ndim)
            ],
            axis=1,
        )
        coord_dst = np.stack(
            [
                _col(df, f"image_coord_dst_{d}", f"image-coord-dst-{d}")
                for d in range(ndim)
            ],
            axis=1,
        )

        node_coord: Dict[int, Tuple[int, ...]] = {}
        for nid_arr, coord_arr in (
            (node_src, coord_src),
            (node_dst, coord_dst),
        ):
            for row, nid in enumerate(nid_arr):
                nid = int(nid)
                if nid not in node_coord:
                    node_coord[nid] = tuple(
                        int(round(v)) for v in coord_arr[row]
                    )

        # Union-find over node ids: every collapsing mechanism below
        # (bridge, wide-region, junction-cluster) unions node ids into
        # this same structure; a branch is absorbed if its two
        # endpoints end up in the same group by ANY combination of
        # them, checked once at the end (see "unified absorption check"
        # below) rather than tracked per-mechanism -- everything stays
        # its own singleton group, and no branch is ever absorbed, if
        # all three thresholds are 0 (reproduces the pre-collapse
        # result exactly).
        parent: Dict[int, int] = {}

        def _find(x: int) -> int:
            root = x
            while parent.get(root, root) != root:
                root = parent[root]
            while parent.get(x, x) != root:
                parent[x], x = root, parent.get(x, x)
            return root

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb

        # --- bridge collapsing ---
        if collapse_length > 0:
            bridge_eligible = (branch_type == 2) & (
                branch_dist < collapse_length
            )
            for row in np.where(bridge_eligible)[0]:
                _union(int(node_src[row]), int(node_dst[row]))

        shape = labels_i.shape

        # --- wide-region collapsing: every node whose coordinate falls
        # inside the same connected wide patch gets merged together
        # (regardless of branch length/type -- a maze can have many
        # nodes in a patch with no single short bridge directly
        # connecting all of them pairwise). ---
        if wide_region_labels is not None:
            def _region_at(coord: Tuple[int, ...]) -> int:
                if all(0 <= c < s for c, s in zip(coord, shape)):
                    return int(wide_region_labels[coord])
                return 0

            node_region: Dict[int, int] = {
                nid: _region_at(coord) for nid, coord in node_coord.items()
            }
            region_to_nodes: Dict[int, List[int]] = {}
            for nid, rid in node_region.items():
                if rid > 0:
                    region_to_nodes.setdefault(rid, []).append(nid)
            for nids in region_to_nodes.values():
                for nid in nids[1:]:
                    _union(nids[0], nid)

        # --- junction-cluster collapsing: any two real (degree>=3)
        # junction nodes within cluster_radius of each other (straight-
        # line, spacing-aware distance, via a KD-tree over just the
        # junction nodes) get merged -- independent of mask width, and
        # independent of whether a branch directly connects them. ---
        if cluster_radius > 0:
            degree_counts: Dict[int, int] = {}
            for nid in np.concatenate([node_src, node_dst]):
                nid = int(nid)
                degree_counts[nid] = degree_counts.get(nid, 0) + 1
            candidates = [
                nid for nid, deg in degree_counts.items() if deg >= 3
            ]
            if len(candidates) >= 2:
                coords_arr = np.array(
                    [node_coord[nid] for nid in candidates], dtype=float
                )
                scaled = coords_arr * np.asarray(spacing, dtype=float)
                tree = cKDTree(scaled)
                for a, b in tree.query_pairs(r=cluster_radius):
                    _union(candidates[a], candidates[b])

        # --- maze/loop-density collapsing: find every independent loop
        # in the graph (a Kruskal-style pass over a FRESH, separate
        # union-find -- any branch whose endpoints are already
        # connected via some other path closes a loop; its two
        # endpoints are that loop's "loop nodes"), then spatially
        # cluster loop nodes the same way junction-cluster clusters
        # junction nodes -- except a cluster only counts if it pulls
        # together loop nodes from 2+ DISTINCT loops. A single loop
        # found in isolation (nothing else nearby) is deliberately left
        # alone: the goal is catching a pileup of loops, not any one
        # real loop on its own. ---
        if loop_radius > 0 and len(branch_dist) > 0:
            cyc_parent: Dict[int, int] = {}

            def _cyc_find(x: int) -> int:
                root = x
                while cyc_parent.get(root, root) != root:
                    root = cyc_parent[root]
                while cyc_parent.get(x, x) != root:
                    cyc_parent[x], x = root, cyc_parent.get(x, x)
                return root

            def _cyc_union(a: int, b: int) -> bool:
                ra, rb = _cyc_find(a), _cyc_find(b)
                if ra == rb:
                    return False
                cyc_parent[ra] = rb
                return True

            # Kruskal-style pass: every edge that DOESN'T close a loop
            # (i.e. its two endpoints weren't already connected) is a
            # spanning-tree edge, recorded here so each qualifying
            # loop's FULL set of nodes -- not just its closing edge's 2
            # endpoints -- can be reconstructed below via a tree-path
            # walk. loop_edges[k] = (src, dst) of the k-th independent
            # loop's closing branch.
            #
            # Processed SHORTEST-first (true Kruskal order), not in
            # whatever order skan's table happens to list branches:
            # which edges end up "closing" a loop vs. being absorbed
            # into the tree is order-dependent, and shortest-first makes
            # the tree greedily soak up short/local edges before longer
            # ones, so a small local loop's closing edge is preferentially
            # one of ITS OWN short edges (a well-localized representative
            # point for clustering) rather than some longer edge that
            # happens to close a large, spatially-spread-out loop instead.
            row_order = np.argsort(branch_dist)
            tree_adj: Dict[int, List[int]] = {}
            loop_edges: List[Tuple[int, int]] = []
            for row in row_order:
                row = int(row)
                s, d = int(node_src[row]), int(node_dst[row])
                if _cyc_union(s, d):
                    tree_adj.setdefault(s, []).append(d)
                    tree_adj.setdefault(d, []).append(s)
                else:
                    loop_edges.append((s, d))

            if len(loop_edges) >= 2:
                # Spatial representative for each loop = its closing
                # edge's 2 endpoints only (cheap; a reasonable stand-in
                # for "where is this loop" without walking its full
                # path yet -- full paths are only reconstructed below,
                # and only for loops that actually turn out to qualify).
                loop_nodes: List[int] = []
                for s, d in loop_edges:
                    loop_nodes.append(s)
                    loop_nodes.append(d)

                coords_arr = np.array(
                    [node_coord[nid] for nid in loop_nodes], dtype=float
                )
                scaled = coords_arr * np.asarray(spacing, dtype=float)
                tree = cKDTree(scaled)
                pairs = list(tree.query_pairs(r=loop_radius))

                # Scratch union-find over loop_nodes POSITIONS (0..
                # len(loop_nodes)-1), separate from every other
                # union-find here -- only used to figure out which
                # spatial clusters involve 2+ distinct loops before
                # touching the real (shared) union-find at all.
                scratch_parent: Dict[int, int] = {}

                def _scr_find(x: int) -> int:
                    root = x
                    while scratch_parent.get(root, root) != root:
                        root = scratch_parent[root]
                    while scratch_parent.get(x, x) != root:
                        scratch_parent[x], x = root, scratch_parent.get(
                            x, x
                        )
                    return root

                def _scr_union(a: int, b: int) -> None:
                    ra, rb = _scr_find(a), _scr_find(b)
                    if ra != rb:
                        scratch_parent[ra] = rb

                for a, b in pairs:
                    _scr_union(a, b)

                group_loop_ids: Dict[int, set] = {}
                for pos in range(len(loop_nodes)):
                    loop_id = pos // 2
                    g = _scr_find(pos)
                    group_loop_ids.setdefault(g, set()).add(loop_id)

                qualifying_groups = {
                    g for g, ids in group_loop_ids.items() if len(ids) >= 2
                }
                # Link each qualifying loop's closing-edge endpoints to
                # every other nearby qualifying loop's.
                for a, b in pairs:
                    if _scr_find(a) in qualifying_groups:
                        _union(loop_nodes[a], loop_nodes[b])

                # Now expand: pull in EVERY node on each qualifying
                # loop's actual cycle (not just its closing edge's 2
                # endpoints), so e.g. a branch that's part of the loop
                # but isn't the closing edge itself still gets absorbed
                # below -- otherwise most of a collapsed loop's own
                # branches would incorrectly survive as "real."
                qualifying_loop_ids = {
                    loop_id
                    for g in qualifying_groups
                    for loop_id in group_loop_ids[g]
                }

                def _tree_path(a: int, b: int) -> List[int]:
                    # BFS on tree_adj (spanning-tree edges only) from a
                    # to b -- guaranteed to exist and be unique, since a
                    # and b were already connected (via tree edges
                    # alone) by the time this loop's closing edge was
                    # processed above.
                    if a == b:
                        return [a]
                    prev: Dict[int, Optional[int]] = {a: None}
                    queue = [a]
                    qi = 0
                    while qi < len(queue):
                        cur = queue[qi]
                        qi += 1
                        if cur == b:
                            break
                        for nxt in tree_adj.get(cur, ()):
                            if nxt not in prev:
                                prev[nxt] = cur
                                queue.append(nxt)
                    if b not in prev:
                        return [a, b]  # defensive; shouldn't happen
                    path = []
                    node: Optional[int] = b
                    while node is not None:
                        path.append(node)
                        node = prev[node]
                    return path

                for loop_id in qualifying_loop_ids:
                    s, d = loop_edges[loop_id]
                    path_nodes = _tree_path(s, d)
                    for n in path_nodes[1:]:
                        _union(path_nodes[0], n)

        # --- unified absorption check: after every mechanism above has
        # had a chance to union nodes, a branch is absorbed (excluded
        # from Branch Count/Branch Length) if its two endpoints now sit
        # in the same union-find group, no matter which mechanism (or
        # combination/chain of mechanisms) put them there. ---
        if len(branch_dist) > 0:
            collapsed = np.array(
                [
                    _find(int(node_src[row])) == _find(int(node_dst[row]))
                    for row in range(len(branch_dist))
                ],
                dtype=bool,
            )
        else:
            collapsed = np.zeros(0, dtype=bool)

        # Diagnostic-overlay mask: every skeleton pixel belonging to an
        # absorbed branch's own traced path, from ANY of the three
        # collapsing mechanisms above (bridge, wide-region, or
        # junction-cluster combined) -- so the Skeleton visualization's
        # overlay layer always shows exactly what's being folded out of
        # Branch Count/Branch Length right now, regardless of which
        # control(s) are responsible. Built from the same
        # ``sk_obj.path_coordinates`` used by spur pruning above; a
        # branch's own endpoint pixel is included like every other
        # pixel on its path (unlike spur pruning, this never removes
        # pixels from ``skel`` itself, only flags them for display).
        collapsed_pixel_mask = np.zeros_like(skel, dtype=bool)
        for row in np.where(collapsed)[0]:
            for coord in sk_obj.path_coordinates(int(row)):
                pixel = tuple(int(round(c)) for c in coord)
                collapsed_pixel_mask[pixel] = True

        # Branch Count/Length: every *surviving* branch, attributed to
        # a body via its own src coordinate -- a collapsed bridge is
        # absorbed into its merged junction and no longer counted as a
        # branch in its own right.
        for row in range(len(branch_dist)):
            if collapsed[row]:
                continue
            coord = tuple(int(round(v)) for v in coord_src[row])
            if all(0 <= c < s for c, s in zip(coord, shape)):
                body_id = int(labels_i[coord])
                if 1 <= body_id <= n_bodies_i:
                    branch_counts[body_id] += 1.0
                    branch_len_totals[body_id] += float(branch_dist[row])

        # Junction Count: degree recomputed per union-find group from
        # surviving branches only (a collapsed bridge's own two
        # endpoint references don't count toward it), so a group that
        # absorbed a bridge is judged on what's left touching it, not
        # on the bridge itself. A group with >=3 surviving references
        # is a junction, placed at the average position of the
        # distinct original nodes folded into it (a single, unmerged
        # node's "average" is just its own coordinate).
        group_counts: Dict[int, int] = {}
        group_nodes: Dict[int, set] = {}
        for row in range(len(branch_dist)):
            if collapsed[row]:
                continue
            for nid in (int(node_src[row]), int(node_dst[row])):
                gid = _find(nid)
                group_counts[gid] = group_counts.get(gid, 0) + 1
                group_nodes.setdefault(gid, set()).add(nid)

        junction_coords_list = []
        for gid, count in group_counts.items():
            if count < 3:
                continue
            member_coords = [node_coord[nid] for nid in group_nodes[gid]]
            centroid = tuple(
                int(round(float(np.mean([c[d] for c in member_coords]))))
                for d in range(ndim)
            )
            if all(0 <= c < s for c, s in zip(centroid, shape)):
                junction_coords_list.append(centroid)
                body_id = int(labels_i[centroid])
                if 1 <= body_id <= n_bodies_i:
                    junction_counts[body_id] += 1.0

        junction_coords = (
            np.array(junction_coords_list, dtype=float)
            if junction_coords_list
            else np.zeros((0, ndim), dtype=float)
        )

        return {
            "skeleton": skel,
            "junction_coords": junction_coords,
            "collapsed_pixel_mask": collapsed_pixel_mask,
            "branch_counts": branch_counts,
            "junction_counts": junction_counts,
            "branch_len_totals": branch_len_totals,
            "body_areas": body_areas,
        }

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
        z_xy_ratio: float = 1.0,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        n = len(masks)
        out: Dict[str, Any] = {}

        def _report(msg: str) -> None:
            # Optional -- callers outside analyze_contacts (e.g. tests)
            # don't need to supply one. Also pumps the Qt event loop
            # via the callback, which is what keeps napari's window
            # responsive during Morphology Network -- by a wide margin
            # the slowest per-channel loop below on large 3D stacks.
            if progress_callback is not None:
                try:
                    progress_callback(msg)
                except Exception:
                    pass

        # Per-body records (one row per surviving body per channel),
        # populated by the Morphology Shape / Morphology Network blocks
        # below (whichever are enabled). Kept separate from ``out``
        # since ``out`` is one row per analysis/ROI, not one row per
        # body -- callers (analyze_contacts) read this back
        # immediately via self._last_bundle_per_body_rows right after
        # each call, before it gets overwritten by the next channel's
        # or next ROI's call.
        per_body_map: Dict[Tuple[int, int], Dict[str, Any]] = {}

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
                _report(f"Fragmentation: channel {i + 1}/{n}")
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
                out[f"Fragmentation Coefficient ({ch_labels[i]})"] = (
                    frag_coef
                )

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
            # Voxel calibration: on a 3D (Z-stack) analysis, ``spacing``
            # is passed to regionprops as (z_xy_ratio, 1.0, 1.0), the
            # same relative Z/XY calibration ratio used for the Contact
            # Threshold distance transform (see _get_z_xy_ratio). Without
            # this, a body that is genuinely round in physical space but
            # spans fewer voxels along Z than XY (whenever the Z step is
            # larger than the XY pixel size) would be reported as more
            # elongated than it actually is, purely as a voxel-grid
            # artifact. 2D analyses are unaffected (XY pixels are assumed
            # square; spacing is a no-op there). Wrapped in a
            # try/except TypeError since ``spacing`` was only added to
            # regionprops in scikit-image 0.19 -- older installations
            # silently fall back to the previous, uncalibrated behavior
            # rather than erroring out.
            for i in range(n):
                _report(f"Morphology Shape: channel {i + 1}/{n}")
                # Fill small interior holes (see _fill_small_holes)
                # before labeling, so Aspect Ratio/Form Factor describe
                # each body's real macro-scale shape rather than being
                # distorted by thresholding-noise pockets. Body Count/
                # Fragmentation Coefficient deliberately keep using the
                # unfilled masks[i] (via the Fragmentation Metrics block
                # above, already computed) -- filling holes doesn't
                # change body count/connectivity either way, only shape.
                shape_mask_i = self._fill_small_holes(
                    masks[i], self.max_hole_size_spinbox.value()
                )
                labels_i, n_bodies_i, _, _ = self._labeled_bodies_for_metrics(
                    shape_mask_i
                )
                ar_vals: List[float] = []
                ff_vals: List[float] = []
                ar_areas: List[float] = []
                ff_areas: List[float] = []
                if n_bodies_i > 0:
                    spacing = (
                        (float(z_xy_ratio), 1.0, 1.0)
                        if labels_i.ndim == 3
                        else None
                    )
                    # Raw (uncalibrated) pixel/voxel count per body --
                    # used for the per-body export's "Area" column, kept
                    # in the same units as every other Area metric in
                    # the plugin (see Limitations & Caveats item 1),
                    # deliberately not the spacing-adjusted regionprops
                    # area used just above as an internal weight.
                    body_areas_raw = np.bincount(
                        labels_i.ravel(), minlength=n_bodies_i + 1
                    )
                    for bid in range(1, n_bodies_i + 1):
                        per_body_map[(i, bid)] = {
                            "Channel": ch_labels[i],
                            "Body ID": bid,
                            "Area": float(body_areas_raw[bid]),
                        }
                    try:
                        try:
                            regionprops_iter = regionprops(
                                labels_i, spacing=spacing
                            )
                        except TypeError:
                            regionprops_iter = regionprops(labels_i)
                        for rp in regionprops_iter:
                            area_i = float(rp.area)
                            body_entry = per_body_map.get(
                                (i, int(rp.label))
                            )
                            try:
                                major = getattr(
                                    rp, "axis_major_length",
                                    getattr(rp, "major_axis_length", None),
                                )
                                minor = getattr(
                                    rp, "axis_minor_length",
                                    getattr(rp, "minor_axis_length", None),
                                )
                                if major is not None and minor and minor > 0:
                                    ar_val = float(major / minor)
                                    ar_vals.append(ar_val)
                                    ar_areas.append(area_i)
                                    if body_entry is not None:
                                        body_entry["Aspect Ratio"] = ar_val
                            except Exception:
                                pass
                            try:
                                perim = rp.perimeter
                                if perim is not None and area_i > 0:
                                    ff_val = float(
                                        (perim ** 2)
                                        / (4 * np.pi * area_i)
                                    )
                                    ff_vals.append(ff_val)
                                    ff_areas.append(area_i)
                                    if body_entry is not None:
                                        body_entry["Form Factor"] = ff_val
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
                out[f"Aspect Ratio Weighted Mean ({ch_labels[i]})"] = (
                    ar_wmean
                )
                out[f"Form Factor Mean ({ch_labels[i]})"] = ff_mean
                out[f"Form Factor SD ({ch_labels[i]})"] = ff_sd
                out[f"Form Factor Weighted Mean ({ch_labels[i]})"] = (
                    ff_wmean
                )

        if self.output_selection.get("Morphology Network", False):
            # Skeleton/graph-based network descriptors, via skan (see
            # _skan_network_analysis's docstring for why -- it builds
            # a proper graph over the skeleton's pixel adjacency
            # instead of clustering nearby junction-flagged pixels, so
            # closely-spaced true junctions in dense/tangled networks
            # are correctly kept distinct rather than merged).
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
            for i in range(n):
                # By a wide margin the slowest per-channel step in this
                # method on large/dense 3D stacks (skeletonize + skan's
                # graph construction) -- report progress before
                # starting it, not after, so the label doesn't sit on
                # the *previous* channel's name for however long this
                # one takes.
                _report(f"Morphology Network: channel {i + 1}/{n}")
                # See the Morphology Shape block above for why: fills
                # tiny thresholding-noise holes so skeletonize traces
                # the real network instead of a ring around every
                # noise pocket. Body Count/Fragmentation Coefficient
                # are unaffected (they use the unfilled masks[i]).
                network_mask_i = self._fill_small_holes(
                    masks[i], self.max_hole_size_spinbox.value()
                )
                labels_i, n_bodies_i, _, binary_i = (
                    self._labeled_bodies_for_metrics(network_mask_i)
                )
                ndim = binary_i.ndim
                spacing = (
                    (float(z_xy_ratio), 1.0, 1.0)
                    if ndim == 3
                    else (1.0, 1.0)
                )
                branch_counts = np.zeros(n_bodies_i + 1, dtype=np.float64)
                junction_counts = np.zeros(n_bodies_i + 1, dtype=np.float64)
                branch_len_totals = np.zeros(
                    n_bodies_i + 1, dtype=np.float64
                )
                body_areas = np.zeros(n_bodies_i + 1, dtype=np.float64)

                if n_bodies_i > 0:
                    try:
                        result = self._skan_network_analysis(
                            binary_i,
                            labels_i,
                            n_bodies_i,
                            spacing,
                            self.prune_branch_length_spinbox.value(),
                            self.collapse_bridge_length_spinbox.value(),
                            self.max_local_width_spinbox.value(),
                            self.collapse_cluster_radius_spinbox.value(),
                            self.collapse_maze_radius_spinbox.value(),
                        )
                        branch_counts = result["branch_counts"]
                        junction_counts = result["junction_counts"]
                        branch_len_totals = result["branch_len_totals"]
                        body_areas = result["body_areas"]
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

                for bid in range(1, n_bodies_i + 1):
                    idx0 = bid - 1
                    entry = per_body_map.get((i, bid))
                    if entry is None:
                        # Morphology Shape wasn't enabled for this run,
                        # so this body has no entry yet -- seed it with
                        # the same raw pixel/voxel Area convention used
                        # in the Shape block above.
                        entry = {
                            "Channel": ch_labels[i],
                            "Body ID": bid,
                            "Area": float(b_areas[idx0]),
                        }
                        per_body_map[(i, bid)] = entry
                    entry["Branch Count"] = float(b_counts[idx0])
                    entry["Junction Count"] = float(j_counts[idx0])
                    entry["Branch Length"] = float(
                        mean_branch_len_per_body[idx0]
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
                    float(
                        np.sum(b_areas[j_counts > 0]) / total_area * 100.0
                    )
                    if total_area > 0
                    else 0.0
                )

                out[f"Branch Count Mean ({ch_labels[i]})"] = bc_mean
                out[f"Branch Count Weighted Mean ({ch_labels[i]})"] = (
                    bc_wmean
                )
                out[f"Junction Count Mean ({ch_labels[i]})"] = jc_mean
                out[f"Junction Count Weighted Mean ({ch_labels[i]})"] = (
                    jc_wmean
                )
                out[f"Branch Length Mean ({ch_labels[i]})"] = bl_mean
                out[f"Branch Length Weighted Mean ({ch_labels[i]})"] = (
                    bl_wmean
                )
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
            # Reuses "union" (already computed above for the "Union"
            # core overlap metric) instead of a separately-passed-in
            # union mask -- these used to be computed twice from the
            # same masks and reported under two different names
            # ("Union" and "Union Signal Area"); now there's only one
            # union area, and this ratio's name spells out what it
            # actually divides.
            avg_dist, cell_area, density_ratio = compute_contact_density(
                contacts, union
            )
            out["Avg Contact Dist"] = float(avg_dist)
            out["Avg Contact Dist / Union Signal Area"] = float(
                density_ratio
            )

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

        self._last_bundle_per_body_rows = list(per_body_map.values())

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

    def _collect_per_body_export_rows(self) -> List[Dict[str, Any]]:
        """Mirrors _collect_export_rows: prefer everything accumulated
        via "Add Analysis" clicks (self.per_body_records, tagged with
        Analysis Name), falling back to just the most recent Analyze
        run (self.last_per_body_rows, untagged) if nothing's been
        added yet."""
        if self.per_body_records:
            return self.per_body_records
        if self.last_per_body_rows:
            return self.last_per_body_rows
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

    def export_per_body_data(self):
        """Export one row per surviving body per channel (Aspect
        Ratio, Form Factor, Branch Count, Junction Count, Branch
        Length, and pixel/voxel Area), tagged with the same Analysis
        Name convention as the aggregated metrics export. Requires
        Morphology Shape and/or Morphology Network to have been
        enabled in Output Selection for at least one stored analysis
        -- those are the only metrics computed at per-body
        granularity; everything else in the plugin is already a
        channel/ROI-level aggregate with nothing further to break out.
        Intended as raw material for building your own histograms
        (Excel, GraphPad, Python, etc.), grouped/filtered by Analysis
        Name (your existing cell/condition labeling convention) and
        Channel."""
        rows = self._collect_per_body_export_rows()
        if not rows:
            QMessageBox.information(
                self,
                "Export Per-Body Data",
                "No per-body data available yet.\n\n"
                "Enable Morphology Shape and/or Morphology Network in "
                "Output Selection, run Analyze, then Add Analysis (or "
                "just Analyze, if you only need the most recent run) "
                "before exporting.",
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Per-Body Data",
            "Per-Body Export.xlsx",
            "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith((".xlsx", ".csv")):
            file_path = file_path + ".xlsx"

        df = pd.DataFrame(rows)
        preferred_order = [
            "Analysis Name",
            "ROI Number",
            "Channel",
            "Body ID",
            "Area",
            "Aspect Ratio",
            "Form Factor",
            "Branch Count",
            "Junction Count",
            "Branch Length",
        ]
        cols = [c for c in preferred_order if c in df.columns] + [
            c for c in df.columns if c not in preferred_order
        ]
        df = df[cols]

        try:
            if file_path.lower().endswith(".csv"):
                df.to_csv(file_path, index=False)
            else:
                df.to_excel(file_path, index=False, engine="openpyxl")
            print(f"Per-body data exported to {file_path}")
        except Exception as e:
            QMessageBox.warning(
                self, "Export Per-Body Data", f"Failed to export:\n{e}"
            )

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

        per_body_start = len(self.per_body_records)

        if isinstance(self.last_metrics, list):
            base_name = self.analysis_name_edit.text().strip() or "Analysis"
            if self.sequential_label_checkbox.isChecked():
                # Same Analysis Name convention used for the aggregated
                # metrics ("<base> <roi+1>"), so per-body rows for a
                # given ROI get tagged to match -- keyed by ROI Number,
                # which _compute_metrics_bundle/analyze_contacts stamps
                # onto both self.last_metrics and
                # self.last_per_body_rows using the same index.
                names_by_roi: Dict[Any, str] = {}
                for idx, metrics in enumerate(self.last_metrics):
                    m = copy.deepcopy(metrics)
                    name = f"{base_name} {idx+1}"
                    m["Analysis Name"] = name
                    self.metrics_list.append(m)
                    names_by_roi[metrics.get("ROI Number", idx)] = name
                for row in self.last_per_body_rows:
                    r = copy.deepcopy(row)
                    r["Analysis Name"] = names_by_roi.get(
                        row.get("ROI Number"), base_name
                    )
                    self.per_body_records.append(r)
            else:
                for metrics in self.last_metrics:
                    m = copy.deepcopy(metrics)
                    m["Analysis Name"] = base_name
                    self.metrics_list.append(m)
                for row in self.last_per_body_rows:
                    r = copy.deepcopy(row)
                    r["Analysis Name"] = base_name
                    self.per_body_records.append(r)
        else:
            name = self.analysis_name_edit.text().strip() or "Analysis"
            m = copy.deepcopy(self.last_metrics)
            m["Analysis Name"] = name
            self.metrics_list.append(m)
            for row in self.last_per_body_rows:
                r = copy.deepcopy(row)
                r["Analysis Name"] = name
                self.per_body_records.append(r)

        self._last_added_per_body_count = (
            len(self.per_body_records) - per_body_start
        )

        self.analysis_count_label.setText(
            f"Analyses Stored: {len(self.metrics_list)}"
        )
        print("Analysis added. Total analyses stored:", len(self.metrics_list))

    def clear_last_analysis(self):
        if self.metrics_list:
            self.metrics_list.pop()
            if self._last_added_per_body_count > 0:
                del self.per_body_records[
                    len(self.per_body_records)
                    - self._last_added_per_body_count :
                ]
            self._last_added_per_body_count = 0
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
        self.per_body_records = []
        self._last_added_per_body_count = 0
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
        show_collapsed: bool = False,
    ):
        """Show the Morphology Network visualization layers (Branch
        Count, Junction Count, Branch Length, % reticular fractions are
        computed from the same underlying skan analysis): a green raw
        skeleton overlay (every skeleton pixel, unaffected by any
        Collapse setting), a blue junction-point layer, and a magenta
        "Skeleton (Collapsed)" overlay -- what the skeleton looks like
        AFTER Collapse Bridges/Wide Regions/Junction Clusters/Maze
        Regions are applied, i.e. the same skeleton with every branch
        absorbed into a merged junction left out, so only surviving
        branches remain.
        Green/blue match the color convention used by MiNA/Fiji's
        Analyze Skeleton so the overlay reads intuitively for anyone
        used to that tool.

        ``show_skeleton``/``show_junctions``/``show_collapsed``
        independently gate which of the three layers actually gets
        created/updated -- the three manual "Skeleton Ch N" /
        "Junction Ch N" / "Collapse Ch N" buttons each request a
        different combination (Collapse requests both show_collapsed
        and show_junctions, since a simplified network reads best with
        its merged-junction markers alongside it); the Auto-Display
        Setup dialog (DisplayLayersDialog) can request any combination.
        The underlying skan analysis always runs regardless of which
        flags are set (junction coordinates and the collapsed-pixel
        mask need the same skeleton either way), only the three napari
        layer calls at the end are skipped when not requested."""
        if not show_skeleton and not show_junctions and not show_collapsed:
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

        # Match the Morphology Network metrics exactly (see the
        # _fill_small_holes call in _compute_metrics_bundle): fill
        # small thresholding-noise holes before skeletonizing, so what
        # you see here is what Branch/Junction Count are computed from.
        network_mask = self._fill_small_holes(
            self.last_masks[ch_index], self.max_hole_size_spinbox.value()
        )
        labels, n_bodies, _, binary = self._labeled_bodies_for_metrics(
            network_mask
        )
        if n_bodies == 0:
            print(f"No bodies to skeletonize for channel {ch_index + 1}.")
            return

        try:
            z_xy_ratio, _, _ = self._get_z_xy_ratio()
            spacing = (
                (float(z_xy_ratio), 1.0, 1.0)
                if binary.ndim == 3
                else (1.0, 1.0)
            )
            # Same skan-based analysis the Morphology Network metrics
            # use (see _skan_network_analysis) -- so what's displayed
            # here always matches what Branch/Junction Count are
            # computed from.
            result = self._skan_network_analysis(
                binary,
                labels,
                n_bodies,
                spacing,
                self.prune_branch_length_spinbox.value(),
                self.collapse_bridge_length_spinbox.value(),
                self.max_local_width_spinbox.value(),
                self.collapse_cluster_radius_spinbox.value(),
                self.collapse_maze_radius_spinbox.value(),
            )
            skel = result["skeleton"]
            junction_coords = result["junction_coords"]
            # The skeleton AFTER collapsing: every pixel belonging to a
            # branch absorbed by ANY of the three collapsing mechanisms
            # (Collapse Bridges, Collapse Wide Regions, Collapse
            # Junction Clusters -- see _skan_network_analysis's unified
            # absorption check) is left out, leaving only the surviving
            # branches -- exactly the pixels Branch Count/Branch Length
            # are actually computed from right now. This is a display-
            # only array computed fresh here; it never modifies ``skel``
            # itself (the raw Skeleton layer above still shows every
            # pixel regardless of collapsing).
            skel_kept = skel & ~result["collapsed_pixel_mask"]
        except Exception as e:
            print(f"Warning: skeletonization failed: {e}")
            return

        ch_label = self.get_channel_labels()[ch_index]
        skel_name = (
            f"Skeleton ({ch_label})"
            if self.use_layer_names_checkbox.isChecked()
            else f"Skeleton Ch {ch_index+1}"
        )
        skel_collapsed_name = (
            f"Skeleton (Collapsed) ({ch_label})"
            if self.use_layer_names_checkbox.isChecked()
            else f"Skeleton (Collapsed) Ch {ch_index+1}"
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

        # Independent of show_skeleton -- the "Collapse Ch N" button
        # requests this on its own, without the raw green layer (it
        # requests show_junctions instead, so the merged-junction
        # markers accompany the simplified network). ``skel_kept`` is a
        # SUBSET of ``skel`` (every collapsed/absorbed pixel removed),
        # so if the raw green Skeleton layer happens to also be visible
        # at the same time (e.g. via Auto-Display Setup requesting
        # both), the two additively-blended layers read as: white =
        # surviving/kept pixels (present in both), plain green =
        # collapsed/absorbed pixels (present only in the raw layer) --
        # a quick visual diff of what collapsing removed, for free.
        if show_collapsed:
            skel_collapsed_data = skel_kept.astype(float)
            if skel_collapsed_name in self.viewer.layers:
                lyr = self.viewer.layers[skel_collapsed_name]
                lyr.data = skel_collapsed_data
                lyr.scale = layer_scale
                lyr.translate = layer_translate
            else:
                self.viewer.add_image(
                    skel_collapsed_data,
                    name=skel_collapsed_name,
                    colormap="magenta",
                    blending="additive",
                    opacity=0.9,
                    scale=layer_scale,
                    translate=layer_translate,
                )

        # junction_coords already holds one point per true junction
        # *node* in skan's graph (see _skan_network_analysis) -- no
        # further pixel-clustering needed here.
        #
        # Pass raw pixel-index coordinates and let the Points layer's
        # own scale/translate do the world-space transform -- the same
        # way the Skeleton Image layer above is positioned. Previously
        # this pre-multiplied the coordinates by the image's physical
        # scale *and* left the Points layer's own scale at its default
        # of 1, so a world-space position got combined with a
        # data-space marker size: on a calibrated image (e.g. ~0.1 um
        # per pixel), a "size=6" marker was rendered as 6 world units
        # (um) wide -- tens of pixels across -- rather than 6 pixels.
        #
        # size=1 makes each marker exactly one pixel/voxel across in
        # data space -- the same footprint as a single skeleton pixel
        # in the green layer above -- rather than a blob spanning
        # several neighboring pixels.
        if show_junctions:
            if junction_name in self.viewer.layers:
                lyr = self.viewer.layers[junction_name]
                lyr.data = junction_coords
                lyr.size = 1
                lyr.scale = layer_scale
                lyr.translate = layer_translate
            else:
                self.viewer.add_points(
                    junction_coords,
                    name=junction_name,
                    face_color="blue",
                    size=1,
                    opacity=0.9,
                    scale=layer_scale,
                    translate=layer_translate,
                )

        shown = []
        if show_skeleton:
            shown.append(f"skeleton ({int(skel.sum())} px)")
        if show_collapsed:
            shown.append(
                f"post-collapse skeleton ({int(skel_kept.sum())} px)"
            )
        if show_junctions:
            shown.append(f"{junction_coords.shape[0]} junction(s)")
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
