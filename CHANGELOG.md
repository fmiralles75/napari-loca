# Changelog

All notable changes to LocA (`napari-loca`) are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version
numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- MiNA's network statistics as Morphology Network outputs (per channel,
  per image or ROI): Summed Branch Length Mean and Median (each body's total
  skeleton length), Pooled Branch Length Mean and Median (every branch
  pooled, MiNA's "branch length"), Branch Count Median, and Donut Count
  (bodies whose skeleton has no free end). They come after the existing
  Network columns; existing outputs are unchanged. Definitions were checked
  against MiNA's and AnalyzeSkeleton's source; see Metric Descriptions for
  how LocA's measurements differ from MiNA's.
- Finish notice: when an analysis finishes (or fails), a small card slides
  into the top-right corner of the screen napari is on, stays a few seconds
  and slides out, so you can work in another app while LocA runs. Click it
  to dismiss. Switch it off with "Notify me when an analysis finishes"
  under the Analyze button (remembered between sessions).

## [0.1.0] - 2026-10-07

First public release.

### Added

- napari widget for colocalization, contact and morphology analysis of 2–4
  fluorescence channels, on single images or confocal Z-stacks, per image or
  per hand-drawn ROI.
- Overlap metrics: Intersection, Union, Intersection/Union (Jaccard) and
  Intersection over each channel's signal area.
- Contact metrics: Contact Area within a user-set distance (Z weighted by the
  voxel calibration), Contact Site Count, Mean Contact Site Size, Contact Site
  nearest-neighbour distance and mean intensity at contacts.
- Contacts method for 3–4 channels: *Overlap-based* (all channels but one
  overlap exactly, the remaining one within the contact distance) or *Focus
  channel* (a chosen channel's signal within the contact distance of every
  other channel). Exports record the method in a Contact Method column.
- Per-channel area metrics: Signal Area, Body Count, Average Area per Body and
  Fragmentation Coefficient.
- Opt-in morphology: Aspect Ratio (in physical proportions for 3D stacks) and
  Form Factor (Crofton perimeter) per body; skeleton-based Branch Count,
  Junction Count and Euclidean Branch Length.
- Before skeletonizing, each body's own small holes and enclosed cavities are
  filled (on by default; Morphology section), so dim voxels inside a
  mitochondrion no longer turn its skeleton into rings and grids of false
  junctions. Only the skeleton uses the filled mask; bodies are never joined.
- Cell (ROI) shape: exact Max/Min Feret diameter, Perimeter, Circularity,
  Roundness and Solidity, using the same definitions as Fiji.
- Thresholding per channel: automatic (Otsu, Li, …), manual on a scaled
  0–1 range, or manual in raw intensity units. Every exported row records the
  threshold in both scaled and raw units.
- Reader for Nikon ND2, TIFF and OME-TIFF that splits channels into layers and
  carries over pixel and Z-step sizes. Zeiss CZI and Leica LIF are available
  as optional extras (`[czi]`, `[lif]`, `[all-formats]`) because their
  readers are GPL-licensed.
- Export to Excel and to GraphPad Prism (one sheet per metric, one column per
  condition).
- Test suite of 190+ tests, including a golden-file regression test, run on
  macOS, Windows and Linux with Python 3.10–3.12.

### Fixed (relative to earlier unreleased versions)

- Junction detection in 3D skeletons: on scikit-image versions where 3D
  `skeletonize` returns 0/255 instead of True/False, nearly every skeleton
  pixel was counted as a junction and branch lengths were inflated. Branch
  Count, Junction Count and Branch Length from earlier builds on 3D stacks
  should not be reused.
- Intensity normalization now ignores the extreme 0.01% tails, so single hot
  pixels no longer compress the threshold range. **Manual thresholds chosen
  with earlier builds need to be re-chosen.**
- "Restrict Z range to signal" keeps the full span from the first to the last
  plane with signal. It previously dropped empty planes in between, which
  made distant planes adjacent and could join bodies across the gap.

### Notes

- Requires Python 3.10–3.12, `numpy>=1.26,<2` and `scikit-image>=0.20,<0.24`.
  These upper bounds are deliberate (see the comments in `pyproject.toml`)
  and will be raised once every dependency supports NumPy 2.
- With 3–4 channels, the default Overlap-based contacts method counts
  nothing where organelles sit close together without overlapping; use the
  Focus channel method for that case.

[Unreleased]: https://github.com/fmiralles75/napari-loca/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/fmiralles75/napari-loca/releases/tag/v0.1.0
