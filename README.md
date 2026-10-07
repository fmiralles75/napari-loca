# LocA: Colocalization & Organelle Contact Analysis for napari

[![License Mozilla Public License 2.0](https://img.shields.io/badge/license-MPL--2.0-green)](https://github.com/fmiralles75/napari-loca/raw/main/LICENSE)
[![tests](https://github.com/fmiralles75/napari-loca/workflows/tests/badge.svg)](https://github.com/fmiralles75/napari-loca/actions)

**LocA** (Colocalization Analysis) is a [napari] plugin that measures how
organelles overlap, touch and are shaped in multichannel fluorescence images,
from single images or confocal Z-stacks. It was built for quantifying
organelle contact sites (e.g. Golgi–mitochondria) and mitochondrial network
morphology, and exports results ready for statistics in Excel or GraphPad
Prism.

<!-- TODO: add a screenshot or short GIF of the widget on a real image, e.g.
![LocA in napari](docs/images/loca-screenshot.png) -->

## What it measures

For 2–4 channels, per image or per hand-drawn ROI (e.g. per cell):

| Group | Metrics |
|---|---|
| **Overlap** | Intersection, Union, Intersection/Union (Jaccard), Intersection / each channel's signal area |
| **Contacts** | Contact Area within a user-set distance (Z weighted by the voxel calibration), Contact Site Count, Mean Contact Site Size, Contact Site nearest-neighbour distance, mean intensity at contacts |
| **Per-channel area** | Signal Area, Body Count, Average Area per Body, Fragmentation Coefficient |
| **Morphology** (opt-in) | Aspect Ratio and Form Factor per body; skeleton-based Branch Count, Junction Count and Branch Length; % of bodies / signal in branched bodies |
| **Intensity** | Mean intensity per channel, plus custom comparisons (one channel's intensity inside another's mask, contacts, etc.) |
| **Cell (ROI) shape** | Max/Min Feret diameter, Perimeter, Circularity, Roundness, Solidity (same definitions as Fiji) |

Every exported row also records the threshold applied to each channel, both
as a scaled value and in raw intensity units, so the analysis can be
reported exactly in a methods section. Full definitions are in the widget
under **Metric Descriptions**.

## Installation

LocA requires Python 3.10–3.12. Into an environment with napari:

    pip install git+https://github.com/fmiralles75/napari-loca.git

Nikon ND2, TIFF and OME-TIFF files open out of the box. Zeiss CZI and Leica
LIF readers are GPL-licensed and therefore optional:

    pip install "napari-loca[czi] @ git+https://github.com/fmiralles75/napari-loca.git"   # Zeiss
    pip install "napari-loca[lif] @ git+https://github.com/fmiralles75/napari-loca.git"   # Leica

## Quick start

1. **Open your image** in napari (*File → Open*). LocA's reader splits
   multichannel files into one layer per channel and carries over the
   physical pixel and Z-step sizes.
2. **Open the widget**: *Plugins → LocA - Colocalization & Contact Analysis*.
3. **Channel & Image Setup**: choose 2, 3 or 4 channels and map each to a
   layer. Check that the Z/XY calibration was detected (or enter it
   manually).
4. **Thresholding**: for each channel pick *Automatic* (Otsu, Li, …),
   *Manual* (a fraction of that channel's intensity range) or
   *Manual (raw intensity)* (the same detector-count cutoff in every image).
   Use the same policy for every condition you compare.
5. **Contact Analysis**: set the contact distance, optionally draw ROIs
   (*Toggle ROI Selection*, tick *Calculate metrics per ROI*), and click
   **Analyze**. Thresholded masks, skeletons and contacts can be shown as
   layers to check them by eye.
6. **Add Analysis** to store each result under a condition name, then
   **Export to Excel** or **Export to GraphPad** (one sheet per metric, one
   column per condition).

## Validation

- **Automated tests**: 190+ tests check each metric against shapes with
  hand-calculated answers (overlap areas, contact distances, skeleton
  topology, branch lengths, aspect ratios, Feret diameters, exports), plus a
  golden-file test that fails if any result on a fixed reference image
  changes. They run on macOS, Windows and Linux with Python 3.10–3.12 on
  every push.
- **Comparison with established tools**: <!-- TODO: summarise your
  MiNA and Coloc 2 / JACoP comparisons and imaging controls here. -->

## Limitations

- One threshold is applied per channel across the whole image, so cells of
  very different brightness in the same field can segment differently.
  Use ROIs, and check the thresholded layers.
- Skeleton metrics (branches, junctions) depend on the segmentation; inspect
  the Skeleton and Junction layers before trusting them.
- With 3–4 channels, Contact Area requires all but one channel to overlap
  exactly; three organelles close together without overlapping are not
  counted.
- Contact distances, lengths and Feret diameters are reported in XY pixels
  (multiply by the pixel size for µm).

## Citing

If you use LocA, please cite this repository. <!-- TODO: add the Zenodo
DOI once v0.1.0 is released. -->

## Contributing and issues

Bug reports and suggestions are welcome via
[GitHub issues](https://github.com/fmiralles75/napari-loca/issues).
To run the tests: `pip install -e ".[testing]"`, then `pytest -v src`.

## License

Distributed under the [Mozilla Public License 2.0](LICENSE).
LocA is free and open source software.

[napari]: https://napari.org
