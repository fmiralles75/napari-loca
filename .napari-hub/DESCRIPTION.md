# LocA: Colocalization & Organelle Contact Analysis

**LocA** (Colocalization Analysis) measures how organelles overlap, touch and
are shaped in multichannel fluorescence images. It works on single images and
on confocal Z-stacks, for the whole field or for individual cells you outline,
and exports tables ready for statistics in Excel or GraphPad Prism.

It was built for quantifying organelle contact sites, such as
Golgi–mitochondria contacts, and mitochondrial network morphology, but works
for any pair (or up to four) of segmentable fluorescent markers.

## Who it is for

Cell biologists who want reproducible colocalization and contact numbers from
confocal data without writing code, and who want to see every segmentation
step as a napari layer before trusting the result.

## What it measures

For 2–4 channels, per image or per hand-drawn ROI:

- **Overlap**: Intersection, Union, Intersection/Union (Jaccard), and the
  fraction of each channel's signal that overlaps the other.
- **Contacts**: Contact Area within a distance you set (Z weighted by the
  voxel size), number of contact sites, their mean size and nearest-neighbour
  spacing, and mean intensity at contacts. With 3–4 channels, contacts are
  either *Overlap-based* (all channels but one overlap exactly, the last
  within the distance) or measured from a *Focus channel* (its signal within
  the distance of every other channel).
- **Per-channel area**: Signal Area, Body Count, Average Area per Body,
  Fragmentation Coefficient.
- **Morphology** (optional): Aspect Ratio and Form Factor per body;
  skeleton-based Branch Count, Junction Count and Branch Length.
- **Intensity**: mean intensity per channel, and one channel's intensity
  inside another channel's mask or at contacts.
- **Cell shape** (per ROI): Max/Min Feret diameter, Perimeter, Circularity,
  Roundness and Solidity, with the same definitions as Fiji.

Each exported row records the threshold used for every channel, in both
scaled and raw intensity units, so the analysis can be reported exactly in a
methods section. Definitions of every metric are in the widget under
**Metric Descriptions**.

## Typical workflow

1. Open a Nikon ND2, TIFF or OME-TIFF file (*File → Open*). LocA splits the
   channels into layers and reads the pixel and Z-step sizes.
2. Open *Plugins → LocA - Colocalization & Contact Analysis*.
3. Choose the number of channels and assign a layer to each; check the
   calibration.
4. Pick a threshold for each channel: automatic (Otsu, Li, Triangle and
   others), manual on a 0–1 scale, or manual in raw intensity units.
5. Set the contact distance, optionally draw ROIs around cells, and click
   **Analyze**. Show the masks, skeletons and contacts as layers to check
   them.
6. Store each result under a condition name with **Add Analysis**, then
   **Export to Excel** or **Export to GraphPad**.

## Installation

Install from the napari plugin manager, or with pip in an environment that
has napari (Python 3.10–3.12):

    pip install napari-loca

Zeiss CZI and Leica LIF files need an optional extra, because those readers
are GPL-licensed:

    pip install "napari-loca[czi]"   # Zeiss
    pip install "napari-loca[lif]"   # Leica

## Good practice and limitations

- Apply the same threshold policy to every condition you compare.
- One threshold is used per channel across the whole image, so cells of very
  different brightness in one field can segment differently. Use ROIs and
  inspect the thresholded layers.
- Skeleton metrics depend on the segmentation; inspect the Skeleton and
  Junction layers before relying on them.
- Contact distances, lengths and Feret diameters are reported in XY pixels.
  Multiply by the pixel size for µm.

## Source, issues and citation

Source code, full documentation and the changelog are on
[GitHub](https://github.com/fmiralles75/napari-loca). Please report problems
on the [issue tracker](https://github.com/fmiralles75/napari-loca/issues).
If you use LocA in published work, please cite it using the citation
information in the repository.
