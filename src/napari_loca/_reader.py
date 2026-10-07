"""
Reader contribution for LocA (napari-loca).

Uses aicsimageio (https://github.com/AllenCellModeling/aicsimageio) to open
microscopy images. With the dependencies declared in ``pyproject.toml``,
this supports:

- TIFF / OME-TIFF (bundled with aicsimageio)
- Nikon ND2 (via the ``nd2`` package, used directly -- see ``_read_nd2``
  below for why this bypasses aicsimageio specifically for this format)
- Zeiss CZI (via ``aicspylibczi``, GPL-licensed, installed separately
  per aicsimageio's own docs since it can't be bundled into an MPL/BSD
  package's default extras)
- Leica LIF (via ``readlif``, also GPL-licensed, same reasoning)

See https://allencellmodeling.github.io/aicsimageio/ for the full list of
formats aicsimageio can be extended to support.

Multi-channel images are split into separate, independently toggleable
image layers (one per channel) via napari's own ``channel_axis`` mechanism
on ``add_image()``, the same approach the community ``napari-aicsimageio``
plugin uses -- and the layer structure this plugin's own widget (the
per-channel dropdowns backed by ``channel_layer_indices`` in ``_widget.py``)
was built around in the first place, rather than one combined
multi-channel array.
"""

import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


def _is_nd2_path(path: str) -> bool:
    return str(path).lower().endswith(".nd2")


def napari_get_reader(path: Union[str, List[str]]):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths. If a list is given, only the first
        path is used -- this reader does not stack multiple files into a
        single layer.

    Returns
    -------
    function or None
        If the file can actually be read, a function that reads it and
        returns napari layer data. Otherwise None, so napari can fall back
        to another reader.
    """
    if isinstance(path, list):
        path = path[0]

    if _is_nd2_path(path):
        try:
            import nd2
        except ImportError:
            return None
        try:
            with nd2.ND2File(path):
                pass
        except Exception:
            return None
        return reader_function

    try:
        from aicsimageio import AICSImage
    except ImportError:
        return None

    # Probe-open the file to confirm aicsimageio can actually read it
    # before committing to this reader.
    test_img = None
    try:
        test_img = AICSImage(path)
    except Exception:
        return None
    finally:
        if test_img is not None and hasattr(test_img, "close"):
            with contextlib.suppress(Exception):
                test_img.close()

    return reader_function


def reader_function(path: Union[str, List[str]]):
    """Read image data from ``path``.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths (only the first is used, see
        ``napari_get_reader``).

    Returns
    -------
    layer_data : list of tuples
        A single-element list of (data, add_kwargs, layer_type) tuples,
        per the napari reader contract. ``add_kwargs`` may include
        ``channel_axis``, in which case napari itself splits the array
        into one layer per channel.
    """
    if isinstance(path, list):
        path = path[0]

    if _is_nd2_path(path):
        return _read_nd2(path)
    return _read_with_aicsimageio(path)


def _channel_axis_kwargs(
    dims_used: str,
    data_shape: tuple,
    path: str,
    channel_names: Optional[List[str]],
    scale_by_letter: Dict[str, Optional[float]],
) -> Dict[str, Any]:
    """Build the ``name`` / ``channel_axis`` / ``scale`` add_kwargs.

    Shared between the ND2 and aicsimageio reading paths so both produce
    the same layer structure: one image layer per channel (via
    ``channel_axis``) when a channel dimension is present, a single
    named layer otherwise, plus physical-pixel ``scale`` for whichever
    of Z/Y/X are present.
    """
    kwargs: Dict[str, Any] = {}

    if "C" in dims_used:
        c_axis = dims_used.index("C")
        n_channels = data_shape[c_axis]
        kwargs["channel_axis"] = c_axis
        if channel_names and len(channel_names) == n_channels:
            kwargs["name"] = channel_names
        else:
            kwargs["name"] = [f"Channel {i + 1}" for i in range(n_channels)]
    else:
        kwargs["name"] = Path(path).stem

    scale = [
        scale_by_letter[letter]
        for letter in dims_used
        if letter in scale_by_letter
    ]
    if scale and all(s is not None and s > 0 for s in scale):
        kwargs["scale"] = tuple(scale)

    return kwargs


def _read_nd2(path: str):
    """Read an ND2 file with the ``nd2`` package directly.

    Deliberately bypasses aicsimageio for this one format. Two earlier
    attempts to fix a "ND2File file not closed before garbage
    collection" warning by working around aicsimageio's internal
    caching turned out to be chasing the wrong cause entirely: the
    warning was traced (via temporary diagnostic prints, since removed)
    to a *different*, separately-installed napari plugin
    (``napari-aicsimageio``) actually being the one napari was using to
    open these files -- not this plugin's own reader. Reading directly
    with ``nd2`` here, in one explicit ``with`` block, avoids depending
    on aicsimageio's ND2 handling regardless.
    """
    import nd2

    with nd2.ND2File(path) as f:
        data = np.asarray(f.asarray())
        native_order = list(f.sizes.keys())  # e.g. ['T', 'C', 'Z', 'Y', 'X']
        try:
            channel_names = [
                c.channel.name for c in (f.metadata.channels or [])
            ] or None
        except Exception:
            channel_names = None
        try:
            voxel = f.voxel_size()  # VoxelSize(x=.., y=.., z=..)
        except Exception:
            voxel = None

    # Reorder nd2's native axes to match whichever of "CZYX" / "ZYX" /
    # "YX" the file actually has all of -- the same cascade the
    # aicsimageio-based path uses -- so downstream code (channel
    # splitting, the ndim>3 squeezes elsewhere in this plugin) sees a
    # consistent axis layout regardless of which reader produced the
    # array. Any other axes nd2 reports (T, P/position, etc.) are
    # moved to the front, to be dropped by the "ndim > 4" squeeze below.
    target = None
    for candidate in ("CZYX", "ZYX", "YX"):
        if all(letter in native_order for letter in candidate):
            target = list(candidate)
            break
    if target is None:
        target = [d for d in native_order if d in "CZYX"]

    extras = [d for d in native_order if d not in target]
    order = extras + target
    data = np.transpose(data, [native_order.index(d) for d in order])
    dims_used = "".join(order)

    while data.ndim > 4:
        data = data[0]
        dims_used = dims_used[1:]

    add_kwargs: Dict[str, Any] = {
        "metadata": {
            "reader": "nd2",
            "filename": path,
            "aics_dims": f"{dims_used}{data.shape}",
        }
    }
    scale_by_letter = (
        {"Z": voxel.z, "Y": voxel.y, "X": voxel.x} if voxel is not None else {}
    )
    add_kwargs.update(
        _channel_axis_kwargs(
            dims_used, data.shape, path, channel_names, scale_by_letter
        )
    )
    return [(data, add_kwargs, "image")]


def _read_with_aicsimageio(path: str):
    """Read a non-ND2 image with aicsimageio."""
    from aicsimageio import AICSImage

    img = None
    try:
        img = AICSImage(path)

        dims_used = "CZYX"
        try:
            data = img.get_image_data(dims_used)
        except Exception:
            try:
                dims_used = "ZYX"
                data = img.get_image_data(dims_used)
            except Exception:
                dims_used = "YX"
                data = img.get_image_data(dims_used)

        data = np.asarray(data)
        while isinstance(data, np.ndarray) and data.ndim > 4:
            data = data[0]
            dims_used = dims_used[1:]

        try:
            channel_names = img.channel_names
        except Exception:
            channel_names = None

        try:
            pps = img.physical_pixel_sizes
            scale_by_letter = {"Z": pps.Z, "Y": pps.Y, "X": pps.X}
        except Exception:
            scale_by_letter = {}

        # NOTE: these must go under the "metadata" key, not be spread
        # directly into add_kwargs -- viewer.add_image() has no
        # "reader"/"filename"/"aics_dims" parameters and would raise
        # TypeError if they were passed as top-level kwargs.
        add_kwargs: Dict[str, Any] = {
            "metadata": {
                "reader": "aicsimageio",
                "filename": path,
                "aics_dims": f"{dims_used}{data.shape}",
            }
        }
        add_kwargs.update(
            _channel_axis_kwargs(
                dims_used, data.shape, path, channel_names, scale_by_letter
            )
        )
        return [(data, add_kwargs, "image")]
    finally:
        if img is not None and hasattr(img, "close"):
            with contextlib.suppress(Exception):
                img.close()
