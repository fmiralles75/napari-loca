"""
Reader contribution for napari-organelle-contact-analyzer.

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
"""

from typing import Any, Dict, List, Union

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
        # TEMPORARY DIAGNOSTIC: see matching note in _read_nd2().
        print("=== napari-organelle-contact-analyzer: napari_get_reader() probing", path, "===")
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
            try:
                test_img.close()
            except Exception:
                pass

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
        per the napari reader contract.
    """
    if isinstance(path, list):
        path = path[0]

    if _is_nd2_path(path):
        return _read_nd2(path)
    return _read_with_aicsimageio(path)


def _read_nd2(path: str):
    """Read an ND2 file with the ``nd2`` package directly.

    Deliberately bypasses aicsimageio for this one format. Two earlier
    attempts to fix the "ND2File file not closed before garbage
    collection" warning by working around aicsimageio's internal
    caching (avoiding ``img.dims``, then pre-warming
    ``Reader.xarray_data`` before aicsimageio's own mosaic-tile check
    could force a delayed read) both turned out to be incomplete --
    aicsimageio's ND2 support goes through several layers of internal
    caching, an unconditional mosaic-tile check, and (for the delayed
    path) nd2's own dask wrapping via ``ResourceBackedDaskArray``,
    and the exact combination that leaves a ``nd2.ND2File`` handle
    open long enough to be garbage-collected wasn't fully pinned down
    even after reading through aicsimageio's and nd2's source directly.
    Reading here with one explicit, fully eager ``with nd2.ND2File(...)``
    block -- no dask, no delayed xarray, no aicsimageio in the loop at
    all -- removes that uncertainty rather than continuing to guess at
    aicsimageio's internals.
    """
    import nd2

    # TEMPORARY DIAGNOSTIC: three fixes in a row haven't stopped the
    # warning, which raises the question of whether this function is
    # even the code path being run for .nd2 files, versus some other
    # installed plugin/reader also handling them. This print is
    # deliberately impossible to miss in the terminal output -- if it
    # does NOT appear right before the warning, that confirms this
    # code isn't the source and the search needs to go elsewhere.
    # Safe to remove once that's settled.
    print("=== napari-organelle-contact-analyzer: _read_nd2() called for", path, "===")

    with nd2.ND2File(path) as f:
        data = np.asarray(f.asarray())
        native_order = list(f.sizes.keys())  # e.g. ['T', 'C', 'Z', 'Y', 'X']
        print(
            "=== napari-organelle-contact-analyzer: nd2.ND2File closed?",
            f.closed,
            "(should be False here, inside the `with` block) ===",
        )

    print(
        "=== napari-organelle-contact-analyzer: after `with` block, "
        "file should now be closed ==="
    )

    # Reorder nd2's native axes to match whichever of "CZYX" / "ZYX" /
    # "YX" the file actually has all of -- the same cascade the
    # aicsimageio-based path used -- so downstream code (channel
    # splitting, the ndim>3 squeezes elsewhere in this plugin) sees a
    # consistent axis layout regardless of which reader produced the
    # array. Any other axes nd2 reports (T, P/position, etc.) are
    # moved to the front, to be dropped by the same "ndim > 4" squeeze
    # used for every other format below.
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

    add_kwargs: Dict[str, Any] = {
        "metadata": {
            "reader": "nd2",
            "filename": path,
            "aics_dims": f"{dims_used}{data.shape}",
        }
    }
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
        return [(data, add_kwargs, "image")]
    finally:
        if img is not None and hasattr(img, "close"):
            try:
                img.close()
            except Exception:
                pass
