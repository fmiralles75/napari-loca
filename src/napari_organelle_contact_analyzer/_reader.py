"""
Reader contribution for napari-organelle-contact-analyzer.

Uses aicsimageio (https://github.com/AllenCellModeling/aicsimageio) to open
microscopy images. With the dependencies declared in ``pyproject.toml``,
this supports:

- TIFF / OME-TIFF (bundled with aicsimageio)
- Nikon ND2 (via the ``aicsimageio[nd2]`` extra)
- Zeiss CZI (via ``aicspylibczi``, GPL-licensed, installed separately
  per aicsimageio's own docs since it can't be bundled into an MPL/BSD
  package's default extras)
- Leica LIF (via ``readlif``, also GPL-licensed, same reasoning)

See https://allencellmodeling.github.io/aicsimageio/ for the full list of
formats aicsimageio can be extended to support.
"""

from typing import Any, Dict, List, Union

import numpy as np


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
        If aicsimageio can open the path, a function that reads it and
        returns napari layer data. Otherwise None, so napari can fall back
        to another reader.
    """
    try:
        from aicsimageio import AICSImage
    except ImportError:
        return None

    if isinstance(path, list):
        path = path[0]

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
    """Read image data from ``path`` using aicsimageio.

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
    from aicsimageio import AICSImage

    if isinstance(path, list):
        path = path[0]

    img = None
    try:
        img = AICSImage(path)

        # Avoiding img.dims wasn't enough on its own (see prior fix):
        # AICSImage.xarray_data -- which get_image_data() below calls
        # into regardless -- unconditionally checks for mosaic tiles
        # before it does anything else, via
        # `DimensionNames.MosaicTile in self.reader.dims.order`. That
        # touches the *reader's own* `dims` property, which (if nothing
        # has populated it yet) forces Reader.xarray_dask_data ->
        # Reader._read_delayed(). For ND2 specifically, the delayed
        # dask array that produces depends on its own nd2.ND2File
        # staying open for later chunk reads -- a documented upstream
        # design choice in the nd2 package (tlambert03/nd2#19) -- and
        # that handle is never explicitly closed by anyone, which is
        # what was producing the "ND2File not closed before garbage
        # collection" warning even after removing our own img.dims use.
        #
        # Reader.xarray_data (the reader's *immediate*, non-delayed
        # property -- for ND2 this opens nd2.ND2File in a single `with`
        # block and closes it right away) has a side effect of also
        # caching a safe, already-in-memory-backed placeholder for
        # Reader.xarray_dask_data. Forcing that here, before touching
        # anything on the AICSImage wrapper, means the mosaic check
        # above finds that placeholder already cached instead of
        # triggering a real delayed nd2 read -- so the leaky path never
        # gets taken in the first place. No data is read twice: once
        # cached, AICSImage.xarray_data's own (non-mosaic) codepath
        # reuses this same object.
        img.reader.xarray_data

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
