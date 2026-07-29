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

        # get_image_data() uses aicsimageio's "immediate" read path
        # (Reader.xarray_data -> Reader._read_immediate()), which for
        # ND2 files opens the underlying nd2.ND2File in a single `with`
        # block and closes it right away. Deliberately never touching
        # `img.dims` (or `.dask_data` / `.xarray_dask_data`, etc.):
        # those force aicsimageio's "delayed" dask path instead, and
        # for ND2 files that hands back a dask array that keeps its
        # own separate nd2.ND2File handle open for later chunk reads --
        # by design, per a known upstream issue in the nd2 package
        # (tlambert03/nd2#19: dask compute depends on the file staying
        # open). That handle is never explicitly closed by anyone and
        # is what was producing the "ND2File not closed before garbage
        # collection" warning, regardless of how carefully img.close()
        # is called on the top-level AICSImage object below.
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
