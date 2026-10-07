"""Reader: axis order, channel splitting and physical scale.

The real readers need ND2/CZI files that can't live in the repo, so the
``nd2`` and ``aicsimageio`` packages are replaced by small fakes that
behave like them. What's tested is LocA's own logic: axis reordering,
which plane is kept, and the scale handed to napari (which feeds the
Z/XY ratio used by every distance-based metric).
"""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from napari_loca import _reader as R


def encoded(sizes):
    """Array whose value at each index encodes its (axis -> index), so
    any axis mix-up is detectable: value = sum(idx * 10**k)."""
    shape = tuple(sizes.values())
    grids = np.indices(shape)
    out = np.zeros(shape, dtype=np.int64)
    for k, g in enumerate(grids):
        out += g * 10 ** (len(shape) - 1 - k)
    return out


def install_fake_nd2(
    monkeypatch, sizes, voxel=(0.10392, 0.10392, 0.15), names=("FITC", "TRITC")
):
    data = encoded(sizes)

    class ND2File:
        def __init__(self, path):
            self.sizes = dict(sizes)
            chans = [
                SimpleNamespace(channel=SimpleNamespace(name=n)) for n in names
            ]
            self.metadata = SimpleNamespace(channels=chans)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def asarray(self):
            return data

        def voxel_size(self):
            return SimpleNamespace(x=voxel[0], y=voxel[1], z=voxel[2])

    mod = types.ModuleType("nd2")
    mod.ND2File = ND2File
    monkeypatch.setitem(sys.modules, "nd2", mod)
    return data


def test_nd2_native_zcyx_is_reordered_to_czyx(monkeypatch):
    sizes = {"Z": 3, "C": 2, "Y": 4, "X": 5}
    data = install_fake_nd2(monkeypatch, sizes)
    assert R.napari_get_reader("x.nd2") is R.reader_function
    [(arr, kw, kind)] = R.reader_function("x.nd2")
    assert kind == "image"
    assert arr.shape == (2, 3, 4, 5)
    # arr[c, z, y, x] must equal native data[z, c, y, x]
    assert np.array_equal(arr, np.transpose(data, (1, 0, 2, 3)))
    assert kw["channel_axis"] == 0
    assert kw["name"] == ["FITC", "TRITC"]
    # Scale is per spatial axis (napari drops the channel axis).
    assert kw["scale"] == pytest.approx((0.15, 0.10392, 0.10392))


def test_nd2_time_axis_keeps_first_timepoint(monkeypatch):
    sizes = {"T": 2, "C": 2, "Z": 3, "Y": 4, "X": 5}
    data = install_fake_nd2(monkeypatch, sizes)
    [(arr, kw, _)] = R.reader_function("x.nd2")
    assert arr.shape == (2, 3, 4, 5)
    assert np.array_equal(arr, data[0])
    assert kw["metadata"]["aics_dims"].startswith("CZYX")


def test_nd2_single_channel_2d(monkeypatch):
    install_fake_nd2(monkeypatch, {"Y": 4, "X": 5}, names=("only",))
    [(arr, kw, _)] = R.reader_function("/a/b/img.nd2")
    assert arr.shape == (4, 5)
    assert "channel_axis" not in kw
    assert kw["name"] == "img"
    assert kw["scale"] == pytest.approx((0.10392, 0.10392))


def test_nd2_channel_name_count_mismatch_falls_back(monkeypatch):
    install_fake_nd2(monkeypatch, {"C": 3, "Y": 4, "X": 5}, names=("a", "b"))
    [(_, kw, _)] = R.reader_function("x.nd2")
    assert kw["name"] == ["Channel 1", "Channel 2", "Channel 3"]


def test_unreadable_nd2_returns_none(monkeypatch):
    mod = types.ModuleType("nd2")

    class ND2File:
        def __init__(self, path):
            raise OSError("corrupt")

    mod.ND2File = ND2File
    monkeypatch.setitem(sys.modules, "nd2", mod)
    assert R.napari_get_reader("bad.nd2") is None


# ----------------------------------------------- aicsimageio path
def install_fake_aics(
    monkeypatch, arr_czyx, pps=(0.3, 0.1, 0.1), names=("A", "B")
):
    class AICSImage:
        def __init__(self, path):
            self.channel_names = list(names)
            self.physical_pixel_sizes = SimpleNamespace(
                Z=pps[0], Y=pps[1], X=pps[2]
            )

        def get_image_data(self, dims):
            assert dims == "CZYX"
            return arr_czyx

        def close(self):
            pass

    mod = types.ModuleType("aicsimageio")
    mod.AICSImage = AICSImage
    monkeypatch.setitem(sys.modules, "aicsimageio", mod)


def test_tiff_path_scale_and_channels(monkeypatch):
    arr = np.zeros((2, 5, 6, 7))
    install_fake_aics(monkeypatch, arr)
    assert R.napari_get_reader("x.tif") is R.reader_function
    [(got, kw, _)] = R.reader_function("x.tif")
    assert got.shape == (2, 5, 6, 7)
    assert kw["channel_axis"] == 0
    assert kw["name"] == ["A", "B"]
    assert kw["scale"] == pytest.approx((0.3, 0.1, 0.1))


def test_missing_z_size_drops_scale_rather_than_guessing(monkeypatch):
    """No physical Z size -> no scale at all, so the widget reports the
    image as uncalibrated instead of silently using 1.0 for Z."""
    install_fake_aics(
        monkeypatch, np.zeros((2, 5, 6, 7)), pps=(None, 0.1, 0.1)
    )
    [(_, kw, _)] = R.reader_function("x.tif")
    assert "scale" not in kw
