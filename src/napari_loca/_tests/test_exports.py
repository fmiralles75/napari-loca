"""Exports: what lands in Excel / CSV / the Prism workbook must be exactly
the numbers computed in memory, under labels that identify them."""

import numpy as np
import pandas as pd
import pytest

from napari_loca import _widget as W


class FakeFileDialog:
    path = ""

    @classmethod
    def getSaveFileName(cls, *a, **k):  # noqa: N802
        return cls.path, ""

    @classmethod
    def getOpenFileName(cls, *a, **k):  # noqa: N802
        return cls.path, ""


class FakeMessageBox:
    messages = []

    @classmethod
    def information(cls, *a, **k):
        cls.messages.append(("info", a))

    @classmethod
    def warning(cls, *a, **k):
        cls.messages.append(("warning", a))


@pytest.fixture
def dialogs(monkeypatch):
    FakeMessageBox.messages = []
    monkeypatch.setattr(W, "QFileDialog", FakeFileDialog)
    monkeypatch.setattr(W, "QMessageBox", FakeMessageBox)
    return FakeFileDialog


def sample_rows():
    """Two conditions x two ROIs, awkward floats, one NaN."""
    return [
        {
            "Intersection": 50,
            "Jaccard": 1 / 3,
            "Form Factor": np.nan,
            "ROI Number": 0,
            "Analysis Name": "WT",
        },
        {
            "Intersection": 61,
            "Jaccard": 0.1 + 0.2,
            "Form Factor": 1.25,
            "ROI Number": 1,
            "Analysis Name": "WT",
        },
        {
            "Intersection": 7,
            "Jaccard": np.pi / 10,
            "Form Factor": 2.0,
            "ROI Number": 0,
            "Analysis Name": "KO",
        },
    ]


# ------------------------------------------------------- sheet names
def test_sanitize_sheet_name(make_harness):
    h = make_harness()
    assert h._sanitize_excel_sheet_name("Signal Area/ROI Area (A)") == (
        "Signal Area_ROI Area (A)"
    )
    assert h._sanitize_excel_sheet_name("a[b]:c*d?e\\f") == "a_b__c_d_e_f"
    assert len(h._sanitize_excel_sheet_name("x" * 50)) == 31
    assert h._sanitize_excel_sheet_name("") == "Sheet"


def test_unique_sheet_names(make_harness):
    h = make_harness()
    used = set()
    names = [h._make_unique_sheet_name("y" * 40, used) for _ in range(3)]
    assert len(set(names)) == 3
    assert all(len(n) <= 31 for n in names)


def _metric_names(two_labels):
    from napari_loca._tests.conftest import (
        _make_harness,
        run_pipeline,
    )

    rng = np.random.default_rng(0)
    a, b = rng.random((6, 24, 24)), rng.random((6, 24, 24))
    out, _, _ = run_pipeline(
        [a, b], _make_harness(), thresholds=[0.7, 0.7], labels=two_labels
    )
    return out


def test_long_per_channel_names_keep_their_channel(make_harness):
    h = make_harness()
    a = h._sanitize_excel_sheet_name("Fragmentation Coefficient (Channel 1)")
    b = h._sanitize_excel_sheet_name("Fragmentation Coefficient (Channel 2)")
    assert a != b
    assert a.endswith("~(Channel 1)") and len(a) <= 31
    assert h._sanitize_excel_sheet_name("Body Count (Mito)") == (
        "Body Count (Mito)"
    )


@pytest.mark.parametrize(
    "labels",
    [
        ["Channel 1", "Channel 2"],
        ["FITC Golgi", "TRITC Mito"],
        ["A very long channel name 1", "A very long channel name 2"],
    ],
)
def test_prism_sheets_identify_their_metric(make_harness, tmp_path, labels):
    """Every metric sheet must be traceable to its exact metric (and so
    its channel) via the Sheet Index, and hold that metric's values.
    Before the fix, default labels made 10 pairs of sheets ambiguous."""
    out = _metric_names(labels)
    rows = [
        dict(out, **{"Analysis Name": "WT"}),
        dict(
            {
                k: (v * 2 if isinstance(v, (int, float)) else v)
                for k, v in out.items()
            },
            **{"Analysis Name": "KO"},
        ),
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "prism.xlsx"
    make_harness()._write_graphpad_workbook_from_dataframe(df, str(path))
    book = pd.read_excel(path, sheet_name=None)
    names = list(book)
    assert names[0] == "Sheet Index" and names[-1] == "All_Metrics"
    index = book["Sheet Index"]
    numeric = [c for c in df.columns if c != "Analysis Name"]
    assert sorted(index["Metric"]) == sorted(numeric)
    assert index["Sheet"].is_unique
    for sheet, metric in zip(index["Sheet"], index["Metric"]):
        got = book[sheet]
        for cond in ("WT", "KO"):
            exp = df.loc[df["Analysis Name"] == cond, metric].dropna()
            assert got[cond].dropna().tolist() == pytest.approx(
                exp.tolist(), rel=1e-14
            ), (sheet, metric)
        # Short enough names are used verbatim (no surprise renames).
        if len(metric) <= 31 and not any(ch in metric for ch in "/[]:*?\\"):
            assert sheet == metric


# ------------------------------------------------ Prism workbook
def test_prism_workbook_round_trip(make_harness, tmp_path):
    df = pd.DataFrame(sample_rows())
    path = tmp_path / "prism.xlsx"
    make_harness()._write_graphpad_workbook_from_dataframe(
        df.copy(), str(path)
    )
    book = pd.read_excel(path, sheet_name=None)
    assert list(book) == [
        "Sheet Index",
        "Intersection",
        "Jaccard",
        "Form Factor",
        "All_Metrics",
    ]
    j = book["Jaccard"]
    assert list(j.columns) == ["WT", "KO"]
    # Excel stores ~15 significant digits; nothing coarser than that.
    assert j["WT"].tolist() == pytest.approx([1 / 3, 0.1 + 0.2], rel=1e-14)
    assert j["KO"].dropna().tolist() == pytest.approx([np.pi / 10], rel=1e-14)
    # NaN values are dropped per column, not written as 0.
    ff = book["Form Factor"]
    assert ff["WT"].dropna().tolist() == [1.25]
    assert 0 not in ff["WT"].tolist()
    # All_Metrics is the full table, unchanged.
    allm = book["All_Metrics"]
    pd.testing.assert_frame_equal(
        allm[df.columns].reset_index(drop=True), df, check_dtype=False
    )


def test_prism_workbook_rejects_empty(make_harness, tmp_path):
    with pytest.raises(ValueError):
        make_harness()._write_graphpad_workbook_from_dataframe(
            pd.DataFrame(), str(tmp_path / "x.xlsx")
        )


def test_export_graphpad_prism_xlsx(make_harness, dialogs, tmp_path):
    h = make_harness(metrics_list=sample_rows())
    dialogs.path = str(tmp_path / "out")  # extension added
    h.export_graphpad_prism()
    book = pd.read_excel(tmp_path / "out.xlsx", sheet_name=None)
    assert book["Intersection"]["WT"].tolist() == [50, 61]
    assert book["Intersection"]["KO"].dropna().tolist() == [7]


def test_export_graphpad_prism_csv_exports_first_metric_only(
    make_harness, dialogs, tmp_path
):
    h = make_harness(metrics_list=sample_rows())
    dialogs.path = str(tmp_path / "out.csv")
    h.export_graphpad_prism()
    df = pd.read_csv(tmp_path / "out.csv")
    assert list(df.columns) == ["WT", "KO"]
    assert df["WT"].tolist() == [50, 61]
    assert any(
        "only the first metric" in str(m[1]) for m in FakeMessageBox.messages
    )


# --------------------------------------------------- Excel / CSV
@pytest.mark.parametrize("ext", [".xlsx", ".csv"])
def test_save_metrics_round_trip(make_harness, dialogs, tmp_path, ext):
    rows = sample_rows()
    h = make_harness(metrics_list=rows)
    dialogs.path = str(tmp_path / f"m{ext}")
    h.save_metrics()
    got = (pd.read_excel if ext == ".xlsx" else pd.read_csv)(dialogs.path)
    assert got.columns[0] == "Analysis Name"
    exp = pd.DataFrame(rows)[got.columns]
    pd.testing.assert_frame_equal(
        got, exp, check_dtype=False, rtol=1e-14, atol=0
    )


def test_save_metrics_uses_last_analysis_when_none_stored(
    make_harness, dialogs, tmp_path
):
    h = make_harness(last_metrics={"Intersection": 3, "Union": 9})
    dialogs.path = str(tmp_path / "m.xlsx")
    h.save_metrics()
    got = pd.read_excel(dialogs.path)
    assert got.to_dict("records") == [{"Intersection": 3, "Union": 9}]


def test_append_to_csv(make_harness, dialogs, tmp_path):
    path = tmp_path / "log.csv"
    pd.DataFrame([{"Analysis Name": "old", "Intersection": 1}]).to_csv(
        path, index=False
    )
    h = make_harness(metrics_list=sample_rows())
    dialogs.path = str(path)
    h.append_to_spreadsheet()
    got = pd.read_csv(path)
    assert len(got) == 4
    assert got["Analysis Name"].tolist() == ["old", "WT", "WT", "KO"]
    assert got["Intersection"].tolist() == [1, 50, 61, 7]


def test_append_to_prism_workbook(make_harness, dialogs, tmp_path):
    path = tmp_path / "prism.xlsx"
    rows = sample_rows()
    make_harness()._write_graphpad_workbook_from_dataframe(
        pd.DataFrame(rows[:2]), str(path)
    )
    h = make_harness(metrics_list=rows[2:])
    dialogs.path = str(path)
    h.append_to_graphpad_prism()
    book = pd.read_excel(path, sheet_name=None)
    assert len(book["All_Metrics"]) == 3
    assert book["Intersection"]["KO"].dropna().tolist() == [7]


# ---------------------------------------------------- add_analysis
def test_add_analysis_labels_rois_sequentially(make_harness):
    from napari_loca._tests.conftest import (
        FakeCheck,
        FakeLineEdit,
    )

    h = make_harness(
        last_metrics=[{"A": 1}, {"A": 2}],
        analysis_name_edit=FakeLineEdit("KO"),
        sequential_label_checkbox=FakeCheck(True),
    )
    h.add_analysis()
    assert [r["Analysis Name"] for r in h.metrics_list] == ["KO 1", "KO 2"]
    # Stored rows are copies: editing the live result can't alter them.
    h.last_metrics[0]["A"] = 999
    assert h.metrics_list[0]["A"] == 1


def test_add_analysis_default_name(make_harness):
    h = make_harness(last_metrics={"A": 1})
    h.add_analysis()
    assert h.metrics_list == [{"A": 1, "Analysis Name": "Analysis"}]
