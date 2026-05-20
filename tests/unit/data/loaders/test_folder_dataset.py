# -*- coding: ascii -*-
"""
Tests for folder dataset.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pathlib import Path

import pandas as pd
import pytest

from pysatl_cpd.data import FolderCsvColumns, load_folder_csv_dataset


def test_load_folder_csv_dataset_loads_multiple_folders_and_series(tmp_path: Path) -> None:
    _write_folder_metadata(
        tmp_path / "folder1",
        """
owner: team-a
nested:
  tags:
    - alpha
    - beta
""",
    )
    _write_folder_metadata(tmp_path / "folder2", "notes: ok\n")

    pd.DataFrame(
        {
            "segment_num": [0, 0, 1, 1],
            "state": ["idle", "idle", "active", "active"],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [5.0, 6.0, 7.0, 8.0],
        }
    ).to_csv(tmp_path / "folder1" / "series1.csv", index=False)
    pd.DataFrame(
        {
            "segment_num": [0, 1, 1],
            "state": ["cold", "hot", "hot"],
            "x": [10.0, 11.0, 12.0],
            "y": [13.0, 14.0, 15.0],
        }
    ).to_csv(tmp_path / "folder2" / "series2.csv", index=False)

    dataset = load_folder_csv_dataset(
        tmp_path,
        FolderCsvColumns(feature_columns=["x", "y"], state_columns=["state"]),
    )

    assert len(dataset) == 2
    assert dataset[0].name == "folder1/series1"
    assert dataset[0].change_points == (2,)
    assert dataset[0].feature_columns == ["x", "y"]
    assert dataset[0].annotation.metadata["owner"] == "team-a"
    assert dataset[0].annotation.metadata["nested"] == {"tags": ("alpha", "beta")}
    assert dataset[1].name == "folder2/series2"
    assert dataset[1].change_points == (1,)


def test_load_folder_csv_dataset_supports_empty_metadata_and_folder_configs(tmp_path: Path) -> None:
    _write_folder_metadata(tmp_path / "folder1", "")
    _write_folder_metadata(tmp_path / "folder2", "kind: alternate\n")

    pd.DataFrame(
        {
            "segment_num": [0, 0, 1],
            "state": ["a", "a", "b"],
            "signal": [1.0, 2.0, 3.0],
        }
    ).to_csv(tmp_path / "folder1" / "series.csv", index=False)
    pd.DataFrame(
        {
            "seg_id": [0, 1, 1],
            "mode": ["x", "y", "y"],
            "value": [5.0, 6.0, 7.0],
        }
    ).to_csv(tmp_path / "folder2" / "series.csv", index=False)

    dataset = load_folder_csv_dataset(
        tmp_path,
        {
            "folder1": FolderCsvColumns(feature_columns=["signal"], state_columns=["state"]),
            "folder2": FolderCsvColumns(feature_columns=["value"], state_columns=["mode"], segment_num_column="seg_id"),
        },
    )

    assert len(dataset) == 2
    assert dict(dataset[0].annotation.metadata) == {"folder_name": "folder1", "file_name": "series.csv"}
    assert dataset[1].annotation.metadata["kind"] == "alternate"
    assert dataset[1].change_points == (1,)


def test_load_folder_csv_dataset_rejects_missing_required_columns(tmp_path: Path) -> None:
    _write_folder_metadata(tmp_path / "folder1", "")
    pd.DataFrame(
        {
            "segment_num": [0, 1],
            "state": ["a", "b"],
        }
    ).to_csv(tmp_path / "folder1" / "series.csv", index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_folder_csv_dataset(tmp_path, FolderCsvColumns(feature_columns=["signal"], state_columns=["state"]))


def test_load_folder_csv_dataset_rejects_nonconstant_state_within_segment(tmp_path: Path) -> None:
    _write_folder_metadata(tmp_path / "folder1", "")
    pd.DataFrame(
        {
            "segment_num": [0, 0, 1],
            "state": ["a", "b", "c"],
            "signal": [1.0, 2.0, 3.0],
        }
    ).to_csv(tmp_path / "folder1" / "series.csv", index=False)

    with pytest.raises(ValueError, match="must stay constant within segment"):
        load_folder_csv_dataset(tmp_path, FolderCsvColumns(feature_columns=["signal"], state_columns=["state"]))


def test_load_folder_csv_dataset_rejects_segment_number_gaps_or_reuse(tmp_path: Path) -> None:
    _write_folder_metadata(tmp_path / "folder1", "")
    pd.DataFrame(
        {
            "segment_num": [0, 0, 2, 2, 0],
            "state": ["a", "a", "b", "b", "c"],
            "signal": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    ).to_csv(tmp_path / "folder1" / "series.csv", index=False)

    with pytest.raises(ValueError, match="must start at 0 and have no gaps"):
        load_folder_csv_dataset(tmp_path, FolderCsvColumns(feature_columns=["signal"], state_columns=["state"]))


def test_load_folder_csv_dataset_rejects_missing_root_path(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"

    with pytest.raises(ValueError, match="Dataset root does not exist"):
        load_folder_csv_dataset(missing_root, FolderCsvColumns(feature_columns=["signal"], state_columns=["state"]))


def test_load_folder_csv_dataset_rejects_root_that_is_not_directory(tmp_path: Path) -> None:
    root_file = tmp_path / "root.csv"
    root_file.write_text("segment_num,state,signal\n0,a,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Dataset root must be a directory"):
        load_folder_csv_dataset(root_file, FolderCsvColumns(feature_columns=["signal"], state_columns=["state"]))


def test_load_folder_csv_dataset_rejects_missing_per_folder_column_configuration(tmp_path: Path) -> None:
    _write_folder_metadata(tmp_path / "folder1", "")
    _write_folder_metadata(tmp_path / "folder2", "")
    pd.DataFrame({"segment_num": [0], "state": ["a"], "signal": [1.0]}).to_csv(
        tmp_path / "folder1" / "series.csv", index=False
    )

    with pytest.raises(ValueError, match="Missing column configuration for folder 'folder2'"):
        load_folder_csv_dataset(
            tmp_path,
            {"folder1": FolderCsvColumns(feature_columns=["signal"], state_columns=["state"])},
        )


def test_load_folder_csv_dataset_rejects_empty_feature_columns(tmp_path: Path) -> None:
    _write_folder_metadata(tmp_path / "folder1", "")

    with pytest.raises(ValueError, match="feature_columns must contain at least one column"):
        load_folder_csv_dataset(tmp_path, FolderCsvColumns(feature_columns=[], state_columns=["state"]))


def test_load_folder_csv_dataset_rejects_empty_state_columns(tmp_path: Path) -> None:
    _write_folder_metadata(tmp_path / "folder1", "")

    with pytest.raises(ValueError, match="state_columns must contain at least one column"):
        load_folder_csv_dataset(tmp_path, FolderCsvColumns(feature_columns=["signal"], state_columns=[]))


def test_load_folder_csv_dataset_rejects_duplicate_configured_columns(tmp_path: Path) -> None:
    _write_folder_metadata(tmp_path / "folder1", "")

    with pytest.raises(ValueError, match="must be distinct"):
        load_folder_csv_dataset(
            tmp_path,
            FolderCsvColumns(feature_columns=["signal"], state_columns=["signal"], segment_num_column="segment_num"),
        )


def test_load_folder_csv_dataset_rejects_missing_metadata_file(tmp_path: Path) -> None:
    (tmp_path / "folder1").mkdir()

    with pytest.raises(ValueError, match="Missing metadata file"):
        load_folder_csv_dataset(tmp_path, FolderCsvColumns(feature_columns=["signal"], state_columns=["state"]))


def test_load_folder_csv_dataset_skips_folders_without_metadata_when_flagged(tmp_path: Path) -> None:
    _write_folder_metadata(tmp_path / "folder_with_meta", "owner: team-a\n")
    pd.DataFrame(
        {
            "segment_num": [0, 0, 1],
            "state": ["a", "a", "b"],
            "signal": [1.0, 2.0, 3.0],
        }
    ).to_csv(tmp_path / "folder_with_meta" / "series.csv", index=False)

    (tmp_path / "folder_without_meta").mkdir()
    pd.DataFrame(
        {
            "segment_num": [0, 1],
            "state": ["x", "y"],
            "signal": [10.0, 20.0],
        }
    ).to_csv(tmp_path / "folder_without_meta" / "series.csv", index=False)

    dataset = load_folder_csv_dataset(
        tmp_path,
        FolderCsvColumns(feature_columns=["signal"], state_columns=["state"]),
        skip_folders_without_metadata=True,
    )

    assert len(dataset) == 1
    assert dataset[0].name == "folder_with_meta/series"


def test_load_folder_csv_dataset_rejects_non_mapping_metadata_yaml(tmp_path: Path) -> None:
    _write_folder_metadata(tmp_path / "folder1", "- not\n- a\n- mapping\n")

    with pytest.raises(ValueError, match="Metadata YAML must contain a mapping"):
        load_folder_csv_dataset(tmp_path, FolderCsvColumns(feature_columns=["signal"], state_columns=["state"]))


def test_load_folder_csv_dataset_rejects_invalid_metadata_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_folder_metadata(tmp_path / "folder1", "owner: team-a\n")

    def _fake_safe_load(_: str) -> dict[str, object]:
        return {"owner": {1, 2}}

    monkeypatch.setattr("pysatl_cpd.data.loaders.folder_dataset.yaml.safe_load", _fake_safe_load)

    with pytest.raises(
        ValueError, match=r"Invalid metadata in .*metadata\.yaml: metadata\.owner: metadata\.owner must be hashable"
    ):
        load_folder_csv_dataset(tmp_path, FolderCsvColumns(feature_columns=["signal"], state_columns=["state"]))


def test_load_folder_csv_dataset_rejects_empty_csv_file(tmp_path: Path) -> None:
    _write_folder_metadata(tmp_path / "folder1", "")
    (tmp_path / "folder1" / "series.csv").write_text("segment_num,state,signal\n", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV file must contain at least one row"):
        load_folder_csv_dataset(tmp_path, FolderCsvColumns(feature_columns=["signal"], state_columns=["state"]))


def test_load_folder_csv_dataset_rejects_nan_segment_values(tmp_path: Path) -> None:
    _write_folder_metadata(tmp_path / "folder1", "")
    pd.DataFrame(
        {
            "segment_num": [0, None, 1],
            "state": ["a", "a", "b"],
            "signal": [1.0, 2.0, 3.0],
        }
    ).to_csv(tmp_path / "folder1" / "series.csv", index=False)

    with pytest.raises(ValueError, match="Segment column 'segment_num' contains missing values"):
        load_folder_csv_dataset(tmp_path, FolderCsvColumns(feature_columns=["signal"], state_columns=["state"]))


def test_load_folder_csv_dataset_rejects_non_integer_segment_values(tmp_path: Path) -> None:
    _write_folder_metadata(tmp_path / "folder1", "")
    pd.DataFrame(
        {
            "segment_num": [0.0, 0.5, 1.0],
            "state": ["a", "a", "b"],
            "signal": [1.0, 2.0, 3.0],
        }
    ).to_csv(tmp_path / "folder1" / "series.csv", index=False)

    with pytest.raises(ValueError, match="must contain integer values"):
        load_folder_csv_dataset(tmp_path, FolderCsvColumns(feature_columns=["signal"], state_columns=["state"]))


def test_load_folder_csv_dataset_rejects_nan_state_values(tmp_path: Path) -> None:
    _write_folder_metadata(tmp_path / "folder1", "")
    pd.DataFrame(
        {
            "segment_num": [0, 0, 1],
            "state": ["a", None, "b"],
            "signal": [1.0, 2.0, 3.0],
        }
    ).to_csv(tmp_path / "folder1" / "series.csv", index=False)

    with pytest.raises(ValueError, match="State column 'state' contains missing values"):
        load_folder_csv_dataset(tmp_path, FolderCsvColumns(feature_columns=["signal"], state_columns=["state"]))


def _write_folder_metadata(folder_path: Path, content: str) -> None:
    folder_path.mkdir()
    (folder_path / "metadata.yaml").write_text(content, encoding="utf-8")


def test_load_folder_csv_dataset_metadata_path_exists_check(tmp_path: Path) -> None:
    """Cover the explicit exists() check inside _load_folder_metadata (line 190)."""
    from pysatl_cpd.data.loaders.folder_dataset import _load_folder_metadata

    non_existent = tmp_path / "non_existent_folder" / "metadata.yaml"
    with pytest.raises(ValueError, match="Missing metadata file"):
        _load_folder_metadata(non_existent)
