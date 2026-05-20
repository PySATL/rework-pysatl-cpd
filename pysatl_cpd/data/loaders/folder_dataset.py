# -*- coding: ascii -*-
"""Folder-based CSV dataset loader."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from pysatl_cpd.data.dataset import Dataset
from pysatl_cpd.data.providers import PandasLabeledData
from pysatl_cpd.data.providers.plain.pd_provider import PandasDataProvider
from pysatl_cpd.data.typedefs import SegmentInfo, StateDescriptor, TimeseriesAnnotation, UnlabeledTimeseriesAnnotation
from pysatl_cpd.typedefs import NumericArray, frozendict


@dataclass(frozen=True, slots=True)
class FolderCsvColumns:
    """Column selection for one folder of segmented CSV series.

    Attributes
    ----------
    feature_columns
        Names of columns containing feature data.
    state_columns
        Names of columns containing state labels.
    segment_num_column
        Name of the column containing segment numbers.
    """

    feature_columns: Sequence[str]
    state_columns: Sequence[str]
    segment_num_column: str = "segment_num"


def load_folder_csv_dataset(
    root: str | Path,
    columns: FolderCsvColumns | Mapping[str, FolderCsvColumns],
    *,
    skip_folders_without_metadata: bool = False,
) -> Dataset[NumericArray, TimeseriesAnnotation]:
    """Load a dataset from folders containing metadata and segmented CSV files.

    Each subfolder under ``root`` is expected to contain a
    ``metadata.yaml`` file and one or more CSV files with columns
    matching the ``FolderCsvColumns`` configuration.

    Parameters
    ----------
    root
        Path to the root directory containing subfolders of CSV series.
    columns
        Either a single column configuration applied to all folders, or
        a mapping keyed by folder name for per-folder configuration.
    skip_folders_without_metadata
        If True, skip folders that don't have a metadata.yaml file
        instead of raising an error.

    Returns
    -------
    dataset
        Dataset containing the loaded labeled series.

    Raises
    ------
    ValueError
        If ``root`` does not exist, is not a directory, or data files
        are missing required columns.
    """

    root_path = Path(root)
    if not root_path.exists():
        raise ValueError(f"Dataset root does not exist: {root_path}")
    if not root_path.is_dir():
        raise ValueError(f"Dataset root must be a directory: {root_path}")

    providers: list[PandasLabeledData[TimeseriesAnnotation]] = []
    folder_paths = sorted(path for path in root_path.iterdir() if path.is_dir())
    columns_by_folder = dict(columns) if isinstance(columns, Mapping) else None

    for folder_path in folder_paths:
        folder_columns = _resolve_folder_columns(folder_path.name, columns, columns_by_folder)
        _validate_column_selection(folder_columns)
        metadata_path = folder_path / "metadata.yaml"
        if not metadata_path.exists():
            if skip_folders_without_metadata:
                continue
            raise ValueError(f"Missing metadata file: {metadata_path}")
        metadata = _load_folder_metadata(metadata_path)

        for csv_path in sorted(folder_path.glob("*.csv")):
            providers.append(_load_csv_series(csv_path, folder_path.name, metadata, folder_columns))

    return Dataset(providers)


def _resolve_folder_columns(
    folder_name: str,
    columns: FolderCsvColumns | Mapping[str, FolderCsvColumns],
    columns_by_folder: dict[str, FolderCsvColumns] | None,
) -> FolderCsvColumns:
    """Resolve the column configuration for a specific folder.

    Uses the per-folder mapping when available; otherwise returns
    the global column configuration.

    Parameters
    ----------
    folder_name
        Name of the folder being processed.
    columns
        Either a single configuration or a mapping keyed by folder name.
    columns_by_folder
        Pre-computed mapping of folder name to column config, or None
        when a single global config is used.

    Returns
    -------
    FolderCsvColumns
        Column configuration for the folder.

    Raises
    ------
    ValueError
        If the folder is not present in the per-folder mapping.
    """
    if columns_by_folder is None:
        return cast(FolderCsvColumns, columns)

    try:
        return columns_by_folder[folder_name]
    except KeyError as exc:
        raise ValueError(f"Missing column configuration for folder '{folder_name}'") from exc


def _validate_column_selection(columns: FolderCsvColumns) -> None:
    """Validate column selection for a folder.

    Checks that at least one feature column and one state column are
    configured, and that feature, state, and segment columns are all
    distinct.

    Parameters
    ----------
    columns
        Column configuration to validate.

    Raises
    ------
    ValueError
        If required columns are missing or contain duplicates.
    """
    if not columns.feature_columns:
        raise ValueError("feature_columns must contain at least one column")
    if not columns.state_columns:
        raise ValueError("state_columns must contain at least one column")

    feature_names = tuple(columns.feature_columns)
    state_names = tuple(columns.state_columns)
    all_names = list(feature_names) + list(state_names) + [columns.segment_num_column]
    duplicates = sorted({name for name in all_names if all_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Configured feature/state/segment columns must be distinct, got duplicates: {duplicates}")


def _load_folder_metadata(metadata_path: Path) -> frozendict[str, Hashable]:
    """Load and validate metadata from a YAML file.

    Parameters
    ----------
    metadata_path
        Path to a ``metadata.yaml`` file.

    Returns
    -------
    frozendict[str, Hashable]
        Normalised metadata dictionary.

    Raises
    ------
    ValueError
        If the file is missing, is not a mapping, or contains
        non-hashable values.
    """
    if not metadata_path.exists():
        raise ValueError(f"Missing metadata file: {metadata_path}")

    loaded = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if loaded is None:
        return frozendict()
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Metadata YAML must contain a mapping at the top level: {metadata_path}")

    normalized: dict[str, Hashable] = {}
    for key, value in loaded.items():
        try:
            normalized[str(key)] = _normalize_metadata_value(value, path=f"metadata.{key}")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid metadata in {metadata_path}: {exc}") from exc
    return frozendict.from_mapping(normalized)


def _normalize_metadata_value(value: object, *, path: str) -> Hashable:
    """Recursively normalise a metadata value to a hashable type.

    Converts mappings to ``frozendict``, sequences to ``tuple``, and
    validates that leaf values are hashable.

    Parameters
    ----------
    value
        Raw value from parsed YAML.
    path
        Dot-separated path for descriptive error messages.

    Returns
    -------
    Hashable
        Normalised hashable value.

    Raises
    ------
    TypeError
        If a leaf value is not hashable.
    ValueError
        If the value or any nested value is not hashable.
    """
    try:
        if isinstance(value, Mapping):
            normalized_items = {
                str(key): _normalize_metadata_value(nested_value, path=f"{path}.{key}")
                for key, nested_value in value.items()
            }
            return frozendict.from_mapping(normalized_items)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            return tuple(_normalize_metadata_value(item, path=f"{path}[{index}]") for index, item in enumerate(value))
        if not isinstance(value, Hashable):
            raise TypeError(f"{path} must be hashable")
        return value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _load_csv_series(
    csv_path: Path,
    folder_name: str,
    metadata: frozendict[str, Hashable],
    columns: FolderCsvColumns,
) -> PandasLabeledData[TimeseriesAnnotation]:
    """Load a single CSV file and produce a labeled data provider.

    Validates columns, builds segment info, and wraps the result
    in a ``PandasLabeledData``.

    Parameters
    ----------
    csv_path
        Path to the CSV file.
    folder_name
        Name of the parent folder for annotation naming.
    metadata
        Metadata associated with the folder.
    columns
        Column configuration for feature and state columns.

    Returns
    -------
    PandasLabeledData
        Labeled data provider for the CSV series.

    Raises
    ------
    ValueError
        If the CSV file is empty.
    """
    frame = pd.read_csv(csv_path)
    _validate_frame_columns(frame, csv_path, columns)
    if frame.empty:
        raise ValueError(f"CSV file must contain at least one row: {csv_path}")

    # normalize segment numbering
    frame[columns.segment_num_column] -= frame[columns.segment_num_column].iloc[0]

    segment_info = _build_segment_info(frame, csv_path, columns)
    feature_frame = frame.loc[:, list(columns.feature_columns)]
    annotation = TimeseriesAnnotation(
        name=f"{folder_name}/{csv_path.stem}",
        source=str(csv_path),
        metadata=frozendict.from_mapping(
            {
                **dict(metadata),
                "folder_name": folder_name,
                "file_name": csv_path.name,
            }
        ),
    )
    unlabeled = PandasDataProvider(
        feature_frame,
        UnlabeledTimeseriesAnnotation(name=annotation.name, source=annotation.source),
    )
    return PandasLabeledData.from_unlabeled_data(unlabeled, segment_info, annotation)


def _validate_frame_columns(frame: pd.DataFrame, csv_path: Path, columns: FolderCsvColumns) -> None:
    """Verify that a CSV frame contains all required columns.

    Parameters
    ----------
    frame
        Loaded CSV data.
    csv_path
        Path used for error messages.
    columns
        Column configuration specifying required columns.

    Raises
    ------
    ValueError
        If any required column is missing from the frame.
    """
    required_columns = set(columns.feature_columns) | set(columns.state_columns) | {columns.segment_num_column}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"CSV file {csv_path} is missing required columns: {missing_columns}")


def _build_segment_info(frame: pd.DataFrame, csv_path: Path, columns: FolderCsvColumns) -> list[SegmentInfo]:
    """Build segment metadata from a CSV frame.

    Identifies contiguous blocks of constant segment numbers and
    constructs ``SegmentInfo`` entries with their start, end, and
    state.

    Parameters
    ----------
    frame
        Loaded CSV data with a segment-number column.
    csv_path
        Path used for error messages.
    columns
        Column configuration specifying segment and state columns.

    Returns
    -------
    list[SegmentInfo]
        Ordered list of segment information.

    Raises
    ------
    ValueError
        If segment numbers contain gaps, missing values, or non-integer
        values.
    """
    segment_series = frame[columns.segment_num_column]
    if segment_series.isna().any():
        raise ValueError(f"Segment column '{columns.segment_num_column}' contains missing values in {csv_path}")

    segment_numbers = pd.to_numeric(segment_series, errors="raise")
    if not (segment_numbers % 1 == 0).all():
        raise ValueError(f"Segment column '{columns.segment_num_column}' must contain integer values in {csv_path}")

    segment_ids = cast(pd.Series, segment_numbers.astype(int))
    change_mask = segment_ids.ne(segment_ids.shift())
    block_starts = list(frame.index[change_mask])
    block_stops = [*block_starts[1:], len(frame)]

    segments: list[SegmentInfo] = []
    for expected_segment_num, (start, stop) in enumerate(zip(block_starts, block_stops, strict=True)):
        end = stop - 1
        segment_num = int(segment_ids.iloc[start])
        if segment_num != expected_segment_num:
            raise ValueError(
                f"Segment numbers in {csv_path} must start at 0 and have no gaps; "
                f"expected {expected_segment_num}, got {segment_num}"
            )
        state = _build_segment_state(frame.iloc[start:stop], csv_path, expected_segment_num, columns.state_columns)
        segments.append(
            SegmentInfo(
                segment_num=segment_num,
                segment_start=int(start),
                segment_end=int(end),
                state=state,
            )
        )
    return segments


def _build_segment_state(
    segment_frame: pd.DataFrame,
    csv_path: Path,
    segment_num: int,
    state_columns: Sequence[str],
) -> StateDescriptor:
    """Extract the state descriptor for a single segment.

    Reads the state columns from a segment's sub-frame and validates
    that each state column is constant within the segment.

    Parameters
    ----------
    segment_frame
        Sub-frame covering a single segment.
    csv_path
        Path used for error messages.
    segment_num
        Segment number for error messages.
    state_columns
        Names of columns that define the state.

    Returns
    -------
    StateDescriptor
        Descriptor containing the state values.

    Raises
    ------
    ValueError
        If any state column contains missing or non-constant values.
    """
    state_values: dict[str, Any] = {}
    for column in state_columns:
        values = segment_frame[column]
        if values.isna().any():
            raise ValueError(f"State column '{column}' contains missing values in segment {segment_num} of {csv_path}")
        first_value = values.iloc[0]
        if not values.eq(first_value).all():
            raise ValueError(f"State column '{column}' must stay constant within segment {segment_num} of {csv_path}")
        state_values[column] = first_value.item() if hasattr(first_value, "item") else first_value
    return StateDescriptor(**state_values)
