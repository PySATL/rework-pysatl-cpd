# -*- coding: ascii -*-
"""Dataset loaders for the data layer.

This module provides file-backed data loading utilities that turn on-disk
segmented CSV files into in-memory ``Dataset`` objects. The loaders
produce the same labeled-provider and dataset abstractions used
throughout the project, so file-backed data participates in the same
workflows as generated or manually constructed data.

The loader expects a directory layout where each subfolder under a root
contains a ``metadata.yaml`` file and one or more CSV files. Each CSV
must include feature columns, state columns, and a segment-number
column that identifies contiguous labeled regions. Segment numbers are
normalized to zero-based indexing during loading.

.. raw:: html

    <h2>Public API</h2>

- ``FolderCsvColumns`` -- Immutable configuration specifying which CSV
  columns contain feature data, state labels, and segment numbers.
- ``load_folder_csv_dataset`` -- Load a ``Dataset`` from a root
  directory of segmented CSV folders, returning a collection of
  ``PandasLabeledData`` providers.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
Load a dataset from a directory of CSV folders using a single column
configuration applied to all folders::

    >>> from pathlib import Path
    >>> from pysatl_cpd.data.loaders import (
    ...     FolderCsvColumns,
    ...     load_folder_csv_dataset,
    ... )
    >>> root = Path("assets/userguide/examples/csv_dataset")
    >>> columns = FolderCsvColumns(
    ...     feature_columns=["value", "aux"],
    ...     state_columns=["state_type", "state_regime"],
    ...     segment_num_column="segment_num",
    ... )
    >>> dataset = load_folder_csv_dataset(root, columns)
    >>> len(dataset)  # doctest: +SKIP
    2
    >>> dataset[0].annotation.name  # doctest: +SKIP
    'demo_source_1/series_alpha'

Use per-folder column configurations when different folders have
different column layouts::

    >>> from pysatl_cpd.data.loaders import (
    ...     FolderCsvColumns,
    ...     load_folder_csv_dataset,
    ... )
    >>> columns_by_folder = {
    ...     "sensor_a": FolderCsvColumns(
    ...         feature_columns=["temperature"],
    ...         state_columns=["regime"],
    ...     ),
    ...     "sensor_b": FolderCsvColumns(
    ...         feature_columns=["pressure", "flow"],
    ...         state_columns=["regime", "quality"],
    ...     ),
    ... }
    >>> dataset = load_folder_csv_dataset(
    ...     "/path/to/sensors", columns_by_folder
    ... )  # doctest: +SKIP

Skip folders that lack a ``metadata.yaml`` file::

    >>> dataset = load_folder_csv_dataset(
    ...     root, columns, skip_folders_without_metadata=True
    ... )  # doctest: +SKIP

.. raw:: html

    <h2>Notes</h2>

Notes
-----
This module depends on ``pandas`` for CSV parsing and ``pyyaml`` for
metadata loading. Both are declared project dependencies.

Each CSV file must contain at least one row. The segment-number column
must contain integer values without gaps, starting from zero (or any
integer, which is normalized to zero-based during loading). State
columns must remain constant within each segment.

The ``metadata.yaml`` file in each folder is optional when
``skip_folders_without_metadata=True`` is set. Metadata values are
recursively normalized to hashable types (mappings become
``frozendict``, sequences become ``tuple``) and attached to each
provider's annotation.

For programmatic data creation without file I/O, use the generator
API in ``pysatl_cpd.data.generator`` instead.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .folder_dataset import FolderCsvColumns, load_folder_csv_dataset

__all__ = ["FolderCsvColumns", "load_folder_csv_dataset"]
