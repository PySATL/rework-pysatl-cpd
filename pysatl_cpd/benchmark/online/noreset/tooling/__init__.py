# -*- coding: ascii -*-
"""Shared tooling for no-reset online benchmarking.

This module provides the helper types used across no-reset benchmark
scenarios: registry entry pickers that filter cached runs by detector
description, provider type, and transition/state identity; an
immutable bisegment-crop model for trimming transition-centered
providers before detection; and a family of analyzers that consume
a populated benchmark registry to produce pandas DataFrames
summarizing classification performance, per-bisegment diagnostics,
and average run length (ARL) across threshold sweeps.

The pickers and bisegment-cut model are internal building blocks
consumed by the benchmark scenario methods and the analyzers.
The analyzers can also be used directly for custom evaluation
pipelines that work with a ``BenchmarkRegistry`` outside the
standard scenario workflow.

.. raw:: html

    <h2>Public API</h2>

- ``OnlineNoResetEntryAlgorithmPicker`` -- Picks registry entries
  whose detector description matches a given benchmark entry and
  whose provider type is ``bisegment``. See the ``pickers`` module
  for additional picker implementations.
- ``NoResetClassificationTableAnalyzer`` -- Computes TP, FP, FN,
  precision, recall, F1, mean delay, and median delay across a
  sequence of thresholds. See the ``analyzers`` subpackage for
  details.
- ``NoResetBisegmentAnalyzer`` -- Computes per-bisegment
  classification metrics at a single fixed threshold. See the
  ``analyzers`` subpackage for details.
- ``NoResetArlAnalyzer`` -- Computes average run length on no-change
  providers for a given state across a sequence of thresholds. See
  the ``analyzers`` subpackage for details.

.. raw:: html

    <h2>Subpackages and Submodules</h2>

- ``analyzers`` -- Analyzer classes that consume a populated
  ``BenchmarkRegistry`` and produce summary DataFrames. This
  subpackage has its own docstring with detailed examples.
- ``pickers`` -- ``BenchmarkEntriesPicker`` implementations for
  filtering registry entries by algorithm, transition, or state.
- ``bisegment_cut`` -- ``BisegmentCut`` dataclass and
  ``NOOP_BISEGMENT_CUT`` constant for validating and parsing
  bisegment crop margins. Not re-exported from this package;
  import directly when needed.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
Use a picker to filter registry entries before running a custom
analysis:

>>> from pysatl_cpd.algorithms.online import ShewhartControlChart
>>> from pysatl_cpd.benchmark.online.noreset import OnlineNoResetBenchmarkEntry
>>> from pysatl_cpd.benchmark.online.noreset.thresholds.ranges import LinearThresholdsRange
>>> from pysatl_cpd.benchmark.online.noreset.tooling import OnlineNoResetEntryAlgorithmPicker
>>> from pysatl_cpd.benchmark.registry import BenchmarkRegistry
>>> from pysatl_cpd.data.generator import preset_dataset
>>> from pysatl_cpd.data.providers.transformers import ColumnsSelectorTransformer
>>> dataset = preset_dataset("mean_shifts", n_series=4, seed=7, series_length=120)
>>> entry = OnlineNoResetBenchmarkEntry(
...     algorithm=ShewhartControlChart(learning_period_size=20, window_size=10),
...     thresholds=LinearThresholdsRange(start=1.5, end=3.0, count=4),
...     data_transformer=ColumnsSelectorTransformer(columns=["feature_0"]),
... )
>>> registry = BenchmarkRegistry()
>>> picker = OnlineNoResetEntryAlgorithmPicker()
>>> picked = picker.pick(list(registry.keys()), entry)
>>> len(picked)
0

Run a classification-table sweep with ``NoResetClassificationTableAnalyzer``:

>>> from pysatl_cpd.benchmark.online.noreset import OnlineNoResetBenchmark
>>> from pysatl_cpd.benchmark.online.noreset import NoResetPolicyKind
>>> from pysatl_cpd.benchmark.online.noreset.tooling import NoResetClassificationTableAnalyzer
>>> benchmark = OnlineNoResetBenchmark(
...     dataset=dataset,
...     registry=registry,
...     max_delay=15,
...     global_policy=NoResetPolicyKind.MIXED,
...     error_margin=(0, 15),
...     policy_strict=False,
... )
>>> _ = benchmark.get_classification_table([entry])
>>> analyzer = NoResetClassificationTableAnalyzer()
>>> analyzer.registry = benchmark.registry
>>> analyzer.classification_report = benchmark.build_classification_report(
...     max_delay=15,
...     global_policy=NoResetPolicyKind.MIXED,
...     error_margin=(0, 15),
...     policy_strict=False,
... )
>>> table = analyzer.analyze(entry, thresholds=[1.5, 2.0, 2.5, 3.0])
>>> list(table.columns)
['threshold', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'mean_delay', 'median_delay']

Compute ARL for a specific state with ``NoResetArlAnalyzer``:

>>> from pysatl_cpd.benchmark.online.noreset.tooling import NoResetArlAnalyzer
>>> arl_analyzer = NoResetArlAnalyzer()
>>> arl_analyzer.registry = benchmark.registry
>>> state = next(iter(dataset.states))
>>> _ = benchmark.get_ARL_table_by_state([entry], state=state, arl_length=60)
>>> arl_table = arl_analyzer.analyze(entry, state=state, thresholds=[1.5, 2.0, 2.5, 3.0], arl_length=60)
>>> list(arl_table.columns)
['threshold', 'arl']

.. raw:: html

    <h2>Notes</h2>

Notes
-----
- All analyzers require a populated ``BenchmarkRegistry`` before
  use. Set it via the ``registry`` property.
- ``NoResetClassificationTableAnalyzer`` and
  ``NoResetBisegmentAnalyzer`` require a ``NoResetClassificationReport``
  to be set; ARL analysis does not use a classification report.
- This module depends on ``pandas`` (for DataFrame outputs) and
  ``tqdm`` (for progress bars in ``NoResetClassificationTableAnalyzer``).
- The ``bisegment_cut`` submodule is not re-exported from this
  package. Import it directly as
  ``from pysatl_cpd.benchmark.online.noreset.tooling.bisegment_cut import BisegmentCut``
  when configuring entry-level crop margins.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.benchmark.online.noreset.tooling.analyzers import (
    NoResetArlAnalyzer,
    NoResetBisegmentAnalyzer,
    NoResetClassificationTableAnalyzer,
)
from pysatl_cpd.benchmark.online.noreset.tooling.pickers import OnlineNoResetEntryAlgorithmPicker

__all__ = [
    "NoResetArlAnalyzer",
    "NoResetBisegmentAnalyzer",
    "NoResetClassificationTableAnalyzer",
    "OnlineNoResetEntryAlgorithmPicker",
]
