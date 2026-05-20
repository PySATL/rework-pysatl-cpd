# -*- coding: ascii -*-
"""No-reset benchmark analyzers.

This module provides analyzers that consume a populated
:class:`~pysatl_cpd.benchmark.registry.BenchmarkRegistry` of cached
online detection runs and produce pandas DataFrames summarizing
classification performance, per-bisegment diagnostics, and average
run length (ARL) across threshold sweeps.

Analyzers are threshold-aware: they evaluate metrics at one or more
threshold values applied to continuous (un-thresholded) detection
traces. Classification semantics (TP/FP/FN, precision, recall, F1)
are driven by a
:class:`~pysatl_cpd.benchmark.online.noreset.metrics.NoResetClassificationReport`
configured with a no-reset policy.

.. raw:: html

    <h2>Public API</h2>

- ``NoResetAnalyzerBase`` -- Abstract base class providing registry
  management and run-picking logic shared by all concrete analyzers.
  See the ``base`` submodule for details.
- ``NoResetBisigementClassificationMixin`` -- Mixin that adds
  classification-report access and bisegment validation helpers.
  See the ``base`` submodule for details.
- ``NoResetClassificationTableAnalyzer`` -- Computes TP, FP, FN,
  precision, recall, F1, mean delay, and median delay across a
  sequence of thresholds, returning one row per threshold.
- ``NoResetBisegmentAnalyzer`` -- Computes per-bisegment classification
  metrics at a single fixed threshold, returning one row per bisegment
  run.
- ``NoResetArlAnalyzer`` -- Computes average run length on no-change
  providers for a given state across a sequence of thresholds.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
>>> from pysatl_cpd.algorithms.online import ShewhartControlChart
>>> from pysatl_cpd.benchmark.online.noreset import OnlineNoResetBenchmark
>>> from pysatl_cpd.benchmark.online.noreset import OnlineNoResetBenchmarkEntry
>>> from pysatl_cpd.benchmark.online.noreset import NoResetPolicyKind
>>> from pysatl_cpd.benchmark.online.noreset.thresholds.ranges import LinearThresholdsRange
>>> from pysatl_cpd.benchmark.online.noreset.tooling.analyzers import (
...     NoResetClassificationTableAnalyzer,
...     NoResetBisegmentAnalyzer,
...     NoResetArlAnalyzer,
... )
>>> from pysatl_cpd.benchmark.registry import BenchmarkRegistry
>>> from pysatl_cpd.data.generator import preset_dataset
>>> from pysatl_cpd.data.providers.transformers import ColumnsSelectorTransformer

Build a benchmark entry and populate the registry:

>>> dataset = preset_dataset("mean_shifts", n_series=4, seed=7, series_length=120)
>>> entry = OnlineNoResetBenchmarkEntry(
...     algorithm=ShewhartControlChart(learning_period_size=20, window_size=10),
...     thresholds=LinearThresholdsRange(start=1.5, end=3.0, count=4),
...     data_transformer=ColumnsSelectorTransformer(columns=["feature_0"]),
... )
>>> benchmark = OnlineNoResetBenchmark(
...     dataset=dataset,
...     registry=BenchmarkRegistry(),
...     max_delay=15,
...     global_policy=NoResetPolicyKind.MIXED,
...     error_margin=(0, 15),
...     policy_strict=False,
... )
>>> _ = benchmark.get_classification_table([entry])

Classification table sweep via ``NoResetClassificationTableAnalyzer``:

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

Per-bisegment diagnostics via ``NoResetBisegmentAnalyzer``:

>>> bisegment_analyzer = NoResetBisegmentAnalyzer()
>>> bisegment_analyzer.registry = benchmark.registry
>>> bisegment_analyzer.classification_report = analyzer.classification_report
>>> bisegment_table = bisegment_analyzer.analyze(entry, threshold=2.0)
>>> list(bisegment_table.columns)
['bisegment_name', 'source', 'transition', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1']

ARL analysis via ``NoResetArlAnalyzer``:

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
- All analyzers require a populated ``BenchmarkRegistry`` before use.
  Set it via the ``registry`` property or by passing it to a
  ``BenchmarkRegistry``-aware orchestrator.
- ``NoResetClassificationTableAnalyzer`` and ``NoResetBisegmentAnalyzer``
  require a ``NoResetClassificationReport`` to be set; ARL analysis
  does not use a classification report.
- Bisegment-backed analyzers validate that every picked run uses a
  ``BISEGMENT`` provider with exactly one true change point.
- ``NoResetArlAnalyzer`` validates that runs use ``NO_CHANGE``
  providers matching the requested state and length.
- This module depends on ``pandas`` and ``tqdm`` (for progress bars
  in ``NoResetClassificationTableAnalyzer``).
"""

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.base import (
    NoResetAnalyzerBase,
    NoResetBisigementClassificationMixin,
)
from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.classification_table import (
    NoResetClassificationTableAnalyzer,
)
from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.individual_bisegment import (
    NoResetBisegmentAnalyzer,
)
from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.state_arl import NoResetArlAnalyzer

__all__ = [
    "NoResetAnalyzerBase",
    "NoResetArlAnalyzer",
    "NoResetBisegmentAnalyzer",
    "NoResetBisigementClassificationMixin",
    "NoResetClassificationTableAnalyzer",
]
