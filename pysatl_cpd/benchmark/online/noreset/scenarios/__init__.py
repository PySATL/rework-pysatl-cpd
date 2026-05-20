# -*- coding: ascii -*-
"""Scenarios for no-reset online benchmark campaigns.

This module provides concrete scenario classes that define how no-reset
online change-point detectors are evaluated against labeled datasets.
Each scenario implements a distinct analysis view: global threshold
sweeps, transition-specific classification, per-bisegment inspection,
and state-based average run length (ARL) evaluation.

All scenarios inherit from ``NoResetBenchmarkScenario``, which lives in
the ``base`` submodule and provides the shared infrastructure for
building ``NoResetOnlineDetector`` instances from benchmark entries.
See ``base`` for details on the base class contract.

.. raw:: html

    <h2>Public API</h2>

- ``NoResetBisegmentsTableScenario``: computes per-bisegment
  classification metrics at a fixed detection threshold. Returns one
  ``pd.DataFrame`` per detector description.
- ``NoResetClassificationTableScenario``: computes classification
  metrics (precision, recall, F1, TP, FP, FN) across all
  transition-centered bisegment runs, sweeping over a threshold range.
  Returns one ``pd.DataFrame`` per detector description.
- ``NoResetClassificationTableByTransitionScenario``: like the global
  classification scenario but restricted to bisegments matching a
  specific ``TransitionDescriptor``. Optionally appends an ARL column
  when ``use_arl=True``.
- ``NoResetArlByStateScenario``: evaluates average run length for
  no-change providers drawn from a single ``StateDescriptor`` across
  a threshold range. Returns one ``pd.DataFrame`` per detector
  description.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
Build a no-reset benchmark entry and run the global classification
scenario:

>>> from pysatl_cpd.algorithms.online import ShewhartControlChart
>>> from pysatl_cpd.benchmark.online.noreset import (
...     NoResetPolicyKind,
...     OnlineNoResetBenchmark,
...     OnlineNoResetBenchmarkEntry,
... )
>>> from pysatl_cpd.benchmark.online.noreset.thresholds.ranges import (
...     LinearThresholdsRange,
... )
>>> from pysatl_cpd.benchmark.online.noreset.scenarios import (
...     NoResetClassificationTableScenario,
... )
>>> from pysatl_cpd.benchmark.registry import BenchmarkRegistry
>>> from pysatl_cpd.data.generator import preset_dataset
>>> from pysatl_cpd.data.providers.transformers import ColumnsSelectorTransformer
>>> dataset = preset_dataset("mean_shifts", n_series=4, seed=42, series_length=120)
>>> feature_transformer = ColumnsSelectorTransformer(columns=["feature_0"])
>>> entry = OnlineNoResetBenchmarkEntry(
...     algorithm=ShewhartControlChart(learning_period_size=20, window_size=10),
...     thresholds=LinearThresholdsRange(start=1.5, end=3.0, count=4),
...     data_transformer=feature_transformer,
... )
>>> benchmark = OnlineNoResetBenchmark(
...     dataset=dataset,
...     registry=BenchmarkRegistry(),
...     max_delay=15,
...     global_policy=NoResetPolicyKind.MIXED,
...     error_margin=(0, 15),
...     policy_strict=False,
... )
>>> scenario = NoResetClassificationTableScenario(entries=[entry])
>>> global_tables = benchmark.get_classification_table([entry])
>>> description, table = next(iter(global_tables.items()))
>>> description  # doctest: +ELLIPSIS
ChangePointDetectorDescription(...)
>>> "threshold" in table.columns
True
>>> "precision" in table.columns
True

Transition-specific classification with optional ARL:

>>> from pysatl_cpd.benchmark.online.noreset.scenarios import (
...     NoResetClassificationTableByTransitionScenario,
... )
>>> transition = next(iter(dataset.transitions))
>>> transition_tables = benchmark.get_classification_table_by_transition(
...     [entry],
...     transition=transition,
...     use_arl=True,
...     arl_length=60,
... )
>>> len(transition_tables)
1

ARL-by-state evaluation:

>>> from pysatl_cpd.benchmark.online.noreset.scenarios import (
...     NoResetArlByStateScenario,
... )
>>> state = next(iter(dataset.states))
>>> arl_tables = benchmark.get_ARL_table_by_state(
...     [entry], state=state, arl_length=60
... )
>>> len(arl_tables)
1

Per-bisegment classification at a fixed threshold:

>>> from pysatl_cpd.benchmark.online.noreset.scenarios import (
...     NoResetBisegmentsTableScenario,
... )
>>> bisegment_tables = benchmark.get_bisegments_table(
...     [entry], threshold=2.0
... )
>>> len(bisegment_tables)
1

.. raw:: html

    <h2>Notes</h2>

Notes
-----
- Scenarios are typically invoked through ``OnlineNoResetBenchmark``
  convenience methods (``get_classification_table``,
  ``get_classification_table_by_transition``, ``get_ARL_table_by_state``,
  ``get_bisegments_table``), which handle job preparation, execution,
  and analysis in one call. Direct scenario instantiation is useful for
  custom pipelines.
- All scenario results are keyed by ``ChangePointDetectorDescription``,
  accessible as ``entry.description`` on each
  ``OnlineNoResetBenchmarkEntry``.
- ARL-by-state runs use no-op bisegment crop semantics regardless of
  any ``bisegment_cut`` set on the entry. Bisegment-backed scenarios
  (global, transition, per-bisegment) honor ``entry.bisegment_cut``.
- Requires ``pandas`` for all result tables.
"""

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .classification_table_global import NoResetClassificationTableScenario
from .classification_table_transition import NoResetClassificationTableByTransitionScenario
from .individual_bisegments_table import NoResetBisegmentsTableScenario
from .state_arl import NoResetArlByStateScenario

__all__ = [
    "NoResetBisegmentsTableScenario",
    "NoResetClassificationTableScenario",
    "NoResetClassificationTableByTransitionScenario",
    "NoResetArlByStateScenario",
]
