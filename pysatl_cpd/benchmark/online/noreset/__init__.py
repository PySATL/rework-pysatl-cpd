# -*- coding: ascii -*-
"""No-reset online benchmarking public API.

This module provides the top-level public interface for no-reset online
change-point detection benchmarking. In no-reset mode the detector runs
continuously without restarting its internal state after each alarm, so
classification of threshold crossings into true positives, false positives,
and false negatives is delegated to policy objects at benchmark time rather
than being baked into the detector itself.

The primary workflow is: build one or more ``OnlineNoResetBenchmarkEntry``
objects (algorithm + threshold range + optional transformer), construct an
``OnlineNoResetBenchmark`` orchestrator with a classification policy, and
call one of its scenario methods to produce threshold-indexed result tables.
All results are keyed by ``ChangePointDetectorDescription``.

.. raw:: html

    <h2>Public API</h2>

Benchmark orchestrator
~~~~~~~~~~~~~~~~~~~~~~

- ``OnlineNoResetBenchmark`` -> Main benchmark class specialised for no-reset
  online detectors. Provides ``get_classification_table``,
  ``get_classification_table_by_transition``, ``get_ARL_table_by_state``,
  ``get_bisegments_table``, and ``get_pr_auc_table``. See the class docstring
  for parameter details.

Benchmark entry
~~~~~~~~~~~~~~~

- ``OnlineNoResetBenchmarkEntry`` -> Dataclass that bundles an
  ``OnlineAlgorithm`` instance, a ``ThresholdsRange``, an optional
  ``IDataTransformer``, and an optional ``BisegmentCut``. The
  ``description`` property yields a ``ChangePointDetectorDescription``
  used as the key for all scenario result mappings.

Detector types
~~~~~~~~~~~~~~

- ``NoResetOnlineDetector`` -> Online detector that omits the post-detection
  ``reset`` call so the detection statistic evolves continuously.
- ``NoResetDetectionTrace`` -> Trace wrapper built from an infinite-threshold
  source trace with policy-selected change points injected.

See the ``detector`` subpackage docstring for detailed usage examples.

Scenario classes
~~~~~~~~~~~~~~~~

- ``NoResetBenchmarkScenario`` -> Abstract base for all no-reset scenarios.
- ``NoResetClassificationTableScenario`` -> Global threshold sweep across all
  transition-centered bisegment runs.
- ``NoResetClassificationTableByTransitionScenario`` -> Same as above but
  filtered to a single ``TransitionDescriptor``, with optional ARL column.
- ``NoResetBisegmentsTableScenario`` -> Per-bisegment classification at a
  fixed threshold.
- ``NoResetArlByStateScenario`` -> Average run length evaluation on no-change
  providers from a single ``StateDescriptor``.

See the ``scenarios`` subpackage docstring for examples of each scenario type.

Policy kinds and classes
~~~~~~~~~~~~~~~~~~~~~~~~

- ``NoResetPolicyKind`` -> ``StrEnum`` with values ``POINT``, ``EVENT``, and
  ``MIXED`` used when building classification reports.
- ``NoResetPolicy`` -> Protocol implemented by all policy classes.
- ``BisegmentPolicyBase`` -> Abstract base for bisegment policies.
- ``PointBasedPolicy`` -> Point-based selection in both true and false regions.
- ``EventBasedPolicy`` -> Event-based selection in both regions.
- ``MixedPolicy`` -> Event-based false region, point-based true region.
- ``NoChangePolicy`` -> Single-detection policy for ARL evaluation.

See the ``metrics`` subpackage docstring for policy semantics and examples.

Metric classes
~~~~~~~~~~~~~~

Base wrappers:

- ``NoResetThresholdMetric`` -> Protocol for threshold-callable evaluation.
- ``NoResetSingleRunMetric`` -> Wraps a classical single-run metric with a
  no-reset policy.
- ``NoResetMultipleRunMetric`` -> Wraps a classical multiple-run metric.
- ``NoResetDerivedMetric`` -> Wraps a derived metric formula with named base
  metrics.
- ``wrap_noreset_single_run_metric`` -> Factory for ``NoResetSingleRunMetric``.
- ``wrap_noreset_multiple_run_metric`` -> Factory for ``NoResetMultipleRunMetric``.
- ``wrap_noreset_derived_metric`` -> Factory for ``NoResetDerivedMetric``.

Classification metrics:

- ``NoResetTotalTPMetric`` / ``NoResetTPMetric`` -> Total true positives.
- ``NoResetTotalFPMetric`` -> Total false positives.
- ``NoResetTotalFNMetric`` -> Total false negatives.
- ``NoResetPrecisionMetric`` -> Precision from configurable TP/FP bases.
- ``NoResetRecallMetric`` -> Recall from configurable TP/FN bases.
- ``NoResetF1Metric`` -> F1 from pre-configured precision and recall.

Online metrics:

- ``NoResetARLMetric`` -> Average run length using ``NoChangePolicy``.
- ``NoResetMeanDelayMetric`` -> Mean detection delay using ``MixedPolicy``.
- ``NoResetMedianDelayMetric`` -> Median detection delay using ``MixedPolicy``.

See the ``metrics`` subpackage docstring for detailed examples.

.. raw:: html

    <h2>Subpackages</h2>

- ``detector`` -> ``NoResetOnlineDetector`` and ``NoResetDetectionTrace``.
  See its module docstring for usage examples.
- ``metrics`` -> Metric wrappers, classification metrics, online metrics,
  and classification policies. See its module docstring for details.
- ``scenarios`` -> Concrete benchmark scenario classes. See its module
  docstring for examples.
- ``thresholds`` -> ``ThresholdsRange`` hierarchy (``LinearThresholdsRange``,
  ``ManualThresholdsRange``, ``AutoThresholdsRange``). See its module
  docstring for details.
- ``tooling`` -> Analyzers, pickers, and the ``BisegmentCut`` model. See its
  module docstring for examples.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
Build a benchmark entry and run a global classification table sweep::

    >>> from pysatl_cpd.algorithms.online import ShewhartControlChart
    >>> from pysatl_cpd.benchmark.online.noreset import (
    ...     NoResetPolicyKind,
    ...     OnlineNoResetBenchmark,
    ...     OnlineNoResetBenchmarkEntry,
    ... )
    >>> from pysatl_cpd.benchmark.online.noreset.thresholds.ranges import (
    ...     LinearThresholdsRange,
    ... )
    >>> from pysatl_cpd.benchmark.online.noreset.tooling.bisegment_cut import (
    ...     BisegmentCut,
    ... )
    >>> from pysatl_cpd.benchmark.registry import BenchmarkRegistry
    >>> from pysatl_cpd.data.generator import preset_dataset
    >>> from pysatl_cpd.data.providers.transformers import ColumnsSelectorTransformer
    >>>
    >>> dataset = preset_dataset("mean_shifts", n_series=4, seed=42, series_length=120)
    >>> transformer = ColumnsSelectorTransformer(columns=["feature_0"])
    >>> entry = OnlineNoResetBenchmarkEntry(
    ...     algorithm=ShewhartControlChart(learning_period_size=20, window_size=10),
    ...     thresholds=LinearThresholdsRange(start=1.5, end=3.0, count=4),
    ...     data_transformer=transformer,
    ...     bisegment_cut=BisegmentCut.parse((8, 0)),
    ... )
    >>> benchmark = OnlineNoResetBenchmark(
    ...     dataset=dataset,
    ...     registry=BenchmarkRegistry(),
    ...     max_delay=15,
    ...     global_policy=NoResetPolicyKind.MIXED,
    ...     error_margin=(0, 15),
    ...     policy_strict=False,
    ... )
    >>> tables = benchmark.get_classification_table([entry])
    >>> description, table = next(iter(tables.items()))
    >>> description == entry.description
    True
    >>> "threshold" in table.columns
    True
    >>> "precision" in table.columns
    True

Run a transition-specific classification with optional ARL::

    >>> transition = next(iter(dataset.transitions))
    >>> transition_tables = benchmark.get_classification_table_by_transition(
    ...     [entry],
    ...     transition=transition,
    ...     use_arl=True,
    ...     arl_length=60,
    ... )
    >>> len(transition_tables)
    1

Compute ARL for a specific state::

    >>> state = next(iter(dataset.states))
    >>> arl_tables = benchmark.get_ARL_table_by_state(
    ...     [entry], state=state, arl_length=60
    ... )
    >>> len(arl_tables)
    1

Inspect per-bisegment classification at a fixed threshold::

    >>> bisegment_tables = benchmark.get_bisegments_table([entry], threshold=2.0)
    >>> len(bisegment_tables)
    1

Compute PR-AUC scores from classification tables::

    >>> pr_auc_scores = benchmark.get_pr_auc_table(tables)
    >>> len(pr_auc_scores)
    1

Build a classification report without instantiating a benchmark::

    >>> report = OnlineNoResetBenchmark.build_classification_report(
    ...     max_delay=15,
    ...     global_policy=NoResetPolicyKind.MIXED,
    ...     error_margin=(0, 15),
    ...     policy_strict=False,
    ... )

.. raw:: html

    <h2>Notes</h2>

Notes
-----
- The detector never calls ``OnlineAlgorithm.reset`` after a detection, so
  the detection statistic accumulates across the full series. This contrasts
  with reset-mode detectors that restart after each declared changepoint.
- Classification semantics (TP/FP/FN, precision, recall, F1) are fixed at
  ``OnlineNoResetBenchmark`` construction via ``max_delay``, ``global_policy``,
  and optional ``precision_policy`` / ``recall_policy`` overrides.
- Bisegment cropping (``entry.bisegment_cut``) affects only
  transition-centered scenarios. ARL-by-state runs use no-op crop semantics.
- All scenario results are keyed by ``ChangePointDetectorDescription``,
  accessible as ``entry.description`` on each
  ``OnlineNoResetBenchmarkEntry``.
- Requires ``pandas`` for all result tables and ``numpy`` for PR-AUC
  computation via ``numpy.trapezoid``.
- Clone detectors via ``detector.clone()`` before use in parallel workers
  to ensure each worker holds an isolated algorithm instance.
"""

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.benchmark.online.noreset.benchmark import OnlineNoResetBenchmark
from pysatl_cpd.benchmark.online.noreset.detector.noreset_detector import NoResetOnlineDetector
from pysatl_cpd.benchmark.online.noreset.detector.noreset_trace import NoResetDetectionTrace
from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry
from pysatl_cpd.benchmark.online.noreset.metrics import (
    BisegmentPolicyBase,
    EventBasedPolicy,
    MixedPolicy,
    NoChangePolicy,
    NoResetARLMetric,
    NoResetDerivedMetric,
    NoResetF1Metric,
    NoResetMeanDelayMetric,
    NoResetMedianDelayMetric,
    NoResetMultipleRunMetric,
    NoResetPolicy,
    NoResetPrecisionMetric,
    NoResetRecallMetric,
    NoResetSingleRunMetric,
    NoResetThresholdMetric,
    NoResetTotalFNMetric,
    NoResetTotalFPMetric,
    NoResetTotalTPMetric,
    NoResetTPMetric,
    PointBasedPolicy,
    wrap_noreset_derived_metric,
    wrap_noreset_multiple_run_metric,
    wrap_noreset_single_run_metric,
)
from pysatl_cpd.benchmark.online.noreset.policy_kind import NoResetPolicyKind
from pysatl_cpd.benchmark.online.noreset.scenarios.base import (
    NoResetBenchmarkScenario,
)
from pysatl_cpd.benchmark.online.noreset.scenarios.classification_table_global import NoResetClassificationTableScenario
from pysatl_cpd.benchmark.online.noreset.scenarios.classification_table_transition import (
    NoResetClassificationTableByTransitionScenario,
)
from pysatl_cpd.benchmark.online.noreset.scenarios.individual_bisegments_table import NoResetBisegmentsTableScenario
from pysatl_cpd.benchmark.online.noreset.scenarios.state_arl import NoResetArlByStateScenario

__all__ = [
    "BisegmentPolicyBase",
    "EventBasedPolicy",
    "MixedPolicy",
    "NoChangePolicy",
    "NoResetArlByStateScenario",
    "NoResetARLMetric",
    "NoResetBenchmarkScenario",
    "NoResetBisegmentsTableScenario",
    "NoResetPolicyKind",
    "NoResetClassificationTableByTransitionScenario",
    "NoResetClassificationTableScenario",
    "NoResetDerivedMetric",
    "NoResetDetectionTrace",
    "NoResetF1Metric",
    "NoResetMeanDelayMetric",
    "NoResetMedianDelayMetric",
    "NoResetMultipleRunMetric",
    "NoResetOnlineDetector",
    "NoResetPolicy",
    "NoResetPrecisionMetric",
    "NoResetRecallMetric",
    "NoResetSingleRunMetric",
    "NoResetThresholdMetric",
    "NoResetTPMetric",
    "NoResetTotalFNMetric",
    "NoResetTotalFPMetric",
    "NoResetTotalTPMetric",
    "OnlineNoResetBenchmark",
    "OnlineNoResetBenchmarkEntry",
    "PointBasedPolicy",
    "wrap_noreset_derived_metric",
    "wrap_noreset_multiple_run_metric",
    "wrap_noreset_single_run_metric",
]
