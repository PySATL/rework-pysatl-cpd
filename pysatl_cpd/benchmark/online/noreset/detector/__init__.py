# -*- coding: ascii -*-
"""No-reset detector types.

Provides the detector and trace classes used for no-reset online change-point
detection, where the algorithm state is never restarted after a changepoint
declaration.

.. raw:: html

    <h2>Public API</h2>

- **NoResetOnlineDetector** -> Online detector that runs the same per-step
  detection loop as the reset variant but omits the post-detection
  ``OnlineAlgorithm.reset`` call. The detection statistic evolves
  continuously across the entire series. Supports optional bisegment-level
  cropping via ``BisegmentCut``.
- **NoResetDetectionTrace** -> Trace wrapper built from an infinite-threshold
  source trace. Injects policy-selected change points and a threshold value
  while copying detection function values and algorithm states from the
  original run.

See ``pysatl_cpd.benchmark.online.noreset.tooling.bisegment_cut`` for the
``BisegmentCut`` model used to crop bisegment providers before detection.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
Create a no-reset detector and run it over a data provider::

    >>> from pysatl_cpd.algorithms.online import ShewhartControlChart
    >>> from pysatl_cpd.benchmark.online.noreset.detector import (
    ...     NoResetDetectionTrace,
    ...     NoResetOnlineDetector,
    ... )
    >>> from pysatl_cpd.benchmark.online.noreset.tooling.bisegment_cut import BisegmentCut
    >>> from pysatl_cpd.data.generator import preset_dataset
    >>> from pysatl_cpd.data.providers.transformers import ColumnsSelectorTransformer
    >>>
    >>> dataset = preset_dataset("mean_shifts", n_series=1, seed=0, series_length=100)
    >>> transformer = ColumnsSelectorTransformer(columns=["feature_0"])
    >>> provider = dataset[0]
    >>> algorithm = ShewhartControlChart(learning_period_size=10, window_size=5)
    >>> detector = NoResetOnlineDetector(
    ...     algorithm,
    ...     data_transformer=transformer,
    ... )
    >>> trace = detector.detect(provider)
    >>> len(trace.detection_function) == len(provider)
    True

Use bisegment cropping to trim provider edges before detection::

    >>> detector = NoResetOnlineDetector(
    ...     algorithm,
    ...     data_transformer=transformer,
    ...     bisegment_cut=BisegmentCut(left_trim=5, right_trim=5),
    ... )
    >>> trace = detector.detect(provider)
    >>> len(trace.detection_function) == len(provider)
    True

Build a no-reset trace from an infinite-threshold source trace::

    >>> detector = NoResetOnlineDetector(algorithm, data_transformer=transformer)
    >>> source_trace = detector.detect(provider)
    >>> noreset_trace = NoResetDetectionTrace.from_inf_trace(
    ...     source_trace=source_trace,
    ...     detected_change_points=[50],
    ...     threshold=2.5,
    ... )
    >>> noreset_trace.detected_change_points
    [50]

Notes
-----
- The detector never calls ``OnlineAlgorithm.reset`` after a detection, so the
  detection statistic accumulates across the full series. This contrasts with
  reset-mode detectors that restart after each declared changepoint.
- ``NoResetDetectionTrace.from_inf_trace`` is the standard way to construct a
  no-reset trace for benchmarking: run the detector with no threshold
  (infinite), then apply a classification policy to select change points and
  wrap the result.
- Bisegment cropping (``BisegmentCut``) affects only transition-centered
  scenarios. ARL-by-state runs are evaluated with no-op crop semantics.
- Clone the detector via ``detector.clone()`` before use in parallel workers
  to ensure each worker holds an isolated algorithm instance.
"""

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.benchmark.online.noreset.detector.noreset_detector import NoResetOnlineDetector
from pysatl_cpd.benchmark.online.noreset.detector.noreset_trace import NoResetDetectionTrace

__all__ = [
    "NoResetDetectionTrace",
    "NoResetOnlineDetector",
]
