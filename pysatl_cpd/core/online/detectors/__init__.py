# -*- coding: ascii -*-
"""Online detection-trace and detector implementations.

This subpackage provides the concrete classes that implement the online
change-point detection runtime: per-step result containers, full-run
traces, the abstract detector base, and a reset-style detector that
reinitialises its algorithm after every declared change point.

The subpackage is organised into three modules, each with its own
docstring and detailed API reference:

- ``online_detection_trace`` -- ``OnlineDetectionStepResult``,
  ``OnlineDetectionTrace``, and the helper ``extract_periods``.
- ``online_detector`` -- the abstract ``OnlineDetector`` base class
  that binds an ``OnlineAlgorithm`` to the ``ChangePointDetector``
  interface.
- ``reset_detector`` -- ``OnlineResetDetector``, a concrete detector
  that applies thresholding, skip periods, and optional run-length
  forcing around any conforming online algorithm.

.. raw:: html

    <h2>Public API</h2>

- ``OnlineDetectionStepResult`` -- dataclass capturing the output of
  processing a single observation (step index, detection flags,
  statistic value, processing time, and optional algorithm state).
- ``OnlineDetectionTrace`` -- dataclass aggregating all step results
  from one detector run; extends ``DetectionTrace`` with per-step
  detection-function values, processing times, forced/signal change
  points, skip periods, learning periods, and algorithm states.
- ``OnlineDetector`` -- abstract base class for online detectors; owns
  an ``OnlineAlgorithm`` and returns an ``OnlineDetectionTrace`` from
  its ``detect`` method. Subclasses must implement ``clone`` and
  ``_detect``.
- ``OnlineResetDetector`` -- concrete detector that resets its
  algorithm after every declared change point. Supports configurable
  ``threshold``, ``skip_period``, ``max_runlength``, and
  ``collect_states``.

.. raw:: html

    <h2>Examples</h2>

Run a reset detector over a univariate provider and inspect the trace:

>>> import numpy as np
>>> from pysatl_cpd.algorithms.online import ShewhartControlChart
>>> from pysatl_cpd.core.online.detectors import (
...     OnlineResetDetector,
...     OnlineDetectionTrace,
... )
>>> from pysatl_cpd.data import NDArrayUnivariateProvider
>>> from pysatl_cpd.data.typedefs import UnlabeledTimeseriesAnnotation
>>> data = np.concatenate([np.random.randn(50), np.random.randn(50) + 3.0])
>>> annotation = UnlabeledTimeseriesAnnotation(name="test")
>>> provider = NDArrayUnivariateProvider(data, annotation)
>>> detector = OnlineResetDetector(
...     ShewhartControlChart(learning_period_size=10, window_size=5),
...     threshold=2.0,
...     skip_period=3,
... )
>>> trace = detector.detect(provider)
>>> isinstance(trace, OnlineDetectionTrace)
True
>>> len(trace.detection_function) == len(data)
True

Cut a trace to a local window and inspect the re-based change points:

>>> local = trace.cut(40, 60)
>>> all(cp >= 0 for cp in local.detected_change_points)
True

Clone a detector for independent reuse:

>>> clone = detector.clone()
>>> clone is not detector
True
>>> clone.threshold == detector.threshold
True

Notes
-----
All change-point indices returned in traces are zero-based.

The ``OnlineDetectionTrace.from_run`` factory requires a
``detector_description`` argument; see its docstring for the full
signature.

Subclasses of ``OnlineDetector`` must implement ``clone`` (returning
an independent copy with a freshly recreated algorithm) and ``_detect``
(the internal detection logic that returns an ``OnlineDetectionTrace``).
"""

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.core.online.detectors.online_detection_trace import (
    OnlineDetectionStepResult,
    OnlineDetectionTrace,
)
from pysatl_cpd.core.online.detectors.online_detector import OnlineDetector
from pysatl_cpd.core.online.detectors.reset_detector import OnlineResetDetector

__all__ = [
    "OnlineDetectionStepResult",
    "OnlineDetectionTrace",
    "OnlineDetector",
    "OnlineResetDetector",
]
