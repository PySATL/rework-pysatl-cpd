# -*- coding: ascii -*-
"""Core online detection components.

This package provides the interfaces and classes for online change-point
detection, including algorithm abstractions, detector implementations,
result containers, and composable wrappers that modify algorithm behavior.

The online API is organized into three layers:

1. Algorithm layer -- the abstract ``OnlineAlgorithm`` protocol and its
   configuration/state dataclasses define the contract for stateful
   streaming detectors.  See the ``ionline_algorithm`` module docstring
   for details.
2. Detector layer -- concrete detectors wrap algorithms with runtime
   policy (thresholding, reset behavior, skip periods) and produce
   durable trace objects.  See the ``detectors`` subpackage docstring
   for details.
3. Wrapper layer -- composable wrappers adapt how observations are
   consumed without modifying the underlying algorithm.  See the
   ``wrappers`` subpackage docstring for details.

.. raw:: html

    <h2>Public API</h2>

Algorithm interface (from ``ionline_algorithm``):

- ``OnlineAlgorithm`` -- Abstract base class for online change-point
  detection algorithms.
- ``OnlineAlgorithmConfiguration`` -- Frozen dataclass holding static
  configuration parameters for an algorithm.
- ``OnlineAlgorithmDescription`` -- Frozen dataclass combining a
  human-readable name with configuration.
- ``OnlineAlgorithmState`` -- Frozen dataclass capturing an algorithm's
  internal state at a point in time.

Detectors and traces (from ``detectors``):

- ``OnlineDetector`` -- Abstract base class for online detectors; binds
  an ``OnlineAlgorithm`` to the ``ChangePointDetector`` interface.
- ``OnlineResetDetector`` -- Concrete detector that resets its algorithm
  after every declared change point.
- ``OnlineDetectionStepResult`` -- Dataclass capturing the output of
  processing a single observation.
- ``OnlineDetectionTrace`` -- Dataclass aggregating all step results
  from one detector run.

Wrappers (from ``wrappers``):

- ``SkippingCondition`` -- Frozen dataclass holding a named predicate
  that decides whether an observation should be skipped.
- ``BatchReducer`` -- Frozen dataclass holding a named function that
  reduces a sequence of raw observations into a single value.
- ``SkippingOnlineAlgorithmWrapper`` -- Wraps an ``OnlineAlgorithm``
  and conditionally bypasses observations.
- ``BatchingOnlineAlgorithmWrapper`` -- Wraps an ``OnlineAlgorithm``
  and feeds it reduced batches of observations.

.. raw:: html

    <h2>Examples</h2>

Examples
--------

Run a reset detector over a univariate provider and inspect the trace:

>>> import numpy as np
>>> from pysatl_cpd.algorithms.online import ShewhartControlChart
>>> from pysatl_cpd.core.online import (
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

Wrap an algorithm with a skipping condition and process observations:

>>> from pysatl_cpd.algorithms.online import ShewhartControlChart
>>> from pysatl_cpd.core.online import (
...     SkippingCondition,
...     SkippingOnlineAlgorithmWrapper,
... )
>>> condition = SkippingCondition(
...     name="large-value",
...     condition=lambda x: abs(float(x)) > 1.0,
... )
>>> wrapper = SkippingOnlineAlgorithmWrapper(
...     ShewhartControlChart(learning_period_size=5, window_size=5),
...     skipping_condition=condition,
... )
>>> wrapper.name
'ShewhartControlChart{skip[on=large-value]}'

Notes
-----
- All change-point indices returned in traces are zero-based.
- Wrappers implement the full ``OnlineAlgorithm`` interface, so wrapped
  instances can be passed directly to ``OnlineResetDetector`` or any
  other consumer expecting an ``OnlineAlgorithm``.
- Subclasses of ``OnlineDetector`` must implement ``clone`` (returning
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
from pysatl_cpd.core.online.ionline_algorithm import (
    OnlineAlgorithm,
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmDescription,
    OnlineAlgorithmState,
)
from pysatl_cpd.core.online.wrappers.online_wrappers import (
    BatchingOnlineAlgorithmWrapper,
    BatchReducer,
    SkippingCondition,
    SkippingOnlineAlgorithmWrapper,
)

__all__ = [
    "OnlineAlgorithm",
    "OnlineAlgorithmState",
    "OnlineAlgorithmConfiguration",
    "OnlineAlgorithmDescription",
    "OnlineDetectionStepResult",
    "OnlineDetectionTrace",
    "OnlineDetector",
    "OnlineResetDetector",
    "SkippingCondition",
    "BatchReducer",
    "SkippingOnlineAlgorithmWrapper",
    "BatchingOnlineAlgorithmWrapper",
]
