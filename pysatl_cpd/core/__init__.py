# -*- coding: ascii -*-
"""
Core module for PySATL CPD.

Provides the fundamental building blocks for both offline (batch) and online
(streaming) change-point detection: abstract detector interfaces, unified
result containers, single-run analysis helpers, and the full online detection
API.

The top-level package re-exports the most commonly used classes and also
exposes the ``online`` subpackage for composable streaming algorithms.

.. raw:: html

    <h2>Public API</h2>

Offline / batch components (from ``change_point_detector``):

- ``ChangePointDetector`` -- Abstract base class for batch change-point
  detectors. Subclasses implement ``_detect`` and ``clone``.
- ``ChangePointDetectorDescription`` -- Frozen dataclass holding a
  human-readable name and parameters for a detector.

Result containers (from ``detection_trace``):

- ``DetectionTrace`` -- Unified container storing detected changepoint
  indices and the detector that produced them.

Single-run analysis (from ``single_run``):

- ``SingleRun`` -- Pairs a detection trace with the labeled provider it
  was produced from.
- ``SingleRunDescription`` -- Hashable description combining detector
  and provider metadata for one run.

Online subpackage (``pysatl_cpd.core.online``):

- ``online`` -- The full online detection API, including algorithm
  interfaces, concrete detectors, traces, and composable wrappers.
  See the ``online`` subpackage docstring for details.

.. raw:: html

    <h2>Examples</h2>

Examples
--------

Run an online reset detector over a univariate provider:

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

Create a DetectionTrace and inspect detected change points:

>>> from pysatl_cpd.core import ChangePointDetector, DetectionTrace
>>> from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
>>> desc = ChangePointDetectorDescription(name="example-detector")
>>> trace = DetectionTrace(
...     detected_change_points=[10, 42, 99],
...     detector_description=desc,
... )
>>> trace.detected_change_points
[10, 42, 99]

Notes
-----
- All change-point indices are zero-based.
- The ``online`` subpackage re-exports algorithm interfaces, detectors,
  traces, and wrappers. Import from ``pysatl_cpd.core.online`` for
  streaming workflows.
- Directories ``offline/``, ``data_providers/``, and ``data_transformers/``
  are legacy placeholders without ``__init__.py`` and are not part of the
  public API.
"""

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.core import online
from pysatl_cpd.core.change_point_detector import ChangePointDetector
from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.core.online import (
    BatchingOnlineAlgorithmWrapper,
    BatchReducer,
    OnlineAlgorithm,
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmDescription,
    OnlineAlgorithmState,
    OnlineDetectionStepResult,
    OnlineDetectionTrace,
    OnlineDetector,
    OnlineResetDetector,
    SkippingCondition,
    SkippingOnlineAlgorithmWrapper,
)
from pysatl_cpd.core.single_run import SingleRun, SingleRunDescription

__all__ = [
    "BatchingOnlineAlgorithmWrapper",
    "BatchReducer",
    "ChangePointDetector",
    "DetectionTrace",
    "online",
    "OnlineAlgorithm",
    "OnlineAlgorithmConfiguration",
    "OnlineAlgorithmDescription",
    "OnlineAlgorithmState",
    "OnlineDetectionStepResult",
    "OnlineDetectionTrace",
    "OnlineDetector",
    "OnlineResetDetector",
    "SingleRun",
    "SingleRunDescription",
    "SkippingCondition",
    "SkippingOnlineAlgorithmWrapper",
]
