# -*- coding: ascii -*-
"""Test support utilities."""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from tests.support.algorithms import (
    CountingAlgorithm,
    CountingAlgorithmConfig,
    CountingAlgorithmState,
    IdentityTransformer,
    RecordingDetector,
    RecordingOnlineDetector,
)
from tests.support.datasets import make_dataset, make_state_dataset, make_state_dataset_source
from tests.support.golden import load_json_golden
from tests.support.metrics import MockDetectionTrace, MockLabeledDataForMetrics, MockOnlineDetectionTrace
from tests.support.providers import (
    make_multivariate_labeled,
    make_multivariate_provider,
    make_no_change_labeled,
    make_pandas_labeled,
    make_pandas_provider,
    make_segments,
    make_state,
    make_transition,
    make_univariate_labeled,
    make_univariate_provider,
)

__all__ = [
    "CountingAlgorithm",
    "CountingAlgorithmConfig",
    "CountingAlgorithmState",
    "IdentityTransformer",
    "RecordingDetector",
    "RecordingOnlineDetector",
    "load_json_golden",
    "make_dataset",
    "make_multivariate_labeled",
    "make_multivariate_provider",
    "make_no_change_labeled",
    "make_pandas_labeled",
    "make_pandas_provider",
    "make_segments",
    "make_state",
    "make_state_dataset",
    "make_state_dataset_source",
    "make_transition",
    "make_univariate_labeled",
    "make_univariate_provider",
    "MockDetectionTrace",
    "MockLabeledDataForMetrics",
    "MockOnlineDetectionTrace",
]
