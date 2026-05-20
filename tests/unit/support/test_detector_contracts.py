# -*- coding: ascii -*-
"""
Tests for detector contracts.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.data.providers.transformers.base import IDataTransformer
from tests.contracts.core.test_change_point_detector_contract import ChangePointDetectorContract
from tests.contracts.core.test_online_detector_contract import OnlineDetectorContract
from tests.support.algorithms import (
    CountingAlgorithm,
    CountingAlgorithmConfig,
    IdentityTransformer,
    RecordingDetector,
    RecordingOnlineDetector,
)


class _BrokenTransformer(IDataTransformer):
    @property
    def annotation(self) -> str:
        return "broken"

    def transform(self, provider):
        return object()


class TestRecordingDetectorContract(ChangePointDetectorContract):
    @pytest.fixture
    def detector(self) -> RecordingDetector:
        return RecordingDetector()

    def test_detect_applies_transformer(self, data_provider) -> None:
        transformer = IdentityTransformer()
        detector = RecordingDetector(data_transformer=transformer)
        detector.detect(data_provider)
        assert transformer.last_provider is data_provider
        assert detector.last_provider is data_provider

    def test_detect_rejects_invalid_transformer_output(self, data_provider) -> None:
        detector = RecordingDetector(data_transformer=_BrokenTransformer())
        with pytest.raises(TypeError, match="DataProvider"):
            detector.detect(data_provider)


class TestRecordingOnlineDetectorContract(OnlineDetectorContract):
    @pytest.fixture
    def detector(self) -> RecordingOnlineDetector:
        return RecordingOnlineDetector(CountingAlgorithm(CountingAlgorithmConfig(threshold=5.0)))
