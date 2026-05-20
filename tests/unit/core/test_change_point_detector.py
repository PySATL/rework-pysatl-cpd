# -*- coding: ascii -*-
"""
Tests for change point detector.
"""

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC

import pytest

from pysatl_cpd.core.change_point_detector import ChangePointDetector, ChangePointDetectorDescription
from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.data.providers.data_provider import DataProvider
from pysatl_cpd.data.providers.transformers.base import IDataTransformer
from pysatl_cpd.data.typedefs import TimeseriesAnnotation
from tests.support.core import MockDataProvider


class ConcreteDetector(ChangePointDetector[float]):
    """Concrete detector for testing."""

    @property
    def description(self) -> ChangePointDetectorDescription:
        return ChangePointDetectorDescription(name="ConcreteDetector")

    def _detect(self, data: DataProvider[float, TimeseriesAnnotation]) -> DetectionTrace:
        self.last_data = data
        return DetectionTrace(
            detected_change_points=[],
            detector_description=self.description,
        )

    def clone(self) -> "ConcreteDetector":
        return ConcreteDetector(data_transformer=self.data_transformer)


type MockProvider = DataProvider[float, TimeseriesAnnotation]


class MockTransformer(IDataTransformer[MockProvider, MockProvider]):
    def __init__(self, transformed: DataProvider[float, TimeseriesAnnotation] | object) -> None:
        self.transformed = transformed

    def transform(self, provider: MockProvider) -> MockProvider:
        return self.transformed  # type: ignore[return-value]

    @property
    def annotation(self) -> str:
        return "mock annotation"


class TestChangePointDetector:
    """Tests for ChangePointDetector abstract source class."""

    def test_is_abstract_base_class(self) -> None:
        assert issubclass(ChangePointDetector, ABC)

    def test_contract_members_are_abstract(self) -> None:
        assert getattr(ChangePointDetector.description.fget, "__isabstractmethod__", False)
        assert getattr(ChangePointDetector._detect, "__isabstractmethod__", False)
        assert getattr(ChangePointDetector.clone, "__isabstractmethod__", False)

    def test_cannot_instantiate_without_required_members(self) -> None:
        with pytest.raises(TypeError):
            ChangePointDetector()

    def test_description_must_be_implemented(self) -> None:
        """Subclasses must implement description property."""
        detector = ConcreteDetector()
        desc = detector.description
        assert isinstance(desc, ChangePointDetectorDescription)
        assert desc.name == "ConcreteDetector"

    def test_detect_must_be_implemented(self) -> None:
        """Subclasses must implement detect method."""
        detector = ConcreteDetector()
        provider = MockDataProvider()
        result = detector.detect(provider)
        assert isinstance(result, DetectionTrace)

    def test_detect_returns_correct_trace(self) -> None:
        """detect() should return a DetectionTrace with correct description."""
        detector = ConcreteDetector()
        provider = MockDataProvider()
        result = detector.detect(provider)
        assert result.detector_description.name == "ConcreteDetector"

    def test_detect_applies_data_transformer(self) -> None:
        provider = MockDataProvider([1.0], name="source")
        transformed = MockDataProvider([2.0], name="transformed")
        detector = ConcreteDetector(data_transformer=MockTransformer(transformed))

        detector.detect(provider)

        assert detector.last_data is transformed

    def test_detect_rejects_invalid_transformer_output(self) -> None:
        detector = ConcreteDetector(data_transformer=MockTransformer(object()))

        with pytest.raises(TypeError, match="data_transformer must return a DataProvider"):
            detector.detect(MockDataProvider())

    def test_clone_preserves_configuration(self) -> None:
        transformer = MockTransformer(MockDataProvider())
        detector = ConcreteDetector(data_transformer=transformer)

        clone = detector.clone()

        assert clone is not detector
        assert clone.data_transformer is transformer
