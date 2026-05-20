# -*- coding: ascii -*-
"""
Tests for change point detector description.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.typedefs import frozendict


class TestChangePointDetectorDescription:
    """Tests for ChangePointDetectorDescription."""

    def test_create_with_name(self) -> None:
        """Should create with name only."""
        desc = ChangePointDetectorDescription(name="TestDetector")
        assert desc.name == "TestDetector"

    def test_create_with_parameters(self) -> None:
        """Should create with parameters."""
        desc = ChangePointDetectorDescription(name="TestDetector", parameters={"threshold": 0.5})
        assert desc.name == "TestDetector"
        assert desc.parameters == {"threshold": 0.5}
        assert isinstance(desc.parameters, frozendict)

    def test_default_parameters_empty(self) -> None:
        """Default parameters should be empty frozen mapping."""
        desc = ChangePointDetectorDescription(name="Test")
        assert desc.parameters == {}
        assert isinstance(desc.parameters, frozendict)

    def test_nested_parameters_are_normalized_to_hashable_values(self) -> None:
        """Nested mutable containers should be normalized on construction."""
        desc = ChangePointDetectorDescription(name="Test", parameters={"x": [1, {"y": 2}]})

        assert desc.parameters == {"x": (1, (("y", 2),))}

    def test_hash_stable(self) -> None:
        """Hash should be stable for same name and params."""
        desc1 = ChangePointDetectorDescription(name="Test", parameters={"x": 1})
        desc2 = ChangePointDetectorDescription(name="Test", parameters={"x": 1})
        assert hash(desc1) == hash(desc2)

    def test_hash_differs_with_diff_params(self) -> None:
        """Hash should differ when parameters differ."""
        desc1 = ChangePointDetectorDescription(name="Test", parameters={"x": 1})
        desc2 = ChangePointDetectorDescription(name="Test", parameters={"x": 2})
        assert hash(desc1) != hash(desc2)

    def test_frozen(self) -> None:
        """Description should be frozen (immutable)."""
        desc = ChangePointDetectorDescription(name="Test")
        with pytest.raises(AttributeError):
            desc.name = "New"  # type: ignore
