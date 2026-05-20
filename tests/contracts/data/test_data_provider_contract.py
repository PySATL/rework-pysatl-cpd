# -*- coding: ascii -*-
"""
Tests for data provider contract.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.data.typedefs import UnlabeledTimeseriesAnnotation


def _iter_values(provider: object) -> list[object]:
    values = list(provider)  # type: ignore[arg-type]
    normalized: list[object] = []
    for value in values:
        if isinstance(value, np.ndarray):
            normalized.append(value.tolist())
        elif hasattr(value, "item"):
            normalized.append(value.item())
        else:
            normalized.append(value)
    return normalized


class DataProviderContract:
    """Reusable behavioral checks for DataProvider implementations."""

    @pytest.fixture
    def provider(self):
        raise NotImplementedError

    @pytest.fixture
    def other_provider(self):
        raise NotImplementedError

    @pytest.fixture
    def custom_annotation(self) -> UnlabeledTimeseriesAnnotation:
        return UnlabeledTimeseriesAnnotation(name="custom")

    def test_len_non_negative(self, provider) -> None:
        assert len(provider) >= 0

    def test_iter_yields_len_items(self, provider) -> None:
        assert len(list(provider)) == len(provider)

    def test_name_matches_annotation_name(self, provider) -> None:
        assert provider.name == provider.annotation.name

    def test_cut_returns_same_concrete_type(self, provider) -> None:
        assert type(provider.cut(1, 3)) is type(provider)

    def test_cut_uses_inclusive_bounds(self, provider) -> None:
        assert len(provider.cut(1, 3)) == 3

    def test_cut_preserves_order(self, provider) -> None:
        assert _iter_values(provider.cut(1, 3)) == _iter_values(provider)[1:4]

    def test_cut_full_range_matches_original(self, provider) -> None:
        assert _iter_values(provider.cut(0, len(provider) - 1)) == _iter_values(provider)

    def test_cut_custom_annotation_is_used(self, provider, custom_annotation: UnlabeledTimeseriesAnnotation) -> None:
        assert provider.cut(1, 3, annotation=custom_annotation).annotation is custom_annotation

    def test_cut_rejects_negative_start(self, provider) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            provider.cut(-1, 1)

    def test_cut_rejects_stop_lt_start(self, provider) -> None:
        with pytest.raises(ValueError, match="greater than or equal"):
            provider.cut(3, 1)

    def test_cut_rejects_stop_overflow(self, provider) -> None:
        with pytest.raises(ValueError, match="exceeds data length"):
            provider.cut(0, len(provider))

    def test_default_slice_annotation_changes_name(self, provider) -> None:
        assert provider.default_slice_annotation(1, 3).name == f"{provider.name}[1:3]"

    def test_merge_empty_raises(self, provider) -> None:
        with pytest.raises(ValueError, match="at least one"):
            type(provider).merge([])

    def test_merge_length_is_additive(self, provider, other_provider) -> None:
        merged = type(provider).merge([provider, other_provider])
        assert len(merged) == len(provider) + len(other_provider)

    def test_merge_preserves_order(self, provider, other_provider) -> None:
        merged = type(provider).merge([provider, other_provider])
        assert _iter_values(merged) == _iter_values(provider) + _iter_values(other_provider)

    def test_merge_single_provider_preserves_data(self, provider) -> None:
        merged = type(provider).merge([provider])
        assert type(merged) is type(provider)
        assert _iter_values(merged) == _iter_values(provider)

    def test_merge_custom_annotation_builder_is_used(self, provider, other_provider) -> None:
        annotation = UnlabeledTimeseriesAnnotation(name="merged-custom", source="contract")
        merged = type(provider).merge([provider, other_provider], annotation_builder=lambda _: annotation)
        assert merged.annotation is annotation

    def test_default_merge_annotation_builder_sets_source(self, provider, other_provider) -> None:
        builder = type(provider).default_merge_annotation_builder(source="contract-source")
        merged = type(provider).merge([provider, other_provider], annotation_builder=builder)
        assert merged.annotation.source == "contract-source"

    def test_or_equivalent_to_merge(self, provider, other_provider) -> None:
        assert _iter_values(provider | other_provider) == _iter_values(type(provider).merge([provider, other_provider]))
