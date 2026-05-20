# -*- coding: ascii -*-
"""Generic synthetic series generator."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping, Sequence

import numpy as np

from pysatl_cpd.data.generator.models import GeneratedSeries
from pysatl_cpd.data.generator.segments import SegmentGenerator, feature_names_for_distribution, sample_distribution
from pysatl_cpd.data.generator.specs import ScenarioSpec
from pysatl_cpd.data.typedefs import NumericArray, SegmentInfo, StateDescriptor, frozendict


class GenericSeriesGenerator:
    """
    Generic series generator for synthetic time series data.

    This generator creates synthetic series from scenario specifications
    or segment generators, supporting various distribution types including
    univariate and multivariate distributions.

    Parameters
    ----------
    seed
        Seed for reproducible generation. None uses system entropy.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def generate_from_scenario(
        self,
        scenario: ScenarioSpec,
        *,
        name: str | None = None,
    ) -> GeneratedSeries:
        """
        Generate a series from a scenario specification.

        Parameters
        ----------
        scenario
            Scenario specification defining segments and distributions.
        name
            Optional name for the generated series.

        Returns
        -------
        GeneratedSeries
            Generated series with data, feature names, and segment info.

        Raises
        ------
        ValueError
            If scenario segments have mismatched feature names or the
            scenario contains no segments.
        """
        parts: list[NumericArray] = []
        segment_rows: list[SegmentInfo] = []
        expected_feature_names: tuple[str, ...] | None = None
        start = 0

        for segment_num, segment_spec in enumerate(scenario.segments):
            segment_plan = scenario.plans[segment_spec.plan_name]
            feature_names = feature_names_for_distribution(segment_plan.distribution)
            if expected_feature_names is None:
                expected_feature_names = feature_names
            elif feature_names != expected_feature_names:
                raise ValueError("All scenario segment distributions must use the same feature names and order")

            sampled = sample_distribution(segment_plan.distribution, segment_spec.length, self._rng)
            end = start + segment_spec.length - 1
            state = segment_plan.state or StateDescriptor(type=segment_spec.plan_name)
            segment_rows.append(
                SegmentInfo(
                    segment_num=segment_num,
                    segment_start=start,
                    segment_end=end,
                    state=state,
                )
            )
            parts.append(sampled)
            start = end + 1

        if expected_feature_names is None:
            raise ValueError("Scenario must contain at least one segment")

        return GeneratedSeries(
            name=name,
            feature_names=expected_feature_names,
            data=np.concatenate(parts, axis=0),
            segments=tuple(segment_rows),
            metadata=scenario.metadata,
        )

    def generate_from_segment_generators(
        self,
        segment_generators: Mapping[str, SegmentGenerator] | Sequence[tuple[str, SegmentGenerator]],
        *,
        name: str | None = None,
    ) -> GeneratedSeries:
        """
        Generate a series from segment generators.

        Parameters
        ----------
        segment_generators
            Mapping or sequence of segment type to generator pairs.
        name
            Optional name for the generated series.

        Returns
        -------
        GeneratedSeries
            Generated series with data, feature names, and segment info.

        Raises
        ------
        ValueError
            If segment generators is empty or have mismatched feature names.
        """
        items = (
            list(segment_generators.items()) if isinstance(segment_generators, Mapping) else list(segment_generators)
        )
        if not items:
            raise ValueError("Segment generators must not be empty")

        parts: list[NumericArray] = []
        segment_rows: list[SegmentInfo] = []
        expected_feature_names: tuple[str, ...] | None = None
        start = 0

        for segment_num, (segment_type, generator) in enumerate(items):
            generated = generator.generate(rng=self._rng)
            if expected_feature_names is None:
                expected_feature_names = generated.feature_names
            elif generated.feature_names != expected_feature_names:
                raise ValueError("All segment generators must use the same feature names and order")

            end = start + generator.length - 1
            state = generated.segment_info.state or StateDescriptor(type=segment_type)
            segment_rows.append(
                SegmentInfo(
                    segment_num=segment_num,
                    segment_start=start,
                    segment_end=end,
                    state=state,
                )
            )
            parts.append(generated.data)
            start = end + 1

        if expected_feature_names is None:
            raise ValueError("Segment generators must not be empty")

        return GeneratedSeries(
            name=name,
            feature_names=expected_feature_names,
            data=np.concatenate(parts, axis=0),
            segments=tuple(segment_rows),
            metadata=frozendict(),
        )
