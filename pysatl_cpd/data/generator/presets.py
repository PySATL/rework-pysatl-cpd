# -*- coding: ascii -*-
"""Preset scenario builders."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence

from pysatl_cpd.data.dataset import Dataset
from pysatl_cpd.data.generator.providers import build_pandas_labeled_data
from pysatl_cpd.data.generator.series import GenericSeriesGenerator
from pysatl_cpd.data.generator.specs import (
    DistributionSpec,
    IndependentColumnsSpec,
    MultivariateNormalSpec,
    NormalSpec,
    ScenarioSpec,
    SegmentPlan,
    SegmentSpec,
)
from pysatl_cpd.data.typedefs import StateDescriptor, frozendict

PRESET_SCENARIOS: frozendict[str, ScenarioSpec] = frozendict()


def get_preset_scenario(name: str) -> ScenarioSpec:
    """
    Retrieve a preset scenario specification by name.

    Parameters
    ----------
    name
        Name of the preset scenario to retrieve.

    Returns
    -------
    spec
        The preset scenario specification.

    Raises
    ------
    ValueError
        If the preset name is unknown.
    """
    if name not in PRESET_SCENARIOS:
        raise ValueError(f"Unknown preset scenario '{name}'")
    return PRESET_SCENARIOS[name]


def build_preset_scenario(
    name: str,
    *,
    feature_names: Sequence[str],
    series_length: int,
    num_segments: int,
) -> ScenarioSpec:
    """
    Build a preset scenario specification with specified parameters.

    Parameters
    ----------
    name
        Name of the preset scenario.
    feature_names
        Names of the features in the series.
    series_length
        Total length of each series.
    num_segments
        Number of segments in the series.

    Returns
    -------
    spec
        The constructed scenario specification.

    Raises
    ------
    ValueError
        If series_length or num_segments is not positive, or feature_names
        is empty.
    """
    if series_length <= 0:
        raise ValueError("Series length must be positive")
    if num_segments <= 0:
        raise ValueError("Number of segments must be positive")
    if not feature_names:
        raise ValueError("Feature names must not be empty")

    segment_lengths = _build_segment_lengths(series_length=series_length, num_segments=num_segments)
    segment_types = tuple(_segment_type_for_index(name, index) for index in range(num_segments))
    segments = tuple(
        SegmentSpec(
            plan_name=segment_type,
            length=segment_lengths[index],
        )
        for index, segment_type in enumerate(segment_types)
    )
    plans = frozendict.from_mapping(
        {
            segment_type: SegmentPlan(
                distribution=_distribution_for_segment_type(
                    preset=name,
                    segment_type=segment_type,
                    feature_names=tuple(feature_names),
                ),
                state=StateDescriptor(type=segment_type),
                name=segment_type,
            )
            for segment_type in dict.fromkeys(segment_types)
        }
    )
    return ScenarioSpec(name=name, segments=segments, plans=plans, metadata=frozendict(preset=name))


def preset_dataset(
    preset: str,
    *,
    n_series: int,
    seed: int,
    series_length: int = 1200,
    n_features: int | None = None,
    num_segments: int = 3,
) -> Dataset:
    """Build a pandas-backed dataset from a preset scenario specification.

    Parameters
    ----------
    preset
        Name of the preset scenario to use.
    n_series
        Number of series to generate.
    seed
        Random seed for reproducibility.
    series_length
        Length of each generated series.
    n_features
        Number of features. Defaults to 3 for ``3d_mean_shifts``
        and 2 for all other presets.
    num_segments
        Number of segments per series.

    Returns
    -------
    dataset
        Dataset containing the generated labeled series.
    """

    if n_features is None:
        n_features = 3 if preset == "3d_mean_shifts" else 2

    feature_names = tuple(f"feature_{index}" for index in range(n_features))
    scenario = build_preset_scenario(
        preset,
        feature_names=feature_names,
        series_length=series_length,
        num_segments=num_segments,
    )
    generator = GenericSeriesGenerator(seed=seed)
    providers = [
        build_pandas_labeled_data(
            generator.generate_from_scenario(scenario, name=f"{preset}_series_{index:04d}"),
            name=f"{preset}_series_{index:04d}",
        )
        for index in range(n_series)
    ]
    return Dataset(providers)


def _build_segment_lengths(*, series_length: int, num_segments: int) -> list[int]:
    """
    Compute segment lengths by dividing series length among segments.

    Parameters
    ----------
    series_length
        Total length of the series.
    num_segments
        Number of segments to divide into.

    Returns
    -------
    lengths
        List of segment lengths.
    """
    base_length, remainder = divmod(series_length, num_segments)
    return [base_length + (1 if index < remainder else 0) for index in range(num_segments)]


def _segment_type_for_index(preset: str, segment_index: int) -> str:
    """
    Determine segment type for a given preset and segment index.

    Parameters
    ----------
    preset
        Name of the preset scenario.
    segment_index
        Index of the segment.

    Returns
    -------
    segment_type
        Type of the segment.
    """
    if preset == "no_shifts":
        return "baseline"
    if preset == "extreme_mean_shifts":
        return ("baseline", "extreme_high", "extreme_low")[segment_index % 3]
    if preset == "mixed_shifts":
        return ("baseline", "mean_shift", "variance_shift")[segment_index % 3]
    return ("baseline", "alternative_1", "alternative_2")[segment_index % 3]


def _distribution_for_segment_type(
    *,
    preset: str,
    segment_type: str,
    feature_names: tuple[str, ...],
) -> DistributionSpec:
    """
    Get distribution specification for a given preset and segment type.

    Parameters
    ----------
    preset
        Name of the preset scenario.
    segment_type
        Type of the segment.
    feature_names
        Names of the features.

    Returns
    -------
    distribution
        Distribution specification for the segment.

    Raises
    ------
    ValueError
        If the preset is not supported.
    """
    if preset in {
        "mean_shifts",
        "no_shifts",
        "variance_shifts",
        "covariance_shifts",
        "extreme_mean_shifts",
        "3d_mean_shifts",
    }:
        return _multivariate_distribution_for_segment_type(
            preset=preset,
            segment_type=segment_type,
            feature_names=feature_names,
        )
    if preset == "mixed_shifts":
        if segment_type == "variance_shift":
            return MultivariateNormalSpec(
                means=frozendict.from_mapping(dict.fromkeys(feature_names, 0.0)),
                covariance=tuple(4.0 for _ in feature_names),
            )
        if segment_type == "mean_shift":
            return IndependentColumnsSpec(
                columns=frozendict.from_mapping(
                    {feature_name: NormalSpec(mean=2.5, std=1.0) for feature_name in feature_names}
                )
            )
        return IndependentColumnsSpec(
            columns=frozendict.from_mapping(
                {feature_name: NormalSpec(mean=0.0, std=1.0) for feature_name in feature_names}
            )
        )
    raise ValueError(f"Unsupported preset '{preset}'")


def _multivariate_distribution_for_segment_type(
    *,
    preset: str,
    segment_type: str,
    feature_names: tuple[str, ...],
) -> MultivariateNormalSpec:
    """
    Get multivariate normal distribution for a preset and segment type.

    Parameters
    ----------
    preset
        Name of the preset scenario.
    segment_type
        Type of the segment.
    feature_names
        Names of the features.

    Returns
    -------
    distribution
        Multivariate normal distribution specification.

    Raises
    ------
    ValueError
        If the preset is not supported.
    """
    n_features = len(feature_names)
    zero_mean = frozendict.from_mapping(dict.fromkeys(feature_names, 0.0))

    if preset == "mean_shifts":
        if segment_type == "alternative_1":
            return MultivariateNormalSpec(
                means=frozendict.from_mapping(dict.fromkeys(feature_names, 2.0)),
                covariance=tuple(1.0 for _ in feature_names),
            )
        if segment_type == "alternative_2":
            return MultivariateNormalSpec(
                means=frozendict.from_mapping(dict.fromkeys(feature_names, -2.0)),
                covariance=tuple(1.0 for _ in feature_names),
            )
        return MultivariateNormalSpec(means=zero_mean, covariance=tuple(1.0 for _ in feature_names))

    if preset == "variance_shifts":
        if segment_type == "alternative_1":
            return MultivariateNormalSpec(means=zero_mean, covariance=tuple(3.0 for _ in feature_names))
        if segment_type == "alternative_2":
            return MultivariateNormalSpec(means=zero_mean, covariance=tuple(0.25 for _ in feature_names))
        return MultivariateNormalSpec(means=zero_mean, covariance=tuple(1.0 for _ in feature_names))

    if preset == "covariance_shifts":
        if segment_type == "alternative_1":
            return MultivariateNormalSpec(
                means=zero_mean,
                covariance=tuple(
                    tuple(value for value in row) for row in _correlated_covariance(n_features, 1.0 / n_features)
                ),
            )
        if segment_type == "alternative_2":
            return MultivariateNormalSpec(
                means=zero_mean,
                covariance=tuple(
                    tuple(value for value in row) for row in _correlated_covariance(n_features, -1.0 / n_features)
                ),
            )
        return MultivariateNormalSpec(means=zero_mean, covariance=tuple(1.0 for _ in feature_names))

    if preset == "no_shifts":
        return MultivariateNormalSpec(means=zero_mean, covariance=tuple(1.0 for _ in feature_names))

    if preset == "extreme_mean_shifts":
        if segment_type == "extreme_high":
            return MultivariateNormalSpec(
                means=frozendict.from_mapping(dict.fromkeys(feature_names, 5.0)),
                covariance=tuple(1.0 for _ in feature_names),
            )
        if segment_type == "extreme_low":
            return MultivariateNormalSpec(
                means=frozendict.from_mapping(dict.fromkeys(feature_names, -5.0)),
                covariance=tuple(1.0 for _ in feature_names),
            )
        return MultivariateNormalSpec(means=zero_mean, covariance=tuple(1.0 for _ in feature_names))

    if preset == "3d_mean_shifts":
        if segment_type == "alternative_1":
            return MultivariateNormalSpec(
                means=frozendict.from_mapping(dict.fromkeys(feature_names, 2.0)),
                covariance=tuple(1.0 for _ in feature_names),
            )
        if segment_type == "alternative_2":
            return MultivariateNormalSpec(
                means=frozendict.from_mapping(dict.fromkeys(feature_names, -2.0)),
                covariance=tuple(1.0 for _ in feature_names),
            )
        return MultivariateNormalSpec(means=zero_mean, covariance=tuple(1.0 for _ in feature_names))

    raise ValueError(f"Unsupported preset '{preset}'")


def _correlated_covariance(n_features: int, correlation: float) -> tuple[tuple[float, ...], ...]:
    """
    Generate a correlation matrix with specified off-diagonal correlation.

    Parameters
    ----------
    n_features
        Number of features (matrix dimension).
    correlation
        Off-diagonal correlation value.

    Returns
    -------
    covariance
        Covariance matrix as tuple of tuples.
    """
    return tuple(
        tuple(1.0 if row == column else correlation for column in range(n_features)) for row in range(n_features)
    )
