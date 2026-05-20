# -*- coding: ascii -*-
"""Formal generator specifications."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Literal

from pysatl_cpd.data.typedefs import NumericArray, StateDescriptor, StateValue, frozendict


@dataclass(frozen=True, slots=True, init=False)
class UnivariateDistributionSpec:
    """
    Core-backed univariate distribution specification.

    Stores the target core family, an optional parametrization name,
    and the distribution parameters passed through as keyword arguments.
    """

    family: str
    parametrization_name: str | None
    parameters: frozendict[str, float]
    kind: Literal["univariate"] = field(init=False, default="univariate")

    def __init__(self, family: str, parametrization_name: str | None = None, **parameters: float) -> None:
        if not family:
            raise ValueError("Univariate distribution family must not be empty")

        normalized_parameters: dict[str, float] = {}
        for key, value in parameters.items():
            if not key:
                raise ValueError("Univariate distribution parameter names must not be empty")
            normalized_parameters[key] = float(value)

        object.__setattr__(self, "kind", "univariate")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "parametrization_name", parametrization_name)
        object.__setattr__(self, "parameters", frozendict.from_mapping(normalized_parameters))


@dataclass(frozen=True, slots=True)
class MultivariateNormalSpec:
    """
    Multivariate normal distribution specification.

    Defines parameters for a multivariate normal distribution
    with named features and covariance structure.
    """

    kind: Literal["multivariate_normal"] = "multivariate_normal"
    means: frozendict[str, float] = field(default_factory=frozendict)
    covariance: NumericArray | tuple[tuple[float, ...], ...] | tuple[float, ...] | float = 1.0

    def __post_init__(self) -> None:
        """Validate multivariate-normal specification fields."""
        if not self.means:
            raise ValueError("Multivariate normal means must not be empty")


@dataclass(frozen=True, slots=True)
class IndependentColumnsSpec:
    """
    Independent columns distribution specification.

    Defines a distribution where each feature column has its
    own independent univariate distribution.
    """

    kind: Literal["independent_columns"] = "independent_columns"
    columns: frozendict[str, UnivariateDistributionSpec] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        """Validate independent-column specification fields."""
        if not self.columns:
            raise ValueError("Independent column distributions must not be empty")


type DistributionSpec = MultivariateNormalSpec | IndependentColumnsSpec | UnivariateDistributionSpec


@dataclass(frozen=True, slots=True)
class SegmentSpec:
    """
    Segment specification within a scenario.

    Defines a single segment with a reference to a segment plan
    and the length of the segment.
    """

    plan_name: str
    length: int

    def __post_init__(self) -> None:
        """Validate segment specification fields."""
        if not self.plan_name:
            raise ValueError("Segment plan name must not be empty")
        if self.length <= 0:
            raise ValueError("Segment length must be positive")


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    """
    Plan for generating a segment.

    Defines the distribution, state, and metadata for a
    specific segment type within a scenario.
    """

    distribution: DistributionSpec
    state: StateDescriptor | None = None
    metadata: frozendict[str, Hashable] = field(default_factory=frozendict)
    name: str | None = None


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """
    Scenario specification for synthetic data generation.

    Defines a complete scenario with named segments, segment
    plans, and metadata for generating synthetic series.
    """

    name: str
    segments: tuple[SegmentSpec, ...]
    plans: frozendict[str, SegmentPlan]
    metadata: frozendict[str, Hashable] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        """Validate scenario specification fields."""
        if not self.name:
            raise ValueError("Scenario name must not be empty")
        if not self.segments:
            raise ValueError("Scenario must contain at least one segment")
        if not self.plans:
            raise ValueError("Scenario plans must not be empty")
        missing = sorted({segment.plan_name for segment in self.segments}.difference(self.plans))
        if missing:
            raise ValueError(f"Scenario is missing segment plans: {missing}")


def freeze_float_mapping(mapping: Mapping[str, float]) -> frozendict[str, float]:
    """
    Freeze a mutable float mapping to a frozendict.

    Parameters
    ----------
    mapping
        Mutable mapping of string keys to float values.

    Returns
    -------
    frozendict
        Immutable frozendict with float values.
    """
    return frozendict.from_mapping({key: float(value) for key, value in mapping.items()})


def freeze_state_mapping(mapping: Mapping[str, StateValue]) -> StateDescriptor:
    """
    Freeze a mutable state mapping to a StateDescriptor.

    Parameters
    ----------
    mapping
        Mutable mapping of state key-value pairs.

    Returns
    -------
    StateDescriptor
        Immutable state descriptor from the mapping.
    """
    return StateDescriptor(**mapping)


def freeze_univariate_mapping(
    mapping: Mapping[str, UnivariateDistributionSpec],
) -> frozendict[str, UnivariateDistributionSpec]:
    """
    Freeze a mutable univariate distribution mapping.

    Parameters
    ----------
    mapping
        Mutable mapping of distribution specifications.

    Returns
    -------
    frozendict
        Immutable frozendict with univariate distributions.
    """
    return frozendict.from_mapping(mapping)


def freeze_distribution_mapping(mapping: Mapping[str, DistributionSpec]) -> frozendict[str, DistributionSpec]:
    """
    Freeze a mutable distribution mapping.

    Parameters
    ----------
    mapping
        Mutable mapping of distribution specifications.

    Returns
    -------
    frozendict
        Immutable frozendict with distributions.
    """
    return frozendict.from_mapping(mapping)
