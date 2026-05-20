# -*- coding: ascii -*-
"""Configuration loaders for generator scenarios."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from pysatl_cpd.data.generator.specs import (
    DistributionSpec,
    ExponentialSpec,
    IndependentColumnsSpec,
    MultivariateNormalSpec,
    NormalSpec,
    ScenarioSpec,
    SegmentPlan,
    SegmentSpec,
    StudentTSpec,
    UniformSpec,
    UnivariateDistributionSpec,
)
from pysatl_cpd.data.typedefs import StateDescriptor, StateValue, frozendict


def scenario_from_yaml(path: str | Path) -> ScenarioSpec:
    """Load one scenario specification from a YAML file.

    Parameters
    ----------
    path
        Path to the YAML file.

    Returns
    -------
    spec
        Parsed scenario specification.

    Raises
    ------
    ValueError
        If the YAML content is not a mapping.
    """

    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("Scenario YAML must contain a mapping at the top level")
    return scenario_from_mapping(loaded)


def scenarios_from_yaml(path: str | Path) -> dict[str, ScenarioSpec]:
    """Load one or more scenario specifications from a YAML file.

    If the YAML contains a top-level ``scenarios`` key, each entry
    under it is parsed as a named scenario. Otherwise the entire
    file is treated as a single scenario and returned under its
    ``name`` key.

    Parameters
    ----------
    path
        Path to the YAML file.

    Returns
    -------
    scenarios
        Dictionary mapping scenario names to parsed specifications.

    Raises
    ------
    ValueError
        If the YAML structure is invalid.
    """

    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(loaded, Mapping) and "scenarios" in loaded:
        scenarios_raw = loaded["scenarios"]
        if not isinstance(scenarios_raw, Mapping):
            raise ValueError("'scenarios' must be a mapping")
        return {
            str(name): scenario_from_mapping(_require_mapping(raw, f"scenarios.{name}"))
            for name, raw in scenarios_raw.items()
        }
    if not isinstance(loaded, Mapping):
        raise ValueError("Scenario YAML must contain a mapping at the top level")
    scenario = scenario_from_mapping(loaded)
    return {scenario.name: scenario}


def scenario_from_mapping(mapping: Mapping[str, Any]) -> ScenarioSpec:
    """Build a scenario specification from a plain mapping.

    Parameters
    ----------
    mapping
        Dictionary containing ``name``, ``segments``, and ``plans`` keys.

    Returns
    -------
    spec
        Parsed scenario specification.
    """

    name = _require_str(mapping.get("name"), "name")
    segments_raw = _require_sequence(mapping.get("segments"), "segments")
    plans_raw = _require_mapping(mapping.get("plans"), "plans")

    segments = tuple(_parse_segment_spec(raw, f"segments[{index}]") for index, raw in enumerate(segments_raw))
    plans = frozendict.from_mapping(
        {str(plan_name): _parse_segment_plan(raw, f"plans.{plan_name}") for plan_name, raw in plans_raw.items()}
    )
    return ScenarioSpec(
        name=name,
        segments=segments,
        plans=plans,
        metadata=frozendict.from_mapping(_parse_hashable_mapping(mapping.get("metadata", {}), "metadata")),
    )


def _parse_segment_spec(raw: object, path: str) -> SegmentSpec:
    """
    Parse a segment specification from raw data.

    Parameters
    ----------
    raw
        Raw data to parse.
    path
        Path for error messages.

    Returns
    -------
    spec
        Parsed segment specification.
    """
    mapping = _require_mapping(raw, path)
    return SegmentSpec(
        plan_name=_require_str(mapping.get("plan_name"), f"{path}.plan_name"),
        length=_require_int(mapping.get("length"), f"{path}.length"),
    )


def _parse_segment_plan(raw: object, path: str) -> SegmentPlan:
    """
    Parse a segment plan from raw data.

    Parameters
    ----------
    raw
        Raw data to parse.
    path
        Path for error messages.

    Returns
    -------
    plan
        Parsed segment plan.
    """
    mapping = _require_mapping(raw, path)
    distribution = parse_distribution_spec(_require_mapping(mapping.get("distribution"), f"{path}.distribution"))
    state_raw = mapping.get("state")
    return SegmentPlan(
        distribution=distribution,
        state=None if state_raw is None else StateDescriptor(**_parse_state_mapping(state_raw, f"{path}.state")),
        metadata=frozendict.from_mapping(_parse_hashable_mapping(mapping.get("metadata", {}), f"{path}.metadata")),
        name=None if mapping.get("name") is None else _require_str(mapping.get("name"), f"{path}.name"),
    )


def parse_distribution_spec(mapping: Mapping[str, Any]) -> DistributionSpec:
    """Build a distribution specification from a plain mapping.

    Parameters
    ----------
    mapping
        Dictionary containing a ``kind`` key and distribution-specific
        parameters.

    Returns
    -------
    spec
        Parsed distribution specification.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the supported distribution types.
    """

    kind = _require_str(mapping.get("kind"), "distribution.kind")
    if kind == "normal":
        return NormalSpec(
            mean=_optional_float(mapping.get("mean"), "distribution.mean", default=0.0),
            std=_optional_float(mapping.get("std"), "distribution.std", default=1.0),
        )
    if kind == "uniform":
        return UniformSpec(
            low=_optional_float(mapping.get("low"), "distribution.low", default=0.0),
            high=_optional_float(mapping.get("high"), "distribution.high", default=1.0),
        )
    if kind == "exponential":
        return ExponentialSpec(scale=_optional_float(mapping.get("scale"), "distribution.scale", default=1.0))
    if kind == "student_t":
        return StudentTSpec(
            df=_optional_float(mapping.get("df"), "distribution.df", default=5.0),
            loc=_optional_float(mapping.get("loc"), "distribution.loc", default=0.0),
            scale=_optional_float(mapping.get("scale"), "distribution.scale", default=1.0),
        )
    if kind == "multivariate_normal":
        means = _parse_float_mapping(mapping.get("means"), "distribution.means")
        covariance = _parse_covariance(mapping.get("covariance", 1.0), "distribution.covariance")
        return MultivariateNormalSpec(means=frozendict.from_mapping(means), covariance=covariance)
    if kind == "independent_columns":
        columns_raw = _require_mapping(mapping.get("columns"), "distribution.columns")
        columns = {
            str(column): _parse_univariate_distribution(_require_mapping(raw, f"distribution.columns.{column}"))
            for column, raw in columns_raw.items()
        }
        return IndependentColumnsSpec(columns=frozendict.from_mapping(columns))
    raise ValueError(f"Unsupported distribution kind '{kind}'")


def _parse_univariate_distribution(mapping: Mapping[str, Any]) -> UnivariateDistributionSpec:
    """
    Parse a univariate distribution specification.

    Parameters
    ----------
    mapping
        Mapping containing distribution specification.

    Returns
    -------
    spec
        Parsed univariate distribution specification.

    Raises
    ------
    ValueError
        If the distribution is not a univariate type.
    """
    distribution = parse_distribution_spec(mapping)
    if not isinstance(distribution, NormalSpec | UniformSpec | ExponentialSpec | StudentTSpec):
        raise ValueError("Independent column distributions must be univariate")
    return distribution


def _parse_float_mapping(raw: object, path: str) -> dict[str, float]:
    """
    Parse a mapping of string keys to float values.

    Parameters
    ----------
    raw
        Raw data to parse.
    path
        Path for error messages.

    Returns
    -------
    mapping
        Dictionary of string to float.
    """
    mapping = _require_mapping(raw, path)
    return {str(key): _require_float(value, f"{path}.{key}") for key, value in mapping.items()}


def _parse_state_mapping(raw: object, path: str) -> dict[str, StateValue]:
    """
    Parse a state mapping from raw data.

    Parameters
    ----------
    raw
        Raw data to parse.
    path
        Path for error messages.

    Returns
    -------
    mapping
        Dictionary of string to state value.

    Raises
    ------
    ValueError
        If any value is not a string, integer, float, or boolean.
    """
    mapping = _require_mapping(raw, path)
    parsed: dict[str, StateValue] = {}
    for key, value in mapping.items():
        if not isinstance(value, str | int | float | bool):
            raise ValueError(f"{path}.{key} must be a string, integer, float, or boolean")
        parsed[str(key)] = value
    return parsed


def _parse_hashable_mapping(raw: object, path: str) -> dict[str, Hashable]:
    """
    Parse a mapping with hashable values.

    Parameters
    ----------
    raw
        Raw data to parse.
    path
        Path for error messages.

    Returns
    -------
    mapping
        Dictionary of string to hashable value.

    Raises
    ------
    ValueError
        If any value is not hashable.
    """
    mapping = _require_mapping(raw, path)
    parsed: dict[str, Hashable] = {}
    for key, value in mapping.items():
        if not isinstance(value, Hashable):
            raise ValueError(f"{path}.{key} must be hashable")
        parsed[str(key)] = value
    return parsed


def _parse_covariance(raw: object, path: str) -> tuple[tuple[float, ...], ...] | tuple[float, ...] | float:
    """
    Parse a covariance matrix from raw data.

    Parameters
    ----------
    raw
        Raw data to parse (number, list, or nested list).
    path
        Path for error messages.

    Returns
    -------
    covariance
        Parsed covariance as float, list, or nested tuple.

    Raises
    ------
    ValueError
        If the raw data is not a valid covariance specification.
    """
    if isinstance(raw, int | float):
        return float(raw)
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError(f"{path} must be a number or sequence")
    values = list(raw)
    if all(isinstance(value, int | float) for value in values):
        return tuple(float(value) for value in values)
    rows: list[tuple[float, ...]] = []
    for row_index, row in enumerate(values):
        if not isinstance(row, Sequence) or isinstance(row, str | bytes):
            raise ValueError(f"{path}[{row_index}] must be a sequence")
        rows.append(tuple(_require_float(value, f"{path}[{row_index}]") for value in row))
    return tuple(rows)


def _require_mapping(raw: object, path: str) -> Mapping[str, Any]:
    """
    Require a mapping at the given path.

    Parameters
    ----------
    raw
        Raw data to validate.
    path
        Path for error messages.

    Returns
    -------
    mapping
        Validated mapping.

    Raises
    ------
    ValueError
        If raw is not a mapping.
    """
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return cast(Mapping[str, Any], raw)


def _require_sequence(raw: object, path: str) -> Sequence[Any]:
    """
    Require a sequence at the given path.

    Parameters
    ----------
    raw
        Raw data to validate.
    path
        Path for error messages.

    Returns
    -------
    sequence
        Validated sequence.

    Raises
    ------
    ValueError
        If raw is not a sequence.
    """
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError(f"{path} must be a sequence")
    return raw


def _require_str(raw: object, path: str) -> str:
    """
    Require a non-empty string at the given path.

    Parameters
    ----------
    raw
        Raw data to validate.
    path
        Path for error messages.

    Returns
    -------
    value
        Validated non-empty string.

    Raises
    ------
    ValueError
        If raw is not a non-empty string.
    """
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{path} must be a non-empty string")
    return raw


def _require_int(raw: object, path: str) -> int:
    """
    Require an integer at the given path.

    Parameters
    ----------
    raw
        Raw data to validate.
    path
        Path for error messages.

    Returns
    -------
    value
        Validated integer.

    Raises
    ------
    ValueError
        If raw is not an integer.
    """
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"{path} must be an integer")
    return raw


def _require_float(raw: object, path: str) -> float:
    """
    Require a number at the given path.

    Parameters
    ----------
    raw
        Raw data to validate.
    path
        Path for error messages.

    Returns
    -------
    value
        Validated float.

    Raises
    ------
    ValueError
        If raw is not a number.
    """
    if not isinstance(raw, int | float) or isinstance(raw, bool):
        raise ValueError(f"{path} must be a number")
    return float(raw)


def _optional_float(raw: object, path: str, *, default: float) -> float:
    """
    Get a float value or default if None.

    Parameters
    ----------
    raw
        Raw data to validate.
    path
        Path for error messages.
    default
        Default value if raw is None.

    Returns
    -------
    value
        Validated float or default.
    """
    return default if raw is None else _require_float(raw, path)
