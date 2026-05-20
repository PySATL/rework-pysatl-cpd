# -*- coding: ascii -*-
"""
Tests for config.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.data.generator.config import (
    _parse_covariance,
    _parse_hashable_mapping,
    _require_float,
    _require_int,
    _require_mapping,
    _require_sequence,
    parse_distribution_spec,
    scenario_from_mapping,
    scenario_from_yaml,
    scenarios_from_yaml,
)
from pysatl_cpd.data.generator.specs import (
    ExponentialSpec,
    IndependentColumnsSpec,
    MultivariateNormalSpec,
    NormalSpec,
    StudentTSpec,
)


def test_scenario_from_mapping_loads_direct_univariate_distribution() -> None:
    scenario = scenario_from_mapping(
        {
            "name": "mean_shift",
            "segments": [
                {"plan_name": "baseline", "length": 10},
                {"plan_name": "shifted", "length": 5},
            ],
            "plans": {
                "baseline": {
                    "state": {"type": "baseline"},
                    "distribution": {"kind": "normal", "mean": 0.0, "std": 1.0},
                },
                "shifted": {
                    "state": {"type": "shifted"},
                    "distribution": {"kind": "normal", "mean": 2.0, "std": 1.0},
                },
            },
        }
    )

    assert scenario.name == "mean_shift"
    assert scenario.segments[0].plan_name == "baseline"
    assert scenario.segments[0].length == 10
    assert isinstance(scenario.plans["baseline"].distribution, NormalSpec)
    assert scenario.plans["baseline"].state is not None
    assert scenario.plans["baseline"].state["type"] == "baseline"


def test_scenario_from_mapping_loads_independent_columns_distribution() -> None:
    scenario = scenario_from_mapping(
        {
            "name": "independent",
            "segments": [{"plan_name": "baseline", "length": 10}],
            "plans": {
                "baseline": {
                    "distribution": {
                        "kind": "independent_columns",
                        "columns": {
                            "x": {"kind": "normal", "mean": 0.0, "std": 1.0},
                            "y": {"kind": "uniform", "low": -1.0, "high": 1.0},
                        },
                    }
                }
            },
        }
    )

    distribution = scenario.plans["baseline"].distribution
    assert isinstance(distribution, IndependentColumnsSpec)
    assert tuple(distribution.columns) == ("x", "y")


def test_scenario_from_mapping_loads_multivariate_normal_distribution() -> None:
    scenario = scenario_from_mapping(
        {
            "name": "multivariate",
            "segments": [{"plan_name": "baseline", "length": 10}],
            "plans": {
                "baseline": {
                    "distribution": {
                        "kind": "multivariate_normal",
                        "means": {"x": 0.0, "y": 1.0},
                        "covariance": [[1.0, 0.2], [0.2, 1.0]],
                    }
                }
            },
        }
    )

    distribution = scenario.plans["baseline"].distribution
    assert isinstance(distribution, MultivariateNormalSpec)
    assert distribution.covariance == ((1.0, 0.2), (0.2, 1.0))


def test_scenario_from_yaml_loads_single_scenario(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "scenario.yaml"
    path.write_text(
        """
name: yaml_scenario
segments:
  - plan_name: baseline
    length: 3
plans:
  baseline:
    distribution:
      kind: normal
      mean: 0.0
      std: 1.0
""",
        encoding="utf-8",
    )

    scenario = scenario_from_yaml(path)

    assert scenario.name == "yaml_scenario"
    assert isinstance(scenario.plans["baseline"].distribution, NormalSpec)


def test_scenarios_from_yaml_loads_mapping(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "scenarios.yaml"
    path.write_text(
        """
scenarios:
  first:
    name: first
    segments:
      - plan_name: baseline
        length: 3
    plans:
      baseline:
        distribution:
          kind: normal
""",
        encoding="utf-8",
    )

    scenarios = scenarios_from_yaml(path)

    assert list(scenarios) == ["first"]
    assert scenarios["first"].name == "first"


def test_scenarios_from_yaml_loads_single_scenario_mapping(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "scenario.yaml"
    path.write_text(
        """
name: fallback
segments:
  - plan_name: baseline
    length: 2
plans:
  baseline:
    distribution:
      kind: normal
""",
        encoding="utf-8",
    )

    scenarios = scenarios_from_yaml(path)

    assert list(scenarios) == ["fallback"]
    assert scenarios["fallback"].name == "fallback"


def test_scenario_from_mapping_rejects_unknown_distribution_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported distribution kind"):
        scenario_from_mapping(
            {
                "name": "invalid",
                "segments": [{"plan_name": "baseline", "length": 10}],
                "plans": {"baseline": {"distribution": {"kind": "missing"}}},
            }
        )


def test_scenario_from_mapping_rejects_missing_distribution_kind() -> None:
    with pytest.raises(ValueError, match="distribution.kind"):
        scenario_from_mapping(
            {
                "name": "invalid",
                "segments": [{"plan_name": "baseline", "length": 10}],
                "plans": {"baseline": {"distribution": {}}},
            }
        )


def test_scenario_from_yaml_rejects_non_mapping_top_level(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "scenario.yaml"
    path.write_text("- invalid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping"):
        scenario_from_yaml(path)


def test_scenarios_from_yaml_rejects_non_mapping_scenarios_key(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "scenarios.yaml"
    path.write_text("scenarios:\n  - invalid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="'scenarios' must be a mapping"):
        scenarios_from_yaml(path)


def test_scenarios_from_yaml_rejects_non_mapping_single_scenario(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "scenarios.yaml"
    path.write_text("invalid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping"):
        scenarios_from_yaml(path)


def test_parse_distribution_spec_loads_exponential_distribution() -> None:
    distribution = parse_distribution_spec({"kind": "exponential", "scale": 2.5})

    assert distribution == ExponentialSpec(scale=2.5)


def test_parse_distribution_spec_loads_student_t_distribution() -> None:
    distribution = parse_distribution_spec({"kind": "student_t", "df": 7.0, "loc": 1.0, "scale": 2.0})

    assert distribution == StudentTSpec(df=7.0, loc=1.0, scale=2.0)


def test_parse_distribution_spec_loads_independent_columns_univariate_variants() -> None:
    distribution = parse_distribution_spec(
        {
            "kind": "independent_columns",
            "columns": {
                "exp": {"kind": "exponential", "scale": 3.0},
                "student": {"kind": "student_t", "df": 4.0},
            },
        }
    )

    assert isinstance(distribution, IndependentColumnsSpec)
    assert distribution.columns["exp"] == ExponentialSpec(scale=3.0)
    assert distribution.columns["student"] == StudentTSpec(df=4.0)


def test_parse_hashable_mapping_rejects_non_hashable_value() -> None:
    with pytest.raises(ValueError, match="metadata.bad must be hashable"):
        _parse_hashable_mapping({"bad": []}, "metadata")


def test_parse_hashable_mapping_accepts_hashable_values() -> None:
    assert _parse_hashable_mapping({"name": "ok", "count": 2}, "metadata") == {"name": "ok", "count": 2}


def test_parse_covariance_accepts_scalar_value() -> None:
    assert _parse_covariance(2, "distribution.covariance") == 2.0


def test_parse_covariance_rejects_non_sequence_non_scalar() -> None:
    with pytest.raises(ValueError, match="distribution.covariance must be a number or sequence"):
        _parse_covariance(object(), "distribution.covariance")


def test_parse_covariance_accepts_flat_sequence() -> None:
    assert _parse_covariance([1, 2.5], "distribution.covariance") == (1.0, 2.5)


def test_parse_covariance_accepts_nested_sequence() -> None:
    assert _parse_covariance([[1, 0.5], [0.5, 1]], "distribution.covariance") == ((1.0, 0.5), (0.5, 1.0))


def test_parse_covariance_rejects_non_sequence_row() -> None:
    with pytest.raises(ValueError, match=r"distribution\.covariance\[1\] must be a sequence"):
        _parse_covariance([[1.0, 0.5], 1.0], "distribution.covariance")


def test_require_mapping_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="field must be a mapping"):
        _require_mapping(["bad"], "field")


def test_require_sequence_rejects_non_sequence() -> None:
    with pytest.raises(ValueError, match="field must be a sequence"):
        _require_sequence(1, "field")


def test_require_int_rejects_non_int() -> None:
    with pytest.raises(ValueError, match="field must be an integer"):
        _require_int(1.5, "field")


def test_require_float_rejects_non_number() -> None:
    with pytest.raises(ValueError, match="field must be a number"):
        _require_float("bad", "field")


def test_scenario_from_mapping_rejects_non_scalar_state_value() -> None:
    with pytest.raises(ValueError, match=r"plans\.baseline\.state\.bad must be a string, integer, float, or boolean"):
        scenario_from_mapping(
            {
                "name": "invalid_state",
                "segments": [{"plan_name": "baseline", "length": 10}],
                "plans": {
                    "baseline": {
                        "state": {"bad": []},
                        "distribution": {"kind": "normal"},
                    }
                },
            }
        )


def test_parse_distribution_spec_rejects_non_univariate_independent_column() -> None:
    with pytest.raises(ValueError, match="must be univariate"):
        parse_distribution_spec(
            {
                "kind": "independent_columns",
                "columns": {
                    "x": {
                        "kind": "multivariate_normal",
                        "means": {"x": 0.0},
                    }
                },
            }
        )
