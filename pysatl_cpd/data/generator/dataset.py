# -*- coding: ascii -*-
"""Dataset generators."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, Protocol

from pysatl_cpd.data.dataset import Dataset
from pysatl_cpd.data.generator.providers import build_plain_multivariate_labeled_data
from pysatl_cpd.data.generator.series import GenericSeriesGenerator
from pysatl_cpd.data.generator.specs import ScenarioSpec
from pysatl_cpd.data.providers.labeled import LabeledData
from pysatl_cpd.data.typedefs import TimeseriesAnnotation, frozendict


class LabeledDataGenerator[DataT: LabeledData[Any, TimeseriesAnnotation]](Protocol):
    """
    Protocol for generating labeled data instances.
    """

    def generate(
        self,
        annotation: TimeseriesAnnotation | None = None,
        name: str | None = None,
    ) -> DataT:
        """
        Generate a labeled data instance.

        Parameters
        ----------
        annotation
            Optional annotation for the generated data.
        name
            Optional name for the generated data.

        Returns
        -------
        data
            Generated labeled data instance.
        """


class ScenarioDatasetGenerator:
    """
    Generator for creating datasets from scenario specifications.

    Parameters
    ----------
    scenarios
        Dictionary mapping scenario names to specifications.
    seed
        Optional random seed for reproducibility.

    Raises
    ------
    ValueError
        If scenarios is empty.
    """

    def __init__(self, scenarios: dict[str, ScenarioSpec], *, seed: int | None = None) -> None:
        if not scenarios:
            raise ValueError("Scenarios must not be empty")
        self._scenarios = dict(scenarios)
        self._series_generator = GenericSeriesGenerator(seed=seed)

    @property
    def scenarios(self) -> dict[str, ScenarioSpec]:
        """
        Get the scenario specifications.

        Returns
        -------
        scenarios
            Dictionary mapping scenario names to specifications.
        """
        return dict(self._scenarios)

    def generate(self, scenario: str, size: int) -> Dataset[Any, TimeseriesAnnotation]:
        """
        Generate a dataset from a scenario.

        Parameters
        ----------
        scenario
            Name of the scenario to generate from.
        size
            Number of data instances to generate.

        Returns
        -------
        dataset
            Generated dataset with labeled data.

        Raises
        ------
        ValueError
            If size is not positive, or the scenario name is unknown.
        """
        if size <= 0:
            raise ValueError("Dataset size must be positive")
        if scenario not in self._scenarios:
            raise ValueError(f"Unknown scenario '{scenario}'")

        scenario_spec = self._scenarios[scenario]
        providers = [
            build_plain_multivariate_labeled_data(
                self._series_generator.generate_from_scenario(scenario_spec, name=f"{scenario}_series_{index:04d}"),
                annotation=TimeseriesAnnotation(
                    name=f"{scenario}_series_{index:04d}",
                    metadata=frozendict(scenario=scenario),
                ),
                name=f"{scenario}_series_{index:04d}",
            )
            for index in range(size)
        ]
        return Dataset(providers)
