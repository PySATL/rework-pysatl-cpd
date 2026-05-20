# -*- coding: ascii -*-
"""
Generalized CUSUM source implementation.

This module provides :class:`GeneralizedCUSUM`, a configurable online detector
that combines three components:

- an estimating schema,
- a monitoring schema,
- and a change-point function (CPF).

Concrete CUSUM algorithms configure these components via factory functions.
"""

from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Self

from pysatl_cpd.algorithms.online.cusum.abstracts.changepoint_func import ICusumChangepointFunc
from pysatl_cpd.algorithms.online.cusum.abstracts.estimator import IEstimatingSchema, ISchemaEstimates
from pysatl_cpd.algorithms.online.cusum.abstracts.monitoring import IMonitoringSchema
from pysatl_cpd.core.online.ionline_algorithm import (
    OnlineAlgorithm,
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
)
from pysatl_cpd.typedefs import Number, stable_hash

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


@dataclass(kw_only=True, frozen=True)
class GeneralizedCUSUMConfiguration(OnlineAlgorithmConfiguration):
    """Configuration for Generalized CUSUM algorithms."""

    adaptive_estimation: bool = True

    def __hash__(self) -> int:
        """Return a stable hash for the CUSUM configuration."""
        return stable_hash(
            (type(self).__module__, type(self).__qualname__, self.learning_period_size, self.adaptive_estimation)
        )


@dataclass(kw_only=True, frozen=True)
class GeneralizedCUSUMState[EstimatesT: ISchemaEstimates](OnlineAlgorithmState):
    """State for Generalized CUSUM algorithms."""

    statistics: ISchemaEstimates

    def __hash__(self) -> int:
        """Return a stable hash for the CUSUM state snapshot."""
        return stable_hash(
            (type(self).__module__, type(self).__qualname__, self.is_in_learning_period, self.statistics)
        )


class GeneralizedCUSUM[
    DataT,
    ConfigurationT: GeneralizedCUSUMConfiguration,
    StateT: GeneralizedCUSUMState,
    EstimatesT: ISchemaEstimates,
    MonitoringT,
](OnlineAlgorithm[DataT, ConfigurationT, StateT]):
    """
    Base class for configurable online CUSUM detectors.

    Parameters
    ----------
    configuration
        Algorithm configuration instance.
    estimating_schema
        Schema that estimates model parameters from data.
    monitoring_schema
        Schema that transforms observations into monitoring-space residuals.
    changepoint_func
        CUSUM change-point function applied to monitoring residuals.
    adaptive_estimation
        Whether to update estimates online after training.
    """

    def __init__(
        self,
        configuration: ConfigurationT,
        estimating_schema: IEstimatingSchema[DataT, EstimatesT],
        monitoring_schema: IMonitoringSchema[DataT, EstimatesT, MonitoringT],
        changepoint_func: ICusumChangepointFunc[MonitoringT],
        adaptive_estimation: bool = True,
    ) -> None:
        self._config = configuration
        self._train_X: list[DataT] = []
        self._is_training = True

        self._estimating_schema: IEstimatingSchema[DataT, EstimatesT] = estimating_schema
        self._monitoring_schema: IMonitoringSchema[DataT, EstimatesT, MonitoringT] = monitoring_schema
        self._changepoint_fun: ICusumChangepointFunc[MonitoringT] = changepoint_func

        self._dim: int = -1

    @property
    @abstractmethod
    def configuration(self) -> ConfigurationT: ...  # pragma: no cover

    @property
    @abstractmethod
    def state(self) -> StateT: ...  # pragma: no cover

    def residual(self, argument: DataT) -> MonitoringT:
        """Transform a raw observation into monitoring-space residual.

        Delegates to the monitoring schema.

        Parameters
        ----------
        argument
            Raw input observation.

        Returns
        -------
        MonitoringT
        """
        return self.monitoring_schema.evaluate(argument, self.estimates)

    @property
    def estimating_schema(self) -> IEstimatingSchema[DataT, EstimatesT]:
        """Estimating schema instance.

        Returns
        -------
        IEstimatingSchema[DataT, EstimatesT]
        """
        return self._estimating_schema

    @property
    def monitoring_schema(self) -> IMonitoringSchema[DataT, EstimatesT, MonitoringT]:
        """Monitoring schema instance.

        Returns
        -------
        IMonitoringSchema[DataT, EstimatesT, MonitoringT]
        """
        return self._monitoring_schema

    @property
    def changepoint_func(self) -> ICusumChangepointFunc[MonitoringT]:
        """CUSUM change-point function instance.

        Returns
        -------
        ICusumChangepointFunc[MonitoringT]
        """
        return self._changepoint_fun

    @property
    def dim(self) -> int | None:
        """Observation dimensionality detected from the first sample.

        Returns
        -------
        int or None
            ``None`` before the first *process* call.
        """
        return self._dim

    @property
    def estimates(self) -> EstimatesT:
        """Current model parameter estimates from the estimating schema.

        Returns
        -------
        EstimatesT
        """
        return self._estimating_schema.estimates

    @property
    def cpf(self) -> float:
        """Current scalar change-point statistic value.

        Returns
        -------
        float
        """
        return self.changepoint_func.value

    def process(self, observation: DataT) -> Number:
        """Ingest one observation and return the change-point statistic.

        During training (*learning_period_size* steps) accumulates samples
        and returns 0. After training, updates the CUSUM change-point
        function and optionally the estimating schema (adaptive mode).

        Parameters
        ----------
        observation
            New data point (first call sets the dimensionality).

        Returns
        -------
        Number
            Change-point statistic value.
        """
        if self._dim is None:
            self._dim = observation.shape[0]

        if self._is_training:
            self._train_X.append(observation)
            if len(self._train_X) == self._config.learning_period_size:
                self.estimating_schema.train(self._train_X)
                self._is_training = False
            return 0.0

        self.changepoint_func.update(self.residual(observation))
        if self._config.adaptive_estimation:
            self.estimating_schema.update(observation)
        return self.cpf

    def reset(self) -> None:
        """Reset the detector to its initial (training) state.

        Clears all component states, training buffer, and flags.

        Returns
        -------
        None
        """
        self.estimating_schema.reset()
        self.monitoring_schema.reset()
        self._changepoint_fun.reset()
        self._train_X = []
        self._is_training = True

    def recreate(self) -> Self:
        """Create a fresh copy with the same configuration and reset state.

        Returns
        -------
        Self
            A new instance in initial (training) state.
        """
        algorithm = deepcopy(self)
        algorithm.reset()
        return algorithm
