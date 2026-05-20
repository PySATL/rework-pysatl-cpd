# -*- coding: ascii -*-
"""Abstract Bayesian base implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
from scipy.special import logsumexp

from pysatl_cpd.algorithms.online.bayesian._enum import BayesianCPFType
from pysatl_cpd.algorithms.online.bayesian.protocol.cpf import IBayesianCPF
from pysatl_cpd.algorithms.online.bayesian.protocol.hazard import IHazard
from pysatl_cpd.algorithms.online.bayesian.protocol.likelihood import ILikelihood
from pysatl_cpd.core.online.ionline_algorithm import (
    OnlineAlgorithm,
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
)
from pysatl_cpd.typedefs import Number, stable_hash

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


@dataclass(kw_only=True, frozen=True)
class BayesianOnlineCPDConfiguration(OnlineAlgorithmConfiguration):
    """Base configuration for Bayesian online CPD algorithms."""

    window: int | None = None
    cpf_type: BayesianCPFType = BayesianCPFType.MAX_RUN_LENGTH

    def __post_init__(self) -> None:
        """Validate configuration fields after initialisation.

        Raises
        ------
        ValueError
            If *learning_period_size* is negative, or *window* is
            non-positive when provided.
        """
        if self.learning_period_size < 0:
            raise ValueError("learning_period_size must be non-negative")
        if self.window is not None and self.window <= 0:
            raise ValueError("window must be positive")

    def __hash__(self) -> int:
        return stable_hash(
            (
                type(self).__module__,
                type(self).__qualname__,
                self.learning_period_size,
                self.window,
                self.cpf_type,
            )
        )


@dataclass(kw_only=True, frozen=True)
class BayesianOnlineCPDState(OnlineAlgorithmState):
    """Base state for Bayesian online CPD algorithms.

    Attributes
    ----------
    t
        Current time step.
    run_length_log_posterior
        Log-posterior distribution over run lengths.
    """

    t: int = 0
    run_length_log_posterior: Any = field(default_factory=list)

    def __hash__(self) -> int:
        posterior = tuple(np.asarray(self.run_length_log_posterior, dtype=np.float64).tolist())
        return stable_hash(
            (type(self).__module__, type(self).__qualname__, self.is_in_learning_period, self.t, posterior)
        )


class AbstractBayesian(OnlineAlgorithm[float, BayesianOnlineCPDConfiguration, BayesianOnlineCPDState], ABC):
    """Base class for configurable online Bayesian detectors.

    Implements the Bayesian online change-point detection (BOCPD) message-passing
    loop. Subclasses must provide the *name* property and wire up the concrete
    *hazard*, *likelihood*, and *cpf* components.

    Parameters
    ----------
    configuration
        Algorithm configuration.
    hazard
        Hazard function defining the prior over run lengths.
    likelihood
        Predictive likelihood model.
    cpf
        Change-point score function.
    """

    def __init__(
        self,
        configuration: BayesianOnlineCPDConfiguration,
        hazard: IHazard,
        likelihood: ILikelihood,
        cpf: IBayesianCPF,
    ) -> None:
        self._config = configuration
        self._hazard = hazard
        self._likelihood = likelihood
        self._cpf = cpf
        self._run_length_log_posterior = np.array([0.0], dtype=np.float64)
        self.t = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable algorithm name."""

    @property
    def configuration(self) -> BayesianOnlineCPDConfiguration:
        """Return the current algorithm configuration.

        Returns
        -------
        BayesianOnlineCPDConfiguration
        """
        return self._config

    @property
    def state(self) -> BayesianOnlineCPDState:
        """Materialise an immutable snapshot of the current state.

        Returns
        -------
        BayesianOnlineCPDState
        """
        return BayesianOnlineCPDState(
            is_in_learning_period=self.t < self._config.learning_period_size,
            t=self.t,
            run_length_log_posterior=self._run_length_log_posterior.copy(),
        )

    def process(self, observation: float) -> Number:
        """Ingest one observation and return the change-point score.

        Implements the BOCPD message-passing update:
          1. Predictive probabilities from the likelihood model.
          2. Hazard-based growth / change-point probabilities.
          3. Normalise and optionally truncate to *window*.
          4. Return CPF score (or 0 during learning period).

        Parameters
        ----------
        observation
            New data point.

        Returns
        -------
        Number
            Change-point score (0.0 during learning period).
        """
        obs = np.float64(observation)

        pred_log_probs = self._likelihood.predict(obs, window=self._config.window)
        self._likelihood.update(obs)

        run_lengths = np.arange(len(self._run_length_log_posterior), dtype=np.intp)
        log_h, log_neg_h = self._hazard.hazard(run_lengths)

        log_growth_probs = self._run_length_log_posterior + log_neg_h + pred_log_probs[1:]
        log_cp_prob = pred_log_probs[0] + logsumexp(self._run_length_log_posterior + log_h)

        new_log_posterior_unnorm = np.append(log_cp_prob, log_growth_probs)
        if self._config.window is not None and len(new_log_posterior_unnorm) > self._config.window:
            new_log_posterior_unnorm = new_log_posterior_unnorm[-self._config.window :]

        self._run_length_log_posterior = new_log_posterior_unnorm - logsumexp(new_log_posterior_unnorm)

        self.t += 1
        if self.t <= self._config.learning_period_size:
            return np.float64(0.0)

        return np.float64(self._cpf.calculate(self._run_length_log_posterior))

    def reset(self) -> None:
        """Reset the algorithm to its initial state.

        Clears step counter, run-length posterior, likelihood, and CPF.

        Returns
        -------
        None
        """
        self.t = 0
        self._run_length_log_posterior = np.array([0.0], dtype=np.float64)
        self._likelihood.clear()
        self._cpf.clear()

    def recreate(self) -> Self:
        """Create a fresh independent copy with identical configuration.

        Returns
        -------
        Self
        """
        algorithm = deepcopy(self)
        algorithm.reset()
        return algorithm
