# -*- coding: ascii -*-
"""
Shewhart control chart algorithm for online change-point detection.

This module provides :class:`ShewhartControlChart`, an online detector that
tracks running mean and variance, computing a standardized deviation of a
sliding-window mean from the global running mean.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Self

import numpy as np

from pysatl_cpd.core.online.ionline_algorithm import (
    OnlineAlgorithm,
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
)
from pysatl_cpd.typedefs import Number, stable_hash


@dataclass(kw_only=True, frozen=True)
class ShewhartControlChartState(OnlineAlgorithmState):
    """
    State snapshot of the Shewhart control chart algorithm.

    This class captures the internal state of the algorithm at a specific point
    in time, allowing for state inspection and debugging.

    Attributes
    ----------
    mean
        Current running mean estimate.
    variance
        Current running variance estimate.
    samples_count
        Number of observations processed so far.
    window_contents
        Current contents of the sliding window in order.
    """

    mean: Number = 0.0
    variance: Number = 0.0
    samples_count: int = 0
    window_contents: list[Number] = field(default_factory=list)

    @property
    def standard_deviation(self) -> Number:
        return np.sqrt(self.variance, dtype=np.float64)  # type: ignore

    @property
    def window_sum(self) -> Number:
        return sum(self.window_contents)

    @property
    def window_size(self) -> int:
        return len(self.window_contents)

    @property
    def window_mean(self) -> Number:
        return self.window_sum / self.window_size if self.window_size > 0 else 0.0

    def __hash__(self) -> int:
        """Return a stable hash for the Shewhart state snapshot."""
        return stable_hash(
            (
                type(self).__module__,
                type(self).__qualname__,
                self.is_in_learning_period,
                self.mean,
                self.variance,
                self.samples_count,
                self.window_contents,
            )
        )


@dataclass(kw_only=True, frozen=True)
class ShewhartControlChartConfiguration(OnlineAlgorithmConfiguration):
    """
    Configuration parameters for the Shewhart control chart algorithm.

    Attributes
    ----------
    window_size
        Size of the sliding window used to compute the local mean statistic.

    Raises
    ------
    ValueError
        If ``learning_period_size`` is not positive.
    ValueError
        If ``window_size`` is not positive.
    ValueError
        If ``window_size`` is greater than ``learning_period_size``.
    """

    window_size: int = 0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.learning_period_size <= 0:
            raise ValueError(f"learning_period_size must be positive, got {self.learning_period_size}")

        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")

        if self.window_size > self.learning_period_size:
            raise ValueError(
                f"window_size ({self.window_size}) must be less than or equal to "
                f"learning_period_size ({self.learning_period_size})"
            )

    def __repr__(self) -> str:
        """Return a short string representation of the configuration."""
        return f"w = {self.window_size}"

    def __hash__(self) -> int:
        """Return a stable hash for the Shewhart configuration."""
        return stable_hash(
            (type(self).__module__, type(self).__qualname__, self.learning_period_size, self.window_size)
        )


class ShewhartControlChart(OnlineAlgorithm[Number, ShewhartControlChartConfiguration, ShewhartControlChartState]):
    """
    Shewhart control chart with sliding-window statistic.

    This algorithm maintains running estimates of mean and variance, and
    computes a standardized deviation between the sliding-window mean and
    the global running mean. The statistic follows the formula:

    .. math::
        S_t = \\frac{|\\bar{x}_w - \\mu| \\sqrt{w}}{\\sigma}

    where:
    - :math:`\\bar{x}_w` is the mean of the last `window_size` observations
    - :math:`\\mu` is the running mean of all observations
    - :math:`w` is the window size
    - :math:`\\sigma` is the running standard deviation

    Parameters
    ----------
    learning_period_size
        Number of initial observations used for training before non-zero
        statistics are emitted. Must be positive.
    window_size
        Size of the sliding window used to compute the local mean statistic.
        Must be positive and less than or equal to learning_period_size.
    """

    def __init__(self, learning_period_size: int, window_size: int) -> None:
        self._configuration = ShewhartControlChartConfiguration(
            learning_period_size=learning_period_size, window_size=window_size
        )

        self._mean: Number = 0.0
        self._previous_mean: Number = 0.0
        self._variance: Number = 0.0
        self._standard_deviation: Number = 0.0
        self._samples_count: int = 0
        self._window: deque[Number] = deque[Number](maxlen=window_size)
        self._window_sum: Number = 0.0
        self._window_mean: Number = 0.0

    @property
    def configuration(self) -> ShewhartControlChartConfiguration:
        """Current algorithm configuration.

        Returns
        -------
        ShewhartControlChartConfiguration
        """
        return self._configuration

    @property
    def state(self) -> ShewhartControlChartState:
        """Materialise an immutable state snapshot.

        Returns
        -------
        ShewhartControlChartState
        """
        return ShewhartControlChartState(
            is_in_learning_period=self._samples_count <= self._configuration.learning_period_size,
            mean=self._mean,
            variance=self._variance,
            samples_count=self._samples_count,
            window_contents=list(self._window),
        )

    def process(self, observation: Number) -> Number:
        """Ingest one observation and return the chart statistic value.

        Computes the standardised absolute deviation between the sliding-window
        mean and the running mean. Returns 0 during the learning period or
        when the running standard deviation is zero.

        Parameters
        ----------
        observation
            New scalar observation.

        Returns
        -------
        Number
        """
        self._samples_count += 1

        # Compute detection statistic after learning period
        detection_func: Number = 0.0
        if self._samples_count > self._configuration.learning_period_size and self._standard_deviation > 0:
            detection_func = np.float64(
                np.abs(self._window_mean - self._mean)
                * (self._configuration.window_size**0.5)
                / self._standard_deviation
            )

        # Maintain sliding window
        if len(self._window) == self._configuration.window_size:
            self._window_sum -= self._window.popleft()

        # Update running statistics
        self._mean, self._previous_mean = self._update_mean(observation)
        self._variance = self._update_variance(observation)
        self._standard_deviation = np.sqrt(self._variance)

        # Update sliding window
        self._window.append(observation)
        self._window_sum += observation
        self._window_mean = self._window_sum / self._configuration.window_size

        return detection_func

    def reset(self) -> None:
        """Reset the algorithm to its initial state.

        Clears all internal statistics, counters, and the sliding window.

        Returns
        -------
        None
        """
        self._mean = 0.0
        self._previous_mean = 0.0
        self._variance = 0.0
        self._standard_deviation = 0.0
        self._samples_count = 0
        self._window.clear()
        self._window_sum = 0.0
        self._window_mean = 0.0

    def _update_mean(self, observation: Number) -> tuple[Number, Number]:
        """Update running mean using Welford's recurrence.

        Parameters
        ----------
        observation
            New scalar observation.

        Returns
        -------
        tuple[Number, Number]
            Pair (new_mean, previous_mean).
        """
        new_mean = self._mean + (observation - self._mean) / self._samples_count
        return new_mean, self._mean

    def _update_variance(self, observation: Number) -> Number:
        """Update running variance estimate using Welford's algorithm.

        Parameters
        ----------
        observation
            New scalar observation.

        Returns
        -------
        Number
            Updated running variance.
        """
        return (
            self._variance
            + ((observation - self._previous_mean) * (observation - self._mean) - self._variance) / self._samples_count
        )

    def recreate(self) -> Self:
        """Create a fresh chart with identical configuration and reset state.

        Returns
        -------
        Self
        """
        algorithm = deepcopy(self)
        algorithm.reset()
        return algorithm
