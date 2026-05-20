# -*- coding: ascii -*-
"""Composable wrappers for online algorithms."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pysatl_cpd.core.online.ionline_algorithm import (
    OnlineAlgorithm,
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
)
from pysatl_cpd.typedefs import Number, stable_hash


@dataclass(frozen=True, kw_only=True)
class SkippingCondition[DataT]:
    """Named condition deciding whether an observation should be skipped.

    Attributes
    ----------
    name
        Human-readable identifier for this condition.
    condition
        Callable that returns True when an observation should be
        skipped (not passed to the underlying algorithm).
    """

    name: str
    condition: Callable[[DataT], bool]

    # Callables are intentionally excluded because their identities are not
    # persistence-stable hash inputs across processes or serialization boundaries.

    def __hash__(self) -> int:
        """Return a stable hash based on the public wrapper name only."""
        return stable_hash((type(self).__module__, type(self).__qualname__, self.name))

    def __post_init__(self) -> None:
        """Validate that the condition name is non-empty.

        Raises
        ------
        ValueError
            If `name` is an empty string.
        """
        if not self.name:
            raise ValueError("SkippingCondition name must be non-empty")


@dataclass(frozen=True, kw_only=True)
class BatchReducer[InT, OutT]:
    """Named reducer converting a block of observations into one observation.

    Attributes
    ----------
    name
        Human-readable identifier for this reducer.
    reducer
        Callable that converts a sequence of raw observations into
        a single value for the wrapped algorithm.
    """

    name: str
    reducer: Callable[[Sequence[InT]], OutT]

    # Callables are intentionally excluded because their identities are not
    # persistence-stable hash inputs across processes or serialization boundaries.

    def __hash__(self) -> int:
        """Return a stable hash based on the public reducer name only."""
        return stable_hash((type(self).__module__, type(self).__qualname__, self.name))

    def __post_init__(self) -> None:
        """Validate that the reducer name is non-empty.

        Raises
        ------
        ValueError
            If `name` is an empty string.
        """
        if not self.name:
            raise ValueError("BatchReducer name must be non-empty")


class SkippingOnlineAlgorithmWrapper[
    DataT,
    ConfigurationT: OnlineAlgorithmConfiguration,
    StateT: OnlineAlgorithmState,
](OnlineAlgorithm[DataT, ConfigurationT, StateT]):
    """Wrap an online algorithm and optionally skip observations.

    When the configured ``skipping_condition`` is met for an observation,
    the wrapper returns the last computed detection statistic without
    passing the observation to the wrapped algorithm.

    Parameters
    ----------
    algorithm
        Algorithm to wrap.
    skipping_condition
        Condition that decides which observations to skip.
    """

    def __init__(
        self,
        algorithm: OnlineAlgorithm[DataT, ConfigurationT, StateT],
        *,
        skipping_condition: SkippingCondition[DataT],
    ) -> None:
        self._algorithm = algorithm
        self._skipping_condition = skipping_condition
        self._last_detection_function: Number = float("nan")

    @property
    def name(self) -> str:
        """Return a composite name reflecting the wrapped algorithm and condition.

        Returns
        -------
        str
            Composite name in the form ``AlgorithmName{skip[on=condition]}``.
        """
        return f"{self._algorithm.name}{{skip[on={self._skipping_condition.name}]}}"

    @property
    def configuration(self) -> ConfigurationT:
        """Return the wrapped algorithm's configuration.

        Returns
        -------
        ConfigurationT
            Configuration of the underlying algorithm.
        """
        return self._algorithm.configuration

    @property
    def state(self) -> StateT:
        """Return the wrapped algorithm's current state.

        Returns
        -------
        StateT
            Current state of the underlying algorithm.
        """
        return self._algorithm.state

    def process(self, observation: DataT) -> Number:
        """Process an observation, skipping it if the condition is met.

        Parameters
        ----------
        observation
            Incoming data point.

        Returns
        -------
        Number
            Detection statistic from the wrapped algorithm, or the last
            computed statistic when the observation is skipped.
        """
        if self._skipping_condition.condition(observation):
            return self._last_detection_function

        self._last_detection_function = self._algorithm.process(observation)
        return self._last_detection_function

    def reset(self) -> None:
        """Reset the wrapper and the underlying algorithm to initial state.

        Clears the cached last detection function value and delegates
        reset to the wrapped algorithm.
        """
        self._last_detection_function = float("nan")
        self._algorithm.reset()

    def recreate(self) -> SkippingOnlineAlgorithmWrapper[DataT, ConfigurationT, StateT]:
        """Return a fresh instance with the same skipping condition.

        Returns
        -------
        SkippingOnlineAlgorithmWrapper
            A new wrapper instance sharing the same skipping condition
            but a freshly recreated underlying algorithm.
        """
        return type(self)(
            _recreate_wrapped_algorithm(self._algorithm),
            skipping_condition=self._skipping_condition,
        )


class BatchingOnlineAlgorithmWrapper[
    DataT,
    BatchT,
    ConfigurationT: OnlineAlgorithmConfiguration,
    StateT: OnlineAlgorithmState,
](OnlineAlgorithm[DataT, ConfigurationT, StateT]):
    """Wrap an online algorithm and feed it reduced observation batches.

    Raw observations are buffered until the batch is full, then reduced
    to a single value via ``BatchReducer`` and passed to the wrapped
    algorithm. Partial batches return the last computed statistic.

    Parameters
    ----------
    algorithm
        Algorithm to wrap (operates on the reduced batch type).
    batch_size
        Number of raw observations to accumulate before feeding a
        reduced value to the wrapped algorithm. Must be positive.
    reducer
        Function that converts a sequence of raw observations into a
        single value for the wrapped algorithm.

    Raises
    ------
    ValueError
        If ``batch_size`` is not positive.
    """

    def __init__(
        self,
        algorithm: OnlineAlgorithm[BatchT, ConfigurationT, StateT],
        *,
        batch_size: int,
        reducer: BatchReducer[DataT, BatchT],
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self._algorithm = algorithm
        self._batch_size = batch_size
        self._reducer = reducer
        self._buffer: list[DataT] = []
        self._last_detection_function: Number = float("nan")

    @property
    def name(self) -> str:
        """Return a composite name reflecting the wrapped algorithm and batch config.

        Returns
        -------
        str
            Composite name in the form
            ``AlgorithmName{batch[size=N, reduce=name]}``.
        """
        return f"{self._algorithm.name}{{batch[size={self._batch_size}, reduce={self._reducer.name}]}}"

    @property
    def configuration(self) -> ConfigurationT:
        """Return the wrapped algorithm's configuration.

        Returns
        -------
        ConfigurationT
            Configuration of the underlying algorithm.
        """
        return self._algorithm.configuration

    @property
    def state(self) -> StateT:
        """Return the wrapped algorithm's current state.

        Returns
        -------
        StateT
            Current state of the underlying algorithm.
        """
        return self._algorithm.state

    def process(self, observation: DataT) -> Number:
        """Buffer an observation and process the batch once full.

        Parameters
        ----------
        observation
            Incoming data point to buffer.

        Returns
        -------
        Number
            Detection statistic. Returns the last computed value until the
            buffer reaches ``batch_size``, then feeds the reduced batch to
            the wrapped algorithm and returns its result.
        """
        self._buffer.append(observation)
        if len(self._buffer) < self._batch_size:
            return self._last_detection_function

        reduced_observation = self._reducer.reducer(tuple(self._buffer))
        self._buffer.clear()
        self._last_detection_function = self._algorithm.process(reduced_observation)
        return self._last_detection_function

    def reset(self) -> None:
        """Reset the buffer, cached statistic, and underlying algorithm.

        Clears the observation buffer and cached detection function,
        then delegates reset to the wrapped algorithm.
        """
        self._buffer.clear()
        self._last_detection_function = float("nan")
        self._algorithm.reset()

    def recreate(self) -> BatchingOnlineAlgorithmWrapper[DataT, BatchT, ConfigurationT, StateT]:
        """Return a fresh instance with the same batch size and reducer.

        Returns
        -------
        BatchingOnlineAlgorithmWrapper
            A new wrapper instance sharing the same batch configuration
            but a freshly recreated underlying algorithm.
        """
        return type(self)(
            _recreate_wrapped_algorithm(self._algorithm),
            batch_size=self._batch_size,
            reducer=self._reducer,
        )


def _recreate_wrapped_algorithm[
    DataT,
    ConfigurationT: OnlineAlgorithmConfiguration,
    StateT: OnlineAlgorithmState,
](
    algorithm: OnlineAlgorithm[DataT, ConfigurationT, StateT],
) -> OnlineAlgorithm[DataT, ConfigurationT, StateT]:
    """Recreate the innermost unwrapped algorithm.

    Walks through nested wrappers by checking the ``recreate`` signature.
    If the wrapped algorithm's ``recreate`` takes no parameters the
    algorithm is considered terminal and is asked to recreate itself.
    Otherwise the wrapper's configuration is used to build a fresh instance.

    Parameters
    ----------
    algorithm
        The (possibly wrapped) algorithm to recreate.

    Returns
    -------
    OnlineAlgorithm
        A fresh instance of the innermost algorithm.
    """
    recreate_params = inspect.signature(algorithm.recreate).parameters
    if not recreate_params:
        return algorithm.recreate()
    return type(algorithm).recreate(algorithm.configuration)  # type: ignore[arg-type]
