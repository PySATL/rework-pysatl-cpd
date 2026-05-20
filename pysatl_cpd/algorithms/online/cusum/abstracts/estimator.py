# -*- coding: ascii -*-
"""
Protocol definition for CUSUM parameter estimating schemas.

This module defines :class:`IEstimatingSchema`, the interface for estimation
components that are trained on a learning sample and optionally updated
online.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypedDict

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


class ISchemaEstimates(TypedDict, total=False): ...


class IEstimatingSchema[DataT, EstimatesT: ISchemaEstimates](ABC):
    """
    Interface for estimating schemas used by generalized CUSUM.

    Implementations estimate model parameters from training data, optionally
    update them with new observations, and expose current estimates.
    """

    @abstractmethod
    def train(self, train_set: Sequence[DataT]) -> None:  # pragma: no cover
        """Fit estimator parameters from a training sample.

        Parameters
        ----------
        train_set
            Learning sample used for initial parameter estimation.
        """

    @abstractmethod
    def update(self, observation: DataT) -> None:  # pragma: no cover
        """Update estimator parameters with a new observation.

        Parameters
        ----------
        observation
            New observation used for adaptive update.
        """

    @abstractmethod
    def reset(self) -> None:  # pragma: no cover
        """Reset estimator state to initial condition.

        Returns
        -------
        None
        """

    @property
    @abstractmethod
    def estimates(self) -> EstimatesT:  # pragma: no cover
        """Current estimated parameters.

        Returns
        -------
        EstimatesT
        """
