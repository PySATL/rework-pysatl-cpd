# -*- coding: ascii -*-
"""
Protocol definition for CUSUM monitoring schemas.

This module defines :class:`IMonitoringSchema`, the interface for components
that map observations and estimated parameters into monitoring-space values.
"""

from abc import ABC, abstractmethod

from pysatl_cpd.algorithms.online.cusum.abstracts.estimator import ISchemaEstimates

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


class IMonitoringSchema[DataT, EstimatesT: ISchemaEstimates, ReturnT](ABC):
    """
    Interface for monitoring schemas used by generalized CUSUM.
    """

    @abstractmethod
    def evaluate(self, observation: DataT, estimates: EstimatesT) -> ReturnT:  # pragma: no cover
        """Evaluate monitoring transform for a new observation.

        Parameters
        ----------
        observation
            New observation.
        estimates
            Current estimated model parameters.

        Returns
        -------
        ReturnT
            Monitoring-space transformed value.
        """

    @abstractmethod
    def reset(self) -> None:  # pragma: no cover
        """Reset monitoring schema state.

        Returns
        -------
        None
        """
