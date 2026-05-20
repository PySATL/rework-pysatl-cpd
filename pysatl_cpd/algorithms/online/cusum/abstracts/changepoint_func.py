# -*- coding: ascii -*-
"""
Protocol definition for CUSUM change-point functions.

This module defines :class:`ICusumChangepointFunc`, a protocol for
CUSUM-compatible statistic updaters used in generalized CUSUM pipelines.
"""

from abc import ABC, abstractmethod

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


class ICusumChangepointFunc[DataT](ABC):
    """
    Interface for CUSUM change-point function objects.

    Implementations maintain an internal statistic updated from monitoring
    observations and expose the current scalar statistic value.
    """

    @abstractmethod
    def update(self, observation: DataT) -> None:  # pragma: no cover
        """Update the internal statistic with a new observation.

        Parameters
        ----------
        observation
            New monitoring-space observation.
        """

    @property
    @abstractmethod
    def value(self) -> float:  # pragma: no cover
        """Current scalar change-point statistic value.

        Returns
        -------
        float
        """

    @abstractmethod
    def reset(self) -> None:  # pragma: no cover
        """Reset internal statistic to initial state.

        Returns
        -------
        None
        """
