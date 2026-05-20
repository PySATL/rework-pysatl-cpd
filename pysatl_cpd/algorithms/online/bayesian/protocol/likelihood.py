# -*- coding: ascii -*-
"""Protocol definition for Bayesian likelihood models."""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Protocol

import numpy as np
import numpy.typing as npt


class ILikelihood(Protocol):
    """Interface for Bayesian predictive likelihood models."""

    def predict(self, observation: np.float64, window: int | None = None) -> npt.NDArray[np.float64]:
        """Return predictive log-probabilities for prior and posterior states.

        Parameters
        ----------
        observation
            New observation to evaluate.
        window
            Maximum number of posterior states to consider.

        Returns
        -------
        npt.NDArray[np.float64]
            Array of log-probabilities: first element is the prior,
            remaining are posterior states.
        """

    def update(self, observation: np.float64) -> None:
        """Update internal posterior state with a new observation.

        Parameters
        ----------
        observation
            Observation to incorporate into the posterior.
        """

    def clear(self) -> None:
        """Reset the likelihood state to its prior.

        Returns
        -------
        None
        """
