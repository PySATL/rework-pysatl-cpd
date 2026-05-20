# -*- coding: ascii -*-
"""Protocol definition for Bayesian change-point score functions."""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Protocol

import numpy as np
import numpy.typing as npt


class IBayesianCPF(Protocol):
    """Interface for scalar Bayesian change-point score functions."""

    def calculate(self, run_length_log_posterior: npt.NDArray[np.float64]) -> float:
        """Convert the current run-length posterior into a scalar score.

        Parameters
        ----------
        run_length_log_posterior
            Log-posterior probabilities for each run length.

        Returns
        -------
        float
            Scalar change-point score for the current step.
        """

    def clear(self) -> None:
        """Reset any internal score state.

        Returns
        -------
        None
        """
