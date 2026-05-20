# -*- coding: ascii -*-
"""Max-run-length Bayesian change-point score function."""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import numpy.typing as npt

from pysatl_cpd.algorithms.online.bayesian.protocol.cpf import IBayesianCPF


class MaxRunLengthCPF(IBayesianCPF):
    """Return one minus the probability of the maximal run length state."""

    def calculate(self, run_length_log_posterior: npt.NDArray[np.float64]) -> float:
        """Convert run-length log-posterior to a change-point score.

        Parameters
        ----------
        run_length_log_posterior
            Log-posterior probabilities for each run length.

        Returns
        -------
        float
            One minus the probability of the longest run length.
        """
        if run_length_log_posterior.size == 0:
            return 0.0
        return 1.0 - float(np.exp(run_length_log_posterior[-1]))

    def clear(self) -> None:
        """Reset internal state.

        Returns
        -------
        None
        """
        return None

    def __repr__(self) -> str:
        return "MaxRunLengthCPF()"
