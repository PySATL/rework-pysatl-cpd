# -*- coding: ascii -*-
"""Drop-based Bayesian change-point score function."""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import numpy.typing as npt

from pysatl_cpd.algorithms.online.bayesian.protocol.cpf import IBayesianCPF


class DropCPF(IBayesianCPF):
    """Return the positive drop in maximal-run probability between steps.

    Sets the previous max-run-log-probability to ``None`` (first call to
    *calculate* will return 0.0).
    """

    def __init__(self) -> None:
        self._previous_log_prob_max_run: float | None = None

    def calculate(self, run_length_log_posterior: npt.NDArray[np.float64]) -> float:
        """Compute the positive drop in max-run probability since the last step.

        Parameters
        ----------
        run_length_log_posterior
            Log-posterior probabilities for each run length.

        Returns
        -------
        float
            Positive drop (clamped at zero) in max-run probability.
        """
        if run_length_log_posterior.size == 0:
            return 0.0

        current_log_prob_max_run = float(run_length_log_posterior[-1])
        if self._previous_log_prob_max_run is None:
            self._previous_log_prob_max_run = current_log_prob_max_run
            return 0.0

        drop = float(np.exp(self._previous_log_prob_max_run)) - float(np.exp(current_log_prob_max_run))
        self._previous_log_prob_max_run = current_log_prob_max_run
        return max(0.0, drop)

    def clear(self) -> None:
        """Reset internal state.

        Returns
        -------
        None
        """
        self._previous_log_prob_max_run = None

    def __repr__(self) -> str:
        return "DropCPF()"
