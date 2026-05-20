# -*- coding: ascii -*-
"""Constant hazard model for Bayesian online change-point detection."""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import numpy.typing as npt

from pysatl_cpd.algorithms.online.bayesian.protocol.hazard import IHazard


class ConstantHazard(IHazard):
    """Constant hazard model with fixed timescale.

    Parameters
    ----------
    lambda_
        Expected run length (must be >= 1.0).

    Raises
    ------
    ValueError
        If *lambda_* < 1.0.
    """

    def __init__(self, lambda_: float) -> None:
        if lambda_ < 1.0:
            raise ValueError("lambda_ must be >= 1.0")

        self._lambda = np.float64(lambda_)
        self._log_h = -np.log(self._lambda)
        self._log_neg_h = np.log(1.0 - (1.0 / self._lambda))

    def hazard(self, run_lengths: npt.NDArray[np.intp]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return constant log-hazard and log-survival for each run length.

        Parameters
        ----------
        run_lengths
            Array of run length indices (used only for shape).

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
            Pair of (log_hazard, log_survival) arrays broadcast to
            match the shape of *run_lengths*.
        """
        n_runs = len(run_lengths)
        return np.full(n_runs, self._log_h), np.full(n_runs, self._log_neg_h)

    def __repr__(self) -> str:
        return f"ConstantHazard(lambda_={float(self._lambda)!r})"
