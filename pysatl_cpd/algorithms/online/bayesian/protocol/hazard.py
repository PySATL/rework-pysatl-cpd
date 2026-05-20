# -*- coding: ascii -*-
"""Protocol definition for Bayesian hazard models."""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Protocol

import numpy as np
import numpy.typing as npt


class IHazard(Protocol):
    """Interface for hazard models in Bayesian online detection."""

    def hazard(self, run_lengths: npt.NDArray[np.intp]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return log hazard and log survival values for each run length.

        Parameters
        ----------
        run_lengths
            Array of run length indices.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
            Pair of (log_hazard, log_survival) arrays.
        """
