# -*- coding: ascii -*-
"""
Factory helpers for generalized CUSUM components.

This module provides internal utility functions used by CUSUM monitoring
schemas, estimating schemas, change-point functions, and algorithm
implementations to normalize incoming observations into a consistent
one-dimensional ``float64`` array representation.

.. raw:: html

    <h2>Public API</h2>

- ``coerce_observation(observation)`` -- converts a scalar, 0-D array, or
  1-D numeric array into a 1-D ``float64`` NumPy array. Raises ``ValueError``
  for inputs with more than one dimension.

Examples
--------
Coerce a scalar observation:

>>> import numpy as np
>>> from pysatl_cpd.algorithms.online.cusum.utils import coerce_observation
>>> coerce_observation(3.14)
array([3.14])

Coerce a 1-D array observation:

>>> coerce_observation(np.array([1.0, 2.0, 3.0]))
array([1., 2., 3.])

Coerce a 0-D array:

>>> coerce_observation(np.array(42.0))
array([42.])

Multi-dimensional inputs are rejected:

>>> coerce_observation(np.array([[1.0, 2.0], [3.0, 4.0]]))
Traceback (most recent call last):
    ...
ValueError: Observations must be vectors or scalars, got shape (2, 2)

Notes
-----
- This module is intended for internal use within the CUSUM algorithm
  package. Consumers of the public CUSUM API (e.g., ``PageTwoSidedCusum``,
  ``CrosierCusum``) do not need to call these helpers directly.
- All outputs are ``float64`` arrays regardless of the input dtype.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np

from pysatl_cpd.typedefs import MultivariateNumericArray, Number, NumericArray, UnivariateNumericArray


def coerce_observation(
    observation: Number | UnivariateNumericArray | MultivariateNumericArray | NumericArray,
) -> NumericArray:
    """Convert one observation to a one-dimensional float array.

    Scalars are reshaped to length-1 vectors. Multi-dimensional arrays
    are rejected.

    Parameters
    ----------
    observation
        Input value (scalar, 1-D array, or 0-D array).

    Returns
    -------
    NumericArray
        1-D float64 array.

    Raises
    ------
    ValueError
        If *observation* has more than one dimension.
    """
    coerced = np.asarray(observation, dtype=np.float64)
    if coerced.ndim == 0:
        coerced = coerced.reshape(1)
    if coerced.ndim != 1:
        raise ValueError(f"Observations must be vectors or scalars, got shape {coerced.shape}")
    return coerced


__all__ = [
    "coerce_observation",
]
