# -*- coding: ascii -*-
"""
Tests for cusum utils.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_cpd.algorithms.online.cusum.utils import coerce_observation


def test_coerce_observation_rejects_multidim() -> None:
    with pytest.raises(ValueError, match="Observations must be vectors or scalars"):
        coerce_observation(np.array([[1.0, 2.0], [3.0, 4.0]]))
