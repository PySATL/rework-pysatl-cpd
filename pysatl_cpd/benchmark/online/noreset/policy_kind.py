# -*- coding: ascii -*-
"""String enum for no-reset bisegment policy selection on benchmarks."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from enum import StrEnum


class NoResetPolicyKind(StrEnum):
    """Policy flavour used when building a no-reset classification report."""

    POINT = "point"
    EVENT = "event"
    MIXED = "mixed"
