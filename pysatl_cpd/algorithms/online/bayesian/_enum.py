# -*- coding: ascii -*-
"""Enums for Bayesian online change-point detection."""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from enum import StrEnum


class BayesianCPFType(StrEnum):
    """Supported Bayesian change-point score functions."""

    MAX_RUN_LENGTH = "max_run_length"
    DROP = "drop"
