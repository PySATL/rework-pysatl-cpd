# -*- coding: ascii -*-
"""
Provider-related type definitions.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from enum import StrEnum


class ProviderType(StrEnum):
    """
    Type identifier for data providers.

    This StrEnum defines the available provider types used throughout
    the data module to identify different categories of data sources
    and their expected behavior.

    The enum values are:
        SEGMENT - Provider for segment-based labeled data
        NO_CHANGE - Provider for sequences with no change points
        BISEGMENT - Provider for bisegment data
        TIMESERIES - Provider for time series data
        UNLABELED - Provider for unlabeled data
    """

    SEGMENT = "segment"
    NO_CHANGE = "no_change"
    BISEGMENT = "bisegment"
    TIMESERIES = "timeseries"
    UNLABELED = "unlabeled"
