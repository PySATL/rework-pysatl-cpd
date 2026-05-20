# -*- coding: ascii -*-
"""Segment generator protocol."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Protocol

import numpy as np

from pysatl_cpd.data.generator.segments.models import GeneratedSegment


class SegmentGenerator(Protocol):
    """
    Protocol for generating synthetic time series segments.

    Implement this protocol to create custom segment generators
    that produce synthetic data with specified distributions.
    """

    @property
    def feature_names(self) -> tuple[str, ...]:
        """
        Get feature names for the generated segment.

        Returns
        -------
        feature_names
            Tuple of feature names.
        """
        ...

    @property
    def length(self) -> int:
        """
        Get length of generated segment.

        Returns
        -------
        length
            Number of time points.
        """
        ...

    def generate(self, rng: np.random.Generator | None = None) -> GeneratedSegment:
        """
        Generate a synthetic segment.

        Parameters
        ----------
        rng
            Random number generator. If None, uses default RNG.

        Returns
        -------
        segment
            Generated synthetic segment.
        """
        ...
