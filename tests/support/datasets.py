# -*- coding: ascii -*-
"""
Tests for datasets.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence

from pysatl_cpd.data import Dataset, StateDataset
from tests.support.providers import make_no_change_labeled, make_pandas_labeled, make_univariate_labeled


def make_dataset() -> Dataset[float, object]:
    """Build a mixed dataset with labeled providers."""
    return Dataset([make_univariate_labeled(name="plain-a"), make_pandas_labeled(name="pandas-b")])


def make_state_dataset() -> StateDataset[float]:
    """Build a direct no-change dataset."""
    state_provider = make_no_change_labeled(name="steady-a")
    other_provider = make_no_change_labeled(data=(5.0, 6.0, 7.0, 8.0), name="steady-b")
    return StateDataset([state_provider, other_provider])


def make_state_dataset_source() -> Dataset[float, object]:
    """Build a source dataset suitable for StateDataset.from_dataset."""
    providers: Sequence[object] = [
        make_univariate_labeled(name="source-a"),
        make_univariate_labeled(data=(5.0, 6.0, 7.0, 8.0, 9.0, 10.0), name="source-b"),
    ]
    return Dataset(providers)  # type: ignore[arg-type]
