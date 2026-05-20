# -*- coding: ascii -*-
"""Base scenario types for no-reset benchmark campaigns."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import logging
from collections.abc import Sequence
from typing import Any, TypeVar

from pysatl_cpd.benchmark.online.noreset.detector.noreset_detector import NoResetOnlineDetector
from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry
from pysatl_cpd.benchmark.scenarios import BenchmarkScenario
from pysatl_cpd.core.online import OnlineDetectionTrace

logger = logging.getLogger(__name__)

DataT = TypeVar("DataT")


class NoResetBenchmarkScenario[DataT, ResultT](BenchmarkScenario[DataT, OnlineDetectionTrace[Any], ResultT]):
    """Base scenario for no-reset online benchmark campaigns.

    Parameters
    ----------
    entries
        Detector entries to benchmark.
    collect_states
        Whether to retain algorithm states during detection (default False).
    """

    def __init__(
        self,
        entries: Sequence[OnlineNoResetBenchmarkEntry],
        collect_states: bool = False,
    ) -> None:
        self.entries = entries
        self.collect_states = collect_states

    def _make_detector(self, entry: OnlineNoResetBenchmarkEntry) -> NoResetOnlineDetector:
        """Build a ``NoResetOnlineDetector`` from a benchmark entry.

        Recreates the algorithm and wraps it together with the entry's
        data transformer into a detector instance.

        Parameters
        ----------
        entry
            Entry containing the algorithm and optional transformer.

        Returns
        -------
        NoResetOnlineDetector
            A fresh detector ready for execution.
        """
        return NoResetOnlineDetector(
            entry.algorithm.recreate(),
            collect_states=self.collect_states,
            data_transformer=entry.data_transformer,
            bisegment_cut=entry.bisegment_cut,
        )
