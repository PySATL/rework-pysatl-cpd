# -*- coding: ascii -*-
"""Registry entry pickers for no-reset benchmarks."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry

from pysatl_cpd.benchmark.tooling import BenchmarkEntriesPicker
from pysatl_cpd.core.single_run import SingleRunDescription
from pysatl_cpd.data.typedefs import BisegmentAnnotation, ProviderType, StateDescriptor, TransitionDescriptor


class OnlineNoResetEntryAlgorithmPicker(BenchmarkEntriesPicker):
    """
    Picks registry keys that match detector description produced by
    NoResetOnlineDetector(entry.algorithm).
    """

    def pick(
        self,
        entries: Sequence[SingleRunDescription],
        benchmark_entry: OnlineNoResetBenchmarkEntry,
    ) -> Sequence[SingleRunDescription]:
        """Pick registry keys matching a benchmark entry's detector description.

        Parameters
        ----------
        entries
            All available registry keys.
        benchmark_entry
            Entry whose detector description is used for matching.

        Returns
        -------
        Sequence[SingleRunDescription]
            Keys whose detector description matches and whose provider
            type is ``bisegment``.
        """
        expected = benchmark_entry.description
        return [
            entry
            for entry in entries
            if entry.detector_description == expected
            and entry.provider_description.provider_type == ProviderType.BISEGMENT
        ]


class OnlineNoResetBisegmentByTransitionPicker(BenchmarkEntriesPicker):
    """
    Like :class:`OnlineNoResetEntryAlgorithmPicker`, but keeps only bisegment runs whose
    annotation matches the given transition.

    Use when the registry may contain other transitions (e.g. after a full benchmark)
    but metrics must refer to a single transition only.

    Parameters
    ----------
    transition
        Transition to filter bisegment runs by.
    """

    def __init__(self, transition: TransitionDescriptor) -> None:
        self._transition = transition
        self._algorithm_picker = OnlineNoResetEntryAlgorithmPicker()

    def pick(
        self,
        entries: Sequence[SingleRunDescription],
        benchmark_entry: OnlineNoResetBenchmarkEntry,
    ) -> Sequence[SingleRunDescription]:
        """Pick keys matching both the algorithm and the configured transition.

        Parameters
        ----------
        entries
            All available registry keys.
        benchmark_entry
            Entry whose detector description is used for matching.

        Returns
        -------
        Sequence[SingleRunDescription]
            Keys whose detector description matches and whose provider
            annotation has the configured transition.
        """
        algorithm_keys = self._algorithm_picker.pick(entries, benchmark_entry)
        return [
            key
            for key in algorithm_keys
            if isinstance(key.provider_description, BisegmentAnnotation)
            and key.provider_description.transition == self._transition
        ]


class OnlineNoResetNoChangeByStatePicker(BenchmarkEntriesPicker):
    """Pick no-change registry keys for a benchmark entry and state.

    Parameters
    ----------
    state
        State descriptor to filter by.
    """

    def __init__(self, state: StateDescriptor) -> None:
        self._state = state

    def pick(
        self,
        entries: Sequence[SingleRunDescription],
        benchmark_entry: OnlineNoResetBenchmarkEntry,
    ) -> Sequence[SingleRunDescription]:
        """Pick keys matching the entry's detector and configured no-change state.

        Parameters
        ----------
        entries
            All available registry keys.
        benchmark_entry
            Entry whose detector description is used for matching.

        Returns
        -------
        Sequence[SingleRunDescription]
        """
        expected = benchmark_entry.description
        return [
            entry
            for entry in entries
            if entry.detector_description == expected
            and entry.provider_description.provider_type == ProviderType.NO_CHANGE
            and getattr(entry.provider_description, "state", None) == self._state
        ]
