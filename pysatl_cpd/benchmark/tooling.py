# -*- coding: ascii -*-
"""Shared tooling types for benchmark modules."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import ParamSpec, Protocol, TypeVar

from pysatl_cpd.core import DetectionTrace
from pysatl_cpd.core.single_run import SingleRun, SingleRunDescription
from pysatl_cpd.data import LabeledData, TimeseriesAnnotation

P = ParamSpec("P")
DataT = TypeVar("DataT")


class BenchmarkEntriesPicker(Protocol[P]):
    """Selects a subset of registry keys by an explicit list and options."""

    def pick(
        self,
        entries: Sequence[SingleRunDescription],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Sequence[SingleRunDescription]:
        """Select a subset of registry entries based on caller-provided arguments.

        Parameters
        ----------
        entries
            Full list of available registry description keys.
        *args
            Positional arguments driving the selection logic.
        **kwargs
            Keyword arguments driving the selection logic.

        Returns
        -------
        Sequence[SingleRunDescription]
            Descriptions of the selected entries.
        """


class BenchmarkEntriesPreparator(Protocol[DataT, P]):
    """Transforms or filters runs (e.g. traces + providers) according to options."""

    def prepare(
        self,
        runs: Sequence[SingleRun[DetectionTrace, LabeledData[DataT, TimeseriesAnnotation]]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Sequence[SingleRun[DetectionTrace, LabeledData[DataT, TimeseriesAnnotation]]]:
        """Transform or filter a collection of runs according to caller options.

        Parameters
        ----------
        runs
            Input runs (trace + provider pairs) to process.
        *args
            Positional arguments controlling the transformation.
        **kwargs
            Keyword arguments controlling the transformation.

        Returns
        -------
        Sequence[SingleRun]
            Processed subset or transformation of the input runs.
        """
