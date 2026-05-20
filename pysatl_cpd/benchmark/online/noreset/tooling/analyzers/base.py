# -*- coding: ascii -*-
"""Base classes and mixins for no-reset benchmark analyzers."""

from __future__ import annotations

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pysatl_cpd.benchmark.online.noreset.entry import OnlineNoResetBenchmarkEntry

from pysatl_cpd.benchmark.online.noreset.metrics import NoResetClassificationReport
from pysatl_cpd.benchmark.online.noreset.tooling.pickers import OnlineNoResetEntryAlgorithmPicker
from pysatl_cpd.benchmark.registry import BenchmarkRegistry
from pysatl_cpd.benchmark.tooling import BenchmarkEntriesPicker
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.core.online.detectors.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.core.single_run import SingleRun
from pysatl_cpd.data import LabeledData, TimeseriesAnnotation
from pysatl_cpd.data.typedefs import ProviderType


class NoResetAnalyzerBase(ABC):
    """Base class for no-reset benchmark analyzers.

    Provides a default ``entries_picker`` and lazily resolved
    ``registry`` property shared by all concrete analyzers.

    Parameters
    ----------
    entries_picker
        Strategy for selecting registry entries. Defaults to
        ``OnlineNoResetEntryAlgorithmPicker`` when not provided.
    """

    def __init__(self, entries_picker: BenchmarkEntriesPicker | None = None) -> None:
        self.entries_picker = entries_picker if entries_picker is not None else OnlineNoResetEntryAlgorithmPicker()
        self._registry: BenchmarkRegistry[Any, OnlineDetectionTrace[Any]] | None = None

    @abstractmethod
    def _set_registry(
        self, registry: BenchmarkRegistry[Any, OnlineDetectionTrace[Any]]
    ) -> None: ...  # pragma: no cover

    @property
    def registry(self) -> BenchmarkRegistry[Any, OnlineDetectionTrace[Any]]:
        """Registry of cached detection runs.

        Raises
        ------
        ValueError
            If the registry has not been set yet.
        """
        if self._registry is None:
            raise ValueError("Registry is not set. Call set_registry(registry) first.")
        return self._registry

    @registry.setter
    def registry(self, registry: BenchmarkRegistry[Any, OnlineDetectionTrace[Any]]) -> None:
        """Set the registry used by this analyzer.

        Parameters
        ----------
        registry
            Registry containing cached detection runs.
        """
        self._set_registry(registry)

    def pick_runs(
        self,
        benchmark_entry: OnlineNoResetBenchmarkEntry,
        *,
        entries_picker: BenchmarkEntriesPicker | None = None,
    ) -> list[SingleRun[OnlineDetectionTrace[Any], Any]]:
        """Retrieve runs matching a benchmark entry from the registry.

        Uses the supplied or default entries picker to select relevant
        keys, then fetches the corresponding ``SingleRun`` objects.

        Parameters
        ----------
        benchmark_entry
            Entry describing the detector configuration.
        entries_picker
            Override picker for this call; falls back to
            ``self.entries_picker`` when None.

        Returns
        -------
        list[SingleRun]
            Runs whose keys were selected by the picker.
        """
        picker = entries_picker if entries_picker is not None else self.entries_picker
        keys = list(self.registry.keys())
        picked_keys = picker.pick(keys, benchmark_entry)
        return [self.registry[k] for k in picked_keys]


class NoResetBisigementClassificationMixin(ABC):
    """Mixin that provides classification-report access and bisegment validation."""

    @abstractmethod
    def _set_classification_report(  # pragma: no cover
        self,
        classification_report: NoResetClassificationReport[Any, Any],
    ) -> None: ...

    @property
    def classification_report(self) -> NoResetClassificationReport[Any, Any]:
        """Classification report containing TP/FP/FN metric definitions.

        Raises
        ------
        ValueError
            If the report has not been set yet.
        """
        report = cast(NoResetClassificationReport, getattr(self, "_classification_report", None))
        if report is None:
            raise ValueError("Classification report is not set. Set classification_repor first.")
        return report

    @classification_report.setter
    def classification_report(
        self,
        classification_report: NoResetClassificationReport[Any, Any],
    ) -> None:
        """Set the classification report used by this analyzer.

        Parameters
        ----------
        classification_report
            Report defining TP/FP/FN source metrics and derived metrics.
        """
        return self._set_classification_report(classification_report)

    @staticmethod
    def validate_bisegment_runs(
        runs: Sequence[SingleRun[OnlineDetectionTrace[Any], LabeledData[Any, TimeseriesAnnotation]]],
        expected_detector: ChangePointDetectorDescription,
    ) -> None:
        """Verify that every run matches the expected detector and is a valid bisegment.

        Checks three conditions: all runs belong to the expected detector,
        all providers are of ``BISEGMENT`` type, and each provider has
        exactly one change point.

        Parameters
        ----------
        runs
            Runs to validate.
        expected_detector
            Detector description that every run must match.

        Raises
        ------
        ValueError
            If any run fails the validation checks.
        """
        for run in runs:
            if run.trace.detector_description != expected_detector:
                raise ValueError("Picked run does not match benchmark entry detector")

            if getattr(run.provider.annotation, "provider_type", None) != ProviderType.BISEGMENT:
                raise ValueError("No-reset classification requires bisegment providers")

            if len(run.provider.change_points) != 1:
                raise ValueError("No-reset classification requires exactly one true change point")
