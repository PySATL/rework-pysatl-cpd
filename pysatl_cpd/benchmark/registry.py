# -*- coding: ascii -*-
"""
In-memory registry of precomputed single runs (traces + labeled providers).

Persistence is opt-in: call :meth:`export_registry` / :meth:`upload_registry` explicitly.
"""

from __future__ import annotations

import pickle
from collections.abc import ItemsView, KeysView, Mapping, Sequence, ValuesView
from pathlib import Path
from typing import Generic, TypeVar, cast

from joblib import Parallel, delayed  # type: ignore[import-untyped]
from tqdm import tqdm

from pysatl_cpd.core import ChangePointDetector, DetectionTrace
from pysatl_cpd.core.single_run import SingleRun, SingleRunDescription
from pysatl_cpd.data import TimeseriesAnnotation
from pysatl_cpd.data.providers.labeled import LabeledData

# Explicit default avoids relying on joblib's contextual / version-specific choice.
DEFAULT_JOB_PARALLEL_BACKEND = "loky"

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

__all__ = ["BenchmarkRegistry", "DEFAULT_JOB_PARALLEL_BACKEND"]

DataT = TypeVar("DataT")
TraceT = TypeVar("TraceT", bound=DetectionTrace, default=DetectionTrace)


def _execute_single_run[DataT, TraceT: DetectionTrace](
    detector: ChangePointDetector[DataT],
    key: SingleRunDescription[TimeseriesAnnotation],
    provider: LabeledData[DataT, TimeseriesAnnotation],
) -> tuple[SingleRunDescription[TimeseriesAnnotation], SingleRun[TraceT, LabeledData[DataT, TimeseriesAnnotation]]]:
    """Run detection on a single provider and package the result.

    Clones the detector to avoid cross-run state pollution, executes
    ``detect`` on the provider, and wraps the outcome in a
    ``SingleRun``.

    Parameters
    ----------
    detector
        Detector to execute (will be cloned internally).
    key
        Registry key identifying this detector-provider pair.
    provider
        Input data for the detection run.

    Returns
    -------
    tuple[SingleRunDescription, SingleRun]
        The registry key paired with the resulting SingleRun.
    """
    trace = cast(TraceT, detector.clone().detect(provider))
    return key, SingleRun(trace=trace, provider=provider)


class BenchmarkRegistry(Generic[DataT, TraceT]):
    """
    Dict-like registry: :class:`SingleRunDescription` -> :class:`SingleRun`.

    ``update`` skips recomputation when the key exists unless ``force_recompute`` is True.
    """

    def __init__(self) -> None:
        self._executions_registry: dict[
            SingleRunDescription[TimeseriesAnnotation],
            SingleRun[TraceT, LabeledData[DataT, TimeseriesAnnotation]],
        ] = {}

    @property
    def executions_registry(
        self,
    ) -> Mapping[
        SingleRunDescription[TimeseriesAnnotation],
        SingleRun[TraceT, LabeledData[DataT, TimeseriesAnnotation]],
    ]:
        """Read-only view of the internal execution registry."""
        return self._executions_registry

    def update(
        self,
        detector: ChangePointDetector[DataT],
        providers: Sequence[LabeledData[DataT, TimeseriesAnnotation]],
        *,
        force_recompute: bool = False,
        n_jobs: int = 1,
        backend: str = DEFAULT_JOB_PARALLEL_BACKEND,
    ) -> None:
        """Register detection results for a detector against a sequence of providers.

        For each provider, skips computation if a matching entry already
        exists in the registry (unless ``force_recompute`` is True).
        Supports sequential and parallel execution modes via the
        ``n_jobs`` parameter.

        Parameters
        ----------
        detector
            Detector to execute against the providers.
        providers
            Providers whose data will be fed into the detector.
        force_recompute
            When True, re-executes detection even for already-registered
            entries.
        n_jobs
            Number of parallel workers (default 1). Serial execution is
            used when set to 1.
        backend
            Joblib parallel backend identifier (default ``"loky"``).
        """
        jobs: list[tuple[SingleRunDescription[TimeseriesAnnotation], LabeledData[DataT, TimeseriesAnnotation]]] = []
        for provider in providers:
            key = SingleRunDescription(
                detector_description=detector.description,
                provider_description=provider.annotation,
            )
            if not force_recompute and key in self._executions_registry:
                continue
            jobs.append((key, provider))

        if n_jobs == 1:
            for key, provider in tqdm(jobs, desc="Benchmarking providers", unit="provider"):
                trace: TraceT = cast(TraceT, detector.detect(provider))  # TODO: Runtime check?
                self._executions_registry[key] = SingleRun(
                    trace=trace,
                    provider=provider,
                )
            return

        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_execute_single_run)(detector, key, provider) for key, provider in jobs
        )
        self._executions_registry.update(dict(results))

    def __getitem__(
        self, key: SingleRunDescription[TimeseriesAnnotation]
    ) -> SingleRun[TraceT, LabeledData[DataT, TimeseriesAnnotation]]:
        """Retrieve a SingleRun by its description key.

        Parameters
        ----------
        key
            Description identifying the detector-provider pair.

        Returns
        -------
        SingleRun
            The cached execution result for the given key.
        """
        return self._executions_registry[key]

    def __len__(self) -> int:
        """Return the number of registered executions.

        Returns
        -------
        int
            Registry size (number of detector-provider pairs stored).
        """
        return len(self._executions_registry)

    def __contains__(self, key: SingleRunDescription[TimeseriesAnnotation]) -> bool:
        """Check whether a given description key exists in the registry.

        Parameters
        ----------
        key
            Description to look up.

        Returns
        -------
        bool
            True if the key is present in the registry.
        """
        return key in self._executions_registry

    def keys(self) -> KeysView[SingleRunDescription[TimeseriesAnnotation]]:
        """Return a view of all registered description keys.

        Returns
        -------
        KeysView[SingleRunDescription]
            Set-like view of registry keys.
        """
        return self._executions_registry.keys()

    def values(self) -> ValuesView[SingleRun[TraceT, LabeledData[DataT, TimeseriesAnnotation]]]:
        """Return a view of all registered SingleRun values.

        Returns
        -------
        ValuesView[SingleRun]
            Collection view of registry values.
        """
        return self._executions_registry.values()

    def items(
        self,
    ) -> ItemsView[
        SingleRunDescription[TimeseriesAnnotation],
        SingleRun[TraceT, LabeledData[DataT, TimeseriesAnnotation]],
    ]:
        """Return a view of all (key, value) pairs in the registry.

        Returns
        -------
        ItemsView[SingleRunDescription, SingleRun]
            Set-like view of registry item pairs.
        """
        return self._executions_registry.items()

    def export_registry(self, export_registry_path: Path) -> None:
        """Serialize the registry to a pickle file on disk.

        Creates parent directories as needed before writing.

        Parameters
        ----------
        export_registry_path
            Destination path for the serialized registry.
        """
        export_registry_path.parent.mkdir(parents=True, exist_ok=True)
        with export_registry_path.open("wb") as f:
            pickle.dump(self._executions_registry, f, protocol=pickle.HIGHEST_PROTOCOL)

    def upload_registry(self, upload_registry_path: Path) -> None:
        """Deserialize and load a registry from a pickle file.

        Validates that the loaded object is a dict and that all entries
        conform to the expected ``SingleRunDescription`` -> ``SingleRun``
        type contract before replacing the in-memory registry.

        Parameters
        ----------
        upload_registry_path
            Path to a pickled registry file previously produced by
            ``export_registry``.

        Raises
        ------
        TypeError
            If the file content is not a dict or contains entries of
            unexpected types.
        """
        with upload_registry_path.open("rb") as f:
            loaded = pickle.load(f)
        if not isinstance(loaded, dict):
            msg = f"Registry file must contain a dict, got {type(loaded).__name__}"
            raise TypeError(msg)
        for k, v in loaded.items():
            if not isinstance(k, SingleRunDescription) or not isinstance(v, SingleRun):
                msg = "Registry file entries must be SingleRunDescription -> SingleRun"
                raise TypeError(msg)
        self._executions_registry = cast(
            dict[
                SingleRunDescription[TimeseriesAnnotation],
                SingleRun[TraceT, LabeledData[DataT, TimeseriesAnnotation]],
            ],
            dict(loaded),
        )
