# -*- coding: ascii -*-
"""
NumPy-backed data providers.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Iterator, Sequence
from typing import Self, cast

import numpy as np

from pysatl_cpd.data.providers.data_provider import DataProvider
from pysatl_cpd.data.typedefs import TimeseriesAnnotation
from pysatl_cpd.typedefs import MultivariateNumericArray, NumericArray, UnivariateNumericArray


class NDArrayMultivariateProvider[AnnotationT: TimeseriesAnnotation](DataProvider[UnivariateNumericArray, AnnotationT]):
    """Data provider for 2-D NumPy arrays.

    Parameters
    ----------
    data
        2-D numeric array containing the timeseries data.
    annotation
        Annotation object associated with the timeseries.

    Raises
    ------
    ValueError
        If the data is not 2-dimensional.
    """

    def __init__(self, data: NumericArray, annotation: AnnotationT) -> None:
        if data.ndim != 2:
            raise ValueError(f"Expected 2 dimensions, got {data.ndim}")
        self._data = cast(MultivariateNumericArray, data)
        self._annotation = annotation

    def __iter__(self) -> Iterator[UnivariateNumericArray]:
        """Iterate over rows of the multivariate series.

        Returns
        -------
        iterator
            Iterator over per-sample feature vectors.
        """
        return iter(self._data)

    def __len__(self) -> int:
        """Return the series length.

        Returns
        -------
        length
            Number of samples in the series.
        """
        return self._data.shape[0]

    @property
    def annotation(self) -> AnnotationT:
        """Return the annotation associated with this timeseries.

        Returns
        -------
        annotation
            The annotation object for this series.
        """
        return self._annotation

    @property
    def raw_data(self) -> MultivariateNumericArray:
        """Return a copy of the underlying data array.

        Returns
        -------
        raw_data
            Copy of the underlying multivariate numeric array.
        """
        return self._data.copy()

    def cut(
        self,
        start: int,
        stop: int,
        *,
        annotation: AnnotationT | None = None,
    ) -> "NDArrayMultivariateProvider[AnnotationT]":
        """
        Extract a slice of the timeseries data.

        Parameters
        ----------
        start
            Start index of the slice (inclusive).
        stop
            Stop index of the slice (inclusive).
        annotation
            Optional annotation for the sliced data.

        Returns
        -------
        provider
            New provider containing the sliced data.
        """
        self._validate_cut_boundaries(start, stop)
        return NDArrayMultivariateProvider(
            self._data[start : stop + 1].copy(),
            annotation if annotation is not None else self.default_slice_annotation(start, stop),
        )

    @classmethod
    def merge(
        cls,
        providers: Sequence[Self],
        annotation_builder: Callable[[Sequence[AnnotationT]], AnnotationT] | None = None,
    ) -> "NDArrayMultivariateProvider[AnnotationT]":
        """
        Merge multiple providers into a single provider.

        Parameters
        ----------
        providers
            Sequence of providers to merge.
        annotation_builder
            Optional function to build the merged annotation.

        Returns
        -------
        provider
            New provider with merged data and annotation.
        """
        cls._validate_merge_inputs(providers)

        if annotation_builder is None:
            annotation_builder = cls.default_merge_annotation_builder()

        merged_data = np.concatenate([p._data for p in providers], axis=0)
        merged_annotation = annotation_builder([p.annotation for p in providers])
        return NDArrayMultivariateProvider[AnnotationT](merged_data, merged_annotation)
