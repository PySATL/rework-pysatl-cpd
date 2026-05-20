# -*- coding: ascii -*-
"""Plain multivariate labeled data provider."""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable
from typing import cast

from pysatl_cpd.data.providers.data_provider import DataProvider
from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData
from pysatl_cpd.data.providers.plain.np_multivariate import NDArrayMultivariateProvider
from pysatl_cpd.data.typedefs import SegmentInfo, TimeseriesAnnotation, UnlabeledTimeseriesAnnotation
from pysatl_cpd.typedefs import MultivariateNumericArray, UnivariateNumericArray


class PlainMultivariateLabeledData[AnnotationT: TimeseriesAnnotation](LabeledData[UnivariateNumericArray, AnnotationT]):
    """Labeled data implementation for multivariate sequences.

    Parameters
    ----------
    unlabeled
        Unlabeled data provider for the multivariate sequence.
    labeling
        Iterable of segment information.
    annotation
        Annotation instance for labeling.
    """

    def __init__(
        self,
        unlabeled: NDArrayMultivariateProvider[UnlabeledTimeseriesAnnotation],
        labeling: Iterable[SegmentInfo],
        annotation: AnnotationT,
    ) -> None:
        super().__init__(unlabeled, labeling, annotation)

    @property
    def unlabeled(self) -> NDArrayMultivariateProvider[UnlabeledTimeseriesAnnotation]:
        """Return the underlying unlabeled data provider.

        Returns
        -------
        unlabeled
            The underlying unlabeled data provider.
        """
        return cast(NDArrayMultivariateProvider[UnlabeledTimeseriesAnnotation], self._unlabeled_data)

    @classmethod
    def from_unlabeled_data[A: TimeseriesAnnotation](
        cls,
        unlabeled: DataProvider[UnivariateNumericArray, UnlabeledTimeseriesAnnotation],
        segment_info: Iterable[SegmentInfo],
        annotation: A,
    ) -> "PlainMultivariateLabeledData[A]":
        """Create labeled data from an unlabeled multivariate provider.

        Parameters
        ----------
        unlabeled
            Unlabeled provider containing the raw series.
        segment_info
            Segment labeling for the series.
        annotation
            Annotation attached to the labeled series.

        Returns
        -------
        labeled_data
            Concrete labeled-data provider built from the inputs.

        Raises
        ------
        TypeError
            If unlabeled is not an NDArrayMultivariateProvider.
        """
        if not isinstance(unlabeled, NDArrayMultivariateProvider):
            raise TypeError("PlainMultivariateLabeledData requires an NDArrayMultivariateProvider")
        return cast("PlainMultivariateLabeledData[A]", cls(unlabeled, segment_info, cast(AnnotationT, annotation)))

    @property
    def raw_data(self) -> MultivariateNumericArray:
        """Return a copy of the underlying data array.

        Returns
        -------
        raw_data
            The underlying numeric data array.
        """
        return self.unlabeled.raw_data
