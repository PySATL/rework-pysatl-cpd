# -*- coding: ascii -*-
"""Plain univariate labeled data provider."""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable
from typing import cast

from pysatl_cpd.data.providers.data_provider import DataProvider
from pysatl_cpd.data.providers.labeled.labeled_data import LabeledData
from pysatl_cpd.data.providers.plain.np_univariate import NDArrayUnivariateProvider
from pysatl_cpd.data.typedefs import SegmentInfo, TimeseriesAnnotation, UnlabeledTimeseriesAnnotation
from pysatl_cpd.typedefs import Number, UnivariateNumericArray


class PlainUnivariateLabeledData[AnnotationT: TimeseriesAnnotation](LabeledData[Number, AnnotationT]):
    """Labeled data implementation for univariate sequences.

    Parameters
    ----------
    unlabeled
        Unlabeled data provider for the sequence.
    labeling
        Iterable of segment information.
    annotation
        Annotation instance for labeling.
    """

    def __init__(
        self,
        unlabeled: NDArrayUnivariateProvider[UnlabeledTimeseriesAnnotation],
        labeling: Iterable[SegmentInfo],
        annotation: AnnotationT,
    ) -> None:
        super().__init__(unlabeled, labeling, annotation)

    @property
    def unlabeled(self) -> NDArrayUnivariateProvider[UnlabeledTimeseriesAnnotation]:
        """Return the underlying unlabeled data provider.

        Returns
        -------
        unlabeled
            The underlying unlabeled data provider.
        """
        return cast(NDArrayUnivariateProvider[UnlabeledTimeseriesAnnotation], self._unlabeled_data)

    @classmethod
    def from_unlabeled_data[A: TimeseriesAnnotation](
        cls,
        unlabeled: DataProvider[Number, UnlabeledTimeseriesAnnotation],
        segment_info: Iterable[SegmentInfo],
        annotation: A,
    ) -> "PlainUnivariateLabeledData[A]":
        """Create labeled data from an unlabeled univariate provider.

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
            If unlabeled is not an NDArrayUnivariateProvider.
        """
        if not isinstance(unlabeled, NDArrayUnivariateProvider):
            raise TypeError("PlainUnivariateLabeledData requires an NDArrayUnivariateProvider")
        return cast("PlainUnivariateLabeledData[A]", cls(unlabeled, segment_info, cast(AnnotationT, annotation)))

    @property
    def raw_data(self) -> UnivariateNumericArray:
        """Return a copy of the underlying data array.

        Returns
        -------
        raw_data
            The underlying numeric data array.
        """
        return self.unlabeled.raw_data
