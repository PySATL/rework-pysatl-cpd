# -*- coding: ascii -*-
"""
Abstract data provider interface.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from typing import Generic, Self, TypeVar

from pysatl_cpd.data.typedefs import TimeseriesAnnotation
from pysatl_cpd.typedefs import frozendict

DataT = TypeVar("DataT")
AnnotationT = TypeVar("AnnotationT", bound=TimeseriesAnnotation, covariant=True)


class DataProvider(ABC, Generic[DataT, AnnotationT]):  # noqa: UP046
    """Abstract source class for sequential data providers."""

    def _validate_cut_boundaries(self, start: int, stop: int) -> None:
        """Validate inclusive cut boundaries against provider length.

        Parameters
        ----------
        start
            Start index of the slice.
        stop
            Stop index of the slice (inclusive).

        Raises
        ------
        ValueError
            If the slice boundaries are outside provider bounds.
        """
        if start < 0:
            raise ValueError("Slice start index must be non-negative")
        if stop < start:
            raise ValueError("Slice stop index must be greater than or equal to start index")
        if stop >= len(self):
            raise ValueError(f"Slice stop index {stop} exceeds data length {len(self)}")

    @classmethod
    def _validate_merge_inputs(cls: type[Self], providers: Sequence[Self]) -> None:
        """Validate common merge preconditions.

        Parameters
        ----------
        providers
            Providers to merge.

        Raises
        ------
        ValueError
            If no providers are supplied.
        TypeError
            If providers do not all share the same concrete type.
        """
        if not providers:
            raise ValueError("merge requires at least one provider")
        for provider in providers:
            if not isinstance(provider, cls):
                raise TypeError(f"All providers must be {cls.__name__}, got {type(provider).__name__}")

    @property
    def name(self) -> str:
        """Name of the data provider.

        Returns
        -------
        name
            Name from the annotation.
        """
        return self.annotation.name

    @property
    @abstractmethod
    def annotation(self) -> AnnotationT:
        """Annotation of the data provider.

        Returns
        -------
        annotation
            Annotation instance for this provider.
        """
        ...  # pragma: no cover

    @abstractmethod
    def __iter__(self) -> Iterator[DataT]:
        """Iterate over data items in the provider.

        Returns
        -------
        iterator
            Iterator yielding data items.
        """
        ...  # pragma: no cover

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of data items in the provider.

        Returns
        -------
        length
            Number of data items.
        """
        ...  # pragma: no cover

    @abstractmethod
    def cut(
        self,
        start: int,
        stop: int,
        *,
        annotation: AnnotationT | None = None,
    ) -> "DataProvider[DataT, AnnotationT]":
        """Cut a slice of data from the provider.

        Parameters
        ----------
        start
            Start index of the slice.
        stop
            Stop index of the slice (inclusive).
        annotation
            Optional annotation for the sliced data.

        Returns
        -------
        provider
            New provider containing the sliced data.
        """
        ...

    def default_slice_annotation(self, start: int, stop: int) -> AnnotationT:
        """Default annotation for sliced data.

        Parameters
        ----------
        start
            Start index of the slice.
        stop
            Stop index of the slice.

        Returns
        -------
        annotation
            Default annotation for the sliced data.
        """
        return replace(
            self.annotation,
            name=f"{self.annotation.name}[{start}:{stop}]",
            source=self.annotation.source,
            # TODO: Fix slicing metadata
            metadata=frozendict(**self.annotation.metadata),  # , slice_start=start, slice_stop=stop),
        )

    @classmethod
    @abstractmethod
    def merge(
        cls: type[Self],
        providers: Sequence[Self],
        annotation_builder: Callable[[Sequence[AnnotationT]], AnnotationT] | None = None,
    ) -> "DataProvider[DataT, AnnotationT]":
        """Merge multiple providers into one.

        Parameters
        ----------
        providers
            Sequence of providers to merge.
        annotation_builder
            Optional callable to build merged annotation.

        Returns
        -------
        provider
            New provider containing merged data.
        """
        ...

    @staticmethod
    def default_merge_annotation_builder(source: str | None = None) -> Callable[[Sequence[AnnotationT]], AnnotationT]:
        """Build a default annotation builder for merged providers.

        Parameters
        ----------
        source
            Optional source string for the merged annotation.

        Returns
        -------
        builder
            Callable that builds annotation from a sequence.
        """

        def builder(annotations: Sequence[AnnotationT]) -> AnnotationT:
            """Merge a sequence of annotations into a single one.

            Uses the annotation of the first element and sets the name
            to ``'merged {first}...{last}'``.
            """
            return replace(
                annotations[0],
                name=f"merged {annotations[0].name}...{annotations[-1].name}",
                source="empty" if source is None else source,
            )

        return builder

    def __or__(self, other: Self) -> "DataProvider[DataT, AnnotationT]":
        """Merge two providers using the | operator.

        Parameters
        ----------
        other
            Other provider to merge with.

        Returns
        -------
        provider
            New provider containing merged data.

        Raises
        ------
        TypeError
            If the other provider is not the same concrete type.
        """
        if type(other) is not type(self):
            raise TypeError(f"Cannot merge {type(self).__name__} with {type(other).__name__}")
        return type(self).merge([self, other])
