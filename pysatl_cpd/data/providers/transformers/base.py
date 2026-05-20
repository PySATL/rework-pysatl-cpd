# -*- coding: ascii -*-
"""Data transformer interfaces."""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from typing import Any

from pysatl_cpd.data.providers import DataProvider
from pysatl_cpd.typedefs import stable_hash


class IDataTransformer[DataProviderInT: DataProvider[Any, Any], DataProviderOutT: DataProvider[Any, Any]](ABC):
    """
    Abstract source class for data transformers.

    Defines the interface for transforming data providers from one type
    to another through a transform operation.
    """

    @abstractmethod
    def transform(self, provider: DataProviderInT) -> DataProviderOutT:
        """
        Transform input data provider to output data provider.

        Parameters
        ----------
        provider
            Input data provider to transform.

        Returns
        -------
        result
            Transformed data provider.
        """

    @property
    @abstractmethod
    def annotation(self) -> str:
        """
        Transformer annotation.

        Returns
        -------
        annotation
            Annotation string describing this transformer.
        """

    def __and__[DataProviderFinalT: DataProvider[Any, Any]](
        self, other: "IDataTransformer[DataProviderOutT, DataProviderFinalT]"
    ) -> "ComposedTransformer[DataProviderInT, DataProviderFinalT]":
        """Compose this transformer with another transformer.

        Parameters
        ----------
        other
            Transformer applied after this transformer.

        Returns
        -------
        transformer
            Composed transformer applying both transforms in order.
        """
        return ComposedTransformer[DataProviderInT, DataProviderFinalT](self, other)  # type: ignore[arg-type]

    def __hash__(self) -> int:
        """Return a stable hash derived from the transformer annotation.

        Returns
        -------
        hash
            Stable integer hash for the transformer instance.
        """
        return stable_hash(self.annotation)


class ComposedTransformer[DataProviderInT: DataProvider[Any, Any], DataProviderOutT: DataProvider[Any, Any]](
    IDataTransformer[DataProviderInT, DataProviderOutT]
):
    """
    Transformer composed of multiple transformers applied in sequence.

    Applies each transformer to the result of the previous one,
    chaining transformations together.

    Parameters
    ----------
    *transformers
        Variable-length list of transformers to apply in order.

    Raises
    ------
    ValueError
        If no transformers are provided.
    """

    def __init__(
        self,
        *transformers: IDataTransformer[DataProvider, DataProvider],
    ) -> None:
        if not transformers:
            raise ValueError("ComposedTransformer requires at least one transformer")
        self._transformers = list(transformers)

    def transform(self, provider: DataProviderInT) -> DataProviderOutT:
        """
        Apply all transformers in sequence to the input provider.

        Parameters
        ----------
        provider
            Input data provider to transform.

        Returns
        -------
        result
            Data provider after all transformations applied.
        """
        result: DataProviderOutT = provider  # type: ignore
        for transformer in self._transformers:
            result = transformer.transform(result)  # type: ignore
        return result

    def __and__[DataProviderFinalT: DataProvider[Any, Any]](
        self, other: "IDataTransformer[DataProviderOutT, DataProviderFinalT]"
    ) -> "ComposedTransformer[DataProviderInT, DataProviderFinalT]":
        """Append another transformer to the composition.

        Parameters
        ----------
        other
            Transformer applied after the current composition.

        Returns
        -------
        transformer
            Extended composed transformer.
        """
        return ComposedTransformer(*self._transformers, other)  # type: ignore[arg-type]

    @property
    def annotation(self) -> str:
        """
        Composed transformer annotation.

        Returns
        -------
        annotation
            Combined annotation from all composed transformers.
        """
        filtered = [t.annotation.strip() for t in self._transformers if t.annotation.strip()]
        return "->".join(filtered) if len(filtered) >= 2 else (filtered[0] if filtered else "")
