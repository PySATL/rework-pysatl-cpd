# -*- coding: ascii -*-
"""Frozen dictionary and shared hashing helpers for PySATL CPD."""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import hashlib
from collections.abc import Hashable, Iterator, Mapping, Sequence, Set
from types import MappingProxyType
from typing import Any, TypeVar

import numpy as np

K = TypeVar("K", bound=Hashable)
V_co = TypeVar("V_co", bound=Hashable, covariant=True)


def make_hashable(value: Any) -> Hashable:
    """Recursively normalize common mutable containers into hashable values."""
    if isinstance(value, np.ndarray):
        return make_hashable(value.tolist())
    if isinstance(value, Mapping):
        return tuple(sorted((key, make_hashable(item)) for key, item in value.items()))
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(make_hashable(item) for item in value)
    if isinstance(value, Set) and not isinstance(value, frozenset):
        return tuple(map(make_hashable, sorted(item for item in value)))
    if isinstance(value, frozenset):
        return tuple(map(make_hashable, sorted(item for item in value)))
    if isinstance(value, Hashable):
        return value
    raise TypeError(f"{type(value).__name__} is not hashable")


def stable_hash(value: object) -> int:
    """Return a deterministic Python-hash value for a normalized object."""
    normalized = make_hashable(value)
    digest = hashlib.sha256(repr(normalized).encode("utf-8")).digest()
    return hash(int.from_bytes(digest[:8], byteorder="big", signed=False))


class frozendict[K, V_co](Mapping[K, V_co]):
    """Immutable dictionary with stable hashing.

    Parameters
    ----------
    **kwargs
        Key-value pairs to populate the dictionary.
    """

    __slots__ = ("_data", "_hash")

    _data: Mapping[K, V_co]
    _hash: int

    def __init__(self: "frozendict[str, V_co]", **kwargs: V_co) -> None:
        copied: dict[str, V_co] = dict(kwargs)

        object.__setattr__(self, "_data", MappingProxyType(copied))
        object.__setattr__(self, "_hash", stable_hash(copied))

    @classmethod
    def from_mapping(cls, data: Mapping[K, V_co]) -> "frozendict[K, V_co]":
        """Create a frozen dictionary from an arbitrary mapping.

        Parameters
        ----------
        data
            Source mapping to copy.

        Returns
        -------
        frozendict
            A new frozen dictionary with the same items.
        """
        obj = cls.__new__(cls)

        copied: dict[K, V_co] = dict(data)

        object.__setattr__(obj, "_data", MappingProxyType(copied))
        object.__setattr__(obj, "_hash", stable_hash(copied))

        return obj

    def __getitem__(self, key: K) -> V_co:
        """Return the value for *key*.

        Parameters
        ----------
        key
            The key to look up.

        Returns
        -------
        value
            The value associated with the key.
        """
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        """Iterate over dictionary keys."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of items."""
        return len(self._data)

    def __hash__(self) -> int:
        """Return the pre-computed hash."""
        return self._hash

    def __repr__(self) -> str:
        """Return a reproducible string representation."""
        return f"{type(self).__name__}({dict(self._data)!r})"

    def __eq__(self, other: object) -> bool:
        """Compare with another mapping for item-wise equality."""
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return NotImplemented

    def __setattr__(self, name: str, value: object) -> None:
        """Block attribute mutation; instances are immutable."""
        raise TypeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        """Block attribute deletion; instances are immutable."""
        raise TypeError(f"{type(self).__name__} is immutable")

    def __getstate__(self) -> dict:
        """Return pickling state as a plain dict."""
        return {"_data": dict(self._data)}

    def __setstate__(self, state: dict) -> None:
        """Restore pickling state."""
        data = state["_data"]
        object.__setattr__(self, "_data", MappingProxyType(data))
        object.__setattr__(self, "_hash", stable_hash(data))


__all__ = [
    "K",
    "V_co",
    "frozendict",
    "make_hashable",
    "stable_hash",
]
