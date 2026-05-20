# -*- coding: ascii -*-
"""
Shared type definitions and hashing utilities for PySATL CPD.

This package provides low-level, zero-dependency type aliases and utility
classes used across the entire project. It sits at the bottom of the
dependency stack -- no submodule in ``pysatl_cpd`` should import from
``core``, ``data``, or ``analysis`` when this module will do.

The package is organized into two submodules:

- ``base`` -- defines numeric type aliases and NumPy array type aliases
  using Python 3.12 ``type`` statements (PEP 695).
- ``frozendict`` -- provides an immutable, hashable ``Mapping`` backed by
  ``MappingProxyType``, along with helpers for converting mutable containers
  into hashable equivalents and computing deterministic hashes.

.. raw:: html

    <h2>Public API</h2>

Numeric type aliases (from ``base``):

- ``NumPyNumber`` -- canonical NumPy scalar type (``np.float64``).
- ``Number`` -- union of ``NumPyNumber | int | float`` for scalar numerics.

Array type aliases (from ``base``):

- ``UnivariateNumericArray`` -- 1-D NumPy array of numerics.
- ``MultivariateNumericArray`` -- 2-D NumPy array of numerics.
- ``NumericArray`` -- generic N-D NumPy array of numerics.
- ``BoolArray`` -- 1-D NumPy boolean array.

Immutable mapping and hashing (from ``frozendict``):

- ``frozendict[K, V_co]`` -- immutable, hashable ``Mapping`` backed by
  ``MappingProxyType``. Supports pickling and provides a stable ``__hash__``
  via SHA-256 truncation.
- ``make_hashable(value)`` -- recursively converts mutable containers
  (lists, dicts, sets, NumPy arrays) into hashable equivalents.
- ``stable_hash(value)`` -- returns a deterministic ``int`` hash for any
  object by normalizing it through ``make_hashable`` and hashing the
  SHA-256 digest.

Type variables (from ``frozendict``):

- ``K`` -- ``TypeVar`` bound to ``Hashable`` for dictionary keys.
- ``V_co`` -- covariant ``TypeVar`` bound to ``Hashable`` for dictionary values.

Examples
--------
Type aliases for annotations:

>>> import numpy as np
>>> from pysatl_cpd.typedefs import Number, UnivariateNumericArray
>>> x: Number = 3.14
>>> arr: UnivariateNumericArray = np.array([1.0, 2.0, 3.0])

Immutable configuration dictionaries:

>>> from pysatl_cpd.typedefs import frozendict
>>> cfg = frozendict(alpha=1, beta=2)
>>> cfg["alpha"]
1
>>> hash(cfg)
1844981020440554400

Hashing utilities:

>>> from pysatl_cpd.typedefs import make_hashable, stable_hash
>>> make_hashable({"a": [1, 2]})
(('a', (1, 2)),)
>>> stable_hash({"key": "value"})
283070982107879554

Notes
-----
- ``frozendict`` is intentionally minimal -- it implements only the
  ``Mapping`` protocol plus immutability guards.
- Hashing is stable across processes because ``stable_hash`` uses
  ``hashlib.sha256`` rather than Python's randomized ``hash()``.
- All array type aliases require NumPy and use Python 3.12 ``type``
  statements (PEP 695).
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.typedefs.base import (
    BoolArray,
    MultivariateNumericArray,
    Number,
    NumericArray,
    NumPyNumber,
    UnivariateNumericArray,
)
from pysatl_cpd.typedefs.frozendict import (
    K,
    V_co,
    frozendict,
    make_hashable,
    stable_hash,
)

__all__ = [
    "BoolArray",
    "K",
    "MultivariateNumericArray",
    "Number",
    "NumericArray",
    "NumPyNumber",
    "UnivariateNumericArray",
    "V_co",
    "frozendict",
    "make_hashable",
    "stable_hash",
]
