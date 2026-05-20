# -*- coding: ascii -*-
"""
Tests for stable hash.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pathlib import Path

import numpy as np

from pysatl_cpd.data.typedefs import StateDescriptor, TimeseriesAnnotation
from pysatl_cpd.typedefs import frozendict, stable_hash
from tests.support.golden import load_json_golden


def test_stable_hash_matches_golden_values() -> None:
    golden = load_json_golden(Path(__file__).resolve().parents[1] / "golden" / "stable_hash.json")
    actual = {
        "int_42": stable_hash(42),
        "tuple_numbers": stable_hash((1, 2, 3)),
        "mapping_sorted": stable_hash({"b": 2, "a": 1}),
        "numpy_array": stable_hash(np.array([1.0, 2.0, 3.0])),
        "frozendict": stable_hash(frozendict(alpha=1, beta=2)),
        "annotation": stable_hash(
            (
                TimeseriesAnnotation(name="series").__class__.__module__,
                TimeseriesAnnotation(name="series").__class__.__qualname__,
                "series",
                None,
                frozendict(),
            )
        ),
        "state": stable_hash(StateDescriptor(label="baseline").data),
    }
    assert actual == golden
