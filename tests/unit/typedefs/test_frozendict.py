# -*- coding: ascii -*-
"""
Tests for frozendict.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections import OrderedDict

import numpy as np
import pytest

from pysatl_cpd.typedefs import frozendict, make_hashable, stable_hash


class TestFrozendictInit:
    def test_init_with_kwargs(self):
        d = frozendict(a=1, b=2)
        assert len(d) == 2
        assert d["a"] == 1
        assert set(d) == {"a", "b"}

    def test_init_empty(self):
        d = frozendict()
        assert len(d) == 0


class TestFrozendictFromMapping:
    def test_from_dict(self):
        orig = {"x": 10, "y": 20}
        d = frozendict.from_mapping(orig)
        assert d == orig

    def test_from_mapping_copies_data(self):
        orig = {"a": 1}
        d = frozendict.from_mapping(orig)
        orig["a"] = 2
        assert d["a"] == 1

    def test_from_other_mapping(self):
        orig = OrderedDict([("b", 2), ("a", 1)])
        d = frozendict.from_mapping(orig)
        assert d == orig


class TestFrozendictImmutability:
    def test_setattr_raises(self):
        d = frozendict(a=1)
        with pytest.raises(TypeError, match="frozendict is immutable"):
            d.new_key = 2

    def test_setattr_existing_attribute_raises(self):
        d = frozendict(a=1)
        with pytest.raises(TypeError, match="frozendict is immutable"):
            d._hash = 0

    def test_delattr_raises(self):
        d = frozendict(a=1)
        with pytest.raises(TypeError, match="frozendict is immutable"):
            del d._data

    def test_underlying_data_immutable(self):
        d = frozendict(a=1)
        with pytest.raises(TypeError):
            d._data["a"] = 2


class TestFrozendictHash:
    def test_hash_consistency(self):
        d1 = frozendict(a=1, b=2)
        d2 = frozendict(a=1, b=2)
        assert hash(d1) == hash(d2)

    def test_hash_different_contents(self):
        d1 = frozendict(a=1)
        d2 = frozendict(a=2)
        assert hash(d1) != hash(d2)

    def test_usable_as_dict_key(self):
        d = frozendict(a=1)
        container = {d: "value"}
        assert container[d] == "value"


class TestHashHelpers:
    def test_make_hashable_normalizes_nested_containers(self):
        value = {
            "mapping": {"b": 2, "a": [1, 2]},
            "set": {3, 1},
        }

        assert make_hashable(value) == (
            ("mapping", (("a", (1, 2)), ("b", 2))),
            ("set", (1, 3)),
        )

    def test_stable_hash_matches_for_equivalent_mappings(self):
        left = {"a": 1, "b": [1, 2], "c": {3, 4}}
        right = {"c": {4, 3}, "b": (1, 2), "a": 1}

        assert stable_hash(left) == stable_hash(right)

    def test_stable_hash_differs_for_different_values(self):
        assert stable_hash({"a": 1}) != stable_hash({"a": 2})

    def test_make_hashable_normalizes_numpy_arrays(self):
        value = np.array([[1, 2], [3, 4]])

        assert make_hashable(value) == ((1, 2), (3, 4))

    def test_make_hashable_normalizes_frozensets(self):
        value = frozenset({("b", 2), ("a", 1)})

        assert make_hashable(value) == (("a", 1), ("b", 2))

    def test_frozendict_hash_matches_stable_hash(self):
        value = frozendict.from_mapping({"b": [1, 2], "a": {"x": 3}})

        assert hash(value) == stable_hash({"b": [1, 2], "a": {"x": 3}})


class TestFrozendictEquality:
    def test_eq_same_frozendict(self):
        d1 = frozendict(a=1, b=2)
        d2 = frozendict(a=1, b=2)
        assert d1 == d2

    def test_eq_dict(self):
        d = frozendict(a=1, b=2)
        assert d == {"a": 1, "b": 2}

    def test_eq_non_mapping(self):
        d = frozendict(a=1)
        assert d != "not a mapping"


class TestFrozendictRepr:
    def test_repr_format(self):
        d = frozendict(a=1, b=2)
        assert repr(d) == "frozendict({'a': 1, 'b': 2})"
