# -*- coding: ascii -*-
"""
Tests for frozendict extra.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.typedefs import make_hashable


def test_make_hashable_rejects_non_hashable_non_container_values() -> None:
    class _NotHashable:
        __hash__ = None

    with pytest.raises(TypeError, match="_NotHashable is not hashable"):
        make_hashable(_NotHashable())
