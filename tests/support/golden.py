# -*- coding: ascii -*-
"""
Tests for golden.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import json
from pathlib import Path
from typing import Any


def load_json_golden(path: Path) -> Any:
    """Load a JSON golden file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
