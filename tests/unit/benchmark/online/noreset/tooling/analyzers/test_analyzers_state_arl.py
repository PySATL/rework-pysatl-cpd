# -*- coding: ascii -*-
"""
Tests for analyzers state arl.
"""

from __future__ import annotations

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from types import SimpleNamespace

import pytest

from pysatl_cpd.benchmark.online.noreset.tooling.analyzers.state_arl import NoResetArlAnalyzer
from pysatl_cpd.core.change_point_detector import ChangePointDetectorDescription
from pysatl_cpd.data.typedefs import ProviderType


class _ProviderWithLen(SimpleNamespace):
    """SimpleNamespace subclass that supports len()."""

    def __init__(self, *, length: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._length = length

    def __len__(self) -> int:
        return self._length


class TestNoResetArlAnalyzer:
    def test_analyze_raises_for_non_positive_arl_length(self) -> None:
        analyzer = NoResetArlAnalyzer()
        with pytest.raises(ValueError, match="arl_length must be positive"):
            analyzer.analyze(
                ChangePointDetectorDescription(name="detector"),
                SimpleNamespace(),
                thresholds=[],
                arl_length=0,
            )

    def test_analyze_raises_for_negative_arl_length(self) -> None:
        analyzer = NoResetArlAnalyzer()
        with pytest.raises(ValueError, match="arl_length must be positive"):
            analyzer.analyze(
                ChangePointDetectorDescription(name="detector"),
                SimpleNamespace(),
                thresholds=[],
                arl_length=-1,
            )

    @staticmethod
    def _make_arl_run(
        *,
        detector_name: str = "detector",
        provider_type: ProviderType = ProviderType.NO_CHANGE,
        state: object = None,
        length: int = 100,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            trace=SimpleNamespace(detector_description=ChangePointDetectorDescription(name=detector_name)),
            provider=_ProviderWithLen(
                length=length,
                annotation=SimpleNamespace(
                    provider_type=provider_type,
                    state=state,
                ),
            ),
        )

    def test_validate_arl_runs_raises_for_empty_runs(self) -> None:
        with pytest.raises(ValueError, match="No no-change .ARL. runs found"):
            NoResetArlAnalyzer.validate_arl_runs(
                [],
                ChangePointDetectorDescription(name="detector"),
                SimpleNamespace(),
                arl_length=100,
            )

    def test_validate_arl_runs_raises_for_detector_mismatch(self) -> None:
        state = SimpleNamespace(type="baseline")
        runs = [self._make_arl_run(detector_name="other", state=state)]
        with pytest.raises(ValueError, match="does not match benchmark entry"):
            NoResetArlAnalyzer.validate_arl_runs(
                runs,
                ChangePointDetectorDescription(name="expected"),
                state,
                arl_length=100,
            )

    def test_validate_arl_runs_raises_for_wrong_provider_type(self) -> None:
        state = SimpleNamespace(type="baseline")
        runs = [
            self._make_arl_run(
                provider_type=ProviderType.BISEGMENT,
                state=state,
            ),
        ]
        with pytest.raises(ValueError, match="requires no-change providers"):
            NoResetArlAnalyzer.validate_arl_runs(
                runs,
                ChangePointDetectorDescription(name="detector"),
                state,
                arl_length=100,
            )

    def test_validate_arl_runs_raises_for_wrong_state(self) -> None:
        state = SimpleNamespace(type="baseline")
        runs = [
            self._make_arl_run(
                state=SimpleNamespace(type="other"),
            ),
        ]
        with pytest.raises(ValueError, match="does not match requested state"):
            NoResetArlAnalyzer.validate_arl_runs(
                runs,
                ChangePointDetectorDescription(name="detector"),
                state,
                arl_length=100,
            )

    def test_validate_arl_runs_raises_for_wrong_length(self) -> None:
        state = SimpleNamespace(type="baseline")
        runs = [self._make_arl_run(state=state, length=50)]
        with pytest.raises(ValueError, match="does not match requested length"):
            NoResetArlAnalyzer.validate_arl_runs(
                runs,
                ChangePointDetectorDescription(name="detector"),
                state,
                arl_length=100,
            )

    def test_validate_arl_runs_passes_for_valid_run(self) -> None:
        state = SimpleNamespace(type="baseline")
        runs = [self._make_arl_run(state=state, length=100)]
        NoResetArlAnalyzer.validate_arl_runs(
            runs,
            ChangePointDetectorDescription(name="detector"),
            state,
            arl_length=100,
        )
