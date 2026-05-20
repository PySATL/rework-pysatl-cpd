# -*- coding: ascii -*-
"""
Tests for data provider contract applications.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pandas as pd
import pytest

from pysatl_cpd.data import NDArrayMultivariateProvider, NDArrayUnivariateProvider
from pysatl_cpd.data.providers.plain.pd_provider import PandasDataProvider
from pysatl_cpd.data.typedefs import TimeseriesAnnotation, UnlabeledTimeseriesAnnotation
from tests.contracts.data.test_data_provider_contract import DataProviderContract
from tests.support.providers import make_multivariate_provider, make_pandas_provider, make_univariate_provider


class TestNDArrayUnivariateProviderContract(DataProviderContract):
    @pytest.fixture
    def provider(self) -> NDArrayUnivariateProvider[UnlabeledTimeseriesAnnotation]:
        return make_univariate_provider(name="left")

    @pytest.fixture
    def other_provider(self) -> NDArrayUnivariateProvider[UnlabeledTimeseriesAnnotation]:
        return make_univariate_provider((7.0, 8.0, 9.0), name="right")


class TestNDArrayMultivariateProviderContract(DataProviderContract):
    @pytest.fixture
    def provider(self) -> NDArrayMultivariateProvider[TimeseriesAnnotation]:
        return make_multivariate_provider(name="left")

    @pytest.fixture
    def other_provider(self) -> NDArrayMultivariateProvider[TimeseriesAnnotation]:
        return make_multivariate_provider(((7.0, 70.0), (8.0, 80.0), (9.0, 90.0)), name="right")


class TestPandasDataProviderContract(DataProviderContract):
    @pytest.fixture
    def provider(self) -> PandasDataProvider[UnlabeledTimeseriesAnnotation]:
        return make_pandas_provider(name="left")

    @pytest.fixture
    def other_provider(self) -> PandasDataProvider[UnlabeledTimeseriesAnnotation]:
        return make_pandas_provider(pd.DataFrame({"x": [7.0, 8.0], "y": [70.0, 80.0]}), name="right")


def test_data_provider_merge_rejects_type_mismatch() -> None:
    with pytest.raises(TypeError, match="All providers"):
        NDArrayUnivariateProvider.merge([make_univariate_provider(), make_pandas_provider()])  # type: ignore[list-item]


def test_data_provider_or_rejects_type_mismatch() -> None:
    with pytest.raises(TypeError, match="Cannot merge"):
        _ = make_univariate_provider() | make_pandas_provider()  # type: ignore[operator]
