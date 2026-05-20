# -*- coding: ascii -*-
"""Synthetic data-generation API for change-point detection.

This package provides a declarative framework for generating synthetic time
series with known regime structure, change points, and state labels. It is
designed for benchmarking, testing, and studying the behavior of change-point
detection algorithms under controlled conditions.

Scenarios are defined as immutable specifications (``ScenarioSpec``) that
separate the ordered sequence of segment occurrences from reusable segment
plans. Each plan carries a distribution specification, a state descriptor,
and optional metadata. Distribution specs support univariate distributions
(normal, uniform, exponential, Student-t), multivariate normal with
correlated features, and independent per-column distributions.

Generated series can be converted into labeled-provider types used by the
data layer via the ``build_*`` helper functions, or assembled into full
datasets via ``ScenarioDatasetGenerator``. Scenarios can also be loaded
from YAML files or plain Python mappings for config-driven workflows.

.. raw:: html

    <h2>Public API</h2>

Distribution specifications
    - ``NormalSpec`` -- Normal (Gaussian) distribution parameters.
    - ``UniformSpec`` -- Uniform distribution over a fixed interval.
    - ``ExponentialSpec`` -- Exponential distribution with scale parameter.
    - ``StudentTSpec`` -- Student's t-distribution parameters.
    - ``UnivariateDistributionSpec`` -- Type alias for the four univariate
      specs above.
    - ``MultivariateNormalSpec`` -- Multivariate normal with named means
      and covariance structure.
    - ``IndependentColumnsSpec`` -- Per-column independent univariate
      distributions.
    - ``DistributionSpec`` -- Type alias covering all distribution specs.

Scenario specifications
    - ``SegmentSpec`` -- Ordered segment occurrence (plan name and length).
    - ``SegmentPlan`` -- Reusable plan defining distribution, state, and
      metadata for a segment type.
    - ``ScenarioSpec`` -- Top-level scenario blueprint combining segments,
      plans, and metadata.

Generators
    - ``GenericSeriesGenerator`` -- Core engine that samples series from
      ``ScenarioSpec`` objects or segment generator sequences.
    - ``SegmentGenerator`` -- Protocol for custom segment generators. See
      the ``segments`` subpackage for details.
    - ``LabeledDataGenerator`` -- Protocol for objects that produce labeled
      data instances.
    - ``ScenarioDatasetGenerator`` -- Builds ``Dataset`` collections from
      named scenario specifications.

Provider builders
    - ``build_pandas_labeled_data`` -- Convert a ``GeneratedSeries`` to a
      multivariate ``PandasLabeledData`` provider.
    - ``build_pandas_univariate_labeled_data`` -- Convert a single feature
      from a ``GeneratedSeries`` to a univariate ``PandasLabeledData``.
    - ``build_plain_multivariate_labeled_data`` -- Convert to a NumPy-backed
      ``PlainMultivariateLabeledData`` provider.
    - ``build_plain_univariate_labeled_data`` -- Convert a single feature
      to a NumPy-backed ``PlainUnivariateLabeledData`` provider.

Configuration loaders
    - ``scenario_from_mapping`` -- Build a ``ScenarioSpec`` from a plain
      Python mapping.
    - ``scenario_from_yaml`` -- Load a single scenario from a YAML file.
    - ``scenarios_from_yaml`` -- Load one or more scenarios from a YAML
      file, returning a name-to-spec dictionary.

Presets
    - ``PRESET_SCENARIOS`` -- Frozen mapping of built-in preset scenario
      specifications (currently empty; presets are constructed on demand).
    - ``preset_dataset`` -- One-call dataset generation from a named preset
      (e.g., ``"mean_shifts"``, ``"variance_shifts"``, ``"covariance_shifts"``).

.. raw:: html

    <h2>Subpackages</h2>

- ``segments`` -- Core segment-generation building blocks, sampling utilities,
  and the ``SegmentGenerator`` protocol. See its module docstring for details.
- ``providers`` -- Provider-builder functions that convert ``GeneratedSeries``
  into labeled-provider types. See its module docstring for details.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
Generate a univariate mean-shift series from a scenario specification::

    >>> from pysatl_cpd.data.generator import (
    ...     GenericSeriesGenerator,
    ...     NormalSpec,
    ...     ScenarioSpec,
    ...     SegmentPlan,
    ...     SegmentSpec,
    ... )
    >>> from pysatl_cpd.data.typedefs import StateDescriptor, frozendict
    >>> scenario = ScenarioSpec(
    ...     name="mean_shift",
    ...     segments=(
    ...         SegmentSpec(plan_name="baseline", length=100),
    ...         SegmentSpec(plan_name="shifted", length=60),
    ...         SegmentSpec(plan_name="baseline", length=40),
    ...     ),
    ...     plans=frozendict(
    ...         baseline=SegmentPlan(
    ...             distribution=NormalSpec(mean=0.0, std=1.0),
    ...             state=StateDescriptor(type="baseline"),
    ...         ),
    ...         shifted=SegmentPlan(
    ...             distribution=NormalSpec(mean=3.0, std=1.0),
    ...             state=StateDescriptor(type="shifted"),
    ...         ),
    ...     ),
    ... )
    >>> series = GenericSeriesGenerator(seed=42).generate_from_scenario(
    ...     scenario, name="example_series",
    ... )
    >>> series.data.shape
    (200, 1)
    >>> series.change_points
    (99, 159)

Generate a multivariate series with independent columns::

    >>> from pysatl_cpd.data.generator import (
    ...     GenericSeriesGenerator,
    ...     IndependentColumnsSpec,
    ...     NormalSpec,
    ...     ScenarioSpec,
    ...     SegmentPlan,
    ...     SegmentSpec,
    ... )
    >>> from pysatl_cpd.data.typedefs import StateDescriptor, frozendict
    >>> mv_scenario = ScenarioSpec(
    ...     name="independent_mv",
    ...     segments=(
    ...         SegmentSpec(plan_name="a", length=50),
    ...         SegmentSpec(plan_name="b", length=30),
    ...     ),
    ...     plans=frozendict(
    ...         a=SegmentPlan(
    ...             distribution=IndependentColumnsSpec(
    ...                 columns=frozendict(
    ...                     x=NormalSpec(mean=0.0, std=1.0),
    ...                     y=NormalSpec(mean=10.0, std=2.0),
    ...                 ),
    ...             ),
    ...             state=StateDescriptor(type="regime_a"),
    ...         ),
    ...         b=SegmentPlan(
    ...             distribution=IndependentColumnsSpec(
    ...                 columns=frozendict(
    ...                     x=NormalSpec(mean=5.0, std=1.0),
    ...                     y=NormalSpec(mean=15.0, std=2.0),
    ...                 ),
    ...             ),
    ...             state=StateDescriptor(type="regime_b"),
    ...         ),
    ...     ),
    ... )
    >>> mv_series = GenericSeriesGenerator(seed=0).generate_from_scenario(
    ...     mv_scenario,
    ... )
    >>> mv_series.feature_names
    ('x', 'y')
    >>> mv_series.data.shape
    (80, 2)

Build a labeled provider from a generated series::

    >>> from pysatl_cpd.data.generator import (
    ...     GenericSeriesGenerator,
    ...     NormalSpec,
    ...     ScenarioSpec,
    ...     SegmentPlan,
    ...     SegmentSpec,
    ...     build_pandas_labeled_data,
    ... )
    >>> from pysatl_cpd.data.typedefs import StateDescriptor, frozendict
    >>> scenario = ScenarioSpec(
    ...     name="provider_example",
    ...     segments=(
    ...         SegmentSpec(plan_name="a", length=50),
    ...         SegmentSpec(plan_name="b", length=30),
    ...     ),
    ...     plans=frozendict(
    ...         a=SegmentPlan(
    ...             distribution=NormalSpec(mean=0.0, std=1.0),
    ...             state=StateDescriptor(type="baseline"),
    ...         ),
    ...         b=SegmentPlan(
    ...             distribution=NormalSpec(mean=3.0, std=1.0),
    ...             state=StateDescriptor(type="shifted"),
    ...         ),
    ...     ),
    ... )
    >>> series = GenericSeriesGenerator(seed=0).generate_from_scenario(scenario)
    >>> provider = build_pandas_labeled_data(series, name="provider_example")
    >>> list(provider.feature_columns)
    ['value']
    >>> provider.change_points
    (50,)

Generate a dataset from multiple named scenarios::

    >>> from pysatl_cpd.data.generator import (
    ...     GenericSeriesGenerator,
    ...     NormalSpec,
    ...     ScenarioDatasetGenerator,
    ...     ScenarioSpec,
    ...     SegmentPlan,
    ...     SegmentSpec,
    ... )
    >>> from pysatl_cpd.data.typedefs import StateDescriptor, frozendict
    >>> s1 = ScenarioSpec(
    ...     name="small_shift",
    ...     segments=(SegmentSpec(plan_name="a", length=50), SegmentSpec(plan_name="b", length=30)),
    ...     plans=frozendict(
    ...         a=SegmentPlan(distribution=NormalSpec(mean=0.0, std=1.0), state=StateDescriptor(type="a")),
    ...         b=SegmentPlan(distribution=NormalSpec(mean=1.0, std=1.0), state=StateDescriptor(type="b")),
    ...     ),
    ... )
    >>> s2 = ScenarioSpec(
    ...     name="large_shift",
    ...     segments=(SegmentSpec(plan_name="a", length=50), SegmentSpec(plan_name="b", length=30)),
    ...     plans=frozendict(
    ...         a=SegmentPlan(distribution=NormalSpec(mean=0.0, std=1.0), state=StateDescriptor(type="a")),
    ...         b=SegmentPlan(distribution=NormalSpec(mean=5.0, std=1.0), state=StateDescriptor(type="b")),
    ...     ),
    ... )
    >>> gen = ScenarioDatasetGenerator({"small": s1, "large": s2}, seed=0)
    >>> dataset = gen.generate("small", size=2)
    >>> len(dataset)
    2

Load a scenario from a Python mapping::

    >>> from pysatl_cpd.data.generator import scenario_from_mapping
    >>> mapping = {
    ...     "name": "from_mapping",
    ...     "segments": [
    ...         {"plan_name": "a", "length": 40},
    ...         {"plan_name": "b", "length": 20},
    ...     ],
    ...     "plans": {
    ...         "a": {"distribution": {"kind": "normal", "mean": 0.0, "std": 1.0}},
    ...         "b": {"distribution": {"kind": "normal", "mean": 2.0, "std": 1.0}},
    ...     },
    ... }
    >>> spec = scenario_from_mapping(mapping)
    >>> spec.name
    'from_mapping'

Generate a dataset from a built-in preset::

    >>> from pysatl_cpd.data.generator import preset_dataset
    >>> ds = preset_dataset("mean_shifts", n_series=2, seed=0, series_length=120)
    >>> len(ds)
    2

.. raw:: html

    <h2>Notes</h2>

Notes
-----
- All change-point indices are zero-based throughout the package.
- Univariate distribution specs (``NormalSpec``, ``UniformSpec``,
  ``ExponentialSpec``, ``StudentTSpec``) produce single-column arrays with
  the default feature name ``"value"``.
- ``MultivariateNormalSpec`` requires non-empty ``means`` with feature names
  as keys. The covariance can be a scalar, a 1-D sequence (diagonal), or a
  nested sequence (full matrix).
- ``IndependentColumnsSpec`` requires each column to reference a univariate
  distribution spec.
- ``GenericSeriesGenerator`` uses NumPy's ``default_rng`` internally. Pass
  ``seed`` for reproducible results.
- The ``preset_dataset`` function supports presets such as ``"mean_shifts"``,
  ``"variance_shifts"``, ``"covariance_shifts"``, ``"no_shifts"``,
  ``"extreme_mean_shifts"``, ``"3d_mean_shifts"``, and ``"mixed_shifts"``.
- YAML loading requires the ``PyYAML`` dependency (included in the project's
  dev installation).
- The ``segments`` and ``providers`` subpackages contain additional utilities
  and protocols; consult their module docstrings for segment-level sampling
  and provider-conversion details.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .config import scenario_from_mapping, scenario_from_yaml, scenarios_from_yaml
from .dataset import LabeledDataGenerator, ScenarioDatasetGenerator
from .presets import PRESET_SCENARIOS, preset_dataset
from .providers import (
    build_pandas_labeled_data,
    build_pandas_univariate_labeled_data,
    build_plain_multivariate_labeled_data,
    build_plain_univariate_labeled_data,
)
from .segments import SegmentGenerator
from .series import GenericSeriesGenerator
from .specs import (
    DistributionSpec,
    ExponentialSpec,
    IndependentColumnsSpec,
    MultivariateNormalSpec,
    NormalSpec,
    ScenarioSpec,
    SegmentPlan,
    SegmentSpec,
    StudentTSpec,
    UniformSpec,
    UnivariateDistributionSpec,
)

__all__ = [
    "DistributionSpec",
    "ExponentialSpec",
    "GenericSeriesGenerator",
    "IndependentColumnsSpec",
    "LabeledDataGenerator",
    "MultivariateNormalSpec",
    "NormalSpec",
    "PRESET_SCENARIOS",
    "ScenarioDatasetGenerator",
    "ScenarioSpec",
    "SegmentGenerator",
    "SegmentPlan",
    "SegmentSpec",
    "StudentTSpec",
    "UniformSpec",
    "UnivariateDistributionSpec",
    "build_pandas_labeled_data",
    "build_pandas_univariate_labeled_data",
    "build_plain_multivariate_labeled_data",
    "build_plain_univariate_labeled_data",
    "preset_dataset",
    "scenario_from_mapping",
    "scenario_from_yaml",
    "scenarios_from_yaml",
]
