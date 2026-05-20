# -*- coding: ascii -*-
"""
Time series visualizers for change-point detection.

This module provides visualizers for rendering univariate and multivariate
time series data with change point annotations, period fills, and ground truth markers.
All visualizers support both Matplotlib and Plotly backends through the ``DrawBackend``
enum and share a common draw pattern: configure a data provider, optionally set style
options, then call ``draw(figure, axes)`` with a backend-appropriate axes mapping.

.. raw:: html

    <h2>Public API</h2>

UnivariateTimeseriesVisualizer
    Visualizer for a single time series. Renders the observations as a line plot
    and accepts optional ``PlotSpec`` and ``LineSpec`` styling. See the class
    docstring for configuration details.

PlainMultivariateTimeseriesVisualizer
    Dimension-oriented visualizer that creates one subplot per data dimension.
    Axis selection supports both integer indices and named columns when the
    provider exposes them. See the class docstring for configuration details.

MultivariateTimeseriesVisualizer
    Compatibility alias for ``PlainMultivariateTimeseriesVisualizer``.

RichMultivariateTimeseriesVisualizer
    Pandas-first visualizer that organizes data by logical plot names rather
    than by dimension. Supports binding named provider columns to series,
    optional custom time columns, and twin y-axis placement. See the class
    docstring for configuration details.

PlotSpec
    ``TypedDict`` defining subplot-level visual metadata (title, axis labels, grid).

LineSpec
    ``TypedDict`` defining line-level visual style (color, linewidth, alpha, label).

.. raw:: html

    <h2>Examples</h2>

Examples
--------
Univariate visualization with Matplotlib::

    >>> import matplotlib.pyplot as plt
    >>> from pysatl_cpd.analysis.visualization import DrawBackend, UnivariateTimeseriesVisualizer
    >>> from pysatl_cpd.data.generator import (
    ...     GenericSeriesGenerator,
    ...     NormalSpec,
    ...     ScenarioSpec,
    ...     SegmentPlan,
    ...     SegmentSpec,
    ...     build_plain_univariate_labeled_data,
    ... )
    >>> from pysatl_cpd.data.typedefs import StateDescriptor, frozendict
    >>> scenario = ScenarioSpec(
    ...     name="demo",
    ...     segments=(SegmentSpec(plan_name="a", length=50), SegmentSpec(plan_name="b", length=50)),
    ...     plans=frozendict(
    ...         a=SegmentPlan(distribution=NormalSpec(mean=0.0, std=1.0), state=StateDescriptor(type="a"), name="a"),
    ...         b=SegmentPlan(distribution=NormalSpec(mean=3.0, std=1.0), state=StateDescriptor(type="b"), name="b"),
    ...     ),
    ... )
    >>> series = GenericSeriesGenerator(seed=42).generate_from_scenario(scenario, name="demo_series")
    >>> provider = build_plain_univariate_labeled_data(series, feature_name="value", name="demo_provider")
    >>> visualizer = UnivariateTimeseriesVisualizer(backend=DrawBackend.MATPLOTLIB)
    >>> visualizer.set_data_provider(provider)
    >>> fig, ax = plt.subplots()
    >>> visualizer.draw(figure=fig, axes={"timeseries": ax})

Plain multivariate visualization with Plotly::

    >>> from plotly.subplots import make_subplots
    >>> from pysatl_cpd.analysis.visualization import DrawBackend, PlainMultivariateTimeseriesVisualizer
    >>> from pysatl_cpd.data.generator import (
    ...     GenericSeriesGenerator,
    ...     IndependentColumnsSpec,
    ...     NormalSpec,
    ...     ScenarioSpec,
    ...     SegmentPlan,
    ...     SegmentSpec,
    ...     build_plain_multivariate_labeled_data,
    ... )
    >>> from pysatl_cpd.data.typedefs import StateDescriptor, frozendict
    >>> scenario = ScenarioSpec(
    ...     name="mv_demo",
    ...     segments=(SegmentSpec(plan_name="base", length=60), SegmentSpec(plan_name="shift", length=40)),
    ...     plans=frozendict(
    ...         base=SegmentPlan(
    ...             distribution=IndependentColumnsSpec(
    ...                 columns=frozendict(x=NormalSpec(mean=0.0, std=1.0), y=NormalSpec(mean=2.0, std=0.5))
    ...             ),
    ...             state=StateDescriptor(type="base"),
    ...             name="base",
    ...         ),
    ...         shift=SegmentPlan(
    ...             distribution=IndependentColumnsSpec(
    ...                 columns=frozendict(x=NormalSpec(mean=3.0, std=1.0), y=NormalSpec(mean=0.5, std=0.5))
    ...             ),
    ...             state=StateDescriptor(type="shift"),
    ...             name="shift",
    ...         ),
    ...     ),
    ... )
    >>> series = GenericSeriesGenerator(seed=10).generate_from_scenario(scenario, name="mv_series")
    >>> provider = build_plain_multivariate_labeled_data(series, name="mv_provider")
    >>> visualizer = PlainMultivariateTimeseriesVisualizer(backend=DrawBackend.PLOTLY, dimensionality=2)
    >>> visualizer.set_data_provider(provider)
    >>> fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    >>> visualizer.draw(figure=fig, axes={"timeseries_0": (1, 1), "timeseries_1": (2, 1)})

Rich multivariate visualization with logical plots::

    >>> import matplotlib.pyplot as plt
    >>> from pysatl_cpd.analysis.visualization import DrawBackend, RichMultivariateTimeseriesVisualizer
    >>> from pysatl_cpd.data.generator import (
    ...     GenericSeriesGenerator,
    ...     IndependentColumnsSpec,
    ...     NormalSpec,
    ...     ScenarioSpec,
    ...     SegmentPlan,
    ...     SegmentSpec,
    ...     build_pandas_labeled_data,
    ... )
    >>> from pysatl_cpd.data.typedefs import StateDescriptor, frozendict
    >>> scenario = ScenarioSpec(
    ...     name="rich_demo",
    ...     segments=(SegmentSpec(plan_name="base", length=60), SegmentSpec(plan_name="shift", length=40)),
    ...     plans=frozendict(
    ...         base=SegmentPlan(
    ...             distribution=IndependentColumnsSpec(
    ...                 columns=frozendict(sensor=NormalSpec(mean=0.0, std=1.0), load=NormalSpec(mean=20.0, std=1.0))
    ...             ),
    ...             state=StateDescriptor(type="base"),
    ...             name="base",
    ...         ),
    ...         shift=SegmentPlan(
    ...             distribution=IndependentColumnsSpec(
    ...                 columns=frozendict(sensor=NormalSpec(mean=3.0, std=1.0), load=NormalSpec(mean=30.0, std=1.0))
    ...             ),
    ...             state=StateDescriptor(type="shift"),
    ...             name="shift",
    ...         ),
    ...     ),
    ... )
    >>> series = GenericSeriesGenerator(seed=7).generate_from_scenario(scenario, name="rich_series")
    >>> provider = build_pandas_labeled_data(series, name="rich_provider")
    >>> visualizer = RichMultivariateTimeseriesVisualizer(backend=DrawBackend.MATPLOTLIB)
    >>> visualizer.set_data_provider(provider)
    >>> visualizer.define_plot("signals", title="Sensors", xlabel="Index", ylabel="Value", grid=True)
    >>> visualizer.define_plot("load", title="Load", xlabel="Index", ylabel="Load", grid=True)
    >>> visualizer.add_series("signals", "sensor", column="sensor", color="tab:blue")
    >>> visualizer.add_series("load", "load", column="load", color="tab:green")
    >>> fig, axes = plt.subplots(2, 1, sharex=True)
    >>> visualizer.draw(figure=fig, axes={"signals": axes[0], "load": axes[1]})

.. raw:: html

    <h2>Notes</h2>

Notes
-----
- All visualizers require ``matplotlib`` and ``plotly`` as dependencies.
- ``PlainMultivariateTimeseriesVisualizer`` requires an explicit ``dimensionality``
  argument at construction and validates that the provider data matches.
- ``RichMultivariateTimeseriesVisualizer`` requires a pandas-backed data provider
  (one that exposes a ``.dataset`` attribute returning a ``pd.DataFrame``).
- The ``draw`` method expects an axes mapping whose keys match the visualizer's
  ``axes`` property. Matplotlib mappings use ``Axes`` objects; Plotly mappings
  use ``(row, col)`` tuples.
- ``MultivariateTimeseriesVisualizer`` is a compatibility alias and should not be
  used in new code; prefer ``PlainMultivariateTimeseriesVisualizer`` directly.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.visualization.specs import LineSpec, PlotSpec
from pysatl_cpd.analysis.visualization.timeseries.multivariate_timeseries_visualizer import (
    MultivariateTimeseriesVisualizer,
)
from pysatl_cpd.analysis.visualization.timeseries.plain_multivariate_timeseries_visualizer import (
    PlainMultivariateTimeseriesVisualizer,
)
from pysatl_cpd.analysis.visualization.timeseries.rich_multivariate_timeseries_visualizer import (
    RichMultivariateTimeseriesVisualizer,
)
from pysatl_cpd.analysis.visualization.timeseries.univariate_timeseries_visualizer import (
    UnivariateTimeseriesVisualizer,
)

__all__ = [
    "MultivariateTimeseriesVisualizer",
    "PlainMultivariateTimeseriesVisualizer",
    "RichMultivariateTimeseriesVisualizer",
    "UnivariateTimeseriesVisualizer",
    "PlotSpec",
    "LineSpec",
]
