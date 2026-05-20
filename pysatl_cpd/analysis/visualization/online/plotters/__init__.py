# -*- coding: ascii -*-
"""Online CPD plotter exports.

Coordinates multiple visualizers and layout strategies into a single,
high-level API for rendering online change-point detection results.

.. raw:: html

    <h2>Public API</h2>

Classes
-------
OnlineCpdPlotter
    Coordinator that bundles a timeseries visualizer, an online trace
    visualizer, and reusable annotation components into one draw call.
    Supports both Matplotlib and Plotly backends.
ILayoutStrategy
    Abstract interface for subplot arrangement. Concrete implementations
    are documented in the ``online_cpd_plotter_layouts`` module.
VerticalLayout
    Stacks all panels in a single column.
SplitLayout
    Places the time series in the left column and trace panels in the
    right column.
DashboardLiteLayout
    Compact two-row layout with uneven column widths.
DashboardLayout
    Full 2x2 grid with time series, state, detection function, and
    processing time panels.
CustomLayout
    Caller-supplied figure factory wrapped behind the layout interface.

.. raw:: html

    <h2>Submodules</h2>

online_cpd_plotter_layouts
    Defines ``ILayoutStrategy`` and all built-in layout implementations.
    See the module docstring for details on each strategy.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
Basic usage with a univariate provider and detection trace::

    >>> from pysatl_cpd.analysis.visualization import DrawBackend, OnlineCpdPlotter
    >>> from pysatl_cpd.core.online import OnlineResetDetector
    >>> from pysatl_cpd.data.generator import (
    ...     GenericSeriesGenerator,
    ...     ScenarioSpec,
    ...     SegmentPlan,
    ...     SegmentSpec,
    ...     UnivariateDistributionSpec,
    ...     build_plain_univariate_labeled_data,
    ... )
    >>> from pysatl_cpd.data.typedefs import StateDescriptor, frozendict
    >>> from plotly.subplots import make_subplots
    >>> scenario = ScenarioSpec(
    ...     name="demo",
    ...     segments=(
    ...         SegmentSpec(plan_name="a", length=100),
    ...         SegmentSpec(plan_name="b", length=80),
    ...     ),
    ...     plans=frozendict(
    ...         a=SegmentPlan(
    ...             distribution=UnivariateDistributionSpec("Normal", "meanStd", mu=0.0, sigma=1.0),
    ...             state=StateDescriptor(type="a"),
    ...             name="a",
    ...         ),
    ...         b=SegmentPlan(
    ...             distribution=UnivariateDistributionSpec("Normal", "meanStd", mu=3.0, sigma=1.0),
    ...             state=StateDescriptor(type="b"),
    ...             name="b",
    ...         ),
    ...     ),
    ... )
    >>> series = GenericSeriesGenerator(seed=42).generate_from_scenario(scenario, name="demo")
    >>> provider = build_plain_univariate_labeled_data(series, feature_name="value", name="demo")
    >>> from pysatl_cpd.algorithms.online import ShewhartControlChart
    >>> detector = OnlineResetDetector(
    ...     ShewhartControlChart(learning_period_size=30, window_size=10),
    ...     threshold=2.0,
    ...     skip_period=8,
    ... )
    >>> trace = detector.detect(provider)
    >>> plotter = OnlineCpdPlotter(
    ...     backend=DrawBackend.PLOTLY,
    ...     data_provider=provider,
    ...     detection_trace=trace,
    ... )
    >>> plotter.set_ground_truth(list(provider.change_points), margin=10)
    OnlineCpdPlotter(...)
    >>> fig = make_subplots(
    ...     rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
    ...     subplot_titles=("Time Series", "Detection Function", "Processing Time"),
    ... )
    >>> fig = plotter.draw(
    ...     figure=fig,
    ...     axes={"timeseries": (1, 1), "detection_function": (2, 1), "processing_time": (3, 1)},
    ... )

Using a named layout string instead of manual figure construction::

    >>> from pysatl_cpd.analysis.visualization import DrawBackend, OnlineCpdPlotter
    >>> plotter = OnlineCpdPlotter(
    ...     backend=DrawBackend.MATPLOTLIB,
    ...     data_provider=provider,
    ...     detection_trace=trace,
    ...     layout="vertical",
    ... )
    >>> fig, ax_mapping = plotter.default_layout()
    >>> fig = plotter.draw(figure=fig, axes=ax_mapping)

Swapping the layout strategy after construction::

    >>> plotter.set_layout("split")
    OnlineCpdPlotter(...)
    >>> plotter.set_layout("dashboard")
    OnlineCpdPlotter(...)

Adding and removing annotation components::

    >>> from pysatl_cpd.analysis.visualization.components import VerticalLineVisualComponent
    >>> custom_lines = VerticalLineVisualComponent(DrawBackend.PLOTLY).set_lines([50, 120]).set_style(
    ...     color="purple", linestyle="dash", linewidth=2, alpha=0.8, label="Custom", legend=True
    ... )
    >>> plotter.add_component("custom_lines", custom_lines, axes_names=["timeseries"])
    OnlineCpdPlotter(...)
    >>> plotter.remove_component("custom_lines")
    OnlineCpdPlotter(...)

Using a custom layout with a caller-supplied figure factory::

    >>> from pysatl_cpd.analysis.visualization.online.plotters import CustomLayout
    >>> def my_layout():
    ...     fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    ...     return fig, {"timeseries": (1, 1), "detection_function": (2, 1)}
    ...
    >>> layout = CustomLayout(create_figure_func=my_layout, required_axes={"timeseries", "detection_function"})
    >>> plotter.set_layout(layout)
    OnlineCpdPlotter(...)

Notes
-----
This package depends on ``matplotlib`` and ``plotly``. Both must be
installed for the full API to function.

The ``OnlineCpdPlotter`` creates default visualizers and components in
its constructor. Callers can replace the timeseries visualizer via
``set_timeseries_visualizer(...)`` or toggle individual components with
``add_component(...)`` and ``remove_component(...)``.

Layout strings ``"vertical"``, ``"split"``, ``"dashboard-lite"``, and
``"dashboard"`` map to the corresponding layout classes. Any unknown
string raises ``ValueError``.

Change-point indices are zero-based throughout. Ground-truth margins
passed to ``set_ground_truth(...)`` must be non-negative.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.visualization.online.plotters.online_cpd_plotter import OnlineCpdPlotter
from pysatl_cpd.analysis.visualization.online.plotters.online_cpd_plotter_layouts import (
    CustomLayout,
    DashboardLayout,
    DashboardLiteLayout,
    ILayoutStrategy,
    SplitLayout,
    VerticalLayout,
)

__all__ = [
    "CustomLayout",
    "DashboardLayout",
    "DashboardLiteLayout",
    "ILayoutStrategy",
    "OnlineCpdPlotter",
    "SplitLayout",
    "VerticalLayout",
]
