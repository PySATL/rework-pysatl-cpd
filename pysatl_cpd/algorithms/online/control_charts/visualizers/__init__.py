# -*- coding: ascii -*-
"""Visualizers for control chart algorithm states.

This subpackage provides visualizers that render the evolution of control
chart algorithm internal state over time. Each visualizer is backend-aware
and can produce plots through either Matplotlib or Plotly.

.. raw:: html

    <h2>Public API</h2>

- ``ShewhartStateVisualizer``: renders Shewhart control chart state evolution,
  including running mean, control limits, and sliding window mean.
- ``ShewhartStatePlotOpts``: TypedDict for general subplot options (labels,
  grid, title).
- ``ShewhartStateDrawOpts``: TypedDict for line styling options (colors,
  linewidths, linestyles, legend labels, fill opacity).
- ``ShewhartStateBandOpts``: TypedDict for control-limit band calculation
  options (multiplier k).

.. raw:: html

    <h2>Submodules</h2>

- ``state_shewhart_chart_visualizer``: contains the
  ``ShewhartStateVisualizer`` class and its associated option TypedDicts.
  See that module's docstring for implementation details.

.. raw:: html

    <h2>Examples</h2>

Create a Shewhart state visualizer and configure it for use with
``OnlineTraceVisualizer``::

    >>> from pysatl_cpd.algorithms.online import ShewhartControlChart
    >>> from pysatl_cpd.algorithms.online.control_charts.visualizers import (
    ...     ShewhartStateVisualizer,
    ... )
    >>> from pysatl_cpd.analysis.visualization import DrawBackend
    >>> from pysatl_cpd.analysis.visualization.online import OnlineTraceVisualizer
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
    >>> scenario = ScenarioSpec(
    ...     name="example",
    ...     segments=(
    ...         SegmentSpec(plan_name="base", length=100),
    ...         SegmentSpec(plan_name="shift", length=80),
    ...     ),
    ...     plans=frozendict(
    ...         base=SegmentPlan(
    ...             distribution=UnivariateDistributionSpec("Normal", "meanStd", mu=0.0, sigma=1.0),
    ...             state=StateDescriptor(type="base"),
    ...             name="base",
    ...         ),
    ...         shift=SegmentPlan(
    ...             distribution=UnivariateDistributionSpec("Normal", "meanStd", mu=3.0, sigma=1.0),
    ...             state=StateDescriptor(type="shift"),
    ...             name="shift",
    ...         ),
    ...     ),
    ... )
    >>> series = GenericSeriesGenerator(seed=42).generate_from_scenario(
    ...     scenario, name="example_series"
    ... )
    >>> provider = build_plain_univariate_labeled_data(
    ...     series, feature_name="value", name="example_provider"
    ... )
    >>> detector = OnlineResetDetector(
    ...     ShewhartControlChart(learning_period_size=30, window_size=10),
    ...     threshold=2.0,
    ...     skip_period=8,
    ...     collect_states=True,
    ... )
    >>> trace = detector.detect(provider)
    >>> state_viz = ShewhartStateVisualizer(DrawBackend.MATPLOTLIB)
    >>> state_viz.set_plot_opts(
    ...     title="Shewhart State", xlabel="Time", ylabel="Value", grid=True
    ... )
    >>> state_viz.set_band_opts(band_size=3.0)
    >>> trace_viz = OnlineTraceVisualizer(
    ...     backend=DrawBackend.MATPLOTLIB,
    ...     state_visualizer=state_viz,
    ... )
    >>> trace_viz.set_trace(trace)

Notes
-----
This subpackage depends on ``matplotlib`` and ``plotly`` for rendering.
The ``ShewhartStateVisualizer`` requires that the detector was created with
``collect_states=True``; otherwise no state snapshots are available to render.
Control limits are computed as ``mu +/- k * sigma / sqrt(w)`` where ``w`` is
the window size at each step.
"""

__author__ = "Danil Totmyanin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.algorithms.online.control_charts.visualizers.state_shewhart_chart_visualizer import (
    ShewhartStateBandOpts,
    ShewhartStateDrawOpts,
    ShewhartStatePlotOpts,
    ShewhartStateVisualizer,
)

__all__ = [
    "ShewhartStateBandOpts",
    "ShewhartStateDrawOpts",
    "ShewhartStatePlotOpts",
    "ShewhartStateVisualizer",
]
