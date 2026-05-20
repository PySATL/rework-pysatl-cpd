# -*- coding: ascii -*-
"""Segment generators for synthetic time series data.

This subpackage provides the core building blocks for generating individual
synthetic time series segments. It defines the data model for generated
segments, a protocol for custom segment generators, and sampling utilities
that draw from distribution specifications defined in
``pysatl_cpd.data.generator.specs``.

.. raw:: html

    <h2>Public API</h2>

- ``GeneratedSegment`` -- Immutable dataclass holding generated segment data,
  feature names, segment metadata, and optional user-defined metadata.
- ``SegmentGenerator`` -- Protocol for implementing custom segment generators
  that produce ``GeneratedSegment`` instances with specified distributions.
- ``sample_distribution`` -- Sample numeric data from a ``DistributionSpec``,
  supporting univariate, multivariate normal, and independent-column distributions.
- ``feature_names_for_distribution`` -- Extract feature names from a
  ``DistributionSpec``.
- ``DEFAULT_UNIVARIATE_FEATURE_NAME`` -- Default feature name (``"value"``)
  used for univariate distribution specs.

.. raw:: html

    <h2>Examples</h2>

Examples
--------
Sample from a univariate distribution spec::

    >>> import numpy as np
    >>> from pysatl_cpd.data.generator import UnivariateDistributionSpec
    >>> from pysatl_cpd.data.generator.segments import sample_distribution
    >>> rng = np.random.default_rng(42)
    >>> spec = UnivariateDistributionSpec("Normal", "meanStd", mu=0.0, sigma=1.0)
    >>> data = sample_distribution(spec, length=5, rng=rng)
    >>> data.shape
    (5, 1)

Sample from a multivariate normal distribution::

    >>> from pysatl_cpd.data.generator import MultivariateNormalSpec
    >>> from pysatl_cpd.data.typedefs import frozendict
    >>> mv_spec = MultivariateNormalSpec(
    ...     means=frozendict(x=0.0, y=1.0),
    ...     covariance=((1.0, 0.0), (0.0, 1.0)),
    ... )
    >>> mv_data = sample_distribution(mv_spec, length=3, rng=rng)
    >>> mv_data.shape
    (3, 2)

Extract feature names from a distribution spec::

    >>> from pysatl_cpd.data.generator.segments import feature_names_for_distribution
    >>> feature_names_for_distribution(mv_spec)
    ('x', 'y')

Implement a custom segment generator using the ``SegmentGenerator`` protocol::

    >>> from pysatl_cpd.data.generator.segments import (
    ...     GeneratedSegment,
    ...     SegmentGenerator,
    ...     sample_distribution,
    ... )
    >>> from pysatl_cpd.data.generator import SegmentPlan, UnivariateDistributionSpec
    >>> from pysatl_cpd.data.typedefs import SegmentInfo, StateDescriptor, frozendict
    >>> import numpy as np
    >>> class SimpleGenerator:
    ...     def __init__(self, plan: SegmentPlan, length: int) -> None:
    ...         self._plan = plan
    ...         self._length = length
    ...     @property
    ...     def feature_names(self) -> tuple[str, ...]:
    ...         from pysatl_cpd.data.generator.segments import feature_names_for_distribution
    ...         return feature_names_for_distribution(self._plan.distribution)
    ...     @property
    ...     def length(self) -> int:
    ...         return self._length
    ...     def generate(self, rng: np.random.Generator | None = None) -> GeneratedSegment:
    ...         rng = rng or np.random.default_rng()
    ...         data = sample_distribution(self._plan.distribution, self._length, rng)
    ...         return GeneratedSegment(
    ...             name=self._plan.name or "segment",
    ...             data=data,
    ...             feature_names=self.feature_names,
    ...             segment_info=SegmentInfo(
    ...                 segment_num=0,
    ...                 segment_start=0,
    ...                 segment_end=self._length,
    ...                 state=self._plan.state or StateDescriptor(),
    ...             ),
    ...             metadata=self._plan.metadata,
    ...         )
    >>> plan = SegmentPlan(
    ...     distribution=UnivariateDistributionSpec("Normal", "meanStd", mu=2.0, sigma=0.5),
    ...     name="shifted",
    ... )
    >>> gen = SimpleGenerator(plan, length=10)
    >>> segment = gen.generate(rng=np.random.default_rng(0))
    >>> segment.name
    'shifted'
    >>> segment.data.shape
    (10, 1)

.. raw:: html

    <h2>Notes</h2>

Notes
-----
- Sampling functions require a ``DistributionSpec`` from
  ``pysatl_cpd.data.generator.specs`` (e.g., ``UnivariateDistributionSpec``,
  ``MultivariateNormalSpec``, ``IndependentColumnsSpec``).
- All returned arrays have shape ``(length, num_features)``. Univariate
  distributions produce a single-column array.
- The ``SegmentGenerator`` protocol is structural (``typing.Protocol``). Any
  class with matching ``feature_names``, ``length``, and ``generate`` attributes
  satisfies it without explicit inheritance.
- This subpackage handles individual segment generation. For full scenario-based
  series generation, see ``pysatl_cpd.data.generator.GenericSeriesGenerator``.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .interface import SegmentGenerator
from .models import GeneratedSegment
from .sampling import DEFAULT_UNIVARIATE_FEATURE_NAME, feature_names_for_distribution, sample_distribution

__all__ = [
    "DEFAULT_UNIVARIATE_FEATURE_NAME",
    "GeneratedSegment",
    "SegmentGenerator",
    "feature_names_for_distribution",
    "sample_distribution",
]
