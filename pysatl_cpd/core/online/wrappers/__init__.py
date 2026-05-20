# -*- coding: ascii -*-
"""Composable wrappers that modify online algorithm behavior.

This subpackage provides wrapper classes and helper dataclasses that decorate
``OnlineAlgorithm`` instances to alter how observations are consumed without
modifying the underlying algorithm's implementation. Wrappers are composable:
they can be nested to combine behaviors such as skipping and batching in a
single processing pipeline.

Each wrapper implements the full ``OnlineAlgorithm`` interface (``process``,
``reset``, ``recreate``, ``name``, ``configuration``, ``state``), so wrapped
instances can be passed directly to ``OnlineResetDetector`` or any other
consumer that expects an ``OnlineAlgorithm``.

.. raw:: html

    <h2>Public API</h2>

- ``SkippingCondition`` -- Frozen dataclass holding a named predicate that
  decides whether an observation should be skipped.
- ``BatchReducer`` -- Frozen dataclass holding a named function that reduces
  a sequence of raw observations into a single value.
- ``SkippingOnlineAlgorithmWrapper`` -- Wraps an ``OnlineAlgorithm`` and
  conditionally bypasses observations, returning the last computed detection
  statistic when the skip condition is met.
- ``BatchingOnlineAlgorithmWrapper`` -- Wraps an ``OnlineAlgorithm`` and
  accumulates raw observations into fixed-size batches, reducing each batch
  to a single value via a ``BatchReducer`` before forwarding it.

.. raw:: html

    <h2>Examples</h2>

Examples
--------

Skip observations whose absolute value exceeds a threshold:

>>> from pysatl_cpd.algorithms.online import ShewhartControlChart
>>> from pysatl_cpd.core.online.wrappers import (
...     SkippingCondition,
...     SkippingOnlineAlgorithmWrapper,
... )
>>> condition = SkippingCondition(
...     name="large-value",
...     condition=lambda x: abs(float(x)) > 1.0,
... )
>>> wrapper = SkippingOnlineAlgorithmWrapper(
...     ShewhartControlChart(learning_period_size=5, window_size=5),
...     skipping_condition=condition,
... )
>>> wrapper.name
'ShewhartControlChart{skip[on=large-value]}'

Batch observations into groups of four, reducing each group by mean:

>>> import numpy as np
>>> from pysatl_cpd.algorithms.online import ShewhartControlChart
>>> from pysatl_cpd.core.online.wrappers import (
...     BatchReducer,
...     BatchingOnlineAlgorithmWrapper,
... )
>>> reducer = BatchReducer(name="mean", reducer=lambda batch: float(np.mean(batch)))
>>> wrapper = BatchingOnlineAlgorithmWrapper(
...     ShewhartControlChart(learning_period_size=3, window_size=3),
...     batch_size=4,
...     reducer=reducer,
... )
>>> wrapper.name
'ShewhartControlChart{batch[size=4, reduce=mean]}'

Compose wrappers by nesting -- the outermost wrapper's name appends last:

>>> from pysatl_cpd.algorithms.online import ShewhartControlChart
>>> from pysatl_cpd.core.online.wrappers import (
...     BatchReducer,
...     BatchingOnlineAlgorithmWrapper,
...     SkippingCondition,
...     SkippingOnlineAlgorithmWrapper,
... )
>>> skip = SkippingCondition(name="nan", condition=lambda x: x != x)
>>> reducer = BatchReducer(name="sum", reducer=sum)
>>> wrapped = BatchingOnlineAlgorithmWrapper(
...     SkippingOnlineAlgorithmWrapper(
...         ShewhartControlChart(learning_period_size=5, window_size=5),
...         skipping_condition=skip,
...     ),
...     batch_size=3,
...     reducer=reducer,
... )
>>> wrapped.name
'ShewhartControlChart{skip[on=nan]}{batch[size=3, reduce=sum]}'

Notes
-----
- The ``recreate`` method walks through nested wrappers to recreate the
  innermost algorithm, preserving the wrapper configuration.
- ``SkippingCondition`` and ``BatchReducer`` hashes are based solely on
  the ``name`` field; callables are intentionally excluded because their
  identities are not stable across processes or serialization boundaries.
"""

__author__ = "Mikhail Mikhailov, Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.core.online.wrappers.online_wrappers import (
    BatchingOnlineAlgorithmWrapper,
    BatchReducer,
    SkippingCondition,
    SkippingOnlineAlgorithmWrapper,
)

__all__ = [
    "BatchingOnlineAlgorithmWrapper",
    "BatchReducer",
    "SkippingCondition",
    "SkippingOnlineAlgorithmWrapper",
]
