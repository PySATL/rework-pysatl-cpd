# -*- coding: ascii -*-
"""
Tests for run length metric.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.metrics.single_run.online.run_lengths import RunLengths


class TestRunLengths:
    """Test cases for RunLengths metric."""

    def test_no_detections(self, make_online_run):
        """No detections should return [len(provider)]."""
        metric = RunLengths()
        run = make_online_run([], [10, 20], length=100)
        result = metric.evaluate(run)
        assert result == [100]

    def test_single_detection(self, make_online_run):
        """Single detection at index 5 should return [5]."""
        metric = RunLengths()
        run = make_online_run([5], [10])
        result = metric.evaluate(run)
        assert result == [5]

    def test_multiple_detections(self, make_online_run):
        """Multiple detections should return distances between them."""
        metric = RunLengths()
        run = make_online_run([3, 7, 10], [5, 15])
        result = metric.evaluate(run)
        assert result == [3, 4, 3]

    def test_unsorted_detections(self, make_online_run):
        """Unsorted detections should be sorted before computing."""
        metric = RunLengths()
        run = make_online_run([10, 3, 7], [5, 15])
        result = metric.evaluate(run)
        assert result == [3, 4, 3]

    def test_first_run_length_from_zero(self, make_online_run):
        """First run length should be from 0 to first detection."""
        metric = RunLengths()
        run = make_online_run([5], [10])
        result = metric.evaluate(run)
        assert result[0] == 5

    def test_ignore_ground_truth(self, make_online_run):
        """RunLengths should ignore true CPs completely."""
        metric = RunLengths()
        run = make_online_run([10, 20], [])
        result = metric.evaluate(run)
        assert result == [10, 10]

    def test_result_is_list_of_int(self, make_online_run):
        """Result should be list of int."""
        metric = RunLengths()
        run = make_online_run([5], [10])
        result = metric.evaluate(run)
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_empty_provider_with_detections(self, make_online_run):
        """Should work even if provider has no change points."""
        metric = RunLengths()
        run = make_online_run([10, 20], [], length=50)
        result = metric.evaluate(run)
        assert result == [10, 10]
