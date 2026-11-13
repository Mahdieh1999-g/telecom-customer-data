"""
Tests for Monitoring Module
TODO: Write pytest tests for the following cases:
"""
import pytest
import numpy as np

# TODO: Import your functions
from src.monitoring.drift_detection import detect_drift_ks, detect_drift_mean_shift


class TestDriftDetection:
    """TODO: Implement tests for drift detection."""

    def test_no_drift_same_distribution(self):
        """Test no drift when reference and current are the same.
        TODO: Pass same data to detect_drift_ks, check drift_detected is False
        """
        reference_data = np.array([
            10, 11, 12, 13, 14,
            15, 16, 17, 18, 19
        ])

        current_data = np.array([
            10, 11, 12, 13, 14,
            15, 16, 17, 18, 19
        ])

        result = detect_drift_ks(
            reference_data,
            current_data
        )

        assert result["drift_detected"] is False

    def test_drift_different_distributions(self):
        """Test drift detected for shifted distribution.
        TODO: Pass shifted data to detect_drift_ks, check drift_detected is True
        """
        reference_data = np.array([
            10, 11, 12, 13, 14,
            15, 16, 17, 18, 19
        ])

        current_data = np.array([
            100, 101, 102, 103, 104,
            105, 106, 107, 108, 109
        ])

        result = detect_drift_ks(
            reference_data,
            current_data
        )

        assert result["drift_detected"] is True

    def test_ks_result_structure(self):
        """Test that KS result has correct keys.
        TODO: Check result contains method, statistic, p_value, drift_detected
        """
        reference_data = np.array([
            10, 11, 12, 13, 14,
            15, 16, 17, 18, 19
        ])

        current_data = np.array([
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20
        ])

        result = detect_drift_ks(
            reference_data,
            current_data
        )

        assert isinstance(result, dict)

        assert "method" in result
        assert "statistic" in result
        assert "p_value" in result
        assert "drift_detected" in result
