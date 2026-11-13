"""
Tests for Data Loading Module
TODO: Write pytest tests for the following cases:
"""
import pytest
import pandas as pd
import numpy as np

# TODO: Import your functions
from src.data.load_data import load_csv, split_data, compute_hash


class TestLoadData:
    """TODO: Implement tests for data loading."""

    def test_load_csv_existing_file(self,tmp_path):
        """Test loading an existing CSV file.
        TODO: Create a temp CSV, load it, check shape
        """
        data = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
            "churn": [0, 1, 0, 1, 0]
        })

        csv_file = tmp_path / "test.csv"

        data.to_csv(csv_file,index=False)

        # Load CSV
        result = load_csv(str(csv_file))

        # Check type
        assert isinstance(result,pd.DataFrame)

        # Check shape
        assert result.shape == (5, 3)

    def test_load_csv_missing_file(self,tmp_path):
        """Test that FileNotFoundError is raised for missing file.
        TODO: Call load_csv with nonexistent file, expect exception
        """
        missing_file = (tmp_path / "does_not_exist.csv")

        with pytest.raises(FileNotFoundError):
            load_csv(str(missing_file))


    def test_compute_hash(self):
        """Test hash computation is deterministic.
        TODO: Compute hash twice on same data, check they match
        """
        data = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6]
        })

        hash_1 = compute_hash(data)
        hash_2 = compute_hash(data)

        assert hash_1 == hash_2

        assert isinstance(
            hash_1,
            str
        )

        assert len(hash_1) > 0


class TestSplitData:
    """TODO: Implement tests for data splitting."""

    def test_split_returns_four_elements(self):
        """Test that split_data returns (X_train, X_test, y_train, y_test).
        TODO: Split sample data, check len(result) == 4
        """
        data = pd.DataFrame({
            "feature1": np.arange(20),
            "feature2": np.arange(20, 40),
            "churn": [0, 1] * 10
        })

        result = split_data(
            data,
            target_column="churn",
            test_size=0.2,
            random_state=42
        )

        assert len(result) == 4

        X_train, X_test, y_train, y_test = result

        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None

    def test_split_sizes(self):
        """Test that split sizes are correct with test_size=0.2.
        TODO: Check train has 80% and test has 20% of samples
        """
        data = pd.DataFrame({
            "feature1": np.arange(100),
            "feature2": np.arange(100, 200),
            "churn": [0, 1] * 50
        })

        X_train, X_test, y_train, y_test = split_data(
            data,
            target_column="churn",
            test_size=0.2,
            random_state=42
        )

        # 80% train
        assert len(X_train) == 80
        assert len(y_train) == 80

        # 20% test
        assert len(X_test) == 20
        assert len(y_test) == 20

    def test_split_reproducible(self):
        """Test that splitting is reproducible with same random_state.
        TODO: Split twice with same random_state, check results match
        """
        data = pd.DataFrame({
            "feature1": np.arange(100),
            "feature2": np.arange(100, 200),
            "churn": [0, 1] * 50
        })

        result_1 = split_data(
            data,
            target_column="churn",
            test_size=0.2,
            random_state=42
        )

        result_2 = split_data(
            data,
            target_column="churn",
            test_size=0.2,
            random_state=42
        )

        X_train_1, X_test_1, y_train_1, y_test_1 = result_1

        X_train_2, X_test_2, y_train_2, y_test_2 = result_2

        # Compare X
        pd.testing.assert_frame_equal(
            X_train_1,
            X_train_2
        )
        pd.testing.assert_frame_equal(
            X_test_1,
            X_test_2
        )

        # Compare y
        pd.testing.assert_series_equal(
            y_train_1,
            y_train_2
        )

        pd.testing.assert_series_equal(
            y_test_1,
            y_test_2
        )
