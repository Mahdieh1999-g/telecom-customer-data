"""
Tests for Model Training Module
TODO: Write pytest tests for the following cases:
"""
import pytest
import pandas as pd
import numpy as np

# TODO: Import your functions
from src.models.train_model import get_model, train_model, evaluate_model


class TestGetModel:
    """TODO: Implement tests for model creation."""

    def test_random_forest(self):
        """Test Random Forest model creation.
        TODO: Call get_model("random_forest", {...}), check it's not None
        """
        model = get_model(
            "random_forest",
            {
                "n_estimators": 10,
                "random_state": 42
            }
        )

        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_invalid_model_type(self):
        """Test that invalid model type raises ValueError.
        TODO: Call get_model("invalid", {}), expect ValueError
        """
        with pytest.raises(ValueError):
            get_model(
                "invalid",
                {}
            )


class TestTrainModel:
    """TODO: Implement tests for model training."""

    def test_train_model(self):
        """Test model training completes.
        TODO: Create sample data, train model, check it has predict method
        """
        X_train = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
            "feature2": [8, 7, 6, 5, 4, 3, 2, 1]
        })

        y_train = np.array([
            0, 0, 0, 0,
            1, 1, 1, 1
        ])

        model = train_model(
            X_train,
            y_train,
            "random_forest",
            {
                "n_estimators": 10,
                "random_state": 42
            }
        )

        assert model is not None
        assert hasattr(model, "predict")


class TestEvaluateModel:
    """TODO: Implement tests for model evaluation."""

    def test_evaluate_returns_metrics(self):
        """Test evaluation returns accuracy and f1_score.
        TODO: Train model, evaluate, check metrics dict has both keys
        """
        X_train = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
            "feature2": [8, 7, 6, 5, 4, 3, 2, 1]
        })

        y_train = np.array([
            0, 0, 0, 0,
            1, 1, 1, 1
        ])

        X_test = pd.DataFrame({
            "feature1": [1.5, 2.5, 6.5, 7.5],
            "feature2": [7.5, 6.5, 2.5, 1.5]
        })

        y_test = np.array([
            0, 0, 1, 1
        ])

        model = train_model(
            X_train,
            y_train,
            "random_forest",
            {
                "n_estimators": 10,
                "random_state": 42
            }
        )
        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        assert isinstance(metrics, dict)

        assert "accuracy" in metrics
        assert "f1_score" in metrics

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
