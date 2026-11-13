import os
import sys

import mlflow
import mlflow.sklearn


sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


from src.data.load_data import (
    load_csv,
    split_data,
    create_data_version
)

from src.features.build_features import (
    scale_features,
    select_features
)

from src.models.train_model import (
    train_model,
    evaluate_model
)


def run_experiment(
    dataset_path,
    data_version,
    model_type,
    model_params,
    scaling="standard",
    k_features=10
):

    # Load data

    df = load_csv(
        dataset_path
    )

    # Split

    X_train, X_test, y_train, y_test = split_data(
        df=df,
        target_column="churn",
        test_size=0.2,
        random_state=42
    )

    # Scale

    X_train_scaled, X_test_scaled, _ = scale_features(
        X_train,
        X_test,
        method=scaling
    )

    # Feature selection

    X_train_selected, X_test_selected, _ = select_features(
        X_train_scaled,
        y_train,
        X_test_scaled,
        k=k_features
    )

    # Start MLflow run

    with mlflow.start_run():

        # Log parameters

        mlflow.log_param(
            "data_version",
            data_version
        )

        mlflow.log_param(
            "model_type",
            model_type
        )

        mlflow.log_param(
            "scaling",
            scaling
        )

        mlflow.log_param(
            "feature_selection_k",
            k_features
        )

        # Log model parameters

        mlflow.log_params(
            model_params
        )

        # Train

        model = train_model(
            X_train=X_train_selected,
            y_train=y_train,
            model_type=model_type,
            params=model_params
        )

        # Evaluate

        metrics = evaluate_model(
            model,
            X_test_selected,
            y_test
        )

        # Log metrics

        mlflow.log_metric(
            "accuracy",
            metrics["accuracy"]
        )

        mlflow.log_metric(
            "f1_score",
            metrics["f1_score"]
        )

        # Log model

        mlflow.sklearn.log_model(
            model,
            name="model"
        )

        print("=" * 50)

        print(
            f"Dataset: {data_version}"
        )

        print(
            f"Model: {model_type}"
        )

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"F1 Score: "
            f"{metrics['f1_score']:.4f}"
        )

        print("=" * 50)

        return metrics


def main():

    # Local MLflow tracking

    mlflow.set_tracking_uri(
        "file:./mlruns"
    )

    mlflow.set_experiment(
        "mlops_homework_01"
    )

    # -------------------------
    # RUN 1
    # Random Forest
    # Dataset v1
    # -------------------------

    run_experiment(
        dataset_path=(
            "data/raw/dataset_v1.csv"
        ),
        data_version="v1",
        model_type="random_forest",
        model_params={
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
    )

    # -------------------------
    # RUN 2
    # Random Forest
    # Different hyperparameter
    # -------------------------

    run_experiment(
        dataset_path=(
            "data/raw/dataset_v1.csv"
        ),
        data_version="v1",
        model_type="random_forest",
        model_params={
            "n_estimators": 200,
            "max_depth": 15,
            "random_state": 42
        }
    )

    # -------------------------
    # RUN 3
    # Logistic Regression
    # Different model
    # -------------------------

    run_experiment(
        dataset_path=(
            "data/raw/dataset_v1.csv"
        ),
        data_version="v1",
        model_type="logistic_regression",
        model_params={
            "max_iter": 1000,
            "random_state": 42
        }
    )


if __name__ == "__main__":
    main()