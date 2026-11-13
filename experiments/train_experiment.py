"""
Training Experiment Script
TODO: Implement the full pipeline below
Read config and run: load data -> features -> train -> evaluate -> save
"""
import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# TODO: Import your modules
from src.data.load_data import load_config, load_csv, split_data, create_data_version
from src.features.build_features import scale_features, select_features
from src.models.train_model import train_model, evaluate_model, save_model



def run_pipeline(config_path="config/config.yaml"):
    """Run the full ML pipeline.
    TODO: Implement the following steps:
    1. Load config from config.yaml
    2. Load data from CSV (or generate sample data if file not found)
    3. Create data version (v1)
    4. Split data into train/test
    5. Scale features
    6. Select features
    7. Train model
    8. Evaluate model
    9. Save model with metadata
    10. Print results
    """

    print("=" * 50)
    print("Starting ML Pipeline")
    print("=" * 50)

    # 1. Load config

    config = load_config(config_path)

    # 2. Load data

    data_path = os.path.join(
        config["data"]["raw_path"],
        config["data"]["file_name"]
        )

    print(f"Loading data from: {data_path}")

    df = load_csv(data_path)

    print(f"Dataset shape: {df.shape}")

    # 3. Create data version

    data_version = create_data_version(
        df=df,
        version_id="v1",
        description=(
            "Reference dataset used "
            "for model training"
        )
    )

    print("Data version created:")

    print(data_version)

    # 4. Split data

    X_train, X_test, y_train, y_test = split_data(
        df=df,
        target_column=config["data"]["target_column"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    print(f"Train size: {X_train.shape}")

    print(f"Test size: {X_test.shape}")

    # 5. Scale features

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test,
        method=config["features"]["scaling"]
    )

    # 6. Select features

    X_train_selected, X_test_selected, selector = select_features(
        X_train_scaled,
        y_train,
        X_test_scaled,
        k=config["features"]["feature_selection_k"]
    )

    print("Selected features:")

    print(list( X_train_selected.columns ))

    # 7. Train model

    model = train_model(
        X_train=X_train_selected,
        y_train=y_train,
        model_type=config["model"]["type"],
        params=config["model"]["params"]
    )

    # 8. Evaluate model

    metrics = evaluate_model(model,X_test_selected,y_test)

    print("=" * 50)
    print("Model Evaluation")
    print("=" * 50)

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"F1 Score: "
        f"{metrics['f1_score']:.4f}"
    )

    # 9. Save model

    save_model(
        model=model,
        metrics=metrics,
        data_version=data_version["version_id"],
        model_version="v1",
        output_dir="models"
    )

    print("=" * 50)
    print("Pipeline completed successfully")
    print("=" * 50)

    return metrics


if __name__ == "__main__":
    run_pipeline()
