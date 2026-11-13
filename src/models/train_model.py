"""
Model Training Module
TODO: Implement the functions below
"""
import os
import json
import joblib
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


def get_model(model_type, params):
    """Get a model instance based on type and parameters.
    TODO: Implement this function
    - model_type: "random_forest", "logistic_regression", or "svm"
    - Return the initialized model
    """
    if model_type == "random_forest":

        return RandomForestClassifier(**params)

    elif model_type == "logistic_regression":

        return LogisticRegression(**params)

    elif model_type == "svm":

        return SVC(probability=True,**params)

    else:

        raise ValueError(
            f"Unsupported model type: {model_type}"
        )


def train_model(X_train, y_train, model_type, params):
    """Train a model.
    TODO: Implement this function
    - Return the trained model
    """
    model = get_model(model_type,params)

    model.fit(X_train,y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model.
    TODO: Implement this function
    - Return dict with "accuracy" and "f1_score"
    """
    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    return {
        "accuracy": float(accuracy),
        "f1_score": float(f1)
    }


def save_model(model, metrics, data_version, model_version, output_dir="models"):
    """Save model and its metadata.
    TODO: Implement this function
    - Save model as model.joblib
    - Save metadata.json with: model_version, created_at, data_version, metric
    """
    os.makedirs(output_dir,exist_ok=True)

    model_path = os.path.join(output_dir,"model.joblib")

    metadata_path = os.path.join(output_dir,"metadata.json")

    joblib.dump(model, model_path)

    metadata = {
        "model_version": model_version,
        "created_at": datetime.now().isoformat(),
        "data_version": data_version,
        "metric": metrics
    }

    with open(metadata_path,"w",encoding="utf-8") as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    return model_path, metadata_path
