"""
Data Loading Module
TODO: Implement the functions below
"""
import os
import hashlib
import json
from datetime import datetime

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file.
    TODO: Implement this function
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def load_csv(file_path):
    """Load a CSV file into a DataFrame.
    TODO: Implement this function
    - Check if file exists, raise FileNotFoundError if not
    - Return the DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"CSV file not found: {file_path}"
        )

    return pd.read_csv(file_path)


def compute_hash(df):
    """Compute MD4 hash of a DataFrame for versioning.
    TODO: Implement this function
    """
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    return hashlib.md5(csv_bytes).hexdigest()


def split_data(df, target_column, test_size=0.2, random_state=42):
    """Split data into train and test sets.
    TODO: Implement this function
    - Use sklearn's train_test_split
    - Return (X_train, X_test, y_train, y_test)
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found"
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_data_version(df, version_id, description, output_dir="data_versions"):
    """Create a versioned snapshot of the data.
    TODO: Implement this function
    - Save metadata to data_versions/metadata.json
    - Metadata should include: version_id, hash, description, n_rows, created_at
    """
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "version_id": version_id,
        "hash": compute_hash(df),
        "description": description,
        "n_rows": int(len(df)),
        "created_at": datetime.now().isoformat()
    }

    metadata_path = os.path.join(
        output_dir,
        f"{version_id}_metadata.json"
    )

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    return metadata
