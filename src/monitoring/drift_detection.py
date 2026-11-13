"""
Monitoring & Drift Detection Module
TODO: Implement the functions below

NOTE: Do NOT store sensitive information in logs.
"""
import json
import os
from datetime import datetime

import numpy as np
from scipy import stats


def log_prediction(prediction, confidence, latency, log_path="experiments/logs"):
    """Log a prediction with structured JSON format.
    TODO: Implement this function
    - Create a log entry with: timestamp, latency, prediction, confidence
    - Append to a daily log file (predictions_YYYY-MM-DD.json)
    - Do NOT store sensitive info (user_id, raw features, etc.)
    """
    os.makedirs(log_path, exist_ok=True)

    timestamp = datetime.now()

    log_entry = {
        "timestamp": timestamp.isoformat(),
        "latency": float(latency),
        "prediction": int(prediction),
        "confidence": float(confidence)
    }

    filename = f"predictions_{timestamp.strftime('%Y-%m-%d')}.json"
    file_path = os.path.join(log_path, filename)

    with open(file_path, "a", encoding="utf-8") as f:
        json.dump(log_entry, f)
        f.write("\n")

    return file_path


def detect_drift_ks(reference_data, current_data, threshold=0.05):
    """Detect drift using Kolmogorov-Smirnov test.
    TODO: Implement this function
    - Return dict with: method, statistic, p_value, drift_detected
    """
    reference_data = np.asarray(reference_data)
    current_data = np.asarray(current_data)

    statistic, p_value = stats.ks_2samp(
        reference_data,
        current_data
    )

    drift_detected = bool(p_value < threshold)

    return {
        "method": "KS Test",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": drift_detected
    }


def detect_drift_mean_shift(reference_data, current_data, threshold=0.1):
    """Detect drift using mean shift detection.
    TODO: Implement this function
    - Return dict with: method, reference_mean, current_mean, drift_detected
    """
    reference_data = np.asarray(reference_data)
    current_data = np.asarray(current_data)

    reference_mean = float(np.mean(reference_data))
    current_mean = float(np.mean(current_data))

    # Relative change between the two means
    if reference_mean == 0:
        mean_shift = abs(current_mean - reference_mean)
    else:
        mean_shift = abs(current_mean - reference_mean) / abs(reference_mean)

    drift_detected = bool(mean_shift > threshold)

    return {
        "method": "Mean Shift",
        "reference_mean": reference_mean,
        "current_mean": current_mean,
        "drift_detected": drift_detected
    }
