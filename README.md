# MLOps Homework 01

## TODO: Fill in your info here
- Name: 
- Student ID:

## Project Structure
```
project/
├── config/config.yaml          # Hyperparameters
├── data/raw/                   # Raw data
├── data/processed/             # Processed data
├── src/
│   ├── data/load_data.py       # TODO: Data loading
│   ├── features/build_features.py  # TODO: Feature engineering
│   ├── models/train_model.py   # TODO: Model training
│   ├── monitoring/drift_detection.py  # TODO: Drift detection
│   └── serving/app.py          # TODO: API serving
├── tests/                      # TODO: pytest tests
├── experiments/train_experiment.py  # TODO: Main pipeline
├── models/                     # Saved models
└── data_versions/metadata.json # Data versioning
```

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python experiments/train_experiment.py
pytest tests/ -v
```
