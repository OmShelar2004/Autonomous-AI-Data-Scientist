# AI Data Scientist

Dataset-agnostic CSV execution layer built around the existing LangGraph workflow.
Python performs profiling, preprocessing, training, validation, model ranking, model
persistence, and prediction; the orchestration layer never invents numerical results.

## Install

```bash
pip install -r requirements.txt
```

## Train from a CSV

## Simple web interface

Run the browser interface after installing the dependencies:

```bash
streamlit run app.py
```

It guides the user through CSV upload, target selection, real model evaluation, choosing
one of the two recommendations, downloading the trained `.joblib` model, and prediction
from either a new CSV or an editable single row.

## Command-line interface

Specify a target when it is known:

```bash
python main.py path/to/data.csv --target target_column
```

If target detection is not confident, the CLI lists the CSV columns and asks you to
choose one. It reports genuine held-out validation metrics, presents the two best
models, and only saves a model after you select option 1 or 2.

For non-interactive execution:

```bash
python main.py path/to/data.csv --target target_column --select 1 --model-path models/my_model.joblib
```

## Predict

```bash
python main.py --model models/my_model.joblib --predict-csv path/to/new_rows.csv
python main.py --model models/my_model.joblib --row-json '{"feature_a": 12, "feature_b": "blue"}'
```

The saved artifact includes date feature engineering, the fitted preprocessing pipeline,
the selected estimator, and (for classification) class-label decoding.

## Execution flow

```
dataset_loader -> analysis -> planner -> executor -> reviewer
```

The executor runs median/mode imputation, one-hot encoding, numeric scaling, safe
datetime components, lightweight candidate training, and metric-based ranking. XGBoost
is used when installed; otherwise the remaining baseline models are still evaluated.
