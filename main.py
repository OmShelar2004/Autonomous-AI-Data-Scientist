"""CLI entry point for the AI Data Scientist execution layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from graph import agent
from tools.model_training import train_final_model
from tools.predict import predict_csv, predict_rows


def _print_metrics(model: dict) -> None:
    metrics = ", ".join(
        f"{name.upper()}: {value:.4f}"
        for name, value in model.items()
        if name != "model_name" and value is not None
    )
    print(f"{model['model_name']}: {metrics}")


def run_training(args: argparse.Namespace) -> None:
    state = {
        "task": "Train and evaluate tabular supervised-learning models.",
        "dataset_path": args.dataset,
        "target_column": args.target,
        "retry_count": 0,
    }
    response = agent.invoke(state)
    if not response.get("target_column"):
        detected = response["target_detection"]
        print(f"\nTarget selection required: {detected['reason']}")
        print("Available columns:")
        print(", ".join(response["dataset_summary"]["column_names"]))
        selected = input("Target column: ").strip()
        if not selected:
            raise SystemExit("No target selected; no model was trained.")
        state["target_column"] = selected
        response = agent.invoke(state)

    print("\n" + "=" * 60)
    print("AI DATA SCIENTIST")
    print("=" * 60)
    print(f"Dataset: {response['dataset_summary']['rows']} rows, {response['dataset_summary']['columns']} columns")
    print(f"Detected target: {response['target_column']}")
    print(f"Problem type: {response['problem_type'].replace('_', ' ')}")
    print(f"Missing values: {sum(response['dataset_summary']['missing_values'].values())}; duplicates: {response['dataset_summary']['duplicate_rows']}")
    print("\nMODEL COMPARISON")
    for result in response["model_results"]:
        _print_metrics(result)
    print("\nTOP 2 RECOMMENDED MODELS")
    for index, model in enumerate(response["top_models"], start=1):
        print(f"{index}. {model['model_name']}")

    choice = args.select
    if choice is None:
        choice = input("Select model (1 or 2, blank to stop without saving): ").strip()
    if str(choice).strip() == "":
        print("No model selected; no model was saved.")
        return
    try:
        selected = response["top_models"][int(choice) - 1]["model_name"]
    except (ValueError, IndexError):
        raise SystemExit("Select 1 or 2.")
    bundle = train_final_model(response["processed_data"], selected)
    output = Path(args.model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    print(f"Training selected model: {selected}")
    print(f"Model saved successfully: {output}")


def run_prediction(args: argparse.Namespace) -> None:
    values = predict_csv(args.model, args.predict_csv) if args.predict_csv else predict_rows(args.model, json.loads(args.row_json))
    for value in values:
        print(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset-agnostic tabular ML execution CLI")
    parser.add_argument("dataset", nargs="?", help="CSV dataset to train on")
    parser.add_argument("--target", help="Target column (asked interactively if uncertain)")
    parser.add_argument("--select", help="Recommended model number to save (1 or 2)")
    parser.add_argument("--model-path", default="models/trained_model.joblib")
    parser.add_argument("--predict-csv", help="Predict an unseen CSV using --model")
    parser.add_argument("--row-json", help="Predict one JSON object using --model")
    parser.add_argument("--model", help="Saved joblib artifact for prediction")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.predict_csv or arguments.row_json:
        if not arguments.model:
            parser.error("--model is required with a prediction option")
        run_prediction(arguments)
    elif arguments.dataset:
        run_training(arguments)
    else:
        parser.print_help()
