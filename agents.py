"""LangGraph nodes orchestrating deterministic Python execution tools."""

from langgraph.graph import END

from state import State
from tools.loader import read_csv
from tools.model_comparison import compare_models
from tools.model_training import train_candidates


def dataset_loader_node(state: State):
    state["df"] = read_csv(state["dataset_path"])
    return state


def planner_node(state: State):
    if not state.get("target_column"):
        state["plan"] = ["Ask the user to select a target column."]
        return state
    state["plan"] = [
        "Apply leakage-safe feature engineering and preprocessing.",
        "Train lightweight candidate models on a held-out validation split.",
        "Rank actual validation metrics and present two models for user selection.",
    ]
    return state


def executor_node(state: State):
    if not state.get("target_column"):
        state["result"] = "Target selection is required before training."
        return state
    training = train_candidates(state["df"], state["target_column"], state["problem_type"])
    table, top_models = compare_models(training["results"], state["problem_type"])
    state["processed_data"] = training
    state["model_results"] = table.to_dict(orient="records")
    state["top_models"] = top_models
    state["result"] = table.to_string(index=False)
    return state


def reviewer_node(state: State):
    state["retry_count"] = state.get("retry_count", 0) + 1
    state["success"] = bool(state.get("top_models"))
    state["feedback"] = "Real validation metrics are available." if state["success"] else "A target column must be selected before execution."
    return state


def router(state: State):
    return END
