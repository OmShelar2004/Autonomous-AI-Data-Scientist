from typing import TypedDict, List, Dict, Any
from pandas import DataFrame


class State(TypedDict, total=False):
    task: str
    dataset_path: str
    df: DataFrame
    dataset_summary: dict
    analysis: dict
    target_column: str
    target_detection: dict
    problem_type: str
    processed_data: dict
    preprocessor: object
    model_results: list
    top_models: list
    selected_model: str
    trained_model: object
    model_path: str
    prediction: list
    plan: List[str]
    result: str
    success: bool
    feedback: str
    retry_count: int
