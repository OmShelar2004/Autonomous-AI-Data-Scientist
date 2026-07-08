from typing import TypedDict, List, Dict, Any
from pandas import DataFrame


class State(TypedDict):
    task: str
    dataset_path: str
    df: DataFrame
    dataset_summary: dict
    analysis : dict
    plan: List[str]
    result: str
    success: bool
    feedback: str
    retry_count: int