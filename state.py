from typing import TypedDict, List, Dict, Any

class State(TypedDict):
    task: str
    dataset_path: str
    dataset_summary: dict
    analysis : dict
    plan: List[str]
    result: str
    success: bool
    feedback: str
    retry_count: int