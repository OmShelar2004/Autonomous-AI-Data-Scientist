"""Analysis node: profile data and detect the supervised-learning objective."""

from state import State
from tools.dataset_profiler import profile_dataset
from tools.problem_detection import detect_problem_type
from tools.target_detection import detect_target


def analysis_agent(state: State):
    df = state["df"]
    summary = profile_dataset(df)
    target_info = detect_target(df, state.get("target_column"))
    state["dataset_summary"] = summary
    state["target_detection"] = target_info
    state["analysis"] = {"profile": summary, "target_detection": target_info}
    if not target_info["requires_user_selection"]:
        problem = detect_problem_type(df, target_info["target_column"])
        state["target_column"] = target_info["target_column"]
        state["problem_type"] = problem["problem_type"]
        state["analysis"]["problem_detection"] = problem
    return state
