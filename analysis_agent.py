from tools.loader import read_csv
from tools.missing_values import get_missing_values
from tools.duplicates import get_duplicate_info
from tools.statistics import get_statistics
from tools.feature_types import get_feature_types
from state import State

def analysis_agent(state: State):
    df = read_csv(state["dataset_path"])

    missing = get_missing_values(df)

    duplicates = get_duplicate_info(df)

    statistics = get_statistics(df)

    features = get_feature_types(df)

    analysis = {
        **missing,
        **duplicates,
        **statistics,
        **features
    }

    state["analysis"] = analysis

    return state