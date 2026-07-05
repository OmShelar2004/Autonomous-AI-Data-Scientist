from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END
from state import State
from models import Plan, Review
from prompts import (
    PLANNER_PROMPT,
    EXECUTOR_PROMPT,
    REVIEWER_PROMPT,
)
from tools.loader import read_csv
from tools.missing_values import get_missing_values
from tools.duplicates import get_duplicate_info
from tools.statistics import get_statistics
from tools.feature_types import get_feature_types


load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

planner = llm.with_structured_output(Plan)
reviewer = llm.with_structured_output(Review)





#-----------------------------
#Dataet Loader Node
#-----------------------------

def dataset_loader_node(state: State):
    print("\n Loading dataset.....")
    summary = read_csv(state["dataset_path"])
    state["dataset_summary"] = summary

    print("\nDataset Loaded Successfully!\n")
    print(summary)

    return state


# ----------------------------
# Planner Node
# ----------------------------
def planner_node(state: State):

    response = planner.invoke(
        PLANNER_PROMPT.format(
            task=state["task"],
            dataset_summary=state["dataset_summary"]
        )
    )

    state["plan"] = response.steps

    return state


# ----------------------------
# Executor Node
# ----------------------------
def executor_node(state: State):

    state["retry_count"] += 1

    results = []
    previous_result = ""

    for step in state["plan"]:

        print(f"\nExecuting: {step}")

        response = llm.invoke(
            EXECUTOR_PROMPT.format(
                dataset_summary=state["dataset_summary"],
                previous_result=previous_result,
                feedback=state["feedback"],
                step=step
            )
        )

        print(response.content)

        results.append(response.content)

        previous_result = response.content

    state["result"] = "\n\n".join(results)

    return state


# ----------------------------
# Reviewer Node
# ----------------------------
def reviewer_node(state: State):

    response = reviewer.invoke(
        REVIEWER_PROMPT.format(
            task=state["task"],
            plan=state["plan"],
            result=state["result"]
        )
    )

    state["success"] = response.success
    state["feedback"] = response.feedback

    print("\nReviewer:", state["success"])
    print("Feedback:", state["feedback"])

    return state


# ----------------------------
# Router
# ----------------------------
def router(state: State):

    if state["success"]:
        return END

    if state["retry_count"] >= 3:
        print("Maximum retry attempts reached.")
        return END

    return "executor"