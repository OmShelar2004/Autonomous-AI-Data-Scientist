from langgraph.graph import StateGraph, START
from analysis_agent import analysis_agent
from state import State
from agents import (
    planner_node,
    executor_node,
    reviewer_node,
    router,
    dataset_loader_node
)

graph = StateGraph(State)

graph.add_node("dataset_loader", dataset_loader_node)
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("reviewer", reviewer_node)
graph.add_node("analysis", analysis_agent)


graph.add_edge(START, "dataset_loader")
graph.add_edge("dataset_loader", "analysis")
graph.add_edge("analysis", "planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor","reviewer")

graph.add_conditional_edges(
    "reviewer",
    router
)

agent = graph.compile()