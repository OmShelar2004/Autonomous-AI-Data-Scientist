"""Optional presentation prompts.

The execution graph does not use these prompts for numerical work. They are kept as
generic templates for a future explanation layer.
"""

PLANNER_PROMPT = """
You are a data-science workflow planner. Use only the supplied dataset facts.
Task: {task}
Dataset summary: {dataset_summary}
Analysis: {analysis}
Return a concise plan. Never claim calculations or training that Python has not run.
"""

EXECUTOR_PROMPT = """
Summarize this completed Python execution step without inventing values.
Task: {task}
Dataset summary: {dataset_summary}
Analysis: {analysis}
Previous result: {previous_result}
Feedback: {feedback}
Step: {step}
"""

REVIEWER_PROMPT = """
Review the supplied data-science result against the supplied evidence only.
Task: {task}
Dataset summary: {dataset_summary}
Analysis: {analysis}
Plan: {plan}
Result: {result}
Do not infer or fabricate metric values.
"""
