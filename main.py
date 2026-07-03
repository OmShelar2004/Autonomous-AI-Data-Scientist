from graph import agent

initial_state = {
    "task": "Analyze the dataset and recommend complete preprocessing and EDA.",
    "dataset_path": "data/Titanic.csv",   
    "dataset_summary": {},
    "plan": [],
    "result": "",
    "success": False,
    "feedback": "",
    "retry_count": 0,
}

response = agent.invoke(initial_state)

print("\n================ FINAL RESULT ================\n")
print(response["result"])

print("\n==============================================")
print("Success :", response["success"])
print("Feedback:", response["feedback"])