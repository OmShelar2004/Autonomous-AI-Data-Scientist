# 🤖 Autonomous AI Data Scientist

An autonomous AI system capable of performing Data Science tasks using **LangGraph**, **LangChain**, **Groq LLM**, and **Python**.

Instead of relying on a single prompt, this project is being developed as a **multi-agent system** where specialized AI agents collaborate with Python tools to analyze datasets, perform preprocessing, generate EDA reports, recommend machine learning models, and eventually train models autonomously.

---

## 🎯 Vision

The long-term goal is to build an AI that can take **any CSV or Excel dataset** and automatically:

- 📂 Understand the dataset
- 🧹 Perform data preprocessing
- 📊 Generate Exploratory Data Analysis (EDA)
- 📈 Create visualizations
- 🧠 Recommend Machine Learning models
- ⚙️ Train and evaluate models
- 📝 Explain every decision like an experienced Data Scientist

The philosophy behind this project is:

> **Don't make the LLM smarter. Make the tools smarter.**

Python performs the computations, while the LLM focuses on reasoning and explanation.

---

# 🚀 Current Features

- ✅ Multi-Agent Architecture
- ✅ Planner Agent
- ✅ Executor Agent
- ✅ Reviewer Agent
- ✅ Router with Retry Logic
- ✅ Dataset Loader
- ✅ CSV Reading Tool
- ✅ Structured Outputs using Pydantic
- ✅ Modular Project Architecture

---

# 🏗 Project Architecture

```
                User
                  │
                  ▼
         Dataset Loader
                  │
                  ▼
              Planner
                  │
                  ▼
             Executor
                  │
                  ▼
             Reviewer
                  │
          ┌───────┴────────┐
          │                │
      Success          Needs Retry
          │                │
          ▼                │
         END ◄─────────────┘
```

---

# 📂 Project Structure

```
AI_DATA_SCIENTIST/

│
├── data/
│   └── Titanic.csv
│
├── docs/
│   └── Development Journal
│
├── agents.py
├── graph.py
├── main.py
├── models.py
├── prompts.py
├── state.py
├── tools.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🛠 Tech Stack

- Python
- LangGraph
- LangChain
- Groq
- Pandas
- Pydantic
- python-dotenv

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Autonomous-AI-Data-Scientist.git

cd Autonomous-AI-Data-Scientist
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
GROQ_API_KEY=YOUR_API_KEY
```

Run the project

```bash
python main.py
```

---

# 📊 Development Progress

## ✅ Version 0.1

- Planner
- Executor
- Reviewer
- Router
- Retry Mechanism
- Dataset Loader
- CSV Reader
- Dataset Summary

---

# 🗺 Roadmap

## Version 0.2
- Specialized Analysis Tools
- Missing Value Detection
- Duplicate Detection
- Statistical Summary
- Outlier Detection

---

## Version 0.3
- Automated EDA
- Histograms
- Boxplots
- Correlation Heatmaps
- Feature Distribution

---

## Version 0.4
- Automated Data Preprocessing
- Missing Value Imputation
- Encoding
- Scaling
- Outlier Handling

---

## Version 0.5
- Machine Learning Advisor
- Model Recommendation
- Feature Selection
- Metric Recommendation

---

## Version 0.6
- Automatic Model Training
- Hyperparameter Tuning
- Model Comparison

---

## Version 1.0

An Autonomous AI Data Scientist capable of:

- Reading any dataset
- Performing preprocessing
- Generating EDA
- Creating visualizations
- Recommending ML models
- Training models
- Explaining every decision
- Working with minimal human intervention

---

# 📖 What I Learned During This Project

This repository documents my journey of learning:

- Agentic AI
- LangGraph
- Multi-Agent Systems
- Prompt Engineering
- Tool Calling
- Data Science Automation
- Software Architecture
- AI System Design

The project is intentionally built incrementally so every version represents a meaningful engineering milestone.

---

# 🤝 Contributing

This project is currently a personal long-term learning and engineering project. Suggestions, issues, and discussions are always welcome.

---

# ⭐ Future Goal

The ultimate objective is to build an AI system that performs the complete workflow of a junior Data Scientist while explaining every decision transparently.

---

## 📜 License

This project is released under the MIT License.
