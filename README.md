# 🎓 VectoSpace — AI Study Coach

## Overview
VectoSpace is an **AI-powered academic coaching platform** that combines machine learning grade prediction with a LangGraph agentic pipeline to deliver personalized study coaching reports.

It analyzes student data (attendance, scores, study habits) using a **Random Forest ML model** to predict grades, then leverages an **AI agent pipeline** to generate comprehensive coaching reports with learning diagnoses, study plans, and resource recommendations.

## Key Features

### 🤖 AI-Powered Coaching (Milestone 2 — NEW)
- **LangGraph Agent Pipeline**: Multi-node agentic workflow that diagnoses → plans → retrieves resources → generates reports
- **Structured AI Outputs**: Pydantic-enforced report schemas ensure consistent, validated recommendations
- **RAG Knowledge Base**: ChromaDB vector store with 40+ curated educational resources (Khan Academy, MIT OCW, Coursera, etc.)
- **Conditional Routing**: At-Risk students receive deeper analysis with subject-by-subject breakdowns
- **PDF Export**: Download professional coaching reports as formatted PDFs
- **Chat Interface**: Ask follow-up questions about any student's coaching report
- **Smart Fallback**: Gracefully reverts to rule-based recommendations if no API key provided

### 📊 ML Grade Prediction (Milestone 1)
- **CSV Data Upload**: Upload batch student records via CSV
- **Grade Prediction**: Random Forest model predicts grades (Grade 0–5)
- **Classification**: Categorize students (At-Risk → Exceptional)
- **Data Visualization**: Distribution charts for grades and classifications
- **Export Results**: Download predictions as CSV

## Architecture

```
CSV Upload → Random Forest Prediction → LangGraph Agent Pipeline → Structured Report + PDF
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
              Diagnose Node            Study Plan Node           Resources Node
              (LLM analysis)           (actionable plan)         (RAG retrieval)
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              ▼
                                    Final Report (Pydantic Schema)
                                    ┌─────────┼─────────┐
                                    ▼         ▼         ▼
                              Streamlit    PDF Export   Chat
```

## Tech Stack
| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| ML Model | scikit-learn (Random Forest) |
| LLM | Groq (Llama 3.3 70B) — free tier |
| Agent Framework | LangGraph |
| LLM Orchestration | LangChain |
| Vector Store | ChromaDB |
| Embeddings | Default (all-MiniLM-L6-v2) |
| Structured Output | Pydantic v2 |
| PDF Export | fpdf2 |

## Project Structure
```text
VectoSpace/
├── app.py                          # Main Streamlit application
├── .env                            # API keys (gitignored)
├── .env.example                    # Environment template
├── requirements.txt                # Python dependencies
├── src/
│   ├── agent/                      # LangGraph agent pipeline
│   │   ├── coach_agent.py          # Multi-node state graph
│   │   └── prompts.py              # Engineered prompt templates
│   ├── models/                     # Pydantic data schemas
│   │   └── schemas.py              # StudyCoachReport, StudentProfile
│   ├── rag/                        # RAG knowledge base
│   │   ├── knowledge_base.py       # ChromaDB operations
│   │   └── resource_data.py        # Curated resource dataset
│   ├── export/                     # Report export
│   │   └── pdf_generator.py        # PDF report generation
│   └── ml/                         # ML models
│       ├── recommender.py          # Rule-based fallback
│       └── models/                 # Trained model files
├── datasets/                       # Student data CSVs
└── notebooks/                      # Analysis notebooks
```

## Setup & Installation

### 1. Clone & Install
```bash
git clone https://github.com/your-username/VectoSpace.git
cd VectoSpace
pip install -r requirements.txt
```

### 2. Get API Key (Free)
1. Go to [console.groq.com](https://console.groq.com)
2. Create a free account
3. Generate an API key

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 4. Run
```bash
streamlit run app.py
```

## Usage

1. **Upload CSV**: Upload student data with columns like `age`, `math_score`, `science_score`, `english_score`, `attendance_percentage`, `study_hours`, etc.
2. **View Predictions**: See ML-predicted grades and classifications for all students
3. **Generate AI Report**: Click "Generate AI Coaching Report" for any student
4. **Review Report**: Read the diagnosis, study plan, goals, and recommended resources
5. **Download PDF**: Export the coaching report as a professional PDF
6. **Ask Questions**: Use the chat interface for follow-up questions about the student

## How the AI Agent Works

The LangGraph agent executes a 4-step pipeline:

1. **Diagnose** → LLM analyzes student data and identifies strengths/weaknesses
   - *Conditional*: At-Risk students get a deeper subject-by-subject analysis
2. **Plan** → Creates a prioritized, time-bound study plan with specific techniques
3. **Find Resources** → RAG searches ChromaDB for relevant educational resources
4. **Generate Report** → Compiles everything into a Pydantic-validated structured report

## License
MIT
