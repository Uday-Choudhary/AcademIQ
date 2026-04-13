"""
VectoSpace — AI Study Coach
Enhanced Streamlit application with LangGraph agentic coaching pipeline.

Features:
- CSV upload with ML grade prediction (preserved from Milestone 1)
- AI-powered coaching reports via LangGraph agent
- RAG-based resource recommendations
- PDF export of coaching reports
- Chat interface for follow-up questions
- Graceful fallback to rule-based recommendations if no API key
"""

# ── Suppress noisy warnings from transformers (pulled in by chromadb) ──
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "ml"))

from recommender import generate_recommendations  # Fallback
from src.models.schemas import StudyCoachReport, StudentProfile, StudyPlanItem, Resource
from src.agent.coach_agent import run_coaching_pipeline, chat_with_coach
from src.export.pdf_generator import generate_pdf

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "ml", "models")

GRADE_MAP = {
    0: "Grade 0", 1: "Grade 1", 2: "Grade 2",
    3: "Grade 3", 4: "Grade 4", 5: "Grade 5",
}
CATEGORY_MAP = {
    0: "At-Risk", 1: "Below-Average", 2: "Average",
    3: "Above-Average", 4: "High-Performing", 5: "Exceptional",
}

RISK_COLORS = {
    "At-Risk": "#DC2626",
    "Below-Average": "#EA580C",
    "Average": "#CA8A04",
    "Above-Average": "#16A34A",
    "High-Performing": "#15803D",
    "Exceptional": "#4F46E5",
}


# ─── MODEL LOADING ───────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
        features = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scale_cols.pkl"), "rb") as f:
        scale_cols = pickle.load(f)
    return model, features, scaler, scale_cols


# ─── DATA PREPROCESSING ─────────────────────────────────────────────────────

def preprocess_raw_data(df, scaler, scale_cols):
    df = df.copy()

    if "student_id" in df.columns:
        df.drop(columns=["student_id"], inplace=True)
    if "final_grade" in df.columns:
        df.drop(columns=["final_grade"], inplace=True)

    str_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    binary_map = {"yes": 1, "no": 0}
    for col in ["internet_access", "extra_activities"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map(binary_map).fillna(0).astype(int)

    travel_map = {"<15 min": 0, "15-30 min": 1, "30-60 min": 2, ">60 min": 3}
    if "travel_time" in df.columns and not pd.api.types.is_numeric_dtype(df["travel_time"]):
        df["travel_time"] = df["travel_time"].map(travel_map).fillna(0).astype(int)

    edu_map = {"no formal": 0, "high school": 1, "diploma": 2, "graduate": 3, "post graduate": 4, "phd": 5}
    if "parent_education" in df.columns and not pd.api.types.is_numeric_dtype(df["parent_education"]):
        df["parent_education"] = df["parent_education"].map(edu_map).fillna(0).astype(int)

    nominal_cols = [c for c in ["gender", "school_type", "study_method"] if c in df.columns]
    if nominal_cols:
        df = pd.get_dummies(df, columns=nominal_cols, drop_first=False, dtype=int)

    if scaler is not None and scale_cols:
        cols_to_scale = [c for c in scale_cols if c in df.columns]
        if cols_to_scale:
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df


def align_columns(df, expected_features):
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    return df[expected_features]


def build_student_profile(row, idx, predicted_grade, classification) -> StudentProfile:
    """Build a StudentProfile from a DataFrame row."""
    # Handle internet_access which can be 'yes'/'no' string or 0/1 int
    raw_internet = row.get("internet_access", 1)
    if pd.notna(raw_internet):
        if isinstance(raw_internet, str):
            internet_val = 1 if raw_internet.strip().lower() == "yes" else 0
        else:
            internet_val = int(raw_internet)
    else:
        internet_val = 1

    # Handle extra_activities similarly
    raw_extra = row.get("extra_activities", "")
    if pd.notna(raw_extra):
        extra_str = str(raw_extra).strip().title()
    else:
        extra_str = None

    return StudentProfile(
        student_index=idx,
        age=row.get("age"),
        gender=str(row.get("gender", "")).title() if pd.notna(row.get("gender")) else None,
        school_type=str(row.get("school_type", "")).title() if pd.notna(row.get("school_type")) else None,
        parent_education=str(row.get("parent_education", "")).title() if pd.notna(row.get("parent_education")) else None,
        study_hours=row.get("study_hours"),
        attendance_percentage=row.get("attendance_percentage"),
        internet_access=internet_val,
        travel_time=str(row.get("travel_time", "")) if pd.notna(row.get("travel_time")) else None,
        extra_activities=extra_str,
        study_method=str(row.get("study_method", "")).title() if pd.notna(row.get("study_method")) else None,
        math_score=row.get("math_score"),
        science_score=row.get("science_score"),
        english_score=row.get("english_score"),
        overall_score=row.get("overall_score"),
        predicted_grade=predicted_grade,
        classification=classification,
    )


# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* ── Theme tokens ─────────────────────────────────────────── */
        :root {
            --iq-text-primary:    #1e293b;
            --iq-text-secondary:  #334155;
            --iq-text-muted:      #64748b;
            --iq-bg-card:         #f8fafc;
            --iq-bg-card-alt:     #f1f5f9;
            --iq-bg-metric:       #ffffff;
            --iq-border:          #e2e8f0;
            --iq-link:            #2563eb;
            --iq-chat-user-bg:    #eff6ff;
            --iq-chat-user-bd:    #bfdbfe;
            --iq-chat-ai-bg:      #f0fdf4;
            --iq-chat-ai-bd:      #bbf7d0;
            --iq-shadow:          rgba(0,0,0,0.08);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --iq-text-primary:    #f1f5f9;
                --iq-text-secondary:  #e2e8f0;
                --iq-text-muted:      #94a3b8;
                --iq-bg-card:         #1e293b;
                --iq-bg-card-alt:     #0f172a;
                --iq-bg-metric:       #1e293b;
                --iq-border:          #334155;
                --iq-link:            #60a5fa;
                --iq-chat-user-bg:    rgba(37,99,235,0.18);
                --iq-chat-user-bd:    #3b82f6;
                --iq-chat-ai-bg:      rgba(22,163,74,0.18);
                --iq-chat-ai-bd:      #22c55e;
                --iq-shadow:          rgba(0,0,0,0.3);
            }
        }

        /* ── Global ───────────────────────────────────────────────── */
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* ── Header ───────────────────────────────────────────────── */
        .main-header {
            background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 50%, #db2777 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .sub-header {
            color: var(--iq-text-muted);
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }

        /* ── Section headers ──────────────────────────────────────── */
        .section-header {
            font-size: 1.15rem;
            font-weight: 600;
            color: var(--iq-text-primary) !important;
            margin: 1rem 0 0.5rem 0;
            padding-bottom: 0.3rem;
            border-bottom: 2px solid var(--iq-border);
        }

        /* ── Report card ──────────────────────────────────────────── */
        .report-card {
            background: var(--iq-bg-card);
            border: 1px solid var(--iq-border);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px -1px var(--iq-shadow);
            color: var(--iq-text-secondary) !important;
        }

        /* ── Metric cards ─────────────────────────────────────────── */
        .metric-card {
            background: var(--iq-bg-metric);
            border-radius: 10px;
            padding: 1rem 1.2rem;
            box-shadow: 0 2px 4px var(--iq-shadow);
            border-left: 4px solid;
            margin-bottom: 0.5rem;
            color: var(--iq-text-secondary) !important;
        }
        .metric-card strong { color: var(--iq-text-primary) !important; }
        .metric-card a      { color: var(--iq-link) !important; text-decoration: underline; }

        .metric-card.strength { border-left-color: #16a34a; }
        .metric-card.weakness { border-left-color: #dc2626; }
        .metric-card.plan     { border-left-color: #2563eb; }
        .metric-card.goal     { border-left-color: #7c3aed; }
        .metric-card.resource { border-left-color: #0891b2; }

        /* ── Risk badge ───────────────────────────────────────────── */
        .risk-badge {
            display: inline-block;
            padding: 4px 16px;
            border-radius: 20px;
            color: #ffffff !important;
            font-weight: 600;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
        }

        /* ── Chat messages ────────────────────────────────────────── */
        .chat-message {
            padding: 0.8rem 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            line-height: 1.5;
            color: var(--iq-text-secondary) !important;
        }
        .chat-user {
            background: var(--iq-chat-user-bg);
            border: 1px solid var(--iq-chat-user-bd);
            margin-left: 2rem;
        }
        .chat-assistant {
            background: var(--iq-chat-ai-bg);
            border: 1px solid var(--iq-chat-ai-bd);
            margin-right: 2rem;
        }

        /* ── AI report gradient border ────────────────────────────── */
        .ai-report-container {
            border: 2px solid transparent;
            border-image: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899) 1;
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
        }

        /* ── Sidebar ──────────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #e2e8f0 !important;
        }

        /* ── Buttons ──────────────────────────────────────────────── */
        .stExpander, .stButton > button { transition: all 0.2s ease; }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .stButton > button { color: var(--iq-text-primary) !important; }

        /* ── Agent pipeline steps ─────────────────────────────────── */
        .agent-step {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.3rem 0;
            color: var(--iq-text-muted);
            font-size: 0.9rem;
        }
        .agent-step.active { color: #3b82f6; font-weight: 600; }
        .agent-step.done   { color: #16a34a; }

        /* ── Rule-based rec cards ─────────────────────────────────── */
        .rule-rec-card {
            background: var(--iq-bg-metric);
            border: 1px solid var(--iq-border);
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.5rem;
            color: var(--iq-text-secondary) !important;
        }
        .rule-rec-card strong { color: var(--iq-text-primary) !important; }

        /* ── Pagination info ──────────────────────────────────────── */
        .page-info {
            text-align: center;
            color: var(--iq-text-muted);
        }
        .page-info strong { color: var(--iq-text-primary); }
    </style>
    """, unsafe_allow_html=True)


# ─── UI COMPONENTS ───────────────────────────────────────────────────────────

def render_risk_badge(classification: str):
    color = RISK_COLORS.get(classification, "#64748B")
    st.markdown(
        f'<span class="risk-badge" style="background: {color};">{classification}</span>',
        unsafe_allow_html=True,
    )


def render_report(report: StudyCoachReport, student_profile: StudentProfile, idx: int):
    """Render a full AI coaching report in the Streamlit UI."""

    # Diagnosis
    st.markdown('<div class="section-header">🔍 Learning Diagnosis</div>', unsafe_allow_html=True)
    render_risk_badge(report.risk_level)
    st.markdown(f"<div class='report-card'>{report.learning_diagnosis}</div>", unsafe_allow_html=True)

    # Strengths & Weaknesses
    col_s, col_w = st.columns(2)
    with col_s:
        st.markdown('<div class="section-header">💪 Key Strengths</div>', unsafe_allow_html=True)
        for s in report.key_strengths:
            st.markdown(f'<div class="metric-card strength">✅ {s}</div>', unsafe_allow_html=True)
    with col_w:
        st.markdown('<div class="section-header">⚠️ Key Weaknesses</div>', unsafe_allow_html=True)
        for w in report.key_weaknesses:
            st.markdown(f'<div class="metric-card weakness">🔴 {w}</div>', unsafe_allow_html=True)

    # Study Plan
    st.markdown('<div class="section-header">📋 Personalized Study Plan</div>', unsafe_allow_html=True)
    for item in report.study_plan:
        st.markdown(
            f'<div class="metric-card plan">'
            f'<strong>Priority {item.priority} — {item.subject}</strong><br>'
            f'📌 {item.action}<br>'
            f'⏱️ {item.duration} &nbsp;|&nbsp; 📖 {item.technique}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Weekly Goals
    st.markdown('<div class="section-header">🎯 Weekly Goals</div>', unsafe_allow_html=True)
    for i, goal in enumerate(report.weekly_goals, 1):
        st.markdown(f'<div class="metric-card goal">🏁 Goal {i}: {goal}</div>', unsafe_allow_html=True)

    # Resources
    if report.resources:
        st.markdown('<div class="section-header">📚 Recommended Resources</div>', unsafe_allow_html=True)
        for resource in report.resources:
            st.markdown(
                f'<div class="metric-card resource">'
                f'<strong><a href="{resource.url}" target="_blank">{resource.title}</a></strong> '
                f'({resource.subject})<br>'
                f'💡 {resource.why}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Motivational Note
    st.markdown('<div class="section-header">🌟 Motivational Note</div>', unsafe_allow_html=True)
    st.success(f"💬 *\"{report.motivational_note}\"*")

    # PDF Download
    pdf_bytes = generate_pdf(report, student_profile)
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_bytes,
        file_name=f"study_coach_report_student_{idx + 1}.pdf",
        mime="application/pdf",
        key=f"pdf_download_{idx}",
    )


def render_chat_interface(student_profile: StudentProfile, report: StudyCoachReport, api_key: str, idx: int):
    """Render a chat interface for follow-up questions about a student."""
    st.markdown('<div class="section-header">💬 Ask Follow-up Questions</div>', unsafe_allow_html=True)

    # Initialize chat history for this student
    chat_key = f"chat_history_{idx}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    # Display chat history
    for msg in st.session_state[chat_key]:
        css_class = "chat-user" if msg["role"] == "user" else "chat-assistant"
        icon = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        st.markdown(
            f'<div class="chat-message {css_class}">{icon} {msg["content"]}</div>',
            unsafe_allow_html=True,
        )

    # Chat input
    question = st.text_input(
        "Ask about this student's report...",
        key=f"chat_input_{idx}",
        placeholder="e.g., How should they prepare for exams? What's the most important thing to focus on?",
    )

    if question and st.button("Send", key=f"chat_send_{idx}"):
        st.session_state[chat_key].append({"role": "user", "content": question})

        with st.spinner("🤖 Thinking..."):
            try:
                answer = chat_with_coach(
                    api_key=api_key,
                    student_profile=student_profile,
                    report=report,
                    question=question,
                )
                st.session_state[chat_key].append({"role": "assistant", "content": answer})
            except Exception as e:
                st.session_state[chat_key].append({
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {str(e)}",
                })

        st.rerun()


# ─── MAIN APP ────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="VectoSpace — AI Study Coach",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # ─── Sidebar ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎓 VectoSpace")
        st.markdown("**AI Study Coach**")
        st.markdown("---")

        # API Key
        api_key = os.getenv("GROQ_API_KEY", "")
        api_key_input = st.text_input(
            "🔑 Groq API Key",
            value=api_key,
            type="password",
            help="Get your free key at console.groq.com",
        )
        if api_key_input:
            api_key = api_key_input

        ai_enabled = bool(api_key and api_key != "your-groq-api-key-here")

        if ai_enabled:
            st.success("✅ AI coaching enabled")
        else:
            st.warning("⚠️ No API key — using rule-based recommendations")

        st.markdown("---")

        # Model Selection
        model_choice = st.selectbox(
            "🧠 LLM Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            help="Select the LLM model for AI coaching",
        )

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "VectoSpace uses a Random Forest ML model for grade prediction "
            "and a LangGraph AI agent for personalized study coaching."
        )
        st.markdown(
            "**Tech Stack:** Streamlit, scikit-learn, LangChain, LangGraph, ChromaDB, Groq"
        )

    # ─── Main Content ────────────────────────────────────────────────
    st.markdown('<h1 class="main-header">🎓 VectoSpace — AI Study Coach</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload student data for AI-powered grade predictions, personalized coaching, and study recommendations.</p>',
        unsafe_allow_html=True,
    )

    # ─── File Upload ─────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📂 Upload Student Data (CSV)",
        type=["csv"],
        help="Upload a CSV file with columns like age, gender, scores, attendance, etc.",
    )

    if uploaded_file is None:
        # Landing state
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🤖 AI-Powered Analysis")
            st.markdown("LangGraph agent pipeline diagnoses strengths, weaknesses, and creates personalized study plans.")
        with col2:
            st.markdown("### 📊 ML Grade Prediction")
            st.markdown("Random Forest model predicts grades based on 22 features including scores, attendance, and habits.")
        with col3:
            st.markdown("### 📚 Smart Resources")
            st.markdown("RAG-powered knowledge base recommends real educational resources tailored to each student.")
        return

    # ─── Process Upload ──────────────────────────────────────────────
    raw_df = pd.read_csv(uploaded_file)
    original_df = raw_df.copy()

    # Reset pagination + cached reports when a new file is uploaded
    file_key = f"loaded_file_{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("_last_file_key") != file_key:
        st.session_state["_last_file_key"] = file_key
        st.session_state["current_page"] = 0
        st.session_state["generated_reports"] = {}
        st.session_state["student_profiles"] = {}

    st.subheader("📄 Uploaded Data")
    st.dataframe(raw_df, width="stretch")

    # Load model
    model, feature_names, scaler, scale_cols = load_model()

    # Preprocess
    has_strings = any(not pd.api.types.is_numeric_dtype(raw_df[c]) for c in raw_df.columns)
    if has_strings:
        processed_df = preprocess_raw_data(raw_df, scaler, scale_cols)
    else:
        processed_df = raw_df.copy()
        if "final_grade" in processed_df.columns:
            processed_df.drop(columns=["final_grade"], inplace=True)

    input_df = align_columns(processed_df, feature_names)

    # Predict
    predictions = model.predict(input_df)

    results_df = original_df.copy()
    results_df["Predicted Grade"] = [GRADE_MAP.get(p, f"Grade {p}") for p in predictions]
    results_df["Classification"] = [CATEGORY_MAP.get(p, "Unknown") for p in predictions]

    # ─── Predictions Table ───────────────────────────────────────────
    st.subheader("📊 Predictions & Classifications")
    st.dataframe(results_df, width="stretch")

    # Distribution Charts
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Grade Distribution**")
        st.bar_chart(results_df["Predicted Grade"].value_counts())
    with col2:
        st.markdown("**Classification Distribution**")
        st.bar_chart(results_df["Classification"].value_counts())

    # ─── CSV Download ────────────────────────────────────────────────
    csv_data = results_df.to_csv(index=False)
    st.download_button(
        "📥 Download Predictions CSV",
        csv_data,
        "predictions.csv",
        "text/csv",
    )

    # ─── AI Coaching Reports ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("🤖 AI Study Coaching Reports")

    if ai_enabled:
        st.info("Select a student below to generate their AI coaching report. The agent will diagnose, plan, and recommend resources.")
    else:
        st.warning("Enter your Groq API key in the sidebar to enable AI coaching. Showing rule-based recommendations instead.")

    # Initialize session state for reports
    if "generated_reports" not in st.session_state:
        st.session_state.generated_reports = {}
    if "student_profiles" not in st.session_state:
        st.session_state.student_profiles = {}

    # ─── Pagination ──────────────────────────────────────────────────
    PAGE_SIZE = 10
    total_students = len(results_df)
    total_pages = max(1, (total_students + PAGE_SIZE - 1) // PAGE_SIZE)

    if "current_page" not in st.session_state:
        st.session_state.current_page = 0

    # Clamp page index in case dataset changed
    st.session_state.current_page = min(st.session_state.current_page, total_pages - 1)

    # Page navigation controls
    nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
    with nav_col1:
        if st.button("◀ Previous", disabled=st.session_state.current_page == 0, key="prev_page"):
            st.session_state.current_page -= 1
            st.rerun()
    with nav_col2:
        start_student = st.session_state.current_page * PAGE_SIZE + 1
        end_student = min(start_student + PAGE_SIZE - 1, total_students)
        st.markdown(
            f"<p class='page-info'>Showing students <strong>{start_student}–{end_student}</strong> of <strong>{total_students}</strong> &nbsp;|&nbsp; Page {st.session_state.current_page + 1} of {total_pages}</p>",
            unsafe_allow_html=True,
        )
    with nav_col3:
        if st.button("Next ▶", disabled=st.session_state.current_page >= total_pages - 1, key="next_page"):
            st.session_state.current_page += 1
            st.rerun()

    # Slice for current page only
    page_start = st.session_state.current_page * PAGE_SIZE
    page_end = page_start + PAGE_SIZE
    page_df = results_df.iloc[page_start:page_end]

    for idx, row in page_df.iterrows():
        pred_grade = row["Predicted Grade"]
        classification = row["Classification"]

        with st.expander(f"🎓 Student {idx + 1} — {pred_grade} ({classification})", expanded=False):

            # Build student profile
            profile = build_student_profile(original_df.iloc[idx], idx, pred_grade, classification)
            st.session_state.student_profiles[idx] = profile

            # Show basic info
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                if profile.math_score is not None:
                    st.metric("Math Score", f"{profile.math_score:.1f}")
                if profile.attendance_percentage is not None:
                    st.metric("Attendance", f"{profile.attendance_percentage:.1f}%")
            with info_col2:
                if profile.science_score is not None:
                    st.metric("Science Score", f"{profile.science_score:.1f}")
                if profile.study_hours is not None:
                    st.metric("Study Hours", f"{profile.study_hours:.1f} hrs/week")
            with info_col3:
                if profile.english_score is not None:
                    st.metric("English Score", f"{profile.english_score:.1f}")
                render_risk_badge(classification)

            st.markdown("---")

            if ai_enabled:
                # Check if report already generated
                if idx in st.session_state.generated_reports:
                    report = st.session_state.generated_reports[idx]
                    render_report(report, profile, idx)
                    st.markdown("---")
                    render_chat_interface(profile, report, api_key, idx)
                else:
                    # Generate report button
                    if st.button(f"🤖 Generate AI Coaching Report", key=f"gen_report_{idx}"):
                        with st.spinner("🔄 Running AI coaching pipeline..."):
                            # Show pipeline steps
                            status = st.status("AI Agent Pipeline", expanded=True)
                            status.write("🔍 **Step 1/4:** Diagnosing learning patterns...")

                            try:
                                report = run_coaching_pipeline(
                                    api_key=api_key,
                                    student_profile=profile,
                                    model=model_choice,
                                )

                                status.write("📋 **Step 2/4:** Creating study plan...")
                                status.write("📚 **Step 3/4:** Finding resources (RAG)...")
                                status.write("📝 **Step 4/4:** Compiling final report...")
                                status.update(label="✅ Report generated!", state="complete")

                                # Store in session state
                                st.session_state.generated_reports[idx] = report

                                # Render the report
                                render_report(report, profile, idx)

                            except Exception as e:
                                status.update(label="❌ Error", state="error")
                                st.error(f"Error generating report: {str(e)}")
                                st.info("Falling back to rule-based recommendations...")

                                # Fallback to rule-based
                                student_data = {
                                    "attendance_percentage": row.get("attendance_percentage", 100),
                                    "study_hours": row.get("study_hours", 10),
                                    "math_score": row.get("math_score", 100),
                                    "science_score": row.get("science_score", 100),
                                    "english_score": row.get("english_score", 100),
                                    "internet_access": row.get("internet_access", 1),
                                }
                                predicted_category = f"Grade {predictions[idx]}"
                                recs = generate_recommendations(student_data, predicted_category)
                                for i, r in enumerate(recs, 1):
                                    st.write(f"{i}. {r}")

            else:
                # Rule-based fallback
                st.markdown("**📋 Rule-Based Recommendations:**")
                student_data = {
                    "attendance_percentage": row.get("attendance_percentage", 100),
                    "study_hours": row.get("study_hours", 10),
                    "math_score": row.get("math_score", 100),
                    "science_score": row.get("science_score", 100),
                    "english_score": row.get("english_score", 100),
                    "internet_access": row.get("internet_access", 1),
                }
                predicted_category = f"Grade {predictions[idx]}"
                recs = generate_recommendations(student_data, predicted_category)
                for i, r in enumerate(recs, 1):
                    st.write(f"{i}. {r}")


if __name__ == "__main__":
    main()