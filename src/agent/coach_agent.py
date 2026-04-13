"""
LangGraph Agentic Study Coach Pipeline.

Multi-node state graph that orchestrates:
  START → diagnose → [deep_diagnose if at-risk] → plan → find_resources → generate_report → END

Each node makes LLM calls with engineered prompts and builds up a comprehensive
coaching report using structured Pydantic outputs.
"""

import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from src.agent.prompts import (
    DIAGNOSIS_PROMPT,
    DEEP_DIAGNOSIS_PROMPT,
    STUDY_PLAN_PROMPT,
    RESOURCE_CURATION_PROMPT,
    REPORT_GENERATION_PROMPT,
    CHAT_FOLLOWUP_PROMPT,
)
from src.models.schemas import StudyCoachReport, StudentProfile, StudyPlanItem, Resource
from src.rag.knowledge_base import initialize_knowledge_base, search_resources_for_student


# ─── STATE DEFINITION ────────────────────────────────────────────────────────

class CoachState(TypedDict):
    """State that flows through the coaching agent graph."""
    student_summary: str          # Formatted student data string
    student_data: dict            # Raw student data dict
    predicted_grade: str          # From Random Forest model
    classification: str           # At-Risk, Average, etc.
    diagnosis: str                # LLM-generated diagnosis
    is_at_risk: bool              # Flag for conditional routing
    study_plan: str               # JSON string of study plan items
    weekly_goals: str             # JSON string of weekly goals
    resources: str                # JSON string of resource recommendations
    final_report: str             # JSON string of final StudyCoachReport


# ─── LLM INITIALIZATION ─────────────────────────────────────────────────────

def get_llm(api_key: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.3):
    """Initialize the Groq LLM."""
    return ChatGroq(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=4096,
    )


# ─── AGENT NODES ─────────────────────────────────────────────────────────────

def diagnose_node(state: CoachState, llm) -> dict:
    """
    Node 1: Analyze student data and produce a learning diagnosis.
    Determines if student is at-risk for conditional routing.
    """
    chain = DIAGNOSIS_PROMPT | llm
    result = chain.invoke({
        "student_summary": state["student_summary"],
    })

    classification = state.get("classification", "")
    is_at_risk = classification in ("At-Risk", "Below-Average")

    return {
        "diagnosis": result.content,
        "is_at_risk": is_at_risk,
    }


def deep_diagnose_node(state: CoachState, llm) -> dict:
    """
    Node 1b (conditional): Deeper analysis for at-risk students.
    Only triggered when classification is At-Risk or Below-Average.
    """
    chain = DEEP_DIAGNOSIS_PROMPT | llm
    result = chain.invoke({
        "student_summary": state["student_summary"],
        "classification": state["classification"],
        "initial_diagnosis": state["diagnosis"],
    })

    # Append deep diagnosis to existing diagnosis
    combined_diagnosis = (
        state["diagnosis"] +
        "\n\n--- DEEP ANALYSIS (At-Risk Intervention) ---\n\n" +
        result.content
    )

    return {"diagnosis": combined_diagnosis}


def plan_node(state: CoachState, llm) -> dict:
    """
    Node 2: Create a personalized study plan based on the diagnosis.
    """
    chain = STUDY_PLAN_PROMPT | llm
    result = chain.invoke({
        "student_summary": state["student_summary"],
        "diagnosis": state["diagnosis"],
    })

    # Parse the LLM output into structured plan and goals
    content = result.content

    # Extract study plan items and weekly goals from the response
    return {
        "study_plan": content,
        "weekly_goals": content,  # Will be parsed in report generation
    }


def find_resources_node(state: CoachState, collection) -> dict:
    """
    Node 3: Use RAG to find relevant educational resources.
    Searches ChromaDB based on student's weak subjects and classification.
    """
    student_data = state.get("student_data", {})
    classification = state.get("classification", "Average")

    # Determine weak subjects from scores
    weak_subjects = []
    if student_data.get("math_score", 100) < 50:
        weak_subjects.append("Math")
    if student_data.get("science_score", 100) < 50:
        weak_subjects.append("Science")
    if student_data.get("english_score", 100) < 50:
        weak_subjects.append("English")

    # If no weak subjects identified, add general study skills
    if not weak_subjects:
        weak_subjects = ["Study Skills", "General"]

    has_internet = bool(student_data.get("internet_access", 1))

    # RAG search
    resources = search_resources_for_student(
        collection=collection,
        weak_subjects=weak_subjects,
        classification=classification,
        has_internet=has_internet,
    )

    # Format resources as readable text
    resource_text = ""
    for i, r in enumerate(resources, 1):
        resource_text += f"\n{i}. {r['title']} ({r['subject']}, {r['level']})\n"
        resource_text += f"   URL: {r['url']}\n"
        resource_text += f"   {r.get('description', '')}\n"

    return {"resources": resource_text}


def generate_report_node(state: CoachState, llm) -> dict:
    """
    Node 4: Compile everything into a final structured StudyCoachReport.
    Uses structured output to guarantee valid JSON matching the Pydantic schema.
    """
    structured_llm = llm.with_structured_output(StudyCoachReport)

    prompt = REPORT_GENERATION_PROMPT
    result = structured_llm.invoke(
        prompt.format_messages(
            student_summary=state["student_summary"],
            diagnosis=state["diagnosis"],
            study_plan=state["study_plan"],
            weekly_goals=state["weekly_goals"],
            resources=state["resources"],
        )
    )

    return {"final_report": result.model_dump_json()}


# ─── CONDITIONAL ROUTING ─────────────────────────────────────────────────────

def route_after_diagnosis(state: CoachState) -> str:
    """Route to deep diagnosis if student is at-risk, otherwise go to plan."""
    if state.get("is_at_risk", False):
        return "deep_diagnose"
    return "plan"


# ─── GRAPH BUILDER ───────────────────────────────────────────────────────────

def build_coach_graph(api_key: str, model: str = "llama-3.3-70b-versatile"):
    """
    Build and compile the LangGraph coaching agent.

    Returns:
        Compiled graph application ready for .invoke()
    """
    llm = get_llm(api_key, model)

    # Initialize RAG knowledge base
    collection = initialize_knowledge_base()

    # Create the state graph
    graph = StateGraph(CoachState)

    # Add nodes (wrapping to inject llm/collection dependencies)
    graph.add_node("diagnose", lambda state: diagnose_node(state, llm))
    graph.add_node("deep_diagnose", lambda state: deep_diagnose_node(state, llm))
    graph.add_node("plan", lambda state: plan_node(state, llm))
    graph.add_node("find_resources", lambda state: find_resources_node(state, collection))
    graph.add_node("generate_report", lambda state: generate_report_node(state, llm))

    # Add edges
    graph.add_edge(START, "diagnose")
    graph.add_conditional_edges("diagnose", route_after_diagnosis, {
        "deep_diagnose": "deep_diagnose",
        "plan": "plan",
    })
    graph.add_edge("deep_diagnose", "plan")
    graph.add_edge("plan", "find_resources")
    graph.add_edge("find_resources", "generate_report")
    graph.add_edge("generate_report", END)

    # Compile
    app = graph.compile()
    return app


# ─── CONVENIENCE FUNCTIONS ───────────────────────────────────────────────────

def run_coaching_pipeline(
    api_key: str,
    student_profile: StudentProfile,
    model: str = "llama-3.3-70b-versatile",
) -> StudyCoachReport:
    """
    Run the full coaching pipeline for a single student.

    Args:
        api_key: Groq API key
        student_profile: StudentProfile with all available data
        model: LLM model name

    Returns:
        StudyCoachReport with complete coaching analysis
    """
    app = build_coach_graph(api_key, model)

    # Build initial state
    initial_state: CoachState = {
        "student_summary": student_profile.to_summary_string(),
        "student_data": {
            "math_score": student_profile.math_score,
            "science_score": student_profile.science_score,
            "english_score": student_profile.english_score,
            "attendance_percentage": student_profile.attendance_percentage,
            "study_hours": student_profile.study_hours,
            "internet_access": student_profile.internet_access,
        },
        "predicted_grade": student_profile.predicted_grade or "",
        "classification": student_profile.classification or "",
        "diagnosis": "",
        "is_at_risk": False,
        "study_plan": "",
        "weekly_goals": "",
        "resources": "",
        "final_report": "",
    }

    # Run the graph
    result = app.invoke(initial_state)

    # Parse the final report
    report = StudyCoachReport.model_validate_json(result["final_report"])
    return report


def chat_with_coach(
    api_key: str,
    student_profile: StudentProfile,
    report: StudyCoachReport,
    question: str,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    """
    Answer a follow-up question about a student's coaching report.

    Args:
        api_key: Groq API key
        student_profile: StudentProfile
        report: Previously generated StudyCoachReport
        question: User's follow-up question
        model: LLM model name

    Returns:
        Answer string
    """
    llm = get_llm(api_key, model, temperature=0.5)
    chain = CHAT_FOLLOWUP_PROMPT | llm

    study_plan_text = "\n".join(
        f"- [{item.priority}] {item.subject}: {item.action} ({item.duration}, {item.technique})"
        for item in report.study_plan
    )
    goals_text = "\n".join(f"- {g}" for g in report.weekly_goals)

    result = chain.invoke({
        "student_summary": student_profile.to_summary_string(),
        "diagnosis": report.learning_diagnosis,
        "study_plan": study_plan_text,
        "weekly_goals": goals_text,
        "user_question": question,
    })

    return result.content
