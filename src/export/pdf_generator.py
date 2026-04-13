"""
PDF Report Generator for AI Study Coach reports.
Uses fpdf2 to create professional, formatted PDF study coaching reports.
"""

import io
from fpdf import FPDF
from src.models.schemas import StudyCoachReport, StudentProfile


class StudyCoachPDF(FPDF):
    """Custom PDF class for study coaching reports."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 58, 138)  # Dark blue
        self.cell(0, 10, "VectoSpace - AI Study Coach Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 58, 138)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str, icon: str = ""):
        self.ln(4)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 58, 138)
        display_title = f"{icon}  {title}" if icon else title
        self.cell(0, 8, display_title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        # Handle encoding — replace characters that Latin-1 can't handle
        safe_text = text.encode("latin-1", errors="replace").decode("latin-1")
        self.set_x(10)  # Reset to left margin
        self.multi_cell(0, 5, safe_text)
        self.ln(2)

    def bullet_item(self, text: str, indent: int = 10):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self.set_x(10 + indent)  # Reset to left margin + indent
        safe_text = text.encode("latin-1", errors="replace").decode("latin-1")
        self.cell(4, 5, chr(149), new_x="END")  # bullet character
        self.multi_cell(0, 5, f"  {safe_text}")
        self.ln(1)

    def colored_badge(self, text: str, r: int, g: int, b: int):
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 10)
        w = self.get_string_width(text) + 10
        self.cell(w, 7, text, fill=True, align="C", new_x="END")
        self.ln(6)


def _risk_color(risk_level: str) -> tuple[int, int, int]:
    """Get color for risk level badge."""
    colors = {
        "At-Risk": (220, 38, 38),       # Red
        "Below-Average": (234, 88, 12),  # Orange
        "Average": (202, 138, 4),        # Yellow
        "Above-Average": (22, 163, 74),  # Green
        "High-Performing": (21, 128, 61),# Dark Green
        "Exceptional": (79, 70, 229),    # Purple
    }
    return colors.get(risk_level, (107, 114, 128))  # Gray default


def generate_pdf(report: StudyCoachReport, student_profile: StudentProfile) -> bytes:
    """
    Generate a PDF report from a StudyCoachReport.

    Args:
        report: The AI-generated coaching report
        student_profile: Student profile data

    Returns:
        PDF bytes that can be downloaded
    """
    pdf = StudyCoachPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # ─── Student Info ────────────────────────────────────────────────
    pdf.section_title("Student Profile", "")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)

    profile_lines = []
    if student_profile.age:
        profile_lines.append(f"Age: {student_profile.age}")
    if student_profile.gender:
        profile_lines.append(f"Gender: {student_profile.gender}")
    if student_profile.school_type:
        profile_lines.append(f"School: {student_profile.school_type}")
    if student_profile.predicted_grade:
        profile_lines.append(f"Predicted Grade: {student_profile.predicted_grade}")

    if profile_lines:
        pdf.body_text(" | ".join(profile_lines))

    # Score summary
    scores = []
    if student_profile.math_score is not None:
        scores.append(f"Math: {student_profile.math_score}")
    if student_profile.science_score is not None:
        scores.append(f"Science: {student_profile.science_score}")
    if student_profile.english_score is not None:
        scores.append(f"English: {student_profile.english_score}")
    if student_profile.overall_score is not None:
        scores.append(f"Overall: {student_profile.overall_score}")
    if scores:
        pdf.body_text("Scores: " + " | ".join(scores))

    # Risk level badge
    r, g, b = _risk_color(report.risk_level)
    pdf.colored_badge(f"  {report.risk_level}  ", r, g, b)
    pdf.ln(4)

    # ─── Learning Diagnosis ──────────────────────────────────────────
    pdf.section_title("Learning Diagnosis", "")
    pdf.body_text(report.learning_diagnosis)

    # ─── Strengths & Weaknesses ──────────────────────────────────────
    pdf.section_title("Key Strengths", "")
    for strength in report.key_strengths:
        pdf.bullet_item(strength)

    pdf.section_title("Key Weaknesses", "")
    for weakness in report.key_weaknesses:
        pdf.bullet_item(weakness)

    # ─── Study Plan ──────────────────────────────────────────────────
    pdf.section_title("Personalized Study Plan", "")
    for item in report.study_plan:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 58, 138)
        safe_subject = item.subject.encode("latin-1", errors="replace").decode("latin-1")
        pdf.cell(0, 6, f"Priority {item.priority}: {safe_subject}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        safe_action = item.action.encode("latin-1", errors="replace").decode("latin-1")
        safe_technique = item.technique.encode("latin-1", errors="replace").decode("latin-1")
        pdf.set_x(10)
        pdf.multi_cell(0, 5, f"  Action: {safe_action}")
        pdf.set_x(10)
        pdf.multi_cell(0, 5, f"  Duration: {item.duration} | Technique: {safe_technique}")
        pdf.ln(3)

    # ─── Weekly Goals ────────────────────────────────────────────────
    pdf.section_title("Weekly Goals", "")
    for i, goal in enumerate(report.weekly_goals, 1):
        pdf.bullet_item(f"Goal {i}: {goal}")

    # ─── Resources ───────────────────────────────────────────────────
    if report.resources:
        pdf.section_title("Recommended Resources", "")
        for resource in report.resources:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 58, 138)
            safe_title = resource.title.encode("latin-1", errors="replace").decode("latin-1")
            pdf.cell(0, 6, safe_title, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(80, 80, 80)
            safe_url = resource.url.encode("latin-1", errors="replace").decode("latin-1")
            safe_why = resource.why.encode("latin-1", errors="replace").decode("latin-1")
            pdf.cell(0, 5, f"  {safe_url}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(50, 50, 50)
            pdf.set_x(10)
            pdf.multi_cell(0, 5, f"  Subject: {resource.subject} | {safe_why}")
            pdf.ln(2)

    # ─── Motivational Note ───────────────────────────────────────────
    pdf.section_title("Motivational Note", "")
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(22, 163, 74)  # Green
    safe_note = report.motivational_note.encode("latin-1", errors="replace").decode("latin-1")
    pdf.set_x(10)
    pdf.multi_cell(0, 6, f'"{safe_note}"')

    # Output
    return bytes(pdf.output())
