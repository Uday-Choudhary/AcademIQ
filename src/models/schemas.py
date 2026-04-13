"""
Pydantic data models for the AI Study Coach structured outputs.
These schemas ensure the LLM returns consistent, validated JSON.
"""

from pydantic import BaseModel, Field
from typing import Optional


class StudyPlanItem(BaseModel):
    """A single step in the personalized study plan."""
    priority: int = Field(description="Priority order (1 = highest)")
    subject: str = Field(description="Subject area (e.g., Math, Science, English, General)")
    action: str = Field(description="Specific actionable study task")
    duration: str = Field(description="Recommended time to spend (e.g., '30 min/day', '2 hrs/week')")
    technique: str = Field(description="Study technique to use (e.g., 'Spaced Repetition', 'Practice Problems', 'Active Recall')")


class Resource(BaseModel):
    """A recommended learning resource."""
    title: str = Field(description="Resource name")
    url: str = Field(description="Resource URL")
    subject: str = Field(description="Subject area it covers")
    why: str = Field(description="Why this resource is recommended for this student")


class StudyCoachReport(BaseModel):
    """Complete AI-generated study coaching report for a student."""
    learning_diagnosis: str = Field(
        description="Comprehensive analysis of the student's academic strengths and weaknesses based on their data"
    )
    risk_level: str = Field(
        description="Student risk classification: 'At-Risk', 'Below-Average', 'Average', 'Above-Average', 'High-Performing', or 'Exceptional'"
    )
    key_strengths: list[str] = Field(
        description="List of identified academic strengths (e.g., 'Strong science performance', 'Consistent attendance')"
    )
    key_weaknesses: list[str] = Field(
        description="List of identified academic weaknesses that need improvement"
    )
    study_plan: list[StudyPlanItem] = Field(
        description="Ordered, multi-step personalized study plan with specific actions"
    )
    weekly_goals: list[str] = Field(
        description="3-5 concrete, measurable weekly milestone goals"
    )
    resources: list[Resource] = Field(
        default_factory=list,
        description="Recommended learning resources with URLs"
    )
    motivational_note: str = Field(
        description="A brief encouraging and motivational message tailored to the student's situation"
    )


class StudentProfile(BaseModel):
    """Cleaned student data for passing between agent nodes."""
    student_index: int = Field(description="Student row index in the uploaded data")
    age: Optional[float] = None
    gender: Optional[str] = None
    school_type: Optional[str] = None
    parent_education: Optional[str] = None
    study_hours: Optional[float] = None
    attendance_percentage: Optional[float] = None
    internet_access: Optional[int] = None
    travel_time: Optional[str] = None
    extra_activities: Optional[str] = None
    study_method: Optional[str] = None
    math_score: Optional[float] = None
    science_score: Optional[float] = None
    english_score: Optional[float] = None
    overall_score: Optional[float] = None
    predicted_grade: Optional[str] = None
    classification: Optional[str] = None

    def to_summary_string(self) -> str:
        """Convert student data to a human-readable summary for the LLM."""
        lines = []
        if self.age is not None:
            lines.append(f"Age: {self.age}")
        if self.gender:
            lines.append(f"Gender: {self.gender}")
        if self.school_type:
            lines.append(f"School Type: {self.school_type}")
        if self.parent_education:
            lines.append(f"Parent Education: {self.parent_education}")
        if self.study_hours is not None:
            lines.append(f"Study Hours/week: {self.study_hours}")
        if self.attendance_percentage is not None:
            lines.append(f"Attendance: {self.attendance_percentage}%")
        if self.internet_access is not None:
            lines.append(f"Internet Access: {'Yes' if self.internet_access else 'No'}")
        if self.travel_time:
            lines.append(f"Travel Time: {self.travel_time}")
        if self.extra_activities:
            lines.append(f"Extra Activities: {self.extra_activities}")
        if self.study_method:
            lines.append(f"Study Method: {self.study_method}")
        if self.math_score is not None:
            lines.append(f"Math Score: {self.math_score}/100")
        if self.science_score is not None:
            lines.append(f"Science Score: {self.science_score}/100")
        if self.english_score is not None:
            lines.append(f"English Score: {self.english_score}/100")
        if self.overall_score is not None:
            lines.append(f"Overall Score: {self.overall_score}/100")
        if self.predicted_grade:
            lines.append(f"ML Predicted Grade: {self.predicted_grade}")
        if self.classification:
            lines.append(f"Classification: {self.classification}")
        return "\n".join(lines)
