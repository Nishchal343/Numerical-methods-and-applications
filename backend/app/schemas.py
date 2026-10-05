from typing import Any, Optional
from pydantic import BaseModel, Field

class SolveRequest(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)

class RecommendationRequest(BaseModel):
    problem: str
    compatible_methods: list[str] = Field(default_factory=list)

class Recommendation(BaseModel):
    recommended_method: str
    reason: str
    alternatives: list[str] = []

class ParseRequest(BaseModel):
    text: str

class ParseResponse(BaseModel):
    problem_type: str
    detected_text: str
    compatible_methods: list[str]
    confidence: str
