from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CalculateOptions(BaseModel):
    round_mode: str = Field(default="half_up")
    precision: int = Field(default=2, ge=0, le=8)


class CalculateContext(BaseModel):
    source_file: str = ""
    source_chunk_id: Optional[int] = None
    source_rule: str = ""


class CalculateRequest(BaseModel):
    request_id: str = Field(..., min_length=1)
    formula_id: str = Field(..., min_length=1)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    options: CalculateOptions = Field(default_factory=CalculateOptions)
    context: CalculateContext = Field(default_factory=CalculateContext)


class ErrorPayload(BaseModel):
    code: str
    message: str


class CalculateResponse(BaseModel):
    ok: bool
    formula_id: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    result: Dict[str, Any] = Field(default_factory=dict)
    formula_text: str = ""
    steps: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    trace: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[ErrorPayload] = None


class FormulaMeta(BaseModel):
    formula_id: str
    category: str
    required_inputs: List[str]
    description: str

