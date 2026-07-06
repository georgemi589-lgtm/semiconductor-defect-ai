# api/schemas/response.py
# Purpose: Output response models
# Defines exactly what format the API returns
# Makes the API contract clear and consistent

from pydantic import BaseModel
from typing import Optional, List

class DefectPrediction(BaseModel):
    """Single defect prediction result."""
    class_name: str
    confidence: float
    confidence_percent: float
    severity: str
    description: str
    pass_fail: str

class PredictionResponse(BaseModel):
    """Full prediction response."""
    status: str
    filename: str
    prediction: DefectPrediction
    inference_time_seconds: float
    model: str
    accuracy_on_test_set: str