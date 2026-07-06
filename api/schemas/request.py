# api/schemas/request.py
# Purpose: Input validation models using Pydantic
# Pydantic automatically validates incoming request data
# and gives clear error messages for invalid inputs

from pydantic import BaseModel
from typing import Optional

class PredictionRequest(BaseModel):
    """Schema for prediction request metadata."""
    inspection_id: Optional[str] = None
    operator_id: Optional[str] = None
    lot_number: Optional[str] = None
    notes: Optional[str] = None