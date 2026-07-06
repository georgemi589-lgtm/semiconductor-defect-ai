# api/routes/metrics.py
# Purpose: Return model performance metrics
# Dashboard reads from this to show real accuracy numbers

from fastapi import APIRouter, HTTPException
from pathlib import Path
import json

router = APIRouter()

@router.get("/metrics")
async def get_metrics():
    """
    Return model performance metrics.
    Reads from models/model_metrics.json
    """
    metrics_path = Path("models/model_metrics.json")
    
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model metrics not found. Run export_model_metrics.py first."
        )
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return {
        "status": "success",
        "metrics": metrics
    }