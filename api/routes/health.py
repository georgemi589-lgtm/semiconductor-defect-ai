# api/routes/health.py
# Purpose: Health check endpoint
# Used by Docker, load balancers, and monitoring
# to verify the API is alive and model is loaded

from fastapi import APIRouter
from pathlib import Path
import torch

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns system status and model availability.
    """
    model_path = Path("models/best.pt")
    model_exists = model_path.exists()
    
    return {
        "status": "healthy",
        "model_loaded": model_exists,
        "model_path": str(model_path),
        "pytorch_version": torch.__version__,
        "api_version": "1.0.0"
    }