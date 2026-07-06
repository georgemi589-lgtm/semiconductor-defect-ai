# api/main.py
# Purpose: Main FastAPI application entry point
# This is the AI brain connector — it receives images,
# runs them through our trained YOLOv8 model,
# and returns real predictions to the dashboard.

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import uuid
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from api.routes.predict import router as predict_router
from api.routes.health import router as health_router
from api.routes.metrics import router as metrics_router

# ─────────────────────────────────────────────
# Initialize FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="Semiconductor Defect Detection API",
    description="""
    Enterprise AI-powered semiconductor defect 
    detection system.
    
    Built for the MIPHI Program by George
    CUBE AI Solutions | 2026
    """,
    version="1.0.0",
    docs_url="/docs",       # Swagger UI at /docs
    redoc_url="/redoc"      # ReDoc at /redoc
)

# ─────────────────────────────────────────────
# CORS Middleware
# Allows Streamlit dashboard to call this API
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Include Routers
# ─────────────────────────────────────────────
app.include_router(health_router, prefix="/api/v1", tags=["Health"])
app.include_router(predict_router, prefix="/api/v1", tags=["Prediction"])
app.include_router(metrics_router, prefix="/api/v1", tags=["Metrics"])

# ─────────────────────────────────────────────
# Root endpoint
# ─────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "Semiconductor Defect Detection API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs",
        "program": "MIPHI Program 2026",
        "organization": "CUBE AI Solutions"
    }

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )