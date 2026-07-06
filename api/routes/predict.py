# api/routes/predict.py
# Purpose: The core prediction endpoint
# Receives an uploaded image, runs it through
# the real YOLOv8 model, returns actual defect prediction

from fastapi import APIRouter, File, UploadFile, HTTPException
from pathlib import Path
import shutil
import uuid
import time
import sys
import os

sys.path.append(str(Path(__file__).parent.parent.parent))

router = APIRouter()

# ─────────────────────────────────────────────
# Load model ONCE when API starts
# (not on every request — that would be very slow)
# ─────────────────────────────────────────────
MODEL_PATH = "models/best.pt"
model = None

def get_model():
    """Load model lazily — only when first prediction is needed."""
    global model
    if model is None:
        try:
            from ultralytics import YOLO
            model = YOLO(MODEL_PATH)
            print(f"✅ Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Model not available: {str(e)}"
            )
    return model

# Class descriptions for each defect type
DEFECT_DESCRIPTIONS = {
    'Center': 'Defects concentrated at wafer center — often contamination during center processing',
    'Donut': 'Ring-shaped defect in center area — typically chemical distribution issue',
    'Edge-Loc': 'Localized defects at wafer edge — handling or edge processing damage',
    'Edge-Ring': 'Ring of defects around entire edge — edge handling or seal issue',
    'Loc': 'Localized cluster of defects — particle contamination',
    'Near-full': 'Almost entire wafer has defects — critical equipment failure',
    'Random': 'Random scattered defect pattern — isolated equipment noise',
    'Scratch': 'Linear scratch across wafer surface — physical handling damage',
    'none': 'No significant defect pattern detected — wafer passes inspection',
}

DEFECT_SEVERITY = {
    'none': 'PASS',
    'Random': 'LOW',
    'Loc': 'MEDIUM',
    'Edge-Loc': 'MEDIUM',
    'Center': 'HIGH',
    'Donut': 'HIGH',
    'Edge-Ring': 'HIGH',
    'Scratch': 'HIGH',
    'Near-full': 'CRITICAL',
}


@router.post("/predict")
async def predict_defect(
    file: UploadFile = File(..., description="Wafer image to inspect")
):
    """
    Run AI defect detection on uploaded wafer image.
    
    Returns:
    - Predicted defect class
    - Confidence score
    - Severity level
    - Defect description
    - Top 3 predictions
    - Inference time
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/bmp", "image/tiff"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Images only."
        )
    
    # Create temp directory
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Save uploaded file with unique name
    temp_filename = f"{uuid.uuid4()}.png"
    temp_path = temp_dir / temp_filename
    
    try:
        # Save file to disk
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load model
        yolo_model = get_model()
        
        # Run inference
        start_time = time.time()
        
        results = yolo_model(
            str(temp_path),
            verbose=False
        )
        
        inference_time = round(time.time() - start_time, 3)
        
        # Parse results
        result = results[0]
        
        # Get top prediction
        probs = result.probs
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        
        # Get class names
        class_names = result.names
        predicted_class = class_names[top1_idx]
        
        # Get top 5 predictions
        top5_indices = probs.top5
        top5_confs = probs.top5conf.tolist()
        
        top5_predictions = [
            {
                "class": class_names[int(idx)],
                "confidence": round(float(conf), 4)
            }
            for idx, conf in zip(top5_indices, top5_confs)
        ]
        
        # Build response
        response = {
            "status": "success",
            "filename": file.filename,
            "prediction": {
                "class": predicted_class,
                "confidence": round(top1_conf, 4),
                "confidence_percent": round(top1_conf * 100, 2),
                "severity": DEFECT_SEVERITY.get(predicted_class, "UNKNOWN"),
                "description": DEFECT_DESCRIPTIONS.get(
                    predicted_class, 
                    "Defect pattern detected"
                ),
                "pass_fail": "PASS" if predicted_class == "none" else "FAIL"
            },
            "top5_predictions": top5_predictions,
            "inference_time_seconds": inference_time,
            "model": "YOLOv8n-cls",
            "accuracy_on_test_set": "92.05%"
        }
        
        return response
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    
    finally:
        # Always clean up temp file
        if temp_path.exists():
            temp_path.unlink()


@router.post("/predict/batch")
async def predict_batch(
    files: list[UploadFile] = File(..., description="Multiple wafer images")
):
    """
    Run batch defect detection on multiple images.
    Returns predictions for all uploaded images.
    """
    if len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 images per batch request"
        )
    
    results = []
    
    for file in files:
        try:
            # Reuse single predict logic
            single_result = await predict_defect(file)
            results.append(single_result)
        except Exception as e:
            results.append({
                "status": "error",
                "filename": file.filename,
                "error": str(e)
            })
    
    # Summary statistics
    total = len(results)
    passed = sum(1 for r in results 
                if r.get('prediction', {}).get('pass_fail') == 'PASS')
    failed = total - passed
    
    return {
        "status": "success",
        "total_inspected": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed/total*100, 1) if total > 0 else 0,
        "results": results
    }