# database/crud.py
# Purpose: All database operations
# CRUD = Create, Read, Update, Delete
# This is the only file that talks to the database

from database.models import Inspection, get_session, create_tables
from datetime import datetime
from typing import List, Optional
import pandas as pd


def save_inspection(
    filename: str,
    defect_type: str,
    confidence: float,
    severity: str,
    pass_fail: str,
    inference_time: float,
    notes: str = None
) -> Inspection:
    """
    Save a new inspection record to database.
    Called every time a prediction is made.
    """
    session = get_session()
    
    try:
        inspection = Inspection(
            filename=filename,
            defect_type=defect_type,
            confidence=confidence,
            severity=severity,
            pass_fail=pass_fail,
            inference_time=inference_time,
            notes=notes,
            timestamp=datetime.utcnow()
        )
        
        session.add(inspection)
        session.commit()
        session.refresh(inspection)
        
        print(f"✅ Saved inspection #{inspection.id}: {defect_type} ({confidence:.1%})")
        return inspection
    
    except Exception as e:
        session.rollback()
        print(f"❌ Failed to save inspection: {e}")
        raise
    
    finally:
        session.close()


def get_all_inspections(limit: int = 100) -> List[dict]:
    """Get all inspection records, most recent first."""
    session = get_session()
    
    try:
        inspections = session.query(Inspection)\
            .order_by(Inspection.timestamp.desc())\
            .limit(limit)\
            .all()
        
        return [
            {
                'id': i.id,
                'filename': i.filename,
                'defect_type': i.defect_type,
                'confidence': round(i.confidence * 100, 1),
                'severity': i.severity,
                'pass_fail': i.pass_fail,
                'inference_time': i.inference_time,
                'timestamp': i.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'model_version': i.model_version
            }
            for i in inspections
        ]
    
    finally:
        session.close()


def get_inspection_stats() -> dict:
    """Get summary statistics for the dashboard."""
    session = get_session()
    
    try:
        total = session.query(Inspection).count()
        
        if total == 0:
            return {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'pass_rate': 0,
                'avg_confidence': 0,
                'most_common_defect': 'N/A'
            }
        
        passed = session.query(Inspection)\
            .filter(Inspection.pass_fail == 'PASS').count()
        failed = total - passed
        
        # Average confidence
        from sqlalchemy import func
        avg_conf = session.query(func.avg(Inspection.confidence)).scalar()
        
        # Most common defect
        from sqlalchemy import func
        most_common = session.query(
            Inspection.defect_type,
            func.count(Inspection.defect_type).label('count')
        ).group_by(Inspection.defect_type)\
         .order_by(func.count(Inspection.defect_type).desc())\
         .first()
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': round(passed / total * 100, 1),
            'avg_confidence': round(avg_conf * 100, 1) if avg_conf else 0,
            'most_common_defect': most_common[0] if most_common else 'N/A'
        }
    
    finally:
        session.close()


def get_inspections_as_dataframe() -> pd.DataFrame:
    """Return all inspections as a pandas DataFrame for charts."""
    inspections = get_all_inspections(limit=500)
    
    if not inspections:
        return pd.DataFrame()
    
    return pd.DataFrame(inspections)