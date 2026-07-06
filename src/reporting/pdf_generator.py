# src/reporting/pdf_generator.py
# Purpose: Generate professional PDF inspection reports
# Every wafer inspection can be exported as a PDF
# for factory quality records and compliance

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
from pathlib import Path
import os


# ─────────────────────────────────────────────
# Brand Colors
# ─────────────────────────────────────────────
DARK_BLUE = colors.HexColor('#0a1628')
ACCENT_BLUE = colors.HexColor('#00d4ff')
SUCCESS_GREEN = colors.HexColor('#2ecc71')
DANGER_RED = colors.HexColor('#e74c3c')
WARNING_ORANGE = colors.HexColor('#f39c12')
LIGHT_GRAY = colors.HexColor('#f5f6fa')
MID_GRAY = colors.HexColor('#636e72')


def generate_inspection_report(
    inspection_data: dict,
    output_path: str = None
) -> str:
    """
    Generate a professional PDF inspection report.
    
    Args:
        inspection_data: Dictionary containing inspection results
        output_path: Where to save the PDF (auto-generated if None)
    
    Returns:
        Path to the generated PDF file
    """
    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Auto-generate filename if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"inspection_report_{timestamp}.pdf"
        output_path = str(reports_dir / filename)
    
    # Create PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    # Build content
    story = []
    styles = getSampleStyleSheet()
    
    # ─────────────────────────────────────────────
    # Custom Styles
    # ─────────────────────────────────────────────
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=22,
        textColor=DARK_BLUE,
        spaceAfter=5,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=MID_GRAY,
        spaceAfter=20,
        alignment=TA_CENTER
    )
    
    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontSize=13,
        textColor=DARK_BLUE,
        spaceBefore=15,
        spaceAfter=8,
        fontName='Helvetica-Bold',
        borderPad=5
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.black,
        spaceAfter=5
    )
    
    # ─────────────────────────────────────────────
    # HEADER
    # ─────────────────────────────────────────────
    story.append(Paragraph(
        "🔬 SEMICONDUCTOR DEFECT INSPECTION REPORT",
        title_style
    ))
    story.append(Paragraph(
        "Enterprise AI-Powered Quality Inspection System | MIPHI Program | CUBE AI Solutions",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT_BLUE))
    story.append(Spacer(1, 0.3*inch))
    
    # ─────────────────────────────────────────────
    # INSPECTION SUMMARY
    # ─────────────────────────────────────────────
    story.append(Paragraph("📋 INSPECTION SUMMARY", section_header_style))
    
    # Determine result color
    is_pass = inspection_data.get('pass_fail', 'FAIL') == 'PASS'
    result_color = SUCCESS_GREEN if is_pass else DANGER_RED
    result_text = "✅ PASS — No significant defect detected" if is_pass else f"❌ FAIL — {inspection_data.get('defect_type', 'Unknown')} defect detected"
    
    # Result banner
    result_table = Table(
        [[result_text]],
        colWidths=[17*cm]
    )
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), SUCCESS_GREEN if is_pass else DANGER_RED),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 14),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [SUCCESS_GREEN if is_pass else DANGER_RED]),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('ROUNDEDCORNERS', [5, 5, 5, 5]),
    ]))
    
    story.append(result_table)
    story.append(Spacer(1, 0.2*inch))
    
    # ─────────────────────────────────────────────
    # PREDICTION DETAILS TABLE
    # ─────────────────────────────────────────────
    story.append(Paragraph("🎯 PREDICTION DETAILS", section_header_style))
    
    severity = inspection_data.get('severity', 'UNKNOWN')
    severity_color = {
        'PASS': SUCCESS_GREEN,
        'LOW': colors.HexColor('#27ae60'),
        'MEDIUM': WARNING_ORANGE,
        'HIGH': colors.HexColor('#e67e22'),
        'CRITICAL': DANGER_RED
    }.get(severity, MID_GRAY)
    
    details_data = [
        ['Field', 'Value'],
        ['Filename', inspection_data.get('filename', 'N/A')],
        ['Defect Classification', inspection_data.get('defect_type', 'N/A')],
        ['Confidence Score', f"{inspection_data.get('confidence', 0)*100:.2f}%"],
        ['Severity Level', severity],
        ['Inspection Result', inspection_data.get('pass_fail', 'N/A')],
        ['Inference Time', f"{inspection_data.get('inference_time', 0):.3f} seconds"],
        ['AI Model', inspection_data.get('model', 'YOLOv8n-cls')],
        ['Model Accuracy', '92.05% (on 9,370 test images)'],
        ['Inspection Timestamp', inspection_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))],
    ]
    
    details_table = Table(details_data, colWidths=[6*cm, 11*cm])
    details_table.setStyle(TableStyle([
        # Header row
        ('BACKGROUND', (0,0), (-1,0), DARK_BLUE),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 11),
        
        # Data rows - alternating colors
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT_GRAY, colors.white]),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,1), (-1,-1), 10),
        ('TEXTCOLOR', (0,1), (0,-1), DARK_BLUE),
        
        # Grid
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dfe6e9')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    
    story.append(details_table)
    story.append(Spacer(1, 0.3*inch))
    
    # ─────────────────────────────────────────────
    # DEFECT DESCRIPTION
    # ─────────────────────────────────────────────
    story.append(Paragraph("📝 DEFECT DESCRIPTION", section_header_style))
    story.append(Paragraph(
        inspection_data.get('description', 'No description available.'),
        normal_style
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # ─────────────────────────────────────────────
    # DEFECT REFERENCE TABLE
    # ─────────────────────────────────────────────
    story.append(Paragraph("📊 DEFECT SEVERITY REFERENCE", section_header_style))
    
    severity_ref = [
        ['Defect Type', 'Severity', 'Action Required'],
        ['none', 'PASS', 'No action needed — wafer approved'],
        ['Random', 'LOW', 'Monitor — likely equipment noise'],
        ['Loc', 'MEDIUM', 'Investigate contamination source'],
        ['Edge-Loc', 'MEDIUM', 'Check edge handling process'],
        ['Center', 'HIGH', 'Inspect center processing equipment'],
        ['Donut', 'HIGH', 'Check chemical distribution system'],
        ['Edge-Ring', 'HIGH', 'Inspect edge seal and handling'],
        ['Scratch', 'HIGH', 'Review physical handling procedures'],
        ['Near-full', 'CRITICAL', 'Halt production — major equipment failure'],
    ]
    
    severity_table = Table(severity_ref, colWidths=[5*cm, 4*cm, 8*cm])
    severity_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), DARK_BLUE),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT_GRAY, colors.white]),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dfe6e9')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    
    story.append(severity_table)
    story.append(Spacer(1, 0.3*inch))
    
    # ─────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT_BLUE))
    story.append(Spacer(1, 0.1*inch))
    
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=MID_GRAY,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph(
        f"Generated by DefectAI v1.0 | MIPHI Program 2026 | CUBE AI Solutions | "
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        footer_style
    ))
    
    # Build PDF
    doc.build(story)
    print(f"✅ PDF report generated: {output_path}")
    
    return output_path


# Quick test
if __name__ == "__main__":
    test_data = {
        'filename': 'wafer_sample_001.png',
        'defect_type': 'Scratch',
        'confidence': 0.923,
        'severity': 'HIGH',
        'pass_fail': 'FAIL',
        'description': 'Linear scratch detected across wafer surface — likely physical handling damage during transport or processing.',
        'inference_time': 0.187,
        'model': 'YOLOv8n-cls',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    output = generate_inspection_report(test_data)
    print(f"✅ Test report saved: {output}")