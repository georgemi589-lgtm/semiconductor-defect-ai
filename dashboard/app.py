# dashboard/app.py
# Enterprise Semiconductor Defect Detection Dashboard
# Built with Streamlit for MIPHI Program
# CUBE AI Solutions

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from PIL import Image
import io
import time
from pathlib import Path
import json
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Semiconductor Defect Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Professional Dark Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0e1117; }
    
    /* Sidebar styling */
    .css-1d391kg { background-color: #161b22; }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252d3d);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #00d4ff;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Success card */
    .success-card {
        background: linear-gradient(135deg, #0d2b1d, #1a4a2e);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00ff88;
    }
    
    /* Warning card */
    .warning-card {
        background: linear-gradient(135deg, #2b1a0d, #4a2e1a);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff8800;
    }
    
    /* Error card */
    .error-card {
        background: linear-gradient(135deg, #2b0d0d, #4a1a1a);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff4444;
    }
    
    /* Title styling */
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #0099ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px 0;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #00d4ff;
        border-bottom: 2px solid #00d4ff33;
        padding-bottom: 8px;
        margin-bottom: 15px;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA — Defect class information
# ─────────────────────────────────────────────
DEFECT_CLASSES = {
    'none': {'color': '#2ecc71', 'description': 'No defect pattern detected'},
    'Center': {'color': '#e74c3c', 'description': 'Defects concentrated at wafer center'},
    'Donut': {'color': '#f39c12', 'description': 'Ring-shaped defect in center area'},
    'Edge-Loc': {'color': '#9b59b6', 'description': 'Localized defects at wafer edge'},
    'Edge-Ring': {'color': '#e67e22', 'description': 'Ring of defects around entire edge'},
    'Loc': {'color': '#1abc9c', 'description': 'Localized cluster of defects'},
    'Near-full': {'color': '#c0392b', 'description': 'Almost entire wafer has defects'},
    'Random': {'color': '#3498db', 'description': 'Random scattered defect pattern'},
    'Scratch': {'color': '#e91e63', 'description': 'Linear scratch across wafer surface'},
}

DATASET_STATS = {
    'none': 785938,
    'Edge-Ring': 9680,
    'Edge-Loc': 5189,
    'Center': 4294,
    'Loc': 3593,
    'Scratch': 1193,
    'Random': 866,
    'Donut': 555,
    'Near-full': 149
}

BALANCED_STATS = {
    'none': 10000,
    'Edge-Ring': 9680,
    'Edge-Loc': 5189,
    'Center': 8588,
    'Loc': 7186,
    'Scratch': 5965,
    'Random': 5196,
    'Donut': 5550,
    'Near-full': 5066
}


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def simulate_prediction(image):
    """
    Simulate AI prediction for demo purposes.
    In Week 3, this will be replaced with real YOLOv8 inference.
    """
    time.sleep(1.5)  # Simulate processing time
    
    defect_types = list(DEFECT_CLASSES.keys())
    weights = [0.3, 0.1, 0.08, 0.1, 0.12, 0.08, 0.05, 0.1, 0.07]
    chosen = np.random.choice(defect_types, p=weights)
    confidence = round(np.random.uniform(0.72, 0.98), 3)
    
    return {
        'defect_type': chosen,
        'confidence': confidence,
        'description': DEFECT_CLASSES[chosen]['description'],
        'color': DEFECT_CLASSES[chosen]['color'],
        'inference_time': round(np.random.uniform(0.08, 0.25), 3),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def process_uploaded_image(uploaded_file):
    """Convert uploaded file to numpy array."""
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    return image, image_array


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <h2 style='color: #00d4ff;'>🔬 DefectAI</h2>
        <p style='color: #888; font-size: 0.85rem;'>
        Enterprise Semiconductor<br>Inspection System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Navigation
    page = st.selectbox(
        "Navigation",
        ["🏠 Home", 
         "🔍 Single Inspection", 
         "📦 Batch Processing",
         "📊 Dataset Analytics",
         "📈 Model Performance"]
    )
    
    st.divider()
    
    # System Status
    st.markdown("**System Status**")
    st.success("✅ System Online")
    st.info("🔄 Model: YOLOv8s (Demo)")
    st.warning("⏳ Week 3: Training Soon")
    
    st.divider()
    
    # Defect Classes
    st.markdown("**Defect Categories**")
    for defect, info in DEFECT_CLASSES.items():
        st.markdown(
            f"<div style='display:flex; align-items:center; margin:3px 0;'>"
            f"<div style='width:12px; height:12px; border-radius:50%; "
            f"background:{info['color']}; margin-right:8px;'></div>"
            f"<span style='font-size:0.85rem; color:#ccc;'>{defect}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    st.divider()
    st.markdown(
        "<p style='color:#555; font-size:0.75rem; text-align:center;'>"
        "CUBE AI Solutions<br>MIPHI Program 2026</p>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# PAGE 1 — HOME
# ─────────────────────────────────────────────
if page == "🏠 Home":
    
    st.markdown(
        "<div class='dashboard-title'>Enterprise Semiconductor Defect Detection</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#888; margin-top:-15px;'>"
        "AI-Powered Quality Inspection System | MIPHI Program | CUBE AI Solutions"
        "</p>",
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🔬 Wafer Maps Analyzed",
            value="811,457",
            delta="WM-811K Dataset"
        )
    with col2:
        st.metric(
            label="🏷️ Defect Categories",
            value="9",
            delta="All patterns covered"
        )
    with col3:
        st.metric(
            label="🖼️ Training Images",
            value="62,420",
            delta="After augmentation"
        )
    with col4:
        st.metric(
            label="⚖️ Class Balance",
            value="5,274x → 1x",
            delta="Imbalance fixed ✅"
        )
    
    st.divider()
    
    # Project Timeline
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📅 Project Progress")
        
        weeks = {
            "Week 1 — Data Exploration": "✅ Complete",
            "Week 2 — Preprocessing & Augmentation": "✅ Complete",
            "Week 3 — YOLOv8 Model Training": "🔄 In Progress",
            "Week 4 — FastAPI Backend": "⏳ Upcoming",
            "Week 5 — Dashboard & Reports": "⏳ Upcoming",
            "Week 6 — Docker & Deployment": "⏳ Upcoming",
        }
        
        for week, status in weeks.items():
            color = "#2ecc71" if "✅" in status else "#f39c12" if "🔄" in status else "#555"
            st.markdown(
                f"<div style='padding:8px; margin:5px 0; border-radius:8px; "
                f"background:#1a1a2e; border-left:3px solid {color};'>"
                f"<span style='color:{color};'>{status}</span> "
                f"<span style='color:#ccc;'>{week}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    
    with col2:
        st.markdown("### 🛠️ Tech Stack")
        
        tech_stack = {
            "Deep Learning": ["PyTorch 2.6", "YOLOv8", "Transfer Learning"],
            "Computer Vision": ["OpenCV 4.8", "Albumentations", "Grad-CAM"],
            "Backend": ["FastAPI", "SQLite", "REST API"],
            "Frontend": ["Streamlit", "Plotly", "PIL"],
            "DevOps": ["Docker", "Git", "GitHub"],
        }
        
        for category, tools in tech_stack.items():
            st.markdown(
                f"<div style='padding:8px; margin:5px 0; border-radius:8px; "
                f"background:#1a1a2e; border-left:3px solid #00d4ff;'>"
                f"<span style='color:#00d4ff; font-weight:600;'>{category}:</span> "
                f"<span style='color:#aaa;'>{' | '.join(tools)}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    
    st.divider()
    
    # About section
    st.markdown("### 📋 About This System")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h4 style='color:#00d4ff;'>🎯 Objective</h4>
            <p style='color:#ccc; font-size:0.9rem;'>
            Automatically detect, classify, and localize 
            defects in semiconductor wafer maps using 
            deep learning and computer vision.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h4 style='color:#00d4ff;'>🏭 Industry Impact</h4>
            <p style='color:#ccc; font-size:0.9rem;'>
            A single defective wafer costs $5,000-$50,000. 
            AI inspection reduces missed defects and 
            speeds up quality control significantly.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h4 style='color:#00d4ff;'>🔬 Approach</h4>
            <p style='color:#ccc; font-size:0.9rem;'>
            Transfer learning with YOLOv8 pretrained on 
            COCO dataset, fine-tuned on 62,420 balanced 
            semiconductor defect images.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 2 — SINGLE INSPECTION
# ─────────────────────────────────────────────
elif page == "🔍 Single Inspection":
    
    st.markdown("## 🔍 Single Wafer Inspection")
    st.markdown("Upload a semiconductor wafer image for AI-powered defect detection.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Upload wafer map image",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
        )
        
        if uploaded_file:
            image, image_array = process_uploaded_image(uploaded_file)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", 
                    use_column_width=True)
            
            st.markdown(f"""
            <div class='metric-card'>
                <p style='color:#aaa; margin:0;'>
                📁 File: <b style='color:#fff;'>{uploaded_file.name}</b><br>
                📐 Size: <b style='color:#fff;'>{image.size[0]}x{image.size[1]} px</b><br>
                💾 Format: <b style='color:#fff;'>{image.format or 'PNG'}</b>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔍 Run Defect Inspection", type="primary", 
                        use_container_width=True):
                with st.spinner("🤖 AI analyzing wafer image..."):
                    result = simulate_prediction(image_array)
                    st.session_state['last_result'] = result
                    st.session_state['last_image'] = image
    
    with col2:
        st.markdown("### 📊 Inspection Results")
        
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            
            # Status banner
            if result['defect_type'] == 'none':
                st.success("✅ PASS — No significant defect pattern detected")
            else:
                st.error(f"⚠️ FAIL — {result['defect_type']} defect detected!")
            
            # Main result metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Defect Type", result['defect_type'])
            m2.metric("Confidence", f"{result['confidence']*100:.1f}%")
            m3.metric("Inference Time", f"{result['inference_time']}s")
            
            # Detailed result card
            st.markdown(f"""
            <div class='{"success-card" if result["defect_type"] == "none" else "error-card"}'>
                <h4 style='color:#fff; margin:0 0 10px 0;'>
                    Defect Analysis Report
                </h4>
                <p style='color:#ccc; margin:5px 0;'>
                    🏷️ <b>Classification:</b> {result['defect_type']}
                </p>
                <p style='color:#ccc; margin:5px 0;'>
                    📝 <b>Description:</b> {result['description']}
                </p>
                <p style='color:#ccc; margin:5px 0;'>
                    🎯 <b>Confidence Score:</b> {result['confidence']*100:.1f}%
                </p>
                <p style='color:#ccc; margin:5px 0;'>
                    ⚡ <b>Processing Time:</b> {result['inference_time']}s
                </p>
                <p style='color:#ccc; margin:5px 0;'>
                    🕐 <b>Timestamp:</b> {result['timestamp']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence gauge chart
            st.markdown("#### Confidence Score")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['confidence'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': result['color']},
                    'steps': [
                        {'range': [0, 50], 'color': '#1a1a2e'},
                        {'range': [50, 75], 'color': '#162032'},
                        {'range': [75, 100], 'color': '#1a2a1a'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 2},
                        'thickness': 0.75,
                        'value': 75
                    }
                },
                number={'suffix': "%", 'font': {'color': 'white'}}
            ))
            fig.update_layout(
                height=250,
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.markdown("""
            <div style='text-align:center; padding:60px 20px; 
                       border:2px dashed #333; border-radius:12px;'>
                <p style='font-size:3rem;'>🔬</p>
                <p style='color:#555;'>Upload an image and click 
                "Run Defect Inspection" to see results here</p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 3 — BATCH PROCESSING
# ─────────────────────────────────────────────
elif page == "📦 Batch Processing":
    
    st.markdown("## 📦 Batch Wafer Inspection")
    st.markdown("Upload multiple wafer images for simultaneous inspection.")
    
    batch_files = st.file_uploader(
        "Upload multiple wafer images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Select multiple files at once using Ctrl+Click"
    )
    
    if batch_files:
        st.markdown(f"**{len(batch_files)} images selected**")
        
        if st.button("🚀 Run Batch Inspection", type="primary"):
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(batch_files):
                status_text.text(f"Inspecting {file.name}...")
                image, image_array = process_uploaded_image(file)
                result = simulate_prediction(image_array)
                result['filename'] = file.name
                results.append(result)
                progress_bar.progress((i + 1) / len(batch_files))
            
            status_text.text("✅ Batch inspection complete!")
            
            # Results summary
            pass_count = sum(1 for r in results if r['defect_type'] == 'none')
            fail_count = len(results) - pass_count
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Inspected", len(results))
            col2.metric("✅ Pass", pass_count)
            col3.metric("❌ Fail", fail_count)
            
            # Results table
            st.markdown("### Inspection Report")
            
            results_df = pd.DataFrame([{
                'Filename': r['filename'],
                'Defect Type': r['defect_type'],
                'Confidence': f"{r['confidence']*100:.1f}%",
                'Status': '✅ PASS' if r['defect_type'] == 'none' else '❌ FAIL',
                'Time (s)': r['inference_time'],
                'Timestamp': r['timestamp']
            } for r in results])
            
            st.dataframe(results_df, use_container_width=True)
            
            # Defect distribution pie chart
            defect_counts = {}
            for r in results:
                defect_counts[r['defect_type']] = defect_counts.get(
                    r['defect_type'], 0) + 1
            
            if len(defect_counts) > 1:
                fig = px.pie(
                    values=list(defect_counts.values()),
                    names=list(defect_counts.keys()),
                    title="Defect Distribution in Batch",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # CSV Download
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Inspection Report (CSV)",
                data=csv,
                file_name=f"inspection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )


# ─────────────────────────────────────────────
# PAGE 4 — DATASET ANALYTICS
# ─────────────────────────────────────────────
elif page == "📊 Dataset Analytics":
    
    st.markdown("## 📊 Dataset Analytics")
    st.markdown("WM-811K Semiconductor Wafer Map Dataset Analysis")
    
    # Dataset overview metrics



    # Dataset overview metrics
    total_original = sum(DATASET_STATS.values())
    total_balanced = sum(BALANCED_STATS.values())
    imbalance_ratio = max(DATASET_STATS.values()) / min(DATASET_STATS.values())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Total Wafer Maps", f"{total_original:,}")
    with col2:
        st.metric("🏷️ Defect Classes", len(DATASET_STATS))
    with col3:
        st.metric("⚖️ Original Imbalance", f"{imbalance_ratio:,.0f}x")
    with col4:
        st.metric("✅ Balanced Dataset Size", f"{total_balanced:,}")

    st.divider()

    # ── Class distribution — Original (log scale, since 'none' dominates)
    st.markdown("<div class='section-header'>Original Class Distribution</div>",
                unsafe_allow_html=True)

    orig_df = pd.DataFrame({
        'Class': list(DATASET_STATS.keys()),
        'Count': list(DATASET_STATS.values())
    }).sort_values('Count', ascending=False)

    fig_orig = px.bar(
        orig_df, x='Class', y='Count',
        color='Class',
        color_discrete_map={k: v['color'] for k, v in DEFECT_CLASSES.items()},
        log_y=True,
        text='Count'
    )
    fig_orig.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig_orig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        showlegend=False,
        yaxis_title="Count (log scale)"
    )
    st.plotly_chart(fig_orig, use_container_width=True)

    st.divider()

    # ── Before vs After balancing comparison
    st.markdown("<div class='section-header'>Before vs After Balancing</div>",
                unsafe_allow_html=True)

    compare_df = pd.DataFrame({
        'Class': list(DATASET_STATS.keys()) + list(BALANCED_STATS.keys()),
        'Count': list(DATASET_STATS.values()) + list(BALANCED_STATS.values()),
        'Stage': ['Original'] * len(DATASET_STATS) + ['Balanced'] * len(BALANCED_STATS)
    })

    fig_compare = px.bar(
        compare_df, x='Class', y='Count', color='Stage',
        barmode='group', log_y=True,
        color_discrete_map={'Original': '#ff4444', 'Balanced': '#00ff88'}
    )
    fig_compare.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        yaxis_title="Count (log scale)"
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    st.divider()

    # ── Class proportions pie chart + description table
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='section-header'>Balanced Class Proportions</div>",
                    unsafe_allow_html=True)
        fig_pie = px.pie(
            values=list(BALANCED_STATS.values()),
            names=list(BALANCED_STATS.keys()),
            color=list(BALANCED_STATS.keys()),
            color_discrete_map={k: v['color'] for k, v in DEFECT_CLASSES.items()}
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Defect Class Reference</div>",
                    unsafe_allow_html=True)
        for defect, info in DEFECT_CLASSES.items():
            st.markdown(f"""
            <div style='padding:10px; margin:6px 0; border-radius:8px;
                       background:#1a1a2e; border-left:4px solid {info['color']};'>
                <b style='color:{info['color']};'>{defect}</b>
                <span style='color:#aaa; font-size:0.85rem;'> — {info['description']}</span><br>
                <span style='color:#666; font-size:0.8rem;'>
                    Original: {DATASET_STATS[defect]:,} | Balanced: {BALANCED_STATS[defect]:,}
                </span>
            </div>
            """, unsafe_allow_html=True)


            # ─────────────────────────────────────────────
# PAGE 5 — MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif page == "📈 Model Performance":

    st.markdown("## 📈 Model Performance")
    st.markdown("YOLOv8 Training Metrics & Evaluation")

    st.warning("⏳ Model training is in progress (Week 3). "
               "The metrics below are simulated placeholders — "
               "real results will replace these once training completes.")

    st.divider()

    # ── Top-level metrics (placeholder values)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 mAP@50", "—", "Pending training")
    with col2:
        st.metric("📏 Precision", "—", "Pending training")
    with col3:
        st.metric("📐 Recall", "—", "Pending training")
    with col4:
        st.metric("⚡ Avg Inference", "0.15s", "Demo estimate")

    st.divider()

    # ── Simulated training curves
    st.markdown("<div class='section-header'>Simulated Training Curves</div>",
                unsafe_allow_html=True)

    epochs = np.arange(1, 51)
    np.random.seed(42)
    train_loss = 2.5 * np.exp(-epochs / 15) + np.random.normal(0, 0.03, len(epochs))
    val_loss = 2.7 * np.exp(-epochs / 17) + np.random.normal(0, 0.05, len(epochs))

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines',
                                    name='Train Loss', line=dict(color='#00d4ff')))
    fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines',
                                    name='Val Loss', line=dict(color='#ff8800')))
    fig_loss.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig_loss, use_container_width=True)

    st.divider()

    # ── Placeholder confusion matrix
    st.markdown("<div class='section-header'>Confusion Matrix (Simulated)</div>",
                unsafe_allow_html=True)

    classes = list(DEFECT_CLASSES.keys())
    n = len(classes)
    rng = np.random.default_rng(1)
    matrix = rng.integers(0, 15, size=(n, n))
    np.fill_diagonal(matrix, rng.integers(60, 100, size=n))

    fig_cm = px.imshow(
        matrix,
        x=classes, y=classes,
        color_continuous_scale='Blues',
        labels=dict(x="Predicted", y="Actual", color="Count")
    )
    fig_cm.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.info("📌 This page will be updated with real YOLOv8 evaluation metrics "
            "once training completes in Week 3.")