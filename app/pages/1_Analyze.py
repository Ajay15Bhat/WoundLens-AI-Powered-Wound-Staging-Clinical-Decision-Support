"""
WoundLens — Wound Analysis Page
================================
Upload or capture a wound image → full AI analysis pipeline.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os
import time
import io
from gtts import gTTS

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.segmentation import segment_wound, draw_wound_contour
from core.tissue_classifier import classify_wound_tissue
from core.staging_engine import classify_stage, get_severity_emoji
from core.treatment_engine import get_treatment
from core.visualization import (
    create_tissue_overlay,
    create_pie_chart,
    create_analysis_dashboard,
    get_tissue_color_legend,
)

# ── Page Config ──
st.set_page_config(page_title="Analyze -- WoundLens", page_icon="", layout="wide")

# ── Premium CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    #MainMenu, footer { visibility: hidden !important; }

    .stApp {
        background: radial-gradient(ellipse at 20% 50%, #1a0a0a 0%, #0a0a14 40%, #060612 100%) !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c0c18 0%, #12101e 50%, #0c0c18 100%) !important;
        border-right: 1px solid rgba(224,64,64,0.08) !important;
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 1rem 1.2rem;
    }
    [data-testid="stMetricValue"] { 
        font-weight: 700 !important; 
        font-size: 1.4rem !important; 
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        color: #aaa !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        padding: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    .stage-badge {
        display: inline-block;
        padding: 0.7rem 2.5rem;
        border-radius: 14px;
        font-weight: 800;
        font-size: 1.6rem;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 6px 24px rgba(0,0,0,0.4);
        letter-spacing: 1px;
    }
    .tissue-bar {
        height: 36px;
        border-radius: 18px;
        display: flex;
        overflow: hidden;
        margin: 1rem 0;
        box-shadow: inset 0 2px 6px rgba(0,0,0,0.4), 0 2px 8px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.05);
    }
    .tissue-segment {
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
        color: white;
        text-shadow: 0 1px 3px rgba(0,0,0,0.7);
    }
    .result-card {
        background: linear-gradient(135deg, rgba(20,14,24,0.9), rgba(16,12,22,0.9));
        border: 1px solid rgba(192,48,48,0.1);
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-title {
        font-size: 0.7rem;
        color: #E06060;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    [data-testid="stSidebarNav"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core.ui import render_sidebar
render_sidebar()

with st.sidebar:
    st.markdown("### Settings")
    use_lab = st.checkbox("Use LAB confirmation", value=True,
                          help="Intersect HSV and LAB masks for higher precision")

    st.markdown("---")
    st.markdown("### Tissue Legend")
    for item in get_tissue_color_legend():
        st.markdown(
            f"<span style='color:{item['color_hex']}'>●</span> "
            f"**{item['name']}** — {item['description']}",
            unsafe_allow_html=True,
        )


# ── Page Header ──
st.markdown("# 📸 Wound Analysis")
st.markdown("Upload a wound image or use your camera for instant AI-powered analysis.")
st.markdown("")

# ── Image Input ──
col_upload, col_camera = st.columns(2)

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload wound image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported formats: JPG, PNG, BMP, WebP",
    )

with col_camera:
    camera_image = st.camera_input("Or capture with camera")

# Determine which image to use
image_source = uploaded_file or camera_image

if image_source is not None:
    # ── Load Image ──
    pil_image = Image.open(image_source).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    st.markdown("---")

    # ── Progress Animation ──
    progress_bar = st.progress(0, text="Initializing analysis...")
    status_text = st.empty()

    # Step 1: Wound Detection + Segmentation
    progress_bar.progress(10, text="Detecting wound region...")
    from models.predict import detect_wound
    bbox = detect_wound(image_bgr)

    progress_bar.progress(20, text="Segmenting wound region...")
    seg_result = segment_wound(image_bgr, bbox_xyxy=bbox)
    time.sleep(0.3)

    # Step 2: Tissue Classification
    progress_bar.progress(40, text="Classifying tissue types...")
    tissue_result = classify_wound_tissue(
        seg_result["preprocessed"],
        wound_mask=seg_result["wound_mask"],
        use_lab_confirmation=use_lab,
    )
    ratios = tissue_result["ratios"]
    time.sleep(0.3)

    # Step 3: Staging
    progress_bar.progress(65, text="Determining wound stage...")
    stage_result = classify_stage(
        ratios["granulation"],
        ratios["slough"],
        ratios["necrotic"],
    )
    time.sleep(0.3)

    # Step 4: Treatment
    progress_bar.progress(85, text="Generating treatment plan...")
    treatment = get_treatment(stage_result["stage"])
    time.sleep(0.3)

    progress_bar.progress(100, text="Analysis complete!")
    time.sleep(0.5)
    progress_bar.empty()

    # ── Store results in session state for the Healing Tracker ──
    st.session_state["last_analysis"] = {
        "stage": stage_result["stage"],
        "severity": stage_result["severity"],
        "granulation": ratios["granulation"],
        "slough": ratios["slough"],
        "necrotic": ratios["necrotic"],
        "wound_area_px": seg_result["wound_area_px"],
        "coverage_pct": seg_result["coverage_pct"],
        "follow_up_days": treatment["follow_up_days"],
    }
    time.sleep(0.5)
    progress_bar.empty()

    # ================================================================
    # ── RESULTS DISPLAY ──
    # ================================================================

    # ── Stage Banner ──
    severity_colors = {
        "LOW": "#4CAF50", "MODERATE": "#FFC107",
        "HIGH": "#FF9800", "CRITICAL": "#F44336",
    }
    stage_color = severity_colors.get(stage_result["severity"], "#FFC107")

    st.markdown(
        f'<div style="text-align:center; margin: 1rem 0;">'
        f'<span class="stage-badge" style="background:{stage_color};">'
        f'{stage_result["stage"]}</span>'
        f'<p style="color:#aaa; margin-top:0.5rem;">{stage_result["description"]}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Voice Summary ──
    with st.expander("🔊 Play Voice Summary"):
        summary_text = (
            f"The wound is classified as {stage_result['stage']}. "
            f"The tissue breakdown is {ratios['granulation']:.0f} percent granulation, "
            f"{ratios['slough']:.0f} percent slough, and {ratios['necrotic']:.0f} percent necrotic tissue. "
            f"Recommended treatment: {treatment['primary_actions'][0]}. "
            f"Suggested dressing: {treatment['dressings'][0]['name']}."
        )
        
        try:
            tts = gTTS(text=summary_text, lang='en', slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            audio_bytes = fp.read()
            st.audio(audio_bytes, format="audio/mp3")
            st.markdown(f"*{summary_text}*")
        except Exception as e:
            st.error(f"Could not generate voice summary: {e}")

    # ── Tissue Composition Bar ──
    gran_pct = ratios["granulation"]
    slough_pct = ratios["slough"]
    necro_pct = ratios["necrotic"]

    st.markdown(
        f'<div class="tissue-bar">'
        f'<div class="tissue-segment" style="width:{max(gran_pct,1)}%; background:#DC5040;">'
        f'{gran_pct:.1f}%</div>'
        f'<div class="tissue-segment" style="width:{max(slough_pct,1)}%; background:#E6D200; color:#333;">'
        f'{slough_pct:.1f}%</div>'
        f'<div class="tissue-segment" style="width:{max(necro_pct,1)}%; background:#323228;">'
        f'{necro_pct:.1f}%</div>'
        f'</div>'
        f'<div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#888;">'
        f'<span>🔴 Granulation (Healing)</span>'
        f'<span>🟡 Slough (Risk)</span>'
        f'<span>⚫ Necrotic (Dead)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ── Two-Column Results ──
    col_img, col_info = st.columns([3, 2])

    with col_img:
        # Tab view for different visualizations
        tab1, tab2, tab3 = st.tabs(["🔬 Tissue Map", "📐 Wound Boundary", "📊 Dashboard"])

        with tab1:
            overlay = create_tissue_overlay(
                seg_result["preprocessed"],
                tissue_result["masks"],
                wound_mask=seg_result["wound_mask"],
            )
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            st.image(overlay_rgb, caption="Color-coded tissue classification", use_container_width=True)

        with tab2:
            contour_img = draw_wound_contour(
                seg_result["preprocessed"], seg_result["wound_mask"],
                color=(0, 255, 130), thickness=2,
            )
            contour_rgb = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)
            st.image(contour_rgb, caption="Wound boundary detection", use_container_width=True)

        with tab3:
            dashboard = create_analysis_dashboard(
                seg_result["preprocessed"], tissue_result, seg_result,
                stage_result, treatment,
            )
            dashboard_rgb = cv2.cvtColor(dashboard, cv2.COLOR_BGR2RGB)
            st.image(dashboard_rgb, caption="Complete analysis dashboard", use_container_width=True)

    with col_info:
        # ── Tissue Breakdown ──
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">Tissue Breakdown</div>', unsafe_allow_html=True)

        pie_img = create_pie_chart(ratios, size=(350, 350))
        pie_rgb = cv2.cvtColor(pie_img, cv2.COLOR_BGR2RGB)
        st.image(pie_rgb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Wound Metrics ──
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">Wound Metrics</div>', unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric("Img Coverage", f"{float(seg_result['coverage_pct']):.1f}%")
        
        # Use the bounding box width and height for size dimensions
        bw, bh = seg_result['wound_bbox'][2:]
        m2.metric("Wound Size", f"{bw} x {bh} px")

        m3, m4 = st.columns(2)
        m3.metric("Severity", stage_result["severity"])
        m4.metric("Follow-up", f"{treatment['follow_up_days']} days")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Treatment Recommendations ──
    st.markdown("---")
    st.markdown("### 💊 Treatment Recommendations")

    urgency_colors = {
        "LOW": "🟢", "MODERATE": "🟡", "HIGH": "🟠", "CRITICAL": "🔴",
    }
    urgency_icon = urgency_colors.get(treatment["urgency"], "⚪")
    st.markdown(f"**Urgency:** {urgency_icon} {treatment['urgency']}")

    # Actions
    with st.expander("📋 Primary Actions", expanded=True):
        for i, action in enumerate(treatment["primary_actions"], 1):
            st.markdown(f"{i}. {action}")

    # Dressings
    with st.expander("🩹 Recommended Dressings", expanded=True):
        for d in treatment["dressings"]:
            st.markdown(f"**{d['name']}**")
            st.markdown(f"  - *Why:* {d['reason']}")
            st.markdown(f"  - *Examples:* {', '.join(d['examples'])}")
            st.markdown("")

    # Additional info columns
    col_n, col_m, col_r = st.columns(3)
    with col_n:
        st.markdown("**🥗 Nutrition**")
        st.info(treatment["nutrition"])
    with col_m:
        st.markdown("**📋 Monitoring**")
        st.info(treatment["monitoring"])
    with col_r:
        st.markdown("**🏥 Referral**")
        st.info(treatment["referral"])

    # Debridement (if applicable)
    if treatment.get("debridement"):
        st.markdown("---")
        st.markdown("### 🔪 Debridement Recommendation")
        st.warning(f"**Method:** {treatment['debridement']['method']}")
        st.markdown(treatment["debridement"]["details"])

    # ── Infection Warning Signs ──
    with st.expander("⚠️ Watch for Infection Signs"):
        for sign in treatment["infection_signs"]:
            st.markdown(f"- {sign}")

    # ── Disclaimer ──
    st.markdown("---")
    st.caption(treatment["disclaimer"])

else:
    # ── Empty State ──
    st.markdown("")
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown(
            '<div style="text-align:center; padding:3rem; '
            'background:#1a1a2e; border-radius:16px; border:2px dashed #2a2a4a;">'
            '<p style="font-size:3rem; margin:0;">📸</p>'
            '<p style="font-size:1.2rem; color:#aaa; margin-top:1rem;">'
            'Upload a wound image or use your camera to begin analysis'
            '</p>'
            '<p style="font-size:0.85rem; color:#666; margin-top:0.5rem;">'
            'Supported: JPG, PNG, BMP, WebP'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )
