"""
WoundLens — About Page
=======================
How the system works, clinical references, and team info.
"""

import streamlit as st

# ── Page Config ──
st.set_page_config(page_title="About — WoundLens", page_icon="ℹ️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    #MainMenu, footer, header { visibility: hidden !important; }
    .stApp {
        background: radial-gradient(ellipse at 20% 50%, #1a0a0a 0%, #0a0a14 40%, #060612 100%) !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c0c18 0%, #12101e 50%, #0c0c18 100%) !important;
        border-right: 1px solid rgba(224,64,64,0.08) !important;
    }

    .arch-step {
        background: linear-gradient(135deg, rgba(20,14,24,0.9), rgba(16,12,22,0.9));
        border: 1px solid rgba(192,48,48,0.1);
        border-radius: 18px;
        padding: 1.5rem;
        text-align: center;
        height: 100%;
    }
    .arch-icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
    .arch-title { font-size: 0.95rem; font-weight: 700; color: #ddd; }
    .arch-desc { font-size: 0.75rem; color: #555; margin-top: 0.3rem; line-height: 1.5; }
    .arrow-col {
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem; color: #C03030;
    }
    .impact-card {
        background: linear-gradient(135deg, rgba(20,14,24,0.9), rgba(16,12,22,0.9));
        border: 1px solid rgba(192,48,48,0.1);
        border-radius: 18px;
        padding: 1.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🩺 WoundLens")
    st.markdown("**About**")
    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown("- [The Problem](#the-problem)")
    st.markdown("- [How It Works](#how-it-works)")
    st.markdown("- [Technology](#technology)")
    st.markdown("- [Clinical Basis](#clinical-basis)")

# ── Header ──
st.markdown("# ℹ️ About WoundLens")
st.markdown("")

# ── The Problem ──
st.markdown("### The Problem")
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    Chronic wounds — diabetic ulcers, pressure injuries, venous leg ulcers —
    affect **28+ million people** worldwide. Accurate wound staging is critical
    for proper treatment, but current methods rely on:

    - **Subjective visual assessment** by nurses with rulers and paper logs
    - **Inconsistent documentation** leading to misdiagnosis
    - **Delayed intervention** resulting in complications like **amputation** or **sepsis**

    The global wound care market exceeds **$25 billion annually**, yet most staging
    is still done manually with significant inter-rater variability.
    """)
with col2:
    st.markdown("""
    <div class="impact-card">
        <p style="font-size:2rem; font-weight:800; background:linear-gradient(135deg,#E04040,#FF8E53);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0.3rem 0;">$25B+</p>
        <p style="color:#666; font-size:0.8rem;">Annual wound care costs</p>
        <p style="font-size:2rem; font-weight:800; background:linear-gradient(135deg,#E04040,#FF8E53);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0.3rem 0;">28M+</p>
        <p style="color:#666; font-size:0.8rem;">People affected globally</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── How It Works ──
st.markdown("### How It Works")
st.markdown("")

steps = [
    ("📸", "Capture", "Upload or photograph the wound via smartphone"),
    ("→", "", ""),
    ("🔬", "Detect", "YOLOv8 AI locates the wound region automatically"),
    ("→", "", ""),
    ("🎨", "Classify", "HSV+LAB color analysis identifies tissue types"),
    ("→", "", ""),
    ("📊", "Stage", "Rule-based engine assigns Stage 1–4"),
    ("→", "", ""),
    ("💊", "Treat", "Evidence-based treatment recommendations"),
]

cols = st.columns(len(steps))
for col, (icon, title, desc) in zip(cols, steps):
    with col:
        if title == "":  # Arrow
            st.markdown(
                f'<div class="arrow-col">{icon}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="arch-step">'
                f'<div class="arch-icon">{icon}</div>'
                f'<div class="arch-title">{title}</div>'
                f'<div class="arch-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

st.markdown("")
st.markdown("---")

# ── Technology ──
st.markdown("### Technology Stack")

tech_col1, tech_col2, tech_col3 = st.columns(3)

with tech_col1:
    st.markdown("#### 🤖 Computer Vision")
    st.markdown("""
    - **YOLOv8** (Ultralytics) for wound detection
    - **OpenCV** HSV/LAB color-space analysis
    - **GrabCut** segmentation algorithm
    - Morphological cleanup & noise removal
    """)

with tech_col2:
    st.markdown("#### 🧠 Clinical Logic")
    st.markdown("""
    - **NPIAP/EPUAP** staging guidelines
    - Rule-based decision tree (tissue → stage)
    - Evidence-based treatment database
    - **TIME** framework integration
    """)

with tech_col3:
    st.markdown("#### 🖥️ Application")
    st.markdown("""
    - **Streamlit** web application
    - **Plotly** interactive visualizations
    - Real-time camera capture
    - Healing progression tracking
    """)

st.markdown("---")

# ── Clinical Basis ──
st.markdown("### Clinical Basis")

st.markdown("#### Tissue Classification")
tissue_data = {
    "Tissue Type": ["Granulation", "Slough", "Necrotic (Eschar)"],
    "Appearance": ["Red/pink, moist, bumpy", "Yellow/cream, stringy", "Black/brown, dry, leathery"],
    "Clinical Meaning": ["Healing — good sign", "Infection risk — needs debridement", "Dead tissue — urgent"],
    "Detection Method": ["HSV red band (H:0-10, 160-180)", "HSV yellow band (H:18-40)", "Low saturation & value"],
}
st.table(tissue_data)

st.markdown("#### NPIAP Staging System")
staging_data = {
    "Stage": ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Unstageable"],
    "Description": [
        "Non-blanchable erythema, intact skin",
        "Partial-thickness skin loss, shallow ulcer",
        "Full-thickness skin loss, fat visible",
        "Full-thickness tissue loss, bone/tendon exposed",
        "Wound bed obscured by slough/eschar",
    ],
    "Tissue Profile": [
        ">80% Granulation",
        ">60% Granulation, <20% Slough",
        "Mixed, <30% Necrotic",
        ">40% Necrotic",
        ">70% non-viable tissue",
    ],
}
st.table(staging_data)

st.markdown("---")

# ── References ──
st.markdown("### References")
st.markdown("""
1. **NPIAP** (2019). *Pressure Injury Staging System.* National Pressure Injury Advisory Panel.
2. **EPUAP/NPIAP/PPPIA** (2019). *Prevention and Treatment of Pressure Ulcers/Injuries: Clinical Practice Guideline.*
3. **Wannous et al.** (2010). *Supervised tissue classification from color images for a remote wound monitoring system.* Journal of Medical Engineering.
4. **TIME Framework** — Tissue, Infection, Moisture, Edge assessment for wound bed preparation.
5. **AAFP** — *Management of Chronic Wounds.* American Academy of Family Physicians.
""")

# ── Disclaimer ──
st.markdown("---")
st.caption(
    "⚠️ WoundLens is an AI-assisted clinical decision support tool for educational "
    "and demonstration purposes. It does NOT replace professional medical judgment. "
    "Always consult a qualified healthcare provider for wound care decisions."
)
