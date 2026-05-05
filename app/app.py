"""
WoundLens -- AI-Powered Wound Staging & Clinical Decision Support
================================================================
Main Streamlit Application Entry Point
"""

import streamlit as st

st.set_page_config(
    page_title="WoundLens -- AI Wound Staging",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================================================================
# MASSIVE CSS OVERHAUL
# ====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    #MainMenu, footer { visibility: hidden !important; }

    /* ---- Deep dark background ---- */
    .stApp {
        background: radial-gradient(ellipse at 20% 50%, #1a0a0a 0%, #0a0a14 40%, #060612 100%) !important;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c0c18 0%, #12101e 50%, #0c0c18 100%) !important;
        border-right: 1px solid rgba(224,64,64,0.08) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown span {
        font-size: 0.9rem;
    }

    /* ---- Buttons ---- */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #C03030 0%, #E05555 50%, #D04040 100%) !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.7rem 2rem !important;
        border-radius: 12px !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 24px rgba(192,48,48,0.25) !important;
        transition: all 0.3s !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 32px rgba(192,48,48,0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* ---- Metric cards ---- */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 1rem 1.2rem;
    }
    [data-testid="stMetricValue"] {
        font-weight: 800 !important;
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        padding: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }

    }

    /* ---- Hide default Streamlit page navigation ---- */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


# ====================================================================
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.ui import render_sidebar

render_sidebar()


# ====================================================================
# HERO
# ====================================================================
st.markdown("")
st.markdown("""<div style="text-align:center; padding:2.5rem 1rem 1.5rem; animation: fadeUp 1s cubic-bezier(0.16, 1, 0.3, 1) forwards; opacity: 0; transform: translateY(20px);">
    <div style="display:inline-block; padding:0.3rem 1.2rem; background: rgba(192,48,48,0.12); border: 1px solid rgba(192,48,48,0.25); border-radius: 20px; font-size:0.7rem; font-weight:700; color:#E06060; letter-spacing:2px; text-transform:uppercase; margin-bottom:1.2rem;">AI-Powered Clinical Decision Support</div>
    <h1 style="font-size:4.5rem; font-weight:900; line-height:1.05; background: linear-gradient(135deg, #C03030 0%, #FF6B6B 40%, #FF9E53 70%, #C03030 100%); background-size: 300% 300%; -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: heroGrad 6s ease infinite; margin:0 0 1rem; letter-spacing:-2px;">WoundLens</h1>
    <p style="font-size:1.1rem; color:#777; font-weight:300; max-width:550px; margin:0 auto; line-height:1.7;">Upload a wound photo. Get instant tissue classification, NPIAP staging, and evidence-based treatment recommendations.</p>
</div>
<style>
@keyframes heroGrad {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes fadeUp {
    to { opacity: 1; transform: translateY(0); }
}
</style>""", unsafe_allow_html=True)

st.markdown("")

# ====================================================================
# STATS
# ====================================================================
col1, col2, col3, col4 = st.columns(4)

stats_data = [
    ("$25B+", "Wound care market", "#C03030"),
    ("28M+", "Chronic wound patients", "#D04A4A"),
    ("4 Stages", "NPIAP classification", "#E06060"),
    ("<3s", "Analysis per image", "#CC4040"),
]

for col, (num, label, color) in zip([col1, col2, col3, col4], stats_data):
    with col:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(20,14,24,0.9), rgba(30,18,28,0.9));
            border: 1px solid rgba(192,48,48,0.12);
            border-radius: 16px;
            padding: 1.8rem 1rem;
            text-align: center;
            transition: all 0.3s;
        ">
            <div style="
                font-size:2.5rem; font-weight:900;
                background: linear-gradient(135deg, {color}, #FF8E53);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                line-height:1.2;
            ">{num}</div>
            <div style="font-size:0.75rem; color:#555; margin-top:0.4rem; font-weight:500; letter-spacing:0.5px;">
                {label}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")
st.markdown("")

# ====================================================================
# PIPELINE - How It Works
# ====================================================================
st.markdown("### How It Works")
st.markdown("")

p1, p2, p3, p4 = st.columns(4)

pipeline = [
    ("1", "Capture", "Upload or photograph the wound with your smartphone", "#C03030"),
    ("2", "Detect", "AI locates the wound region and isolates it from background", "#D04A4A"),
    ("3", "Classify", "Computer vision classifies granulation, slough, and necrotic tissue", "#E06060"),
    ("4", "Treat", "Rule engine assigns NPIAP stage and recommends dressings", "#CC4040"),
]

for col, (num, title, desc, color) in zip([p1, p2, p3, p4], pipeline):
    with col:
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.015);
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: 16px;
            padding: 1.5rem 1rem;
            text-align: center;
            min-height: 180px;
            transition: all 0.3s;
            position: relative;
        ">
            <div style="
                display:inline-block;
                width:36px; height:36px; line-height:36px;
                border-radius:50%;
                background: linear-gradient(135deg, {color}, #FF8E53);
                color:white; font-size:0.85rem; font-weight:800;
                margin-bottom:0.8rem;
                box-shadow: 0 4px 12px rgba(192,48,48,0.2);
            ">{num}</div>
            <div style="font-size:1.05rem; font-weight:700; color:#ddd; margin-bottom:0.4rem;">{title}</div>
            <div style="font-size:0.78rem; color:#555; line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")
st.markdown("")



st.markdown("")
st.markdown("---")
st.markdown(
    '<div style="color:#333; text-align:center; font-size:0.7rem; padding:0.5rem 0; line-height:1.7;">'
    "WoundLens is an AI-assisted clinical decision support tool. "
    "It does NOT replace professional medical judgment. "
    "Always consult a qualified healthcare provider."
    "</div>",
    unsafe_allow_html=True,
)
