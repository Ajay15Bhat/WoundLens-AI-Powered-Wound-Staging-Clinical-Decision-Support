import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:1rem 0 0.5rem;">
            <div style="
                width:50px; height:50px; margin:0 auto 0.5rem;
                background: linear-gradient(135deg, #C03030, #E06060);
                border-radius: 14px;
                display:flex; align-items:center; justify-content:center;
                font-size:1.5rem; color:white; font-weight:900;
                box-shadow: 0 4px 16px rgba(192,48,48,0.3);
            ">W</div>
            <div style="font-size:1.1rem; font-weight:700; color:#eee;">WoundLens</div>
            <div style="font-size:0.65rem; color:#555; margin-top:0.15rem; letter-spacing:1px; text-transform:uppercase;">Clinical AI Platform</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### 🚀 Quick Access")
        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/1_Analyze.py", label="Analyze Wound", icon="🔍")
        st.page_link("pages/2_Tracker.py", label="Healing Tracker", icon="📈")
        st.page_link("pages/3_About.py", label="About System", icon="ℹ️")
        st.markdown("---")
        st.markdown(
            "<div style='color:#333; font-size:0.65rem; text-align:center; letter-spacing:0.5px;'>"
            "Built for Hackathon 2026</div>",
            unsafe_allow_html=True,
        )
