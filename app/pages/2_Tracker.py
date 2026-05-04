"""
WoundLens -- Healing Tracker Page
=================================
Generates a projected healing timeline based on the ACTUAL wound analysis
results from the Analyze page. If no analysis has been done yet, prompts
the user to analyze a wound first.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib

# -- Page Config --
st.set_page_config(page_title="Healing Tracker -- WoundLens", page_icon="", layout="wide")

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
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 1rem 1.2rem;
    }
    [data-testid="stMetricValue"] { font-weight: 800 !important; }
</style>
""", unsafe_allow_html=True)

# -- Sidebar --
with st.sidebar:
    st.markdown("## WoundLens")
    st.markdown("**Healing Tracker**")
    st.markdown("---")
    st.markdown("### Projection Settings")
    weeks = st.slider("Projection Period (weeks)", 4, 24, 12)
    healing_rate = st.select_slider(
        "Expected Healing Response",
        options=["Poor", "Moderate", "Good", "Excellent"],
        value="Good",
    )


# -- Check for analysis data --
analysis = st.session_state.get("last_analysis", None)

if analysis is None:
    st.markdown("# Healing Tracker")
    st.markdown("")
    st.info(
        "No wound analysis found. Please go to the **Analyze** page first, "
        "upload a wound image, and run the analysis. The healing tracker will "
        "then project a healing timeline based on your actual wound data."
    )
    st.stop()


# -- Extract actual wound data --
current_stage = analysis["stage"]
current_gran = analysis["granulation"]
current_slough = analysis["slough"]
current_necro = analysis["necrotic"]
coverage_pct = analysis["coverage_pct"]
wound_area_px = analysis["wound_area_px"]

# Estimate wound area in cm2 (rough: assume 10px/mm at 640px image)
estimated_area_cm2 = round(wound_area_px / 1000.0, 1)
if estimated_area_cm2 < 0.5:
    estimated_area_cm2 = 0.5

# -- Generate healing projection from ACTUAL tissue ratios --
def generate_healing_projection(initial_gran, initial_slough, initial_necro,
                                 initial_area, weeks, healing_rate, seed_str):
    """Project healing based on actual wound tissue composition."""
    rate_map = {"Poor": 0.4, "Moderate": 0.7, "Good": 1.0, "Excellent": 1.4}
    rate = rate_map[healing_rate]

    # Use a deterministic seed based on the wound data so the same wound
    # always produces the same projection (no random flicker on rerun)
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16) % (2**31)
    rng = np.random.RandomState(seed)

    dates, areas = [], []
    gran_list, slough_list, necro_list = [], [], []
    stages = []

    base_date = datetime.now()

    for week in range(weeks + 1):
        dates.append(base_date + timedelta(weeks=week))

        # Area decay from actual starting area
        decay = np.exp(-0.07 * rate * week)
        noise = rng.uniform(-0.3, 0.3)
        area = max(0.3, initial_area * decay + noise)
        areas.append(round(area, 1))

        # Tissue shifts from actual starting composition
        progress = min(1.0, (week / max(weeks, 1)) * rate)

        gran = min(98, initial_gran + progress * (95 - initial_gran) + rng.uniform(-2, 2))
        necro = max(0, initial_necro * (1 - progress * 1.3) + rng.uniform(-1, 0.5))
        slough = max(0, initial_slough * (1 - progress * 1.1) + rng.uniform(-1, 1))

        # Normalize to 100%
        total = gran + slough + necro
        if total > 0:
            gran = (gran / total) * 100
            slough = (slough / total) * 100
            necro = (necro / total) * 100

        gran_list.append(round(gran, 1))
        slough_list.append(round(slough, 1))
        necro_list.append(round(necro, 1))

        # Stage from tissue
        if necro > 40:
            stages.append("Stage 4")
        elif necro > 15 or slough > 35:
            stages.append("Stage 3")
        elif gran > 60 and necro < 10:
            stages.append("Stage 2")
        else:
            stages.append("Stage 2")
        if area < 2.0 and gran > 80:
            stages[-1] = "Stage 1"
        if area < 0.8 and gran > 90:
            stages[-1] = "Healed"

    return pd.DataFrame({
        "Date": dates,
        "Week": list(range(weeks + 1)),
        "Area_cm2": areas,
        "Granulation": gran_list,
        "Slough": slough_list,
        "Necrotic": necro_list,
        "Stage": stages,
    })


# Create a seed string from the actual analysis data
seed_str = f"{current_gran:.1f}_{current_slough:.1f}_{current_necro:.1f}_{coverage_pct}"
df = generate_healing_projection(
    current_gran, current_slough, current_necro,
    estimated_area_cm2, weeks, healing_rate, seed_str
)

# -- Page Header --
st.markdown("# Healing Tracker")
st.markdown("Projected healing timeline based on your analyzed wound.")
st.markdown("")

# -- Current Wound Summary --
st.markdown("### Current Wound Assessment")
col_a, col_b, col_c, col_d, col_e = st.columns(5)
col_a.metric("Stage", current_stage)
col_b.metric("Granulation", f"{current_gran:.1f}%")
col_c.metric("Slough", f"{current_slough:.1f}%")
col_d.metric("Necrotic", f"{current_necro:.1f}%")
col_e.metric("Est. Area", f"{estimated_area_cm2} cm2")

st.markdown("---")

# -- Projected Summary --
st.markdown("### Projected Outcome")
p1, p2, p3, p4 = st.columns(4)

area_change_pct = ((df["Area_cm2"].iloc[-1] - df["Area_cm2"].iloc[0]) / df["Area_cm2"].iloc[0]) * 100
gran_change = df["Granulation"].iloc[-1] - df["Granulation"].iloc[0]

p1.metric("Starting Area", f"{df['Area_cm2'].iloc[0]} cm2")
p2.metric(f"Area at Week {weeks}", f"{df['Area_cm2'].iloc[-1]} cm2", f"{area_change_pct:.1f}%")
p3.metric("Projected Stage", df["Stage"].iloc[-1])
p4.metric("Projected Granulation", f"{df['Granulation'].iloc[-1]:.0f}%", f"+{gran_change:.0f}%")

st.markdown("")

# -- Chart 1: Wound Area Over Time --
st.markdown("### Wound Area Projection")

fig_area = go.Figure()
fig_area.add_trace(go.Scatter(
    x=df["Date"], y=df["Area_cm2"],
    mode="lines+markers",
    name="Wound Area",
    line=dict(color="#FF4B4B", width=3),
    marker=dict(size=6),
    fill="tozeroy",
    fillcolor="rgba(255, 75, 75, 0.1)",
))
# Mark today


fig_area.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    yaxis_title="Area (cm2)",
    xaxis_title="Date",
    height=350,
    margin=dict(l=20, r=20, t=10, b=20),
    font=dict(family="Inter"),
)
st.plotly_chart(fig_area, use_container_width=True)

# -- Chart 2: Tissue Composition Over Time --
st.markdown("### Tissue Composition Projection")

fig_tissue = go.Figure()
fig_tissue.add_trace(go.Scatter(
    x=df["Date"], y=df["Granulation"],
    mode="lines", name="Granulation",
    stackgroup="one",
    line=dict(color="#DC5040"),
    fillcolor="rgba(220, 80, 64, 0.6)",
))
fig_tissue.add_trace(go.Scatter(
    x=df["Date"], y=df["Slough"],
    mode="lines", name="Slough",
    stackgroup="one",
    line=dict(color="#E6D200"),
    fillcolor="rgba(230, 210, 0, 0.6)",
))
fig_tissue.add_trace(go.Scatter(
    x=df["Date"], y=df["Necrotic"],
    mode="lines", name="Necrotic",
    stackgroup="one",
    line=dict(color="#323228"),
    fillcolor="rgba(50, 50, 40, 0.8)",
))


fig_tissue.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    yaxis_title="Tissue %",
    xaxis_title="Date",
    yaxis=dict(range=[0, 100]),
    height=350,
    margin=dict(l=20, r=20, t=10, b=20),
    font=dict(family="Inter"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_tissue, use_container_width=True)

# -- Chart 3: Stage Timeline --
st.markdown("### Stage Progression")

stage_colors = {
    "Stage 4": "#F44336", "Stage 3": "#FF9800",
    "Stage 2": "#FFC107", "Stage 1": "#4CAF50", "Healed": "#00E676",
}

fig_stage = go.Figure()
for stage_name, color in stage_colors.items():
    mask = df["Stage"] == stage_name
    if mask.any():
        fig_stage.add_trace(go.Scatter(
            x=df.loc[mask, "Date"],
            y=df.loc[mask, "Stage"],
            mode="markers",
            name=stage_name,
            marker=dict(size=14, color=color, symbol="circle"),
        ))


fig_stage.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    yaxis_title="Stage",
    xaxis_title="Date",
    height=250,
    margin=dict(l=20, r=20, t=10, b=20),
    font=dict(family="Inter"),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_stage, use_container_width=True)

# -- Raw Data Table --
with st.expander("View Raw Projection Data"):
    st.dataframe(
        df.style.format({
            "Area_cm2": "{:.1f}",
            "Granulation": "{:.1f}%",
            "Slough": "{:.1f}%",
            "Necrotic": "{:.1f}%",
        }),
        use_container_width=True,
    )

# -- Disclaimer --
st.markdown("---")
st.caption(
    "This healing projection is generated based on the analyzed wound's actual tissue "
    "composition and estimated area. Projections assume standard healing rates and may "
    "not reflect individual patient outcomes. Always consult a healthcare provider."
)
