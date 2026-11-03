import sys
from pathlib import Path

# Make the project root importable (fixes "No module named 'src'")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

from src.data.load_data import load_data


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_DIR = ROOT / "models"
API_URL = "http://127.0.0.1:8000"
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
RISK_ORDER = ["CRITICAL", "HIGH", "MODERATE", "LOW"]
RISK_COLORS = {
    "CRITICAL": "#ff4b4b",
    "HIGH": "#ff9f43",
    "MODERATE": "#feca57",
    "LOW": "#2ecc71",
}
# Bright, high-contrast colours that stay readable on a dark background
PALETTE = [
    "#4f8cff", "#ff6b6b", "#2ecc71", "#feca57", "#a55eea",
    "#26de81", "#fd9644", "#45aaf2", "#fc5c65", "#2bcbba",
]

st.set_page_config(
    page_title="PredictX | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)


# --------------------------------------------------
# STYLING
# --------------------------------------------------

st.markdown(
    """
<style>
    /* Streamlit's fixed top toolbar overlaps content, so leave room for it */
    .block-container { padding-top: 4.5rem; padding-bottom: 2rem; max-width: 1400px; }

    div[data-testid="stMetric"] {
        background-color: #161b26;
        padding: 16px 18px;
        border-radius: 12px;
        border: 1px solid #263044;
    }
    div[data-testid="stMetricLabel"] p { color: #9aa4b5; font-size: 0.85rem; }
    div[data-testid="stMetricValue"] { font-size: 1.7rem; }

    section[data-testid="stSidebar"] { border-right: 1px solid #263044; }

    .hero-title { font-size: 2.2rem; font-weight: 700; line-height: 1.3; margin: 0; padding-top: 0.25rem; }
    .hero-sub   { color: #9aa4b5; margin-top: 0.2rem; }

    .alert-card {
        background-color: #161b26;
        padding: 20px 24px;
        border-radius: 12px;
        border: 1px solid #263044;
    }
    .alert-card h4 { margin: 0; color: #9aa4b5; font-weight: 500; font-size: 0.9rem; }
    .alert-card h2 { margin: 6px 0 10px 0; }
    .alert-card p  { margin: 0; color: #9aa4b5; }

    .badge {
        display: inline-block; padding: 2px 12px; border-radius: 999px;
        font-weight: 600; font-size: 0.85rem; color: #0e1117;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def get_dataset():
    _, test, _ = load_data()
    return test


@st.cache_data
def get_health_report():
    path = MODEL_DIR / "engine_health_report.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


test_data = get_dataset()
health_report = get_health_report()


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def get_risk_color(risk):
    return RISK_COLORS.get(str(risk).upper(), "#808080")


def style_fig(fig, height=380):
    """Apply one consistent look to every chart."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        # Title lives at the top; legend sits below the plot so they never collide
        margin=dict(l=10, r=10, t=70, b=105),
        title=dict(x=0, xanchor="left", y=0.96, yanchor="top", pad=dict(t=0, b=10)),
        # Anchor the legend to the bottom edge of the whole figure (not the plot
        # area), so it can never land on top of the x-axis title or tick labels.
        legend=dict(
            orientation="h",
            yref="container",
            yanchor="bottom",
            y=0.01,
            xref="container",
            xanchor="left",
            x=0.01,
            title_text="",
        ),
        font=dict(size=13),
    )
    fig.update_xaxes(gridcolor="#263044", zeroline=False, title_standoff=12)
    fig.update_yaxes(gridcolor="#263044", zeroline=False)
    return fig


def normalize(df, cols):
    """Min-max scale each column to 0-1 so different sensors are comparable."""
    out = df.copy()
    for c in cols:
        span = out[c].max() - out[c].min()
        out[c] = 0.0 if span == 0 else (out[c] - out[c].min()) / span
    return out


def sensor_panels(df, sensors, title):
    """One stacked panel per sensor, each with its own y-axis scale."""
    fig = make_subplots(
        rows=len(sensors),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
    )
    for i, s in enumerate(sensors, start=1):
        fig.add_trace(
            go.Scatter(
                x=df["cycle"],
                y=df[s],
                mode="lines",
                name=s,
                line=dict(color=PALETTE[(i - 1) % len(PALETTE)], width=2),
            ),
            row=i,
            col=1,
        )
        fig.update_yaxes(title_text=s, row=i, col=1)
    fig.update_xaxes(title_text="Operating cycle", row=len(sensors), col=1)
    fig.update_layout(title=title, showlegend=False)
    return style_fig(fig, height=max(320, 190 * len(sensors) + 80))


def predict_engine(engine_id):
    engine_history = (
        test_data[test_data["unit"] == engine_id]
        .sort_values("cycle")
        .drop(columns=["unit"])
    )

    history = [
        {k: int(v) if k == "cycle" else float(v) for k, v in rec.items()}
        for rec in engine_history.to_dict(orient="records")
    ]

    response = requests.post(
        f"{API_URL}/predict",
        json={"engine_id": int(engine_id), "history": history},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def rul_gauge(rul, color):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(rul),
            number={"suffix": " cycles", "font": {"size": 34}},
            title={"text": "Remaining Useful Life"},
            gauge={
                "axis": {"range": [0, max(150, float(rul) * 1.25)]},
                "bar": {"color": color},
                "bgcolor": "#161b26",
                "borderwidth": 0,
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=20, r=20, t=60, b=10),
    )
    return fig


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ PredictX")
st.sidebar.caption("Predictive Maintenance Platform")

page = st.sidebar.radio(
    "Navigation",
    ["Fleet Overview", "Engine Analysis", "Sensor Analytics"],
)

st.sidebar.divider()
st.sidebar.markdown("**System Status**")

try:
    requests.get(f"{API_URL}/health", timeout=5).raise_for_status()
    st.sidebar.success("Prediction API online")
except requests.RequestException:
    st.sidebar.error("Prediction API offline")

st.sidebar.caption("Dataset: NASA C-MAPSS FD004")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown('<p class="hero-title">PredictX</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Remaining Useful Life prediction · '
    "Anomaly detection · Engine health monitoring</p>",
    unsafe_allow_html=True,
)
st.divider()


# --------------------------------------------------
# PAGE 1: FLEET OVERVIEW
# --------------------------------------------------

if page == "Fleet Overview":

    st.subheader("Fleet Health Overview")

    if health_report.empty:
        st.error("Health report not found. Run the health report generation script first.")
        st.stop()

    total = len(health_report)
    counts = health_report["risk_level"].value_counts()
    anomalies = int(health_report["anomaly_detected"].astype(bool).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Engines", total)
    c2.metric("Critical", int(counts.get("CRITICAL", 0)))
    c3.metric("High Risk", int(counts.get("HIGH", 0)))
    c4.metric("Anomalies", anomalies)
    if "predicted_RUL" in health_report.columns:
        c5.metric("Avg Predicted RUL", f"{health_report['predicted_RUL'].mean():.0f} cycles")

    st.write("")
    left, right = st.columns(2)

    with left:
        risk_counts = (
            health_report["risk_level"]
            .value_counts()
            .reindex(RISK_ORDER, fill_value=0)
            .reset_index()
        )
        risk_counts.columns = ["Risk Level", "Engines"]
        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Engines",
            hole=0.55,
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            title="Fleet Risk Distribution",
        )
        fig.update_traces(textinfo="label+percent")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with right:
        if "predicted_RUL" in health_report.columns:
            fig = px.histogram(
                health_report,
                x="predicted_RUL",
                color="risk_level",
                nbins=30,
                color_discrete_map=RISK_COLORS,
                category_orders={"risk_level": RISK_ORDER},
                title="Predicted RUL Distribution",
                labels={"predicted_RUL": "Predicted RUL (cycles)", "risk_level": "Risk"},
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("Engine Health Report")

    selected_risks = st.multiselect(
        "Filter by risk level", RISK_ORDER, default=RISK_ORDER, key="risk_filter"
    )
    table = health_report[health_report["risk_level"].isin(selected_risks)]
    if "predicted_RUL" in table.columns:
        table = table.sort_values("predicted_RUL")

    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download report (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name="engine_health_report.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# PAGE 2: ENGINE ANALYSIS
# --------------------------------------------------

elif page == "Engine Analysis":

    st.subheader("Individual Engine Analysis")

    engine_ids = sorted(test_data["unit"].unique())

    sel_col, btn_col = st.columns([3, 1], vertical_alignment="bottom")
    selected_engine = sel_col.selectbox("Select Engine ID", engine_ids, key="engine_select")
    analyze = btn_col.button("Analyze Engine", type="primary", use_container_width=True)

    # Run prediction only on click, store it so later widget changes
    # (e.g. adding sensors) do NOT wipe the result.
    if analyze:
        with st.spinner("Analyzing engine sensor history..."):
            try:
                st.session_state["prediction"] = {
                    "engine": selected_engine,
                    "result": predict_engine(selected_engine),
                    "error": None,
                }
            except requests.RequestException as e:
                st.session_state["prediction"] = {
                    "engine": selected_engine,
                    "result": None,
                    "error": f"Could not connect to prediction API: {e}",
                }
            except Exception as e:
                st.session_state["prediction"] = {
                    "engine": selected_engine,
                    "result": None,
                    "error": f"Prediction failed: {e}",
                }

    pred = st.session_state.get("prediction")

    if pred and pred["engine"] == selected_engine:
        if pred["error"]:
            st.error(pred["error"])
        else:
            result = pred["result"]
            risk = result["risk_level"]
            color = get_risk_color(risk)

            g_col, m_col = st.columns([1, 1])

            with g_col:
                st.plotly_chart(
                    rul_gauge(result["predicted_rul"], color),
                    use_container_width=True,
                )

            with m_col:
                m1, m2 = st.columns(2)
                m1.metric("Predicted RUL", f'{result["predicted_rul"]:.2f} cycles')
                m2.metric(
                    "Anomaly Status",
                    "Detected" if result["anomaly_detected"] else "Normal",
                )

                try:
                    score = f'{float(result["anomaly_score"]):.4f}'
                except (TypeError, ValueError):
                    score = str(result["anomaly_score"])

                st.markdown(
                    f"""
                    <div class="alert-card" style="border-left:6px solid {color}; margin-top:12px;">
                        <h4>MAINTENANCE RECOMMENDATION</h4>
                        <h2 style="color:{color}">{result["maintenance_alert"]}</h2>
                        <p>Risk level:
                            <span class="badge" style="background:{color}">{risk}</span>
                            &nbsp; Anomaly score: {score}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("Select an engine and click **Analyze Engine** to get its RUL prediction.")

    # ---- Sensor history (independent of the prediction) ----
    st.divider()
    st.subheader("Engine Sensor History")

    engine_history = (
        test_data[test_data["unit"] == selected_engine].sort_values("cycle")
    )

    ctl1, ctl2 = st.columns([3, 1], vertical_alignment="bottom")
    selected_sensors = ctl1.multiselect(
        "Choose sensors to visualize",
        SENSORS,
        default=["sensor_11", "sensor_13", "sensor_15"],
        key="history_sensors",
    )
    view = ctl2.radio(
        "View",
        ["Separate panels", "Normalized overlay"],
        key="history_view",
    )

    if not selected_sensors:
        st.info("Select at least one sensor to plot.")
    else:
        title = f"Engine {selected_engine} sensor trends"
        if view == "Separate panels":
            fig = sensor_panels(engine_history, selected_sensors, title)
        else:
            melted = normalize(engine_history, selected_sensors).melt(
                id_vars="cycle",
                value_vars=selected_sensors,
                var_name="Sensor",
                value_name="Normalized reading (0-1)",
            )
            fig = px.line(
                melted,
                x="cycle",
                y="Normalized reading (0-1)",
                color="Sensor",
                color_discrete_sequence=PALETTE,
                title=title + " (each scaled 0-1)",
            )
            fig.update_traces(line=dict(width=2))
            style_fig(fig, height=460)

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Sensors have very different scales (e.g. ~2388 vs ~8), so a single "
            "axis would flatten the small ones. Panels give each sensor its own axis."
        )


# --------------------------------------------------
# PAGE 3: SENSOR ANALYTICS
# --------------------------------------------------

elif page == "Sensor Analytics":

    st.subheader("Sensor Analytics")

    engine_ids = sorted(test_data["unit"].unique())

    f1, f2, f3 = st.columns([1, 2, 1], vertical_alignment="bottom")
    sensor = f1.selectbox("Select Sensor", SENSORS, index=10, key="analytics_sensor")
    selected_engines = f2.multiselect(
        "Select Engines", engine_ids, default=engine_ids[:5], key="analytics_engines"
    )
    scale = f3.toggle("Normalize 0-1", value=False, key="analytics_norm")

    if not selected_engines:
        st.info("Select at least one engine to view sensor analytics.")
    else:
        sensor_data = test_data[test_data["unit"].isin(selected_engines)].copy()
        sensor_data["unit"] = sensor_data["unit"].astype(str)

        y_col = sensor
        if scale:
            sensor_data = normalize(sensor_data, [sensor])

        fig = px.line(
            sensor_data.sort_values(["unit", "cycle"]),
            x="cycle",
            y=y_col,
            color="unit",
            color_discrete_sequence=PALETTE,
            title=f"{sensor} trends across engines",
            labels={"unit": "Engine", "cycle": "Operating cycle", y_col: "Sensor reading"},
        )
        fig.update_traces(line=dict(width=2))
        st.plotly_chart(style_fig(fig, height=460), use_container_width=True)

        st.subheader("Sensor Statistics")
        stats = (
            test_data[test_data["unit"].isin(selected_engines)]
            .groupby("unit")[sensor]
            .agg(["mean", "std", "min", "max"])
            .round(3)
            .reset_index()
        )
        st.dataframe(stats, use_container_width=True, hide_index=True)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "PredictX · Built with Python, Scikit-learn, FastAPI, Streamlit and Plotly · "
    "NASA C-MAPSS FD004"
)