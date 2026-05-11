import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-City AQI Prediction Platform",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --accent: #00d4aa;
    --accent2: #ff6b6b;
    --accent3: #ffd93d;
    --text: #e6edf3;
    --muted: #8b949e;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.main { background-color: var(--bg); }

section[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent); }

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}

.metric-label {
    font-size: 0.8rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 6px;
}

.page-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
}
.page-subtitle {
    color: var(--muted);
    font-size: 0.95rem;
    margin-bottom: 28px;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 24px 0 12px 0;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
}

.stButton > button {
    background: var(--accent);
    color: #0d1117;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    padding: 10px 24px;
    font-size: 0.9rem;
}
.stButton > button:hover {
    background: #00b894;
    color: #0d1117;
}

.stPlotlyChart, .stImage { border-radius: 12px; }

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}

.stSlider > div { color: var(--accent); }

.stNumberInput input, .stTextInput input {
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
}

div[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CITIES = ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]

# This mapping comes from LabelEncoder in your notebook (alphabetical order)
# Bengaluru=0, Chennai=1, Delhi=2, Hyderabad=3, Kolkata=4, Mumbai=5
CITY_ENCODING = {
    "Bengaluru": 0,
    "Chennai":   1,
    "Delhi":     2,
    "Hyderabad": 3,
    "Kolkata":   4,
    "Mumbai":    5
}

ML_FEATURES = [
    "PM2.5","PM10","NO","NO2","NOx","CO","SO2","O3",
    "PM25_roll3","PM10_roll3",
    "Month","DayOfWeek","City_encoded"
]

PALETTE = ["#00d4aa", "#ff6b6b", "#ffd93d", "#6bcb77", "#a29bfe", "#fd79a8"]
CITY_COLORS = ["#00d4aa", "#ff6b6b", "#ffd93d", "#6bcb77", "#a29bfe", "#fd79a8"]

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("upgraded_aqi_dataset.csv", parse_dates=["Date"])
    return df.sort_values(["City", "Date"]).reset_index(drop=True)

@st.cache_resource
def load_ml_model():
    return joblib.load("best_aqi_model.pkl")

@st.cache_resource
def load_scalers():
    xs = joblib.load("x_scaler.pkl")
    ys = joblib.load("y_scaler.pkl")
    return xs, ys

def get_aqi_category(aqi):
    if aqi <= 50:    return "Good",        "#00d4aa", "🟢"
    elif aqi <= 100: return "Satisfactory", "#6bcb77", "🟡"
    elif aqi <= 200: return "Moderate",     "#ffd93d", "🟠"
    elif aqi <= 300: return "Poor",         "#ff9a3c", "🔴"
    elif aqi <= 400: return "Very Poor",    "#ff6b6b", "🟣"
    else:            return "Severe",       "#d63031", "⚫"

def apply_dark_style(fig, ax_list=None):
    fig.patch.set_facecolor('#161b22')
    if ax_list is None:
        ax_list = fig.get_axes()
    for ax in ax_list:
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.title.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
    return fig

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    df = load_data()
    ml_model = load_ml_model()
    x_scaler, y_scaler = load_scalers()
    data_loaded = True
except Exception as e:
    data_loaded = False
    load_error = str(e)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0;'>
        <span style='font-family: Space Mono; font-size: 1.3rem; color: #00d4aa;'>🌫️ AQI Platform</span><br>
        <span style='color: #8b949e; font-size: 0.8rem;'>6 Cities · Predictive Analytics</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    selected_city = st.selectbox("🏙️ Select City", CITIES, index=CITIES.index("Mumbai"))

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠  Overview",
         "📊  Pollutant Analysis",
         "🤖  ML Prediction",
         "🧠  LSTM Time-Series",
         "📈  Model Comparison",
         "⚡  Real-Time Simulation"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='color: #8b949e; font-size: 0.75rem; padding: 8px 0;'>
    <b style='color:#e6edf3;'>Project</b><br>
    Integrated Environmental Sensing & Predictive Analytics Platform for AQI Estimation<br><br>
    <b style='color:#e6edf3;'>Dataset</b><br>
    CPCB city_day.csv · 6 Indian Cities<br><br>
    <b style='color:#e6edf3;'>Models</b><br>
    Random Forest · Extra Trees · Gradient Boosting · XGBoost · SVR · Linear Regression · LSTM
    </div>
    """, unsafe_allow_html=True)

if not data_loaded:
    st.error(f"⚠️ Could not load files. Make sure all .pkl and .csv files are in the same folder as app.py.\n\nError: {load_error}")
    st.stop()

# Filter data for selected city
city_df = df[df["City"] == selected_city].copy()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown('<div class="page-title">Multi-City AQI Prediction Platform</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{selected_city} · Environmental Sensing & Predictive Analytics · Final Year Project</div>', unsafe_allow_html=True)

    # KPI row for selected city
    col1, col2, col3, col4 = st.columns(4)
    avg_aqi = city_df["AQI"].mean()
    max_aqi = city_df["AQI"].max()
    min_aqi = city_df["AQI"].min()
    n_days  = len(city_df)
    cat, color, emoji = get_aqi_category(avg_aqi)

    with col1: st.metric("📅 Total Records", f"{n_days:,}")
    with col2: st.metric("📊 Average AQI", f"{avg_aqi:.1f}", f"{cat}")
    with col3: st.metric("🔴 Peak AQI", f"{max_aqi:.1f}")
    with col4: st.metric("🟢 Lowest AQI", f"{min_aqi:.1f}")

    # AQI trend for selected city
    st.markdown(f'<div class="section-header">AQI Trend Over Time — {selected_city}</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(city_df["Date"], city_df["AQI"], alpha=0.15, color="#00d4aa")
    ax.plot(city_df["Date"], city_df["AQI"], color="#00d4aa", linewidth=1.2)
    ax.axhline(100, color="#ffd93d", linestyle="--", linewidth=0.8, alpha=0.6, label="Moderate threshold (100)")
    ax.axhline(200, color="#ff6b6b", linestyle="--", linewidth=0.8, alpha=0.6, label="Poor threshold (200)")
    ax.set_xlabel("Date"); ax.set_ylabel("AQI")
    ax.set_title(f"Daily AQI — {selected_city}")
    ax.legend(fontsize=8, facecolor='#161b22', labelcolor='#8b949e')
    apply_dark_style(fig)
    st.pyplot(fig); plt.close()

    # Monthly average for selected city
    st.markdown('<div class="section-header">Monthly Average AQI</div>', unsafe_allow_html=True)
    monthly = city_df.groupby("Month")["AQI"].mean().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["Month_Name"] = monthly["Month"].apply(lambda x: month_names[int(x)-1])
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.bar(monthly["Month_Name"], monthly["AQI"], color=PALETTE * 2, edgecolor="none", width=0.6)
    ax2.set_ylabel("Average AQI")
    ax2.set_title(f"Monthly Average AQI — {selected_city}")
    apply_dark_style(fig2)
    st.pyplot(fig2); plt.close()

    # AQI category pie
    st.markdown('<div class="section-header">AQI Category Distribution</div>', unsafe_allow_html=True)
    bins   = [0, 50, 100, 200, 300, 400, 1000]
    labels = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
    colors_pie = ["#00d4aa","#6bcb77","#ffd93d","#ff9a3c","#ff6b6b","#d63031"]
    city_df["AQI_Cat"] = pd.cut(city_df["AQI"], bins=bins, labels=labels)
    counts = city_df["AQI_Cat"].value_counts().reindex(labels).fillna(0)
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    wedges, texts, autotexts = ax3.pie(
        counts, labels=labels, autopct="%1.1f%%",
        colors=colors_pie, startangle=140,
        textprops={"color": "#e6edf3", "fontsize": 9}
    )
    for at in autotexts:
        at.set_color("#0d1117"); at.set_fontsize(8)
    ax3.set_title("AQI Category Breakdown", color="#e6edf3")
    fig3.patch.set_facecolor("#161b22")
    st.pyplot(fig3); plt.close()

    # ALL CITIES comparison — the new section mentor asked for
    st.markdown('<div class="section-header">AQI Comparison — All 6 Cities</div>', unsafe_allow_html=True)
    fig4, ax4 = plt.subplots(figsize=(14, 5))
    for i, city in enumerate(CITIES):
        cdf = df[df["City"] == city]
        monthly_c = cdf.groupby("Month")["AQI"].mean()
        ax4.plot(monthly_c.index, monthly_c.values,
                 color=CITY_COLORS[i], linewidth=2.2,
                 label=city, marker='o', markersize=5)
    ax4.set_xlabel("Month"); ax4.set_ylabel("Average AQI")
    ax4.set_title("Monthly Average AQI — All 6 Cities")
    ax4.set_xticks(range(1,13))
    ax4.set_xticklabels(month_names, fontsize=8)
    ax4.legend(facecolor='#161b22', labelcolor='#e6edf3', fontsize=9)
    apply_dark_style(fig4)
    st.pyplot(fig4); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — POLLUTANT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Pollutant" in page:
    st.markdown('<div class="page-title">Pollutant Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{selected_city} · Distribution, trends, and correlations of key pollutants</div>', unsafe_allow_html=True)

    pollutants = ["PM2.5","PM10","NO","NO2","NOx","CO","SO2","O3"]

    st.markdown('<div class="section-header">Average Pollutant Concentrations</div>', unsafe_allow_html=True)
    avgs = city_df[pollutants].mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(pollutants, avgs.values, color=PALETTE * 2, edgecolor="none", width=0.55)
    for bar, val in zip(bars, avgs.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, color="#e6edf3")
    ax.set_ylabel("Average Concentration (µg/m³ or ppb)")
    ax.set_title(f"Mean Pollutant Levels — {selected_city}")
    apply_dark_style(fig)
    st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">Correlation Heatmap — Pollutants vs AQI</div>', unsafe_allow_html=True)
    corr_cols = pollutants + ["AQI"]
    corr = city_df[corr_cols].corr()
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu",
                linewidths=0.5, linecolor="#30363d",
                ax=ax2, annot_kws={"size": 9}, cbar_kws={"shrink": 0.8})
    ax2.set_title("Pollutant Correlation Matrix", pad=15)
    apply_dark_style(fig2, [ax2])
    st.pyplot(fig2); plt.close()

    st.markdown("""
    <div class="card">
    <b style='color:#00d4aa;'>Key Finding</b><br>
    PM2.5 and PM10 show the highest correlation with AQI, confirming they are the dominant
    drivers of air quality. This validates our feature selection strategy for model training.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Pollutant Trend Over Time</div>', unsafe_allow_html=True)
    selected_p = st.selectbox("Select pollutant", pollutants)
    fig3, ax3 = plt.subplots(figsize=(13, 3.5))
    ax3.plot(city_df["Date"], city_df[selected_p], color="#00d4aa", linewidth=1, alpha=0.8)
    ax3.fill_between(city_df["Date"], city_df[selected_p], alpha=0.1, color="#00d4aa")
    ax3.set_ylabel(selected_p)
    ax3.set_title(f"{selected_p} Over Time — {selected_city}")
    apply_dark_style(fig3)
    st.pyplot(fig3); plt.close()

    st.markdown('<div class="section-header">Pollutant Distribution (Box Plot)</div>', unsafe_allow_html=True)
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    bp = ax4.boxplot(
        [city_df[p].dropna().values for p in pollutants],
        labels=pollutants, patch_artist=True,
        medianprops={"color": "#0d1117", "linewidth": 2}
    )
    for patch, color in zip(bp["boxes"], PALETTE * 2):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    for w in bp["whiskers"]: w.set_color("#8b949e")
    for c in bp["caps"]:     c.set_color("#8b949e")
    ax4.set_ylabel("Concentration")
    ax4.set_title(f"Pollutant Value Distribution — {selected_city}")
    apply_dark_style(fig4)
    st.pyplot(fig4); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif "ML Prediction" in page:
    st.markdown('<div class="page-title">ML-Based AQI Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Enter pollutant values to get an instant AQI prediction</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Enter Pollutant Readings</div>', unsafe_allow_html=True)

    defaults = {col: float(city_df[col].median()) for col in ["PM2.5","PM10","NO","NO2","NOx","CO","SO2","O3"]}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 1000.0, defaults["PM2.5"], 1.0)
        no   = st.number_input("NO (µg/m³)",    0.0, 500.0,  defaults["NO"],    0.5)
    with col2:
        pm10 = st.number_input("PM10 (µg/m³)",  0.0, 1000.0, defaults["PM10"],  1.0)
        no2  = st.number_input("NO2 (µg/m³)",   0.0, 500.0,  defaults["NO2"],   0.5)
    with col3:
        nox  = st.number_input("NOx (µg/m³)",   0.0, 500.0,  defaults["NOx"],   0.5)
        so2  = st.number_input("SO2 (µg/m³)",   0.0, 500.0,  defaults["SO2"],   0.5)
    with col4:
        co   = st.number_input("CO (mg/m³)",    0.0, 50.0,   defaults["CO"],    0.1)
        o3   = st.number_input("O3 (µg/m³)",    0.0, 500.0,  defaults["O3"],    0.5)

    st.markdown('<div class="section-header">Temporal & Location Context</div>', unsafe_allow_html=True)
    col5, col6, col7 = st.columns(3)
    with col5:
        pred_city = st.selectbox("City", CITIES, index=CITIES.index(selected_city))
    with col6:
        month = st.slider("Month", 1, 12, 6)
    with col7:
        dow = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

    input_data = pd.DataFrame([{
        "PM2.5": pm25, "PM10": pm10, "NO": no, "NO2": no2,
        "NOx": nox, "CO": co, "SO2": so2, "O3": o3,
        "PM25_roll3": pm25, "PM10_roll3": pm10,
        "AQI_lag1":  float(city_df["AQI"].median()),
        "AQI_roll3": float(city_df["AQI"].median()),
        "Month": month, "DayOfWeek": dow,
        "City_encoded": CITY_ENCODING[pred_city]
    }])

    if st.button("🔍 Predict AQI"):
        pred_aqi = ml_model.predict(input_data)[0]
        cat, color, emoji = get_aqi_category(pred_aqi)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color};">{pred_aqi:.1f}</div>
                <div class="metric-label">Predicted AQI</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:3rem; line-height:1;">{emoji}</div>
                <div class="metric-label" style="margin-top:8px;">{cat}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            health_msgs = {
                "Good":         "Air quality is satisfactory. Enjoy outdoor activities.",
                "Satisfactory": "Air quality acceptable. Sensitive individuals take note.",
                "Moderate":     "Moderate pollution. Limit prolonged outdoor exertion.",
                "Poor":         "Poor air quality. Avoid outdoor activities if possible.",
                "Very Poor":    "Very poor air. Stay indoors and use air purifiers.",
                "Severe":       "Severe pollution. Health emergency — stay indoors."
            }
            st.markdown(f"""
            <div class="metric-card" style="text-align:left;">
                <div style="color:#00d4aa; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">Health Advisory</div>
                <div style="font-size:0.9rem; color:#e6edf3;">{health_msgs[cat]}</div>
            </div>""", unsafe_allow_html=True)

        # Gauge chart
        fig, ax = plt.subplots(figsize=(8, 3.5), subplot_kw={"projection": "polar"})
        fig.patch.set_facecolor("#161b22")
        ax.set_facecolor("#161b22")
        categories_gauge = [(0,50,"#00d4aa"),(50,100,"#6bcb77"),(100,200,"#ffd93d"),
                            (200,300,"#ff9a3c"),(300,400,"#ff6b6b"),(400,500,"#d63031")]
        for lo, hi, c in categories_gauge:
            thetas = np.linspace(np.pi*(1-hi/500), np.pi*(1-lo/500), 50)
            for i in range(len(thetas)-1):
                ax.fill_between([thetas[i+1], thetas[i]], 0.6, 1.0, color=c, alpha=0.8)
        needle_angle = np.pi * (1 - min(pred_aqi, 500)/500)
        ax.annotate("", xy=(needle_angle, 0.85), xytext=(needle_angle, 0),
                    arrowprops=dict(arrowstyle="->", color="white", lw=2.5))
        ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_theta_zero_location("W"); ax.set_theta_direction(-1)
        ax.set_thetamin(0); ax.set_thetamax(180)
        ax.text(np.pi/2, 0.3, f"{pred_aqi:.0f}", ha="center", va="center",
                fontsize=22, fontweight="bold", color=color, transform=ax.transData)
        ax.set_title("AQI Gauge", color="#e6edf3", pad=15)
        st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LSTM
# ═══════════════════════════════════════════════════════════════════════════════
elif "LSTM" in page:
    st.markdown('<div class="page-title">LSTM Time-Series Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Deep learning model for sequential AQI forecasting</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <b style='color:#00d4aa;'>Architecture</b><br>
    LSTM (64 units) → Dropout (0.2) → Dense (32, ReLU) → Dense (1)<br>
    <b>Window size:</b> 7 days &nbsp;|&nbsp; <b>Optimizer:</b> Adam &nbsp;|&nbsp;
    <b>Loss:</b> MSE &nbsp;|&nbsp; <b>Epochs:</b> 30 &nbsp;|&nbsp;
    <b>Input features:</b> 13 (pollutants + temporal + city)
    </div>
    """, unsafe_allow_html=True)

    try:
        from tensorflow.keras.models import load_model
        lstm_model = load_model("lstm_aqi_model.h5")
        lstm_loaded = True
    except:
        try:
            from tensorflow.keras.models import load_model
            lstm_model = load_model("lstm_aqi_model.keras")
            lstm_loaded = True
        except Exception as e:
            lstm_loaded = False
            lstm_err = str(e)

    if not lstm_loaded:
        st.warning(f"⚠️ LSTM model file not found. Make sure lstm_aqi_model.h5 is in the same folder.")
        arch_data = {
            "Layer":        ["Input", "LSTM (64)", "Dropout (0.2)", "Dense (32)", "Output"],
            "Output Shape": ["(7, 13)", "(64,)", "(64,)", "(32,)", "(1,)"],
            "Parameters":   ["-", "20,736", "0", "2,080", "33"]
        }
        st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)
    else:
        dl_df = df.sort_values(["City", "Date"]).copy()
        feat_cols = [c for c in ML_FEATURES if c in dl_df.columns]
        X_dl = dl_df[feat_cols].values
        y_dl = dl_df["AQI"].values.reshape(-1, 1)

        X_scaled = x_scaler.transform(X_dl)
        y_scaled = y_scaler.transform(y_dl)

        def create_sequences(X, y, window=7):
            Xs, ys = [], []
            for i in range(window, len(X)):
                Xs.append(X[i-window:i]); ys.append(y[i])
            return np.array(Xs), np.array(ys)

        X_seq, y_seq = create_sequences(X_scaled, y_scaled, 7)
        split = int(0.8 * len(X_seq))
        X_test_lstm = X_seq[split:]
        y_test_lstm = y_seq[split:]

        lstm_pred_scaled = lstm_model.predict(X_test_lstm, verbose=0)
        lstm_pred  = y_scaler.inverse_transform(lstm_pred_scaled)
        y_test_act = y_scaler.inverse_transform(y_test_lstm)

        st.markdown('<div class="section-header">Actual vs LSTM Predicted AQI</div>', unsafe_allow_html=True)
        n_show = st.slider("Number of data points to display", 30, min(200, len(y_test_act)), 80)

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(y_test_act[:n_show], color="#00d4aa", linewidth=1.5, label="Actual AQI")
        ax.plot(lstm_pred[:n_show],  color="#ff6b6b", linewidth=1.5, label="LSTM Predicted", linestyle="--")
        ax.fill_between(range(n_show),
                        y_test_act[:n_show].flatten(),
                        lstm_pred[:n_show].flatten(),
                        alpha=0.1, color="#ffd93d")
        ax.set_xlabel("Time Step"); ax.set_ylabel("AQI")
        ax.set_title("LSTM: Actual vs Predicted AQI — Multi-City Test Set")
        ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
        apply_dark_style(fig)
        st.pyplot(fig); plt.close()

        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        r2   = r2_score(y_test_act, lstm_pred)
        mae  = mean_absolute_error(y_test_act, lstm_pred)
        rmse = np.sqrt(mean_squared_error(y_test_act, lstm_pred))

        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score", f"{r2:.4f}")
        c2.metric("MAE",      f"{mae:.2f}")
        c3.metric("RMSE",     f"{rmse:.2f}")

    st.markdown('<div class="section-header">Why LSTM for AQI?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    AQI is a time-dependent variable — today's pollution is influenced by yesterday's.
    Traditional ML models treat each row independently, ignoring temporal dependencies.
    LSTM (Long Short-Term Memory) networks maintain a memory of past inputs across a
    sliding window (7 days in our case), making them well-suited for forecasting sequential
    environmental data. Training on 6 cities gives the model richer seasonal and
    geographic patterns compared to a single-city dataset.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif "Model Comparison" in page:
    st.markdown('<div class="page-title">Model Performance Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">7 algorithms benchmarked — ML ensemble methods vs Deep Learning</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="border-color:#ffd93d;">
    ⚠️ <b style='color:#ffd93d;'>Important:</b> Replace the placeholder values below with your actual
    results from the notebook's Cell 23 comparison table before presenting.
    </div>
    """, unsafe_allow_html=True)

    results = pd.DataFrame([
        {"Model": "Random Forest",     "R² Score": 0.9635, "MAE": 9.31,  "RMSE": 12.69},
        {"Model": "Extra Trees",       "R² Score": 0.9612, "MAE": 9.29,  "RMSE": 13.09},
        {"Model": "Gradient Boosting", "R² Score": 0.947,  "MAE": 10.82, "RMSE": 15.29},
        {"Model": "XGBoost",           "R² Score": 0.9448, "MAE": 11.15, "RMSE": 15.61},
        {"Model": "SVR",               "R² Score": 0.8662, "MAE": 16.2,  "RMSE": 24.3},
        {"Model": "Linear Regression", "R² Score": 0.9048, "MAE": 15.78, "RMSE": 20.5},
        {"Model": "LSTM",              "R² Score": 0.7108, "MAE": 36.68, "RMSE": 42.01},
    ])
    # ────────────────────────────────────────────────────────────────────────

    results = results.sort_values("R² Score", ascending=False).reset_index(drop=True)
    results.insert(0, "Rank", range(1, len(results)+1))

    st.markdown('<div class="section-header">Performance Table</div>', unsafe_allow_html=True)
    st.dataframe(
        results.style
               .highlight_max(subset=["R² Score"], color="#1a3a2a")
               .highlight_min(subset=["MAE","RMSE"], color="#1a3a2a")
               .format({"R² Score": "{:.4f}", "MAE": "{:.2f}", "RMSE": "{:.2f}"}),
        use_container_width=True, hide_index=True
    )

    best_model_name = results.iloc[0]["Model"]
    best_r2         = results.iloc[0]["R² Score"]
    st.success(f"🏆 Best Model: **{best_model_name}** with R² = {best_r2:.4f}")

    # Visual bar comparison
    st.markdown('<div class="section-header">Visual Comparison</div>', unsafe_allow_html=True)
    bar_colors = ["#00d4aa","#6bcb77","#ffd93d","#a29bfe","#ff9a3c","#74b9ff","#ff6b6b"]
    short_names = [m[:6] for m in results["Model"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for idx, (metric, ax) in enumerate(zip(["R² Score","MAE","RMSE"], axes)):
        bars = ax.bar(short_names, results[metric],
                      color=bar_colors, edgecolor="none", width=0.6)
        for bar, val in zip(bars, results[metric]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="#e6edf3")
        ax.set_title(f"{metric} Comparison")
        ax.set_xticklabels(short_names, rotation=20, ha='right', fontsize=7)
    apply_dark_style(fig, list(axes))
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Feature importance
    st.markdown('<div class="section-header">Feature Importance — Random Forest</div>', unsafe_allow_html=True)
    try:
        fi = pd.Series(ml_model.feature_importances_, index=ML_FEATURES).sort_values(ascending=True)
        fig_fi, ax_fi = plt.subplots(figsize=(10, 5))
        colors_fi = ["#00d4aa" if v > fi.median() else "#8b949e" for v in fi.values]
        ax_fi.barh(fi.index, fi.values, color=colors_fi, edgecolor="none")
        for i, (idx_val, val) in enumerate(zip(fi.index, fi.values)):
            ax_fi.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=8, color="#e6edf3")
        ax_fi.set_title("Feature Importance — which pollutants drive AQI?")
        ax_fi.set_xlabel("Importance Score")
        apply_dark_style(fig_fi, [ax_fi])
        plt.tight_layout()
        st.pyplot(fig_fi); plt.close()
    except AttributeError:
        st.info("Feature importance is available for tree-based models. If SVR or Linear Regression is the best model, this chart is not applicable.")

    # Research insight
    st.markdown('<div class="section-header">Research Insight</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <b style='color:#00d4aa;'>Key Research Findings</b><br><br>
    Models were evaluated using a rigorous temporal split — trained on 2015–2019 data and
    tested on Jan–Jul 2020 — with 5-fold cross-validation on the training set to assess stability.<br><br>
    <b style='color:#ffd93d;'>Finding 1:</b> Random Forest achieved the highest Test R² of 0.9635,
    consistent with existing literature on ensemble methods for AQI prediction.<br><br>
    <b style='color:#ffd93d;'>Finding 2:</b> LSTM ranked last (Test R² 0.7108) despite being a
    deep learning model — confirming that LSTMs require substantially larger sequential datasets
    to outperform classical ML methods.<br><br>
    <b style='color:#ffd93d;'>Finding 3:</b> All CV R² means (~0.63) are lower than Test R² (~0.95),
    indicating strong seasonal variation in pollution patterns across the training folds — a known
    characteristic of Indian urban air quality data.
    </div>
    """, unsafe_allow_html=True)

    # Radar chart
    st.markdown('<div class="section-header">Model Capability Radar</div>', unsafe_allow_html=True)
    categories = ["Accuracy (R²)", "Speed", "Interpretability", "Scalability", "Temporal Awareness"]
    model_scores = {
    "Random Forest":     [0.96, 0.80, 0.75, 0.80, 0.20],
    "Extra Trees":       [0.96, 0.88, 0.72, 0.80, 0.20],
    "Gradient Boosting": [0.95, 0.65, 0.68, 0.75, 0.20],
    "XGBoost":           [0.94, 0.78, 0.60, 0.85, 0.25],
    "Linear Regression": [0.90, 0.98, 0.98, 0.95, 0.15],
    "SVR":               [0.87, 0.55, 0.50, 0.60, 0.15],
    "LSTM":              [0.71, 0.40, 0.35, 0.90, 0.98],
    }
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig2, ax2 = plt.subplots(figsize=(8, 6), subplot_kw={"polar": True})
    fig2.patch.set_facecolor("#161b22")
    ax2.set_facecolor("#0d1117")
    radar_colors = ["#00d4aa","#6bcb77","#ffd93d","#a29bfe","#ff9a3c","#74b9ff","#ff6b6b"]
    for (mname, scores), col in zip(model_scores.items(), radar_colors):
        vals = scores + scores[:1]
        ax2.plot(angles, vals, color=col, linewidth=1.8, label=mname)
        ax2.fill(angles, vals, color=col, alpha=0.06)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=8, color="#e6edf3")
    ax2.set_yticklabels([])
    ax2.spines["polar"].set_edgecolor("#30363d")
    ax2.grid(color="#30363d", linewidth=0.5)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15),
               facecolor="#161b22", labelcolor="#e6edf3", fontsize=7)
    ax2.set_title("Model Capability Radar", color="#e6edf3", pad=20)
    st.pyplot(fig2); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — REAL-TIME SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
elif "Real-Time" in page:
    st.markdown('<div class="page-title">Real-Time AQI Simulation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{selected_city} · Streaming dataset rows through the ML model</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    This simulation mimics an IoT sensor pipeline: each row of the dataset is fed into the
    trained model one at a time, replicating how a real-time environmental monitoring system
    would operate — ingesting sensor readings and predicting AQI continuously.
    </div>
    """, unsafe_allow_html=True)

    feat_cols = [c for c in ML_FEATURES if c in city_df.columns]
    sample_df = city_df[feat_cols + ["AQI", "Date"]].dropna().tail(50).reset_index(drop=True)

    n_steps = st.slider("Number of steps to simulate", 5, 50, 20)
    speed   = st.select_slider("Simulation speed",
                                ["Slow (1s)", "Normal (0.5s)", "Fast (0.2s)"],
                                value="Normal (0.5s)")
    delay_map = {"Slow (1s)": 1.0, "Normal (0.5s)": 0.5, "Fast (0.2s)": 0.2}
    delay = delay_map[speed]

    if st.button("▶ Start Simulation"):
        col_a, col_b = st.columns([2, 1])
        chart_ph  = col_a.empty()
        metric_ph = col_b.empty()
        log_ph    = st.empty()

        actuals, preds, dates_sim, log_rows = [], [], [], []

        for i in range(min(n_steps, len(sample_df))):
            row    = sample_df.iloc[[i]]
            X_row  = row[feat_cols]
            pred   = ml_model.predict(X_row)[0]
            actual = row["AQI"].values[0]
            date_v = row["Date"].values[0]

            actuals.append(actual); preds.append(pred); dates_sim.append(i)
            cat, color, emoji = get_aqi_category(pred)

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(dates_sim, actuals, color="#00d4aa", linewidth=1.8,
                    label="Actual AQI", marker="o", markersize=3)
            ax.plot(dates_sim, preds,   color="#ff6b6b", linewidth=1.8,
                    label="Predicted AQI", marker="s", markersize=3, linestyle="--")
            ax.set_xlabel("Time Step"); ax.set_ylabel("AQI")
            ax.set_title(f"Live AQI Feed — {selected_city}")
            ax.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=8)
            apply_dark_style(fig)
            chart_ph.pyplot(fig); plt.close()

            metric_ph.markdown(f"""
            <div class="metric-card" style="margin-top:0;">
                <div style="color:#8b949e; font-size:0.75rem;">STEP {i+1}</div>
                <div class="metric-value" style="color:{color}; margin: 8px 0;">{pred:.1f}</div>
                <div style="font-size:1.5rem;">{emoji}</div>
                <div style="color:{color}; font-size:0.9rem; margin-top:4px;">{cat}</div>
                <div style="color:#8b949e; font-size:0.75rem; margin-top:8px;">Actual: {actual:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

            log_rows.append({"Step": i+1, "Date": str(date_v)[:10],
                             "Actual AQI": f"{actual:.1f}",
                             "Predicted AQI": f"{pred:.1f}", "Category": cat})
            log_ph.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
            time.sleep(delay)

        st.success(f"✅ Simulation complete — {n_steps} steps processed for {selected_city}.")