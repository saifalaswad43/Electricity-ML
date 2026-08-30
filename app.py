
import calendar
import glob
import os
import warnings
import json
from datetime import datetime
import hashlib

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Config & Styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="⚡ Electricity Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Professional
st.markdown("""
<style>
    /* Global */
    .main { background: #f8f9fa; }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0,0,0,0.2);
    }
    .metric-card .value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.5rem 0;
        letter-spacing: -1px;
    }
    .metric-card .label {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 500;
    }
    .metric-card .sub {
        font-size: 0.85rem;
        opacity: 0.7;
        margin-top: 0.25rem;
    }
    
    /* Color Variants */
    .metric-card.blue { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .metric-card.green { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    .metric-card.orange { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .metric-card.purple { background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%); }
    .metric-card.dark { background: linear-gradient(135deg, #2d3436 0%, #000000 100%); }
    .metric-card.gold { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); color: #2d3436; }
    .metric-card.red { background: linear-gradient(135deg, #eb4d4b 0%, #f0932b 100%); }
    
    /* Section Title */
    .section-title {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3436;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 4px solid #667eea;
        display: inline-block;
    }
    .section-sub {
        font-size: 1.1rem;
        color: #636e72;
        margin-bottom: 2rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 50px;
        transition: all 0.3s ease;
        width: 100%;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #f1f3f5;
        padding: 0.5rem;
        border-radius: 50px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        color: #636e72;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: #667eea;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Warning Box */
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 5px solid #f9ca24;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(249, 202, 36, 0.2);
    }
    .warning-box .title {
        font-weight: 700;
        font-size: 1.2rem;
        color: #2d3436;
    }
    .warning-box .content {
        color: #636e72;
        margin-top: 0.5rem;
    }
    .warning-box .icon {
        font-size: 2rem;
        margin-right: 0.5rem;
    }
    
    /* Info Box */
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .info-box.success { border-left-color: #43e97b; }
    .info-box.warning { border-left-color: #f9ca24; }
    .info-box.danger { border-left-color: #eb4d4b; }
    
    /* History Card */
    .history-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
        transition: transform 0.2s ease;
    }
    .history-card:hover {
        transform: translateX(5px);
    }
    .history-card .date {
        color: #636e72;
        font-size: 0.85rem;
    }
    .history-card .value {
        font-weight: 700;
        font-size: 1.1rem;
    }
    .history-card .badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-low { background: #43e97b; color: white; }
    .badge-medium { background: #f9ca24; color: #2d3436; }
    .badge-high { background: #f0932b; color: white; }
    .badge-critical { background: #eb4d4b; color: white; }
    
    /* Progress Bar */
    .progress-container {
        background: #f1f3f5;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .progress-bar {
        height: 12px;
        border-radius: 10px;
        background: #e0e0e0;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .progress-bar .fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1.5s ease;
    }
    .progress-bar .fill.good { background: linear-gradient(90deg, #43e97b, #38f9d7); }
    .progress-bar .fill.warning { background: linear-gradient(90deg, #f9ca24, #f0932b); }
    .progress-bar .fill.danger { background: linear-gradient(90deg, #f0932b, #eb4d4b); }
    
    /* Comparison Box */
    .comparison-box {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    
    /* Sidebar */
    .css-1d391kg { background: #ffffff; }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 10px;
        font-weight: 600;
        color: #2d3436;
    }
    
    /* About Page */
    .about-section {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    .about-section h3 {
        color: #2d3436;
        margin-bottom: 1rem;
    }
    .about-section .emoji-big {
        font-size: 4rem;
        text-align: center;
        display: block;
        margin: 1rem 0;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
</style>
""", unsafe_allow_html=True)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

CANDIDATE_DIRS = [
    os.path.join(APP_DIR, "artifacts"),
    APP_DIR,
    os.getcwd(),
]

ARTIFACT_SUFFIXES = {
    "selected_features": "selected_features.pkl",
    "usage_level_encoder": "usage_level_encoder.pkl",
    "best_model": "best_model.pkl",
    "categorical_imputer": "categorical_imputer.pkl",
    "cost_per_kwh_bounds": "cost_per_kwh_bounds.pkl",
    "feature_columns_config": "feature_columns_config.pkl",
    "is_heavy_appliance_encoder": "is_heavy_appliance_encoder.pkl",
    "month_encoder": "month_encoder.pkl",
    "numeric_imputer": "numeric_imputer.pkl",
    "onehot_encoder": "onehot_encoder.pkl",
    "robust_scaler": "robust_scaler.pkl",
    "scaled_columns": "scaled_columns.pkl",
}

# Text cleaning maps
MONTH_MAP = {
    "jan": "January", "january": "January",
    "feb": "February", "february": "February",
    "mar": "March", "march": "March",
    "apr": "April", "april": "April",
    "may": "May",
    "jun": "June", "june": "June",
    "jul": "July", "july": "July",
    "aug": "August", "august": "August",
    "sep": "September", "september": "September",
    "oct": "October", "october": "October",
    "nov": "November", "november": "November",
    "dec": "December", "december": "December",
}
ROOM_MAP = {
    "bedroom": "Bedroom",
    "kitchen": "Kitchen",
    "living room": "Living Room",
    "drawing room": "Drawing Room",
    "bathroom": "Bathroom",
    "dining room": "Dining Room",
}
DEVICE_MAP = {
    "light": "Light",
    "fan": "Fan",
    "tv": "TV",
    "ac": "AC",
    "computer": "Computer",
    "heater": "Heater",
    "washing machine": "Washing Machine",
    "microwave": "Microwave",
    "fridge": "Fridge",
}
SEASONS = ["Autumn", "Spring", "Summer", "Winter"]
USAGE_LEVELS = ["low", "medium", "high"]

MONTH_TO_NUM = {name: num for num, name in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"], start=1
)}

# --------------------------------------------------------------------------
# Session State for History
# --------------------------------------------------------------------------
if 'history' not in st.session_state:
    st.session_state.history = []

if 'predictions_count' not in st.session_state:
    st.session_state.predictions_count = 0


# --------------------------------------------------------------------------
# Loading artifacts
# --------------------------------------------------------------------------
def _find_artifact(suffix: str):
    for directory in CANDIDATE_DIRS:
        if not os.path.isdir(directory):
            continue
        matches = glob.glob(os.path.join(directory, f"*{suffix}"))
        if matches:
            return matches[0]
        try:
            for fname in os.listdir(directory):
                if fname.lower().endswith(suffix.lower()):
                    return os.path.join(directory, fname)
        except OSError:
            continue
    return None


@st.cache_resource(show_spinner="⏳ Loading model & preprocessing artifacts...")
def load_artifacts():
    artifacts = {}
    missing = []
    for key, suffix in ARTIFACT_SUFFIXES.items():
        path = _find_artifact(suffix)
        if path is None:
            missing.append(suffix)
        else:
            artifacts[key] = joblib.load(path)
    if missing:
        raise FileNotFoundError(f"Missing artifact(s): {', '.join(missing)}")
    return artifacts


# --------------------------------------------------------------------------
# Preprocessing (exact copy from original working code)
# --------------------------------------------------------------------------
def apply_basic_cleaning(df_raw: pd.DataFrame, cost_bounds: dict) -> pd.DataFrame:
    d = df_raw.copy()

    if "month" in d.columns:
        d["month"] = d["month"].astype(str).str.strip().str.lower().map(MONTH_MAP)

    if "room_name" in d.columns:
        d["room_name"] = (
            d["room_name"].astype(str).str.strip().str.lower()
            .str.replace("_", " ", regex=False).map(ROOM_MAP)
        )

    if "device_name" in d.columns:
        d["device_name"] = (
            d["device_name"].astype("string").str.strip().str.lower()
            .str.replace("_", " ", regex=False).map(DEVICE_MAP)
        )

    if "watts" in d.columns:
        d["watts"] = d["watts"].replace(["?", "error", "unknown", "-"], np.nan)
        d["watts"] = pd.to_numeric(d["watts"], errors="coerce")
        d.loc[d["watts"] >= 10000, "watts"] = d.loc[d["watts"] >= 10000, "watts"] / 100
        d.loc[d["watts"] < 0, "watts"] = np.nan

    if "hours" in d.columns:
        d["hours"] = d["hours"].replace(["?", "error", "unknown", "nan"], np.nan)
        d["hours"] = pd.to_numeric(d["hours"], errors="coerce")
        d.loc[d["hours"] < 0, "hours"] = np.nan
        d.loc[d["hours"] > 24, "hours"] = np.nan

    if "quantity" in d.columns:
        d["quantity"] = d["quantity"].astype(str).str.replace(".0.0", ".0", regex=False)
        d["quantity"] = pd.to_numeric(d["quantity"], errors="coerce")
        d.loc[d["quantity"] > 10, "quantity"] = np.nan

    if "cost_per_kwh" in d.columns and cost_bounds is not None:
        lb, ub = cost_bounds["lower_bound"], cost_bounds["upper_bound"]
        d.loc[(d["cost_per_kwh"] < lb) | (d["cost_per_kwh"] > ub), "cost_per_kwh"] = np.nan

    if "usage_level" in d.columns:
        d["usage_level"] = (
            d["usage_level"].astype(str).str.lower()
            .replace({"med": "medium"}).replace("nan", np.nan)
        )

    return d


def predict_monthly_kwh(raw_input: dict, artifacts: dict) -> float:
    cfg = artifacts["feature_columns_config"]

    df = pd.DataFrame([raw_input])[cfg["raw_input_columns"]]
    df = apply_basic_cleaning(df, artifacts["cost_per_kwh_bounds"])

    num_cols = cfg["numeric_impute_cols"]
    df[num_cols] = artifacts["numeric_imputer"].transform(df[num_cols])

    cat_cols = cfg["categorical_impute_cols"]
    df[cat_cols] = artifacts["categorical_imputer"].transform(df[cat_cols])

    df["month"] = df["month"].map(artifacts["month_encoder"])

    ohe_cols = cfg["ohe_cols"]
    encoded_arr = artifacts["onehot_encoder"].transform(df[ohe_cols])
    encoded_df = pd.DataFrame(
        encoded_arr,
        columns=artifacts["onehot_encoder"].get_feature_names_out(ohe_cols),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    df["usage_level"] = df["usage_level"].map(artifacts["usage_level_encoder"])
    df["is_heavy_appliance"] = df["is_heavy_appliance"].map(artifacts["is_heavy_appliance_encoder"])

    scaled_cols = cfg["scaled_columns"]
    for c in scaled_cols:
        if c not in df.columns:
            df[c] = 0
    df = df[scaled_cols]
    df[scaled_cols] = artifacts["robust_scaler"].transform(df[scaled_cols])

    final_df = df[artifacts["selected_features"]]
    prediction = artifacts["best_model"].predict(final_df)[0]
    return float(prediction)


# --------------------------------------------------------------------------
# Formula-based estimate (plain physics)
# --------------------------------------------------------------------------
def days_in_month(month_name: str, year: int = None) -> int:
    if year is None:
        year = pd.Timestamp.now().year
    month_num = MONTH_TO_NUM.get(month_name, 1)
    return calendar.monthrange(year, month_num)[1]


def formula_based_estimate(watts: float, hours: float, quantity: int,
                            cost_per_kwh: float, month: str) -> dict:
    n_days = days_in_month(month)
    daily_kwh = (watts * hours * quantity) / 1000
    monthly_kwh = daily_kwh * n_days
    daily_bill = daily_kwh * cost_per_kwh
    monthly_bill = monthly_kwh * cost_per_kwh
    return {
        "daily_kwh": daily_kwh,
        "daily_bill": daily_bill,
        "monthly_kwh": monthly_kwh,
        "monthly_bill": monthly_bill,
        "days_in_month": n_days,
    }


# --------------------------------------------------------------------------
# Analysis Functions
# --------------------------------------------------------------------------
def analyze_consumption(monthly_kwh: float, cost_per_kwh: float) -> dict:
    daily_kwh = monthly_kwh / 30
    
    if daily_kwh <= 2:
        tier = "Low"
        badge = "badge-low"
        color = "#43e97b"
        suggestion = "🌟 Excellent! You're an energy champion!"
        tips = [
            "Keep up the great habits!",
            "Share your tips with family and friends",
            "Consider solar panels for even more savings"
        ]
    elif daily_kwh <= 5:
        tier = "Medium"
        badge = "badge-medium"
        color = "#f9ca24"
        suggestion = "⚠️ Moderate consumption - room for improvement"
        tips = [
            "Use LED bulbs instead of traditional ones",
            "Unplug devices when not in use",
            "Use natural light during daytime"
        ]
    elif daily_kwh <= 10:
        tier = "High"
        badge = "badge-high"
        color = "#f0932b"
        suggestion = "🔴 High consumption - review your appliances"
        tips = [
            "Set AC to 24-26°C for optimal efficiency",
            "Clean AC filters regularly",
            "Replace old appliances with energy-efficient ones",
            "Use timers and smart plugs"
        ]
    else:
        tier = "Very High"
        badge = "badge-critical"
        color = "#eb4d4b"
        suggestion = "🚨 Excessive consumption - immediate action needed"
        tips = [
            "Perform an energy audit of your home",
            "Consider upgrading to energy-efficient appliances",
            "Check for standby power consumption",
            "Install a smart meter to monitor usage"
        ]
    
    if monthly_kwh > 300:
        saving_potential = monthly_kwh * 0.25
        saving_percentage = 25
    elif monthly_kwh > 150:
        saving_potential = monthly_kwh * 0.15
        saving_percentage = 15
    else:
        saving_potential = 0
        saving_percentage = 0
    
    saving_cost = saving_potential * cost_per_kwh
    co2_kg = monthly_kwh * 0.5
    trees_needed = co2_kg * 12 / 21
    
    return {
        "tier": tier,
        "badge": badge,
        "color": color,
        "suggestion": suggestion,
        "tips": tips,
        "saving_potential": saving_potential,
        "saving_cost": saving_cost,
        "saving_percentage": saving_percentage,
        "co2_kg": co2_kg,
        "trees_needed": trees_needed,
    }


def is_physically_plausible(monthly_kwh: float, watts: float, hours: float, quantity: int) -> tuple:
    """Check if the prediction is physically plausible"""
    # Minimum plausible: at least 0.1 kWh per month (if device is on)
    if watts > 0 and hours > 0 and quantity > 0:
        min_possible = (watts * hours * quantity * 30) / 1000 * 0.01  # 1% of theoretical minimum
        if monthly_kwh < min_possible:
            return False, f"Predicted {monthly_kwh:.2f} kWh is too low for a {watts}W device running {hours}h/day"
    
    # Maximum plausible: no more than 2x theoretical maximum (with 24h usage)
    max_possible = (watts * 24 * quantity * 30) / 1000 * 1.2
    if monthly_kwh > max_possible:
        return False, f"Predicted {monthly_kwh:.2f} kWh exceeds theoretical maximum of {max_possible:.2f} kWh"
    
    # Zero consumption check
    if monthly_kwh == 0:
        return False, "Prediction is zero - this is physically impossible for an active device"
    
    return True, "Physically plausible"


# --------------------------------------------------------------------------
# Save to History
# --------------------------------------------------------------------------
def save_to_history(raw_input: dict, ml_monthly: float, formula_monthly: float, 
                    tier: str, is_plausible: bool):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "month": raw_input["month"],
        "room": raw_input["room_name"],
        "device": raw_input["device_name"],
        "watts": raw_input["watts"],
        "hours": raw_input["hours"],
        "quantity": raw_input["quantity"],
        "cost_per_kwh": raw_input["cost_per_kwh"],
        "usage_level": raw_input["usage_level"],
        "season": raw_input["season"],
        "ml_prediction": round(ml_monthly, 2),
        "formula_prediction": round(formula_monthly, 2),
        "tier": tier,
        "is_plausible": is_plausible,
        "id": len(st.session_state.history) + 1
    }
    st.session_state.history.append(entry)
    st.session_state.predictions_count += 1


# ======================================================================
# Pages
# ======================================================================
def page_about():
    st.markdown('<div class="section-title">ℹ️ About</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="about-section">
            <span class="emoji-big">⚡</span>
            <h3>Smart Energy Predictor v2.0</h3>
            <p>
                This application predicts monthly electricity consumption using 
                <strong>two different approaches</strong>:
            </p>
            <ul>
                <li><strong>🤖 Machine Learning Model</strong> - Trained on historical consumption data using a Stacking Regressor</li>
                <li><strong>🧮 Formula Estimate</strong> - Pure physics calculation: <code>watts × hours × quantity × days / 1000</code></li>
            </ul>
            <p>
                The app provides a comprehensive analysis including daily consumption,
                monthly bills, saving tips, and environmental impact assessment.
            </p>
            <hr>
            <h4>🛠️ How It Works</h4>
            <ol>
                <li>Enter your device details (watts, hours, quantity, etc.)</li>
                <li>Click "Predict & Analyze"</li>
                <li>View ML prediction and formula estimate side-by-side</li>
                <li>Get personalized saving tips and environmental impact</li>
                <li>Track all predictions in the History page</li>
            </ol>
            <hr>
            <h4>📊 Features</h4>
            <ul>
                <li>✅ Daily consumption tracking</li>
                <li>✅ Monthly bill estimation</li>
                <li>✅ Smart saving recommendations</li>
                <li>✅ Environmental impact (CO₂, trees)</li>
                <li>✅ Historical predictions tracking</li>
                <li>✅ Physical plausibility checks</li>
                <li>✅ Interactive charts and visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="about-section">
            <h3>📦 Tech Stack</h3>
            <ul>
                <li>🐍 Python 3.9+</li>
                <li>📊 Streamlit</li>
                <li>🤖 Scikit-learn</li>
                <li>📈 Plotly</li>
                <li>🐼 Pandas</li>
                <li>🔢 NumPy</li>
            </ul>
            <hr>
            <h3>📁 Artifacts</h3>
            <ul>
                <li>Stacking Regressor</li>
                <li>OneHot Encoder</li>
                <li>Robust Scaler</li>
                <li>Feature Config</li>
            </ul>
            <hr>
            <h3>📞 Contact</h3>
            <p>
                Built with ❤️ for smart energy management
            </p>
        </div>
        """, unsafe_allow_html=True)


def page_history():
    st.markdown('<div class="section-title">📜 Prediction History</div>', unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("No predictions yet. Start by making a prediction on the main page!")
        return
    
    st.markdown(f"**Total predictions:** {len(st.session_state.history)}")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_tier = st.selectbox("Filter by tier", ["All", "Low", "Medium", "High", "Very High"])
    with col2:
        filter_plausible = st.selectbox("Filter by plausibility", ["All", "Plausible", "Not Plausible"])
    with col3:
        sort_by = st.selectbox("Sort by", ["Date (newest)", "Date (oldest)", "ML Prediction (high-low)", "ML Prediction (low-high)"])
    
    # Filter data
    history = st.session_state.history.copy()
    
    if filter_tier != "All":
        history = [h for h in history if h["tier"] == filter_tier]
    
    if filter_plausible != "All":
        is_plausible = filter_plausible == "Plausible"
        history = [h for h in history if h["is_plausible"] == is_plausible]
    
    # Sort
    if sort_by == "Date (newest)":
        history = sorted(history, key=lambda x: x["timestamp"], reverse=True)
    elif sort_by == "Date (oldest)":
        history = sorted(history, key=lambda x: x["timestamp"])
    elif sort_by == "ML Prediction (high-low)":
        history = sorted(history, key=lambda x: x["ml_prediction"], reverse=True)
    elif sort_by == "ML Prediction (low-high)":
        history = sorted(history, key=lambda x: x["ml_prediction"])
    
    # Display history cards
    for entry in history:
        badge_class = f"badge-{entry['tier'].lower().replace(' ', '-')}"
        badge_color = {
            "Low": "#43e97b", "Medium": "#f9ca24", 
            "High": "#f0932b", "Very High": "#eb4d4b"
        }.get(entry['tier'], "#636e72")
        
        plausibility_icon = "✅" if entry['is_plausible'] else "⚠️"
        
        st.markdown(f"""
        <div class="history-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span class="date">#{entry['id']} · {entry['timestamp']}</span>
                    <span style="margin-left:1rem;">
                        <span class="badge {badge_class}">{entry['tier']}</span>
                    </span>
                    <span style="margin-left:0.5rem;">{plausibility_icon}</span>
                </div>
                <div>
                    <span class="value">{entry['ml_prediction']:.1f} kWh</span>
                    <span style="color:#636e72; font-size:0.9rem;">(ML)</span>
                </div>
            </div>
            <div style="margin-top:0.5rem; color:#636e72; font-size:0.9rem;">
                {entry['device']} · {entry['room']} · {entry['watts']}W · {entry['hours']}h/day · {entry['season']} · {entry['usage_level']}
            </div>
            <div style="margin-top:0.25rem; color:#b2bec3; font-size:0.8rem;">
                Formula: {entry['formula_prediction']:.1f} kWh · Cost/kWh: {entry['cost_per_kwh']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Statistics
    st.markdown("---")
    st.markdown("### 📊 History Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    all_predictions = [h["ml_prediction"] for h in st.session_state.history]
    avg_pred = np.mean(all_predictions) if all_predictions else 0
    max_pred = max(all_predictions) if all_predictions else 0
    min_pred = min(all_predictions) if all_predictions else 0
    
    plausible_count = sum(1 for h in st.session_state.history if h["is_plausible"])
    
    with col1:
        st.metric("📊 Average", f"{avg_pred:.1f} kWh")
    with col2:
        st.metric("📈 Highest", f"{max_pred:.1f} kWh")
    with col3:
        st.metric("📉 Lowest", f"{min_pred:.1f} kWh")
    with col4:
        st.metric("✅ Plausible", f"{plausible_count}/{len(st.session_state.history)}")
    
    # Clear history button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.predictions_count = 0
        st.rerun()


def page_analysis():
    st.markdown('<div class="section-title">📊 Detailed Analysis</div>', unsafe_allow_html=True)
    st.markdown("Comprehensive analysis of your energy consumption patterns")
    
    if not st.session_state.history:
        st.info("No prediction data available. Please make a prediction first on the main page!")
        return
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Consumption Trends",
        "🏠 Device Analysis",
        "💰 Cost Analysis",
        "🌍 Environmental Impact"
    ])
    
    with tab1:
        st.markdown("#### Monthly Consumption Trend")
        
        # Prepare data
        df_history = pd.DataFrame(st.session_state.history)
        df_history['timestamp_dt'] = pd.to_datetime(df_history['timestamp'])
        df_history = df_history.sort_values('timestamp_dt')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_history['timestamp_dt'],
            y=df_history['ml_prediction'],
            mode='lines+markers',
            name='ML Prediction',
            line=dict(color='#667eea', width=3),
            marker=dict(size=12, color='#764ba2'),
        ))
        
        fig.add_trace(go.Scatter(
            x=df_history['timestamp_dt'],
            y=df_history['formula_prediction'],
            mode='lines+markers',
            name='Formula Estimate',
            line=dict(color='#f0932b', width=2, dash='dash'),
            marker=dict(size=10, color='#f0932b'),
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            xaxis_title="Date",
            yaxis_title="Consumption (kWh)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tier distribution
        st.markdown("#### Consumption Tier Distribution")
        tier_counts = df_history['tier'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=tier_counts.index,
            values=tier_counts.values,
            hole=0.4,
            marker=dict(colors=['#43e97b', '#f9ca24', '#f0932b', '#eb4d4b']),
            textinfo='label+percent',
            textposition='auto'
        )])
        fig.update_layout(
            height=350,
            margin=dict(t=0, b=0, l=0, r=0),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Device Usage Analysis")
        
        device_stats = df_history.groupby('device').agg({
            'ml_prediction': ['mean', 'count', 'max', 'min']
        }).round(2)
        device_stats.columns = ['Avg kWh', 'Count', 'Max', 'Min']
        st.dataframe(device_stats, use_container_width=True)
        
        # Device comparison chart
        fig = px.bar(
            device_stats.reset_index(),
            x='device',
            y='Avg kWh',
            color='Avg kWh',
            color_continuous_scale='Blues',
            title='Average Consumption by Device',
            labels={'Avg kWh': 'Average kWh', 'device': 'Device'}
        )
        fig.update_layout(
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### Cost Analysis")
        
        cost_stats = df_history.groupby('tier').agg({
            'ml_prediction': 'mean',
            'cost_per_kwh': 'mean'
        }).round(2)
        cost_stats['Estimated Cost'] = cost_stats['ml_prediction'] * cost_stats['cost_per_kwh']
        
        st.dataframe(cost_stats, use_container_width=True)
        
        # Cost distribution
        fig = go.Figure(data=[go.Bar(
            x=cost_stats.index,
            y=cost_stats['Estimated Cost'],
            marker_color=['#43e97b', '#f9ca24', '#f0932b', '#eb4d4b'],
            text=cost_stats['Estimated Cost'].round(2),
            textposition='outside',
        )])
        fig.update_layout(
            title='Average Monthly Cost by Consumption Tier',
            xaxis_title='Tier',
            yaxis_title='Estimated Cost (units)',
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("#### Environmental Impact Summary")
        
        total_co2 = df_history['ml_prediction'].sum() * 0.5
        avg_co2 = df_history['ml_prediction'].mean() * 0.5
        total_trees = total_co2 * 12 / 21
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌍 Total CO₂", f"{total_co2:.1f} kg")
        with col2:
            st.metric("📊 Avg CO₂/mo", f"{avg_co2:.1f} kg")
        with col3:
            st.metric("🌳 Trees Needed", f"{total_trees:.1f}")


# ======================================================================
# Main App
# ======================================================================
try:
    artifacts = load_artifacts()
except FileNotFoundError as e:
    st.error("❌ Couldn't load model artifacts — see details below.")
    st.code(str(e))
    st.stop()

# ======================================================================
# Navigation
# ======================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/light-on.png", width=80)
st.sidebar.title("⚡ Smart Energy")

page = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Home", "📊 Analysis", "📜 History", "ℹ️ About"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**📊 Predictions:** {st.session_state.predictions_count}")

if st.sidebar.button("🗑️ Clear History", use_container_width=True):
    st.session_state.history = []
    st.session_state.predictions_count = 0
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**🕐 Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ======================================================================
# Page Router
# ======================================================================
if page == "ℹ️ About":
    page_about()
elif page == "📜 History":
    page_history()
elif page == "📊 Analysis":
    page_analysis()
else:  # Home
    # Main content
    st.markdown('<div class="section-title">⚡ Electricity Consumption Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Predict monthly consumption using ML or formula-based estimation</div>', unsafe_allow_html=True)

    # Input Form
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            month = st.selectbox(
                "📅 Month",
                ["January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"],
                index=datetime.now().month - 1
            )
            room_name = st.selectbox("🏠 Room", list(ROOM_MAP.values()))
            device_name = st.selectbox("📱 Device", list(DEVICE_MAP.values()))
        
        with col2:
            season = st.selectbox("🌤️ Season", SEASONS, index=2)
            usage_level = st.selectbox("📊 Usage Level", USAGE_LEVELS, index=1)
            watts = st.number_input("⚡ Power (Watts)", min_value=0.0, value=100.0, step=10.0)
        
        with col3:
            hours = st.slider("⏰ Hours/day", 0.0, 24.0, 6.0, 0.5)
            quantity = st.number_input("🔢 Quantity", 1, 10, 1)
            cost_per_kwh = st.number_input("💰 Cost/kWh", 0.0, value=8.0, step=0.5)
            is_heavy_appliance = st.checkbox("⚙️ Heavy Appliance", value=False)

        if st.button("🔮 Predict & Analyze", use_container_width=True, type="primary"):
            raw_input = {
                "month": month,
                "room_name": room_name,
                "device_name": device_name,
                "watts": watts,
                "hours": hours,
                "quantity": quantity,
                "cost_per_kwh": cost_per_kwh,
                "usage_level": usage_level,
                "season": season,
                "is_heavy_appliance": is_heavy_appliance,
            }

            try:
                # ---- ML Prediction ----
                raw_prediction = predict_monthly_kwh(raw_input, artifacts)
                ml_monthly = max(0.0, raw_prediction)
                n_days = days_in_month(month)
                ml_daily = ml_monthly / n_days if n_days > 0 else 0
                ml_cost = ml_monthly * cost_per_kwh
                ml_daily_cost = ml_daily * cost_per_kwh

                # ---- Formula Estimate ----
                formula = formula_based_estimate(watts, hours, quantity, cost_per_kwh, month)
                
                # ---- Physical Plausibility Check ----
                is_plausible, plausibility_msg = is_physically_plausible(ml_monthly, watts, hours, quantity)
                
                # ---- Analysis ----
                analysis = analyze_consumption(ml_monthly, cost_per_kwh)
                
                # ---- Save to History ----
                save_to_history(raw_input, ml_monthly, formula['monthly_kwh'], analysis['tier'], is_plausible)

                # ==============================================================
                # Results Display
                # ==============================================================
                st.markdown("---")
                st.markdown("### 📊 Results")

                # ---- Plausibility Warning ----
                if not is_plausible or ml_monthly == 0:
                    st.markdown(f"""
                    <div class="warning-box">
                        <div>
                            <span class="icon">⚠️</span>
                            <span class="title">Physically Unusual Result Detected!</span>
                        </div>
                        <div class="content">
                            <strong>{plausibility_msg}</strong>
                            <br><br>
                            This suggests the input combination is unusual compared to the training data,
                            causing the ML model to extrapolate poorly.
                            <br><br>
                            <strong>💡 Recommendation:</strong> Check your input values. 
                            For a {watts}W device running {hours}h/day, expected consumption should be 
                            between {(watts * hours * quantity * n_days * 0.5) / 1000:.1f} and 
                            {(watts * 24 * quantity * n_days) / 1000:.1f} kWh/month.
                            <br><br>
                            <span style="color:#636e72; font-size:0.9rem;">
                                The value below has been clipped to 0 for display, but the actual model output was {raw_prediction:.4f} kWh.
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Use formula for display if ML is zero
                    display_monthly = formula['monthly_kwh'] if ml_monthly == 0 else ml_monthly
                    display_daily = display_monthly / n_days
                    display_cost = display_monthly * cost_per_kwh
                    display_daily_cost = display_daily * cost_per_kwh
                    
                    st.warning(f"⚠️ Using formula-based estimate ({display_monthly:.2f} kWh) instead of ML prediction for display")
                else:
                    display_monthly = ml_monthly
                    display_daily = ml_daily
                    display_cost = ml_cost
                    display_daily_cost = ml_daily_cost
                    st.success("✅ Prediction is physically plausible")

                # ---- 4 Main Cards ----
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card green">
                        <div class="label">📅 Daily Consumption</div>
                        <div class="value">{display_daily:.1f}</div>
                        <div class="label">kWh / day</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card blue">
                        <div class="label">📆 Monthly Consumption</div>
                        <div class="value">{display_monthly:.1f}</div>
                        <div class="label">kWh / month</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card orange">
                        <div class="label">💰 Monthly Bill</div>
                        <div class="value">{display_cost:,.0f}</div>
                        <div class="label">Currency units</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card dark">
                        <div class="label">📊 Consumption Level</div>
                        <div class="value" style="color:{analysis['color']}; font-size:2rem;">{analysis['tier']}</div>
                        <div class="label"><span class="{analysis['badge']}">{analysis['tier']}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                # ---- Comparison Section ----
                st.markdown("### 🔄 ML Model vs. Formula Estimate")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="comparison-box">
                        <h4>🤖 ML Model Prediction</h4>
                        <p><strong>Monthly:</strong> {ml_monthly:.2f} kWh</p>
                        <p><strong>Daily:</strong> {ml_daily:.2f} kWh</p>
                        <p><strong>Cost:</strong> {ml_cost:.2f} units</p>
                        <p style="font-size:0.85rem; color:#636e72;">Trained on historical data</p>
                        <p style="font-size:0.85rem; color:#636e72;">Raw output: {raw_prediction:.4f} kWh</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="comparison-box">
                        <h4>🧮 Formula Estimate</h4>
                        <p><strong>Monthly:</strong> {formula['monthly_kwh']:.2f} kWh</p>
                        <p><strong>Daily:</strong> {formula['daily_kwh']:.2f} kWh</p>
                        <p><strong>Cost:</strong> {formula['monthly_bill']:.2f} units</p>
                        <p style="font-size:0.85rem; color:#636e72;">watts × hours × quantity × days / 1000</p>
                        <p style="font-size:0.85rem; color:#636e72;">{formula['days_in_month']} days in {month}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Difference
                diff_kwh = ml_monthly - formula['monthly_kwh']
                diff_pct = (diff_kwh / formula['monthly_kwh'] * 100) if formula['monthly_kwh'] else 0
                diff_color = "#43e97b" if diff_kwh > 0 else "#eb4d4b"
                diff_emoji = "📈" if diff_kwh > 0 else "📉"
                
                st.markdown(f"""
                <div class="info-box">
                    <strong>{diff_emoji} Difference:</strong> 
                    <span style="color:{diff_color}; font-weight:700;">{diff_kwh:+.2f} kWh ({diff_pct:+.1f}%)</span>
                    <span style="color:#636e72; font-size:0.9rem; margin-left:1rem;">
                        ML predicts {abs(diff_kwh):.2f} kWh {'more' if diff_kwh > 0 else 'less'} than formula
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # ---- Tabs for details ----
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📋 Bill Details",
                    "💡 Saving Tips",
                    "🌍 Environmental Impact",
                    "📊 Charts"
                ])

                with tab1:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown("#### Detailed Breakdown")
                        bill_data = {
                            "Metric": [
                                "Daily Consumption",
                                "Monthly Consumption",
                                "Yearly Consumption",
                                "Daily Cost",
                                "Monthly Cost",
                                "Yearly Cost",
                                "Potential Savings",
                                "Savings Value"
                            ],
                            "Value": [
                                f"{display_daily:.2f} kWh",
                                f"{display_monthly:.2f} kWh",
                                f"{display_monthly * 12:.2f} kWh",
                                f"{display_daily_cost:.2f} units",
                                f"{display_cost:.2f} units",
                                f"{display_cost * 12:.2f} units",
                                f"{analysis['saving_potential']:.1f} kWh",
                                f"{analysis['saving_cost']:.2f} units"
                            ]
                        }
                        st.dataframe(pd.DataFrame(bill_data), hide_index=True, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Cost Distribution")
                        fig = go.Figure(data=[go.Pie(
                            labels=['Current Cost', 'Potential Savings'],
                            values=[display_cost, analysis['saving_cost']],
                            hole=0.4,
                            marker=dict(colors=['#667eea', '#f0932b']),
                            textinfo='label+percent',
                            textposition='auto'
                        )])
                        fig.update_layout(
                            height=300,
                            margin=dict(t=0, b=0, l=0, r=0),
                            showlegend=False,
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    st.markdown(f"#### {analysis['suggestion']}")
                    st.markdown("##### Recommended Actions:")
                    for tip in analysis['tips']:
                        st.markdown(f"🔹 {tip}")
                    
                    if analysis['saving_cost'] > 0:
                        st.markdown(f"""
                        <div class="info-box success">
                            <strong>💰 Potential Monthly Savings:</strong> {analysis['saving_cost']:.2f} units
                            <br><small>By implementing the recommendations above</small>
                        </div>
                        """, unsafe_allow_html=True)

                with tab3:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 🌿 Carbon Footprint")
                        st.markdown(f"""
                        <div style="background:#f8f9fa; padding:1.5rem; border-radius:15px;">
                            <div style="display:flex; align-items:center; gap:1rem;">
                                <span style="font-size:2.5rem;">🌍</span>
                                <div>
                                    <div style="font-size:1.5rem; font-weight:700;">{analysis['co2_kg']:.1f}</div>
                                    <div style="color:#636e72;">kg CO₂ per month</div>
                                </div>
                            </div>
                            <div style="margin-top:0.5rem; border-top:1px solid #e0e0e0; padding-top:0.5rem;">
                                <span style="color:#636e72;">Yearly: {(analysis['co2_kg'] * 12):.1f} kg CO₂</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("#### 🌳 Trees Needed")
                        st.markdown(f"""
                        <div style="background:#f8f9fa; padding:1.5rem; border-radius:15px;">
                            <div style="display:flex; align-items:center; gap:1rem;">
                                <span style="font-size:2.5rem;">🌳</span>
                                <div>
                                    <div style="font-size:1.5rem; font-weight:700;">{analysis['trees_needed']:.1f}</div>
                                    <div style="color:#636e72;">trees to offset yearly CO₂</div>
                                </div>
                            </div>
                            <div style="margin-top:0.5rem; border-top:1px solid #e0e0e0; padding-top:0.5rem;">
                                <span style="color:#636e72; font-size:0.85rem;">Each tree absorbs ~21 kg CO₂/year</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                with tab4:
                    st.markdown("#### 📈 Monthly Consumption Trend")
                    
                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    
                    seasonal_factors = {
                        'Winter': [1.1, 1.0, 0.9, 0.8, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.9, 1.1],
                        'Spring': [0.9, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.0, 0.9, 0.9, 0.9, 0.9],
                        'Summer': [0.9, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9],
                        'Autumn': [1.0, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.1, 1.0, 1.0]
                    }
                    
                    factors = seasonal_factors.get(season, [1.0] * 12)
                    monthly_trend = [display_monthly * f for f in factors]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=months,
                        y=monthly_trend,
                        mode='lines+markers',
                        name='ML Prediction',
                        line=dict(color='#667eea', width=3),
                        marker=dict(size=10, color='#764ba2'),
                        fill='tozeroy',
                        fillcolor='rgba(102, 126, 234, 0.2)'
                    ))
                    
                    fig.add_hline(
                        y=formula['monthly_kwh'],
                        line_dash="dash",
                        line_color="#f0932b",
                        annotation_text=f"Formula: {formula['monthly_kwh']:.1f} kWh",
                        annotation_position="bottom right"
                    )
                    
                    avg_trend = sum(monthly_trend) / len(monthly_trend)
                    fig.add_hline(
                        y=avg_trend,
                        line_dash="dot",
                        line_color="#43e97b",
                        annotation_text=f"Avg: {avg_trend:.1f} kWh",
                        annotation_position="top left"
                    )
                    
                    fig.update_layout(
                        height=350,
                        margin=dict(t=50, b=50, l=50, r=50),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        hovermode='x unified',
                        xaxis_title="Month",
                        yaxis_title="Consumption (kWh)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # ---- Raw Data ----
                with st.expander("📄 Raw Input & Debug Data"):
                    st.json(raw_input)
                    st.markdown(f"**Raw ML output:** {raw_prediction:.4f} kWh")
                    st.markdown(f"**Days in {month}:** {days_in_month(month)}")
                    st.markdown(f"**Is Plausible:** {is_plausible}")
                    st.markdown(f"**Plausibility Message:** {plausibility_msg}")

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.code(str(e))

    # ======================================================================
    # Footer
    # ======================================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#b2bec3; font-size:0.9rem;">
        ⚡ Smart Energy Predictor v2.0 | ML + Physics-Based Estimation | Built with Streamlit
    </div>
    """, unsafe_allow_html=True)