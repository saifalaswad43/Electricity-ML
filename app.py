"""
Streamlit App — Monthly Electricity Consumption (kWh) Predictor
-----------------------------------------------------------------
Version with Hugging Face integration
"""

import calendar
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")


# ==========================================================================
# Config & Styling
# ==========================================================================

st.set_page_config(
    page_title="⚡ Electricity Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================================
# CSS
# ==========================================================================

st.markdown("""
<style>
    .main { background: #f8f9fa; }

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

    .metric-card.blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }

    .metric-card.green {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }

    .metric-card.orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }

    .metric-card.dark {
        background: linear-gradient(135deg, #2d3436 0%, #000000 100%);
    }

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

    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 5px solid #f9ca24;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }

    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }

    .info-box.success {
        border-left-color: #43e97b;
    }

    .history-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
    }

    .badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .badge-low {
        background: #43e97b;
        color: white;
    }

    .badge-medium {
        background: #f9ca24;
        color: #2d3436;
    }

    .badge-high {
        background: #f0932b;
        color: white;
    }

    .badge-very-high {
        background: #eb4d4b;
        color: white;
    }

    .comparison-box {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }

    .about-section {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================================================
# Hugging Face Configuration
# ==========================================================================

HF_REPO_ID = "saifalaswad/electricity-consumption-model"

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


# ==========================================================================
# Text Cleaning Maps
# ==========================================================================

MONTH_MAP = {
    "jan": "January",
    "january": "January",
    "feb": "February",
    "february": "February",
    "mar": "March",
    "march": "March",
    "apr": "April",
    "april": "April",
    "may": "May",
    "jun": "June",
    "june": "June",
    "jul": "July",
    "july": "July",
    "aug": "August",
    "august": "August",
    "sep": "September",
    "september": "September",
    "oct": "October",
    "october": "October",
    "nov": "November",
    "november": "November",
    "dec": "December",
    "december": "December",
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

SEASONS = [
    "Autumn",
    "Spring",
    "Summer",
    "Winter"
]

USAGE_LEVELS = [
    "low",
    "medium",
    "high"
]

MONTH_TO_NUM = {
    name: num
    for num, name in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December"
        ],
        start=1
    )
}


# ==========================================================================
# Session State
# ==========================================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "predictions_count" not in st.session_state:
    st.session_state.predictions_count = 0


# ==========================================================================
# Loading Artifacts from Hugging Face
# ==========================================================================

@st.cache_resource(show_spinner="⏳ Loading model & preprocessing artifacts from Hugging Face...")
def load_artifacts():
    artifacts = {}

    with st.status("📥 Downloading artifacts from Hugging Face...", expanded=True) as status:
        for key, filename in ARTIFACT_SUFFIXES.items():
            try:
                status.write(f"⏳ Downloading: {filename}")
                
                path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=filename,
                    repo_type="model"
                )

                artifacts[key] = joblib.load(path)
                status.write(f"✅ Loaded: {filename}")

            except Exception as e:
                status.write(f"❌ Failed to download {filename}: {e}")
                raise

        status.write("✅ All artifacts loaded successfully!")

    return artifacts


# ==========================================================================
# Preprocessing
# ==========================================================================

def apply_basic_cleaning(df_raw: pd.DataFrame, cost_bounds: dict) -> pd.DataFrame:
    d = df_raw.copy()

    if "month" in d.columns:
        d["month"] = (
            d["month"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(MONTH_MAP)
        )

    if "room_name" in d.columns:
        d["room_name"] = (
            d["room_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace("_", " ", regex=False)
            .map(ROOM_MAP)
        )

    if "device_name" in d.columns:
        d["device_name"] = (
            d["device_name"]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace("_", " ", regex=False)
            .map(DEVICE_MAP)
        )

    if "watts" in d.columns:
        d["watts"] = d["watts"].replace(
            ["?", "error", "unknown", "-"],
            np.nan
        )
        d["watts"] = pd.to_numeric(
            d["watts"],
            errors="coerce"
        )
        d.loc[
            d["watts"] >= 10000,
            "watts"
        ] = (
            d.loc[
                d["watts"] >= 10000,
                "watts"
            ] / 100
        )
        d.loc[d["watts"] < 0, "watts"] = np.nan

    if "hours" in d.columns:
        d["hours"] = d["hours"].replace(
            ["?", "error", "unknown", "nan"],
            np.nan
        )
        d["hours"] = pd.to_numeric(
            d["hours"],
            errors="coerce"
        )
        d.loc[d["hours"] < 0, "hours"] = np.nan
        d.loc[d["hours"] > 24, "hours"] = np.nan

    if "quantity" in d.columns:
        d["quantity"] = (
            d["quantity"]
            .astype(str)
            .str.replace(".0.0", ".0", regex=False)
        )
        d["quantity"] = pd.to_numeric(
            d["quantity"],
            errors="coerce"
        )
        d.loc[d["quantity"] > 10, "quantity"] = np.nan

    if "cost_per_kwh" in d.columns and cost_bounds is not None:
        lb = cost_bounds["lower_bound"]
        ub = cost_bounds["upper_bound"]
        d.loc[
            (d["cost_per_kwh"] < lb)
            | (d["cost_per_kwh"] > ub),
            "cost_per_kwh"
        ] = np.nan

    if "usage_level" in d.columns:
        d["usage_level"] = (
            d["usage_level"]
            .astype(str)
            .str.lower()
            .replace({"med": "medium"})
            .replace("nan", np.nan)
        )

    return d


# ==========================================================================
# ML Prediction
# ==========================================================================

def predict_monthly_kwh(raw_input: dict, artifacts: dict) -> float:
    cfg = artifacts["feature_columns_config"]

    df = pd.DataFrame([raw_input])[cfg["raw_input_columns"]]
    df = apply_basic_cleaning(df, artifacts["cost_per_kwh_bounds"])

    # Numeric Imputation
    num_cols = cfg["numeric_impute_cols"]
    df[num_cols] = artifacts["numeric_imputer"].transform(df[num_cols])

    # Categorical Imputation
    cat_cols = cfg["categorical_impute_cols"]
    df[cat_cols] = artifacts["categorical_imputer"].transform(df[cat_cols])

    # Month Encoding
    df["month"] = df["month"].map(artifacts["month_encoder"])

    # One Hot Encoding
    ohe_cols = cfg["ohe_cols"]
    encoded_arr = artifacts["onehot_encoder"].transform(df[ohe_cols])
    encoded_df = pd.DataFrame(
        encoded_arr,
        columns=artifacts["onehot_encoder"].get_feature_names_out(ohe_cols),
        index=df.index
    )
    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    # Usage Level Encoding
    df["usage_level"] = df["usage_level"].map(artifacts["usage_level_encoder"])

    # Heavy Appliance Encoding
    df["is_heavy_appliance"] = df["is_heavy_appliance"].map(artifacts["is_heavy_appliance_encoder"])

    # Scaling
    scaled_cols = cfg["scaled_columns"]
    for c in scaled_cols:
        if c not in df.columns:
            df[c] = 0
    df = df[scaled_cols]
    df[scaled_cols] = artifacts["robust_scaler"].transform(df[scaled_cols])

    # Selected Features
    final_df = df[artifacts["selected_features"]]
    prediction = artifacts["best_model"].predict(final_df)[0]

    return float(prediction)


# ==========================================================================
# Formula Estimate
# ==========================================================================

def days_in_month(month_name: str, year: int = None) -> int:
    if year is None:
        year = pd.Timestamp.now().year
    month_num = MONTH_TO_NUM.get(month_name, 1)
    return calendar.monthrange(year, month_num)[1]


def formula_based_estimate(
    watts: float,
    hours: float,
    quantity: int,
    cost_per_kwh: float,
    month: str
) -> dict:
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
        "days_in_month": n_days
    }


# ==========================================================================
# Consumption Analysis
# ==========================================================================

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
        badge = "badge-very-high"
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


# ==========================================================================
# Physical Plausibility
# ==========================================================================

def is_physically_plausible(
    monthly_kwh: float,
    watts: float,
    hours: float,
    quantity: int
) -> tuple:
    if watts > 0 and hours > 0 and quantity > 0:
        min_possible = (watts * hours * quantity * 30) / 1000 * 0.01
        if monthly_kwh < min_possible:
            return (
                False,
                f"Predicted {monthly_kwh:.2f} kWh is too low "
                f"for a {watts}W device running {hours}h/day"
            )

    max_possible = (watts * 24 * quantity * 30) / 1000 * 1.2
    if monthly_kwh > max_possible:
        return (
            False,
            f"Predicted {monthly_kwh:.2f} kWh exceeds "
            f"theoretical maximum of {max_possible:.2f} kWh"
        )

    if monthly_kwh == 0:
        return (
            False,
            "Prediction is zero - this is physically impossible "
            "for an active device"
        )

    return True, "Physically plausible"


# ==========================================================================
# Save History
# ==========================================================================

def save_to_history(
    raw_input: dict,
    ml_monthly: float,
    formula_monthly: float,
    tier: str,
    is_plausible: bool
):
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


# ==========================================================================
# ABOUT PAGE
# ==========================================================================

def page_about():
    st.markdown(
        '<div class="section-title">ℹ️ About</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="about-section">
            <span style="font-size:4rem;text-align:center;display:block;">⚡</span>
            <h3>Smart Energy Predictor v2.0</h3>
            <p>
            This application predicts monthly electricity consumption
            using two different approaches.
            </p>
            <ul>
                <li>
                <strong>🤖 Machine Learning Model</strong>
                - Trained on historical consumption data
                </li>
                <li>
                <strong>🧮 Formula Estimate</strong>
                - Physics-based calculation
                </li>
            </ul>
            <hr>
            <h4>🛠️ How It Works</h4>
            <ol>
                <li>Enter your device details</li>
                <li>Click Predict & Analyze</li>
                <li>Compare ML and formula predictions</li>
                <li>Review saving recommendations</li>
                <li>Track predictions in History</li>
            </ol>
            <hr>
            <h4>📊 Features</h4>
            <ul>
                <li>Daily consumption tracking</li>
                <li>Monthly bill estimation</li>
                <li>Smart saving recommendations</li>
                <li>Environmental impact</li>
                <li>Historical predictions</li>
                <li>Physical plausibility checks</li>
                <li>Interactive charts</li>
            </ul>
            <hr>
            <h4>🤗 Hugging Face</h4>
            <p>
            Model artifacts hosted at:<br>
            <code>saifalaswad/electricity-consumption-model</code>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-section">
            <h3>📦 Tech Stack</h3>
            <ul>
                <li>🐍 Python</li>
                <li>📊 Streamlit</li>
                <li>🤖 Scikit-learn</li>
                <li>📈 Plotly</li>
                <li>🐼 Pandas</li>
                <li>🔢 NumPy</li>
                <li>🤗 Hugging Face</li>
            </ul>
            <hr>
            <h3>📁 Repository</h3>
            <p>
            Source code on GitHub
            </p>
            <hr>
            <h3>📄 License</h3>
            <p>MIT</p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================================================
# HISTORY PAGE
# ==========================================================================

def page_history():
    st.markdown(
        '<div class="section-title">📜 Prediction History</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.history:
        st.info("No predictions yet. Start by making a prediction!")
        return

    st.markdown(f"**Total predictions:** {len(st.session_state.history)}")

    col1, col2, col3 = st.columns(3)

    with col1:
        filter_tier = st.selectbox(
            "Filter by tier",
            ["All", "Low", "Medium", "High", "Very High"]
        )

    with col2:
        filter_plausible = st.selectbox(
            "Filter by plausibility",
            ["All", "Plausible", "Not Plausible"]
        )

    with col3:
        sort_by = st.selectbox(
            "Sort by",
            [
                "Date (newest)",
                "Date (oldest)",
                "ML Prediction (high-low)",
                "ML Prediction (low-high)"
            ]
        )

    history = st.session_state.history.copy()

    if filter_tier != "All":
        history = [h for h in history if h["tier"] == filter_tier]

    if filter_plausible != "All":
        is_plausible = filter_plausible == "Plausible"
        history = [h for h in history if h["is_plausible"] == is_plausible]

    if sort_by == "Date (newest)":
        history = sorted(history, key=lambda x: x["timestamp"], reverse=True)
    elif sort_by == "Date (oldest)":
        history = sorted(history, key=lambda x: x["timestamp"])
    elif sort_by == "ML Prediction (high-low)":
        history = sorted(history, key=lambda x: x["ml_prediction"], reverse=True)
    else:
        history = sorted(history, key=lambda x: x["ml_prediction"])

    for entry in history:
        badge_class = "badge-" + entry["tier"].lower().replace(" ", "-")
        plausibility_icon = "✅" if entry["is_plausible"] else "⚠️"

        st.markdown(
            f"""
            <div class="history-card">
                <div style="display:flex;justify-content:space-between;">
                    <div>
                        <span style="color:#636e72;">
                        #{entry['id']} · {entry['timestamp']}
                        </span>
                        <span style="margin-left:1rem;" class="badge {badge_class}">
                        {entry['tier']}
                        </span>
                        <span style="margin-left:0.5rem;">{plausibility_icon}</span>
                    </div>
                    <div>
                        <strong style="font-size:1.1rem;">
                        {entry['ml_prediction']:.1f} kWh
                        </strong>
                        <span style="color:#636e72;">(ML)</span>
                    </div>
                </div>
                <div style="margin-top:0.5rem;color:#636e72;">
                    {entry['device']} · {entry['room']} · {entry['watts']}W · {entry['hours']}h/day · {entry['season']} · {entry['usage_level']}
                </div>
                <div style="margin-top:0.25rem;color:#b2bec3;font-size:0.8rem;">
                    Formula: {entry['formula_prediction']:.1f} kWh · Cost/kWh: {entry['cost_per_kwh']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

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

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.predictions_count = 0
        st.rerun()


# ==========================================================================
# ANALYSIS PAGE
# ==========================================================================

def page_analysis():
    st.markdown(
        '<div class="section-title">📊 Detailed Analysis</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.history:
        st.info("No prediction data available. Please make a prediction first!")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Consumption Trends",
        "🏠 Device Analysis",
        "💰 Cost Analysis",
        "🌍 Environmental Impact"
    ])

    df_history = pd.DataFrame(st.session_state.history)
    df_history["timestamp_dt"] = pd.to_datetime(df_history["timestamp"])
    df_history = df_history.sort_values("timestamp_dt")

    with tab1:
        st.markdown("#### Monthly Consumption Trend")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_history["timestamp_dt"],
            y=df_history["ml_prediction"],
            mode="lines+markers",
            name="ML Prediction"
        ))
        fig.add_trace(go.Scatter(
            x=df_history["timestamp_dt"],
            y=df_history["formula_prediction"],
            mode="lines+markers",
            name="Formula Estimate"
        ))

        fig.update_layout(
            height=400,
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Consumption (kWh)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Consumption Tier Distribution")
        tier_counts = df_history["tier"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=tier_counts.index,
            values=tier_counts.values,
            hole=0.4,
            textinfo="label+percent"
        )])
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### Device Usage Analysis")
        device_stats = (
            df_history
            .groupby("device")
            .agg({"ml_prediction": ["mean", "count", "max", "min"]})
            .round(2)
        )
        device_stats.columns = ["Avg kWh", "Count", "Max", "Min"]
        st.dataframe(device_stats, use_container_width=True)

        fig = px.bar(
            device_stats.reset_index(),
            x="device",
            y="Avg kWh",
            color="Avg kWh",
            title="Average Consumption by Device"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### Cost Analysis")
        cost_stats = (
            df_history
            .groupby("tier")
            .agg({"ml_prediction": "mean", "cost_per_kwh": "mean"})
            .round(2)
        )
        cost_stats["Estimated Cost"] = cost_stats["ml_prediction"] * cost_stats["cost_per_kwh"]
        st.dataframe(cost_stats, use_container_width=True)

        fig = px.bar(
            cost_stats.reset_index(),
            x="tier",
            y="Estimated Cost",
            title="Average Monthly Cost by Consumption Tier"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("#### Environmental Impact Summary")
        total_co2 = df_history["ml_prediction"].sum() * 0.5
        avg_co2 = df_history["ml_prediction"].mean() * 0.5
        total_trees = total_co2 * 12 / 21

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌍 Total CO₂", f"{total_co2:.1f} kg")
        with col2:
            st.metric("📊 Avg CO₂/mo", f"{avg_co2:.1f} kg")
        with col3:
            st.metric("🌳 Trees Needed", f"{total_trees:.1f}")


# ==========================================================================
# Load Model
# ==========================================================================

try:
    artifacts = load_artifacts()
except Exception as e:
    st.error("❌ Couldn't load model artifacts from Hugging Face.")
    st.error("Make sure the repository 'saifalaswad/electricity-consumption-model' exists.")
    st.code(str(e))
    st.stop()


# ==========================================================================
# Sidebar
# ==========================================================================

st.sidebar.image("https://img.icons8.com/fluency/96/light-on.png", width=80)
st.sidebar.title("⚡ Smart Energy")

page = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Home", "📊 Analysis", "📜 History", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**📊 Predictions:** {st.session_state.predictions_count}")

if st.sidebar.button("🗑️ Clear History", use_container_width=True):
    st.session_state.history = []
    st.session_state.predictions_count = 0
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**🕐 Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.sidebar.markdown("---")
st.sidebar.markdown("**🤗 Hugging Face**")
st.sidebar.caption("saifalaswad/electricity-consumption-model")


# ==========================================================================
# Page Router
# ==========================================================================

if page == "ℹ️ About":
    page_about()
elif page == "📜 History":
    page_history()
elif page == "📊 Analysis":
    page_analysis()
else:
    # ======================================================================
    # HOME
    # ======================================================================

    st.markdown(
        '<div class="section-title">⚡ Electricity Consumption Predictor</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-sub">'
        'Predict monthly consumption using ML or formula-based estimation'
        '</div>',
        unsafe_allow_html=True
    )

    # ==================================================================
    # INPUTS
    # ==================================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        month = st.selectbox(
            "📅 Month",
            [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ],
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
        cost_per_kwh = st.number_input("💰 Cost/kWh", min_value=0.0, value=8.0, step=0.5)
        is_heavy_appliance = st.checkbox("⚙️ Heavy Appliance", value=False)

    # ==================================================================
    # PREDICT BUTTON
    # ==================================================================

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
            "is_heavy_appliance": is_heavy_appliance
        }

        try:
            # ==========================================================
            # ML Prediction
            # ==========================================================

            raw_prediction = predict_monthly_kwh(raw_input, artifacts)
            ml_monthly = max(0.0, raw_prediction)
            n_days = days_in_month(month)
            ml_daily = ml_monthly / n_days if n_days > 0 else 0
            ml_cost = ml_monthly * cost_per_kwh
            ml_daily_cost = ml_daily * cost_per_kwh

            # ==========================================================
            # Formula
            # ==========================================================

            formula = formula_based_estimate(
                watts, hours, quantity, cost_per_kwh, month
            )

            # ==========================================================
            # Plausibility
            # ==========================================================

            is_plausible, plausibility_msg = is_physically_plausible(
                ml_monthly, watts, hours, quantity
            )

            # ==========================================================
            # Analysis
            # ==========================================================

            analysis = analyze_consumption(ml_monthly, cost_per_kwh)

            # ==========================================================
            # History
            # ==========================================================

            save_to_history(
                raw_input,
                ml_monthly,
                formula["monthly_kwh"],
                analysis["tier"],
                is_plausible
            )

            # ==========================================================
            # Results
            # ==========================================================

            st.markdown("---")
            st.markdown("### 📊 Results")

            if not is_plausible or ml_monthly == 0:
                st.markdown(
                    f"""
                    <div class="warning-box">
                        <h4>⚠️ Physically Unusual Result Detected!</h4>
                        <strong>{plausibility_msg}</strong>
                        <br><br>
                        The ML model may be extrapolating poorly for this input combination.
                        <br><br>
                        For a {watts}W device running {hours}h/day, expected consumption
                        is approximately between
                        {(watts * hours * quantity * n_days * 0.5) / 1000:.1f}
                        and {(watts * 24 * quantity * n_days) / 1000:.1f} kWh/month.
                        <br><br>
                        Raw model output: <strong>{raw_prediction:.4f} kWh</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                display_monthly = formula["monthly_kwh"] if ml_monthly == 0 else ml_monthly
                display_daily = display_monthly / n_days
                display_cost = display_monthly * cost_per_kwh
                display_daily_cost = display_daily * cost_per_kwh

                st.warning(
                    f"⚠️ Using formula-based estimate "
                    f"({display_monthly:.2f} kWh) for display"
                )
            else:
                display_monthly = ml_monthly
                display_daily = ml_daily
                display_cost = ml_cost
                display_daily_cost = ml_daily_cost
                st.success("✅ Prediction is physically plausible")

            # ==========================================================
            # Cards
            # ==========================================================

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card green">
                        <div class="label">📅 Daily Consumption</div>
                        <div class="value">{display_daily:.1f}</div>
                        <div class="label">kWh / day</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card blue">
                        <div class="label">📆 Monthly Consumption</div>
                        <div class="value">{display_monthly:.1f}</div>
                        <div class="label">kWh / month</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col3:
                st.markdown(
                    f"""
                    <div class="metric-card orange">
                        <div class="label">💰 Monthly Bill</div>
                        <div class="value">{display_cost:,.0f}</div>
                        <div class="label">Currency units</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col4:
                st.markdown(
                    f"""
                    <div class="metric-card dark">
                        <div class="label">📊 Consumption Level</div>
                        <div class="value" style="font-size:2rem;color:{analysis['color']};">
                            {analysis['tier']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ==========================================================
            # Comparison
            # ==========================================================

            st.markdown("### 🔄 ML Model vs. Formula Estimate")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                    <div class="comparison-box">
                        <h4>🤖 ML Model Prediction</h4>
                        <p><strong>Monthly:</strong> {ml_monthly:.2f} kWh</p>
                        <p><strong>Daily:</strong> {ml_daily:.2f} kWh</p>
                        <p><strong>Cost:</strong> {ml_cost:.2f} units</p>
                        <p style="color:#636e72;">Trained on historical data</p>
                        <p style="color:#636e72;">Raw output: {raw_prediction:.4f} kWh</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    f"""
                    <div class="comparison-box">
                        <h4>🧮 Formula Estimate</h4>
                        <p><strong>Monthly:</strong> {formula['monthly_kwh']:.2f} kWh</p>
                        <p><strong>Daily:</strong> {formula['daily_kwh']:.2f} kWh</p>
                        <p><strong>Cost:</strong> {formula['monthly_bill']:.2f} units</p>
                        <p style="color:#636e72;">watts × hours × quantity × days / 1000</p>
                        <p style="color:#636e72;">{formula['days_in_month']} days in {month}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ==========================================================
            # Difference
            # ==========================================================

            diff_kwh = ml_monthly - formula["monthly_kwh"]
            diff_pct = (diff_kwh / formula["monthly_kwh"] * 100) if formula["monthly_kwh"] else 0
            diff_emoji = "📈" if diff_kwh > 0 else "📉"

            st.markdown(
                f"""
                <div class="info-box">
                    <strong>{diff_emoji} Difference:</strong>
                    <span style="font-weight:700;">{diff_kwh:+.2f} kWh ({diff_pct:+.1f}%)</span>
                    <span style="color:#636e72;">
                        ML predicts {abs(diff_kwh):.2f} kWh {'more' if diff_kwh > 0 else 'less'} than formula.
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ==========================================================
            # Detailed Tabs
            # ==========================================================

            tab1, tab2, tab3, tab4 = st.tabs([
                "📋 Bill Details",
                "💡 Saving Tips",
                "🌍 Environmental Impact",
                "📊 Charts"
            ])

            with tab1:
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
                st.dataframe(
                    pd.DataFrame(bill_data),
                    hide_index=True,
                    use_container_width=True
                )

            with tab2:
                st.markdown(f"#### {analysis['suggestion']}")
                st.markdown("##### Recommended Actions:")
                for tip in analysis["tips"]:
                    st.markdown(f"🔹 {tip}")

                if analysis["saving_cost"] > 0:
                    st.success(
                        f"💰 Potential Monthly Savings: "
                        f"{analysis['saving_cost']:.2f} units"
                    )

            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "🌍 Carbon Footprint",
                        f"{analysis['co2_kg']:.1f} kg CO₂/month"
                    )
                    st.caption(f"Yearly: {analysis['co2_kg'] * 12:.1f} kg CO₂")
                with col2:
                    st.metric(
                        "🌳 Trees Needed",
                        f"{analysis['trees_needed']:.1f}"
                    )
                    st.caption("Trees needed to offset yearly CO₂")

            with tab4:
                st.markdown("#### 📈 Monthly Consumption Trend")

                months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

                seasonal_factors = {
                    "Winter": [1.1, 1.0, 0.9, 0.8, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.9, 1.1],
                    "Spring": [0.9, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.0, 0.9, 0.9, 0.9, 0.9],
                    "Summer": [0.9, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9],
                    "Autumn": [1.0, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.1, 1.0, 1.0]
                }

                factors = seasonal_factors.get(season, [1.0] * 12)
                monthly_trend = [display_monthly * f for f in factors]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=months,
                    y=monthly_trend,
                    mode="lines+markers",
                    name="ML Prediction",
                    fill="tozeroy"
                ))

                fig.add_hline(
                    y=formula["monthly_kwh"],
                    line_dash="dash",
                    annotation_text=f"Formula: {formula['monthly_kwh']:.1f} kWh"
                )

                fig.update_layout(
                    height=350,
                    hovermode="x unified",
                    xaxis_title="Month",
                    yaxis_title="Consumption (kWh)"
                )
                st.plotly_chart(fig, use_container_width=True)

            # ==========================================================
            # Debug
            # ==========================================================

            with st.expander("📄 Raw Input & Debug Data"):
                st.json(raw_input)
                st.markdown(f"**Raw ML output:** {raw_prediction:.4f} kWh")
                st.markdown(f"**Days in {month}:** {days_in_month(month)}")
                st.markdown(f"**Is Plausible:** {is_plausible}")
                st.markdown(f"**Plausibility Message:** {plausibility_msg}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.code(str(e))


# ==========================================================================
# Footer
# ==========================================================================

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#b2bec3;font-size:0.9rem;">
    ⚡ Smart Energy Predictor v2.0 | ML + Physics-Based Estimation | 🤗 Hugging Face
</div>
""", unsafe_allow_html=True)