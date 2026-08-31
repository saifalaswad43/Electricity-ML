<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,16,20&height=260&section=header&text=Electricity%20Consumption%20AI&fontSize=44&fontColor=ffffff&animation=twinkling&fontAlignY=35&desc=Predict%20%E2%80%A2%20Analyze%20%E2%80%A2%20Save%20Energy%20with%20Machine%20Learning&descAlignY=55&descSize=18" width="100%"/>

<a href="https://github.com/saifalaswad43/Electricity-ML">
  <img src="https://readme-typing-svg.demolab.com/?lines=%E2%9A%A1+Smart+Monthly+Electricity+(kWh)+Predictor;%F0%9F%A4%96+ML+Model+Hosted+on+Hugging+Face+Hub;%F0%9F%A7%AE+ML+Prediction+vs+Physics-Based+Formula;%F0%9F%8C%8D+Cost%2C+CO%E2%82%82+%26+Savings+Insights+Built-in&font=Fira+Code&center=true&width=750&height=50&color=00D4AA&vCenter=true&size=21&pause=1500"/>
</a>

<br/>

<img src="https://img.shields.io/github/stars/saifalaswad43/Electricity-ML?style=for-the-badge&color=FFD700&logo=github&logoColor=white" />
<img src="https://img.shields.io/github/forks/saifalaswad43/Electricity-ML?style=for-the-badge&color=8A2BE2&logo=github&logoColor=white" />
<img src="https://img.shields.io/github/last-commit/saifalaswad43/Electricity-ML?style=for-the-badge&color=00C7B7&logo=git&logoColor=white" />
<img src="https://img.shields.io/badge/license-MIT-brightgreen?style=for-the-badge" />
<img src="https://img.shields.io/badge/status-active-success?style=for-the-badge" />

<br/><br/>

<a href="https://electricity-ml-5zhdpjyvagfwxxzsknhxjv.streamlit.app/">
  <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit" />
</a>

<br/><br/>

<img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/Scikit--learn-1.7.2-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/XGBoost-EC4E20?style=flat-square" />
<img src="https://img.shields.io/badge/CatBoost-FFCC00?style=flat-square&logoColor=black" />
<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-yellow?style=flat-square" />

</div>

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">
</div>

## ⚡ Overview

🔗 **Live demo:** [electricity-ml-5zhdpjyvagfwxxzsknhxjv.streamlit.app](https://electricity-ml-5zhdpjyvagfwxxzsknhxjv.streamlit.app/)

**Electricity Consumption AI** is a Streamlit web app that predicts your **monthly electricity consumption (kWh)** for any device, room, or usage pattern — combining a trained **Machine Learning model** with a transparent **physics-based formula**, side by side. Model artifacts are pulled live from the **Hugging Face Hub**, so the app always runs the latest trained pipeline without shipping heavy files in the repo. It comes packed with cost estimation, CO₂ impact tracking, saving recommendations, and rich interactive analytics — now with a redesigned, animated interface.

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/216122041-518ac897-8d92-4c6b-9b3f-ca01dcaf38ee.gif" width="480">
</div>

---

## 🆕 What's new in this update

- **Redesigned UI** — animated gradient hero header, glowing tier badges, and hover-responsive metric cards for a more professional feel.
- **Smoother motion** — fade-in transitions on results, a pulsing usage-tier badge, and shimmering background gradients instead of static panels.
- **Reproducibility fix** — the training notebook previously hard-coded a local Windows path (`D:\saifproject\...`) to load `electricity_dataset.csv`; it now uses a relative path so the notebook runs for anyone who clones the repo.
- **Pipeline parity** — the app's inference code now mirrors the notebook's exact `apply_basic_cleaning → impute → encode → scale → select → predict` order, so predictions match the notebook's own end-to-end test.

---

## ✨ Features

<table>
<tr>
<td width="50%" valign="top">

### 🏠 Predict
- Full device / room / usage input form
- Live **ML prediction** vs **formula estimate**
- Physical plausibility check (catches impossible predictions)
- Animated gradient metric cards

</td>
<td width="50%" valign="top">

### 📊 Analysis
- Consumption distribution across the dataset (Plotly)
- Session tier-distribution donut chart
- Device-level breakdown & averages
- Dark-themed, animated charts

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 📜 History
- Session-based prediction log
- Filter by tier & plausibility
- Sort by date or predicted usage
- One-click history reset

</td>
<td width="50%" valign="top">

### 🌍 Insights
- 💰 Monthly bill estimation
- 🌳 CO₂ emissions & trees-needed equivalent
- 💡 Personalized energy-saving tips by tier
- 🏷️ Low / Medium / High / Very High usage badges with animated glow

</td>
</tr>
</table>

---

## 🧠 How It Works

```mermaid
flowchart LR
    A[User Input<br/>device · room · watts · hours] --> B[Data Cleaning<br/>& Validation]
    B --> C[Imputation<br/>Numeric · Categorical]
    C --> D[Encoding<br/>OneHot · Month · Usage Level]
    D --> E[Robust Scaling]
    E --> F[Feature Selection]
    F --> G["🤗 Model from Hugging Face Hub"]
    G --> H[Monthly kWh Prediction]
    H --> I[🧮 Compared vs Formula Estimate]
    I --> J[💰 Cost · 🌍 CO₂ · 💡 Tips]
```

The app cross-checks every ML prediction against a **physics-based sanity formula** (`watts × hours × quantity × days ÷ 1000`) and flags results that fall outside a physically plausible range — so you never get a silently broken prediction.

**Consumption tiers:** `Low` → `Medium` → `High` → `Very High`, each with tailored saving tips and a color-coded, glowing badge.

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/saifalaswad43/Electricity-ML.git
cd Electricity-ML
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501` ⚡ — model artifacts are downloaded automatically from Hugging Face on first launch.

---

## 📂 Project Structure

```
Electricity-ML/
├── app.py                       # Streamlit application (predict · analysis · history · about)
├── electricity_notebook.ipynb   # EDA, preprocessing & model training notebook
├── electricity_dataset.csv      # Household electricity consumption dataset
└── requirements.txt             # Dependencies
```

> 🤗 Trained model & preprocessing artifacts (encoders, scaler, imputers, selected features) are hosted separately on the **Hugging Face Hub** at `saifalaswad/electricity-consumption-model` and downloaded on demand — keeping this repo lightweight.

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://skillicons.dev/icons?i=python)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/-XGBoost-EC4E20?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/-CatBoost-FFCC00?style=for-the-badge&logoColor=black)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![HuggingFace](https://img.shields.io/badge/-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

</div>

---

## 📊 Dataset & Inputs

The model is trained on household electricity usage records capturing device-level behavior. Key input fields:

| Field | Description |
|:--|:--|
| `month` / `season` | Time-of-year context |
| `room_name` | Bedroom, Kitchen, Living Room, Bathroom, etc. |
| `device_name` | Light, Fan, TV, AC, Fridge, Heater, Washing Machine, etc. |
| `watts` · `hours` · `quantity` | Device power, daily runtime, and count |
| `cost_per_kwh` | Local electricity tariff |
| `usage_level` | Low / Medium / High |
| `is_heavy_appliance` | Flags high-draw devices |

**Target:** Predicted monthly consumption in **kWh**.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

<img src="https://user-images.githubusercontent.com/74038190/213866269-5d00981c-7c98-46d7-8a8e-16f462f15227.gif" width="100%">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,16,20&height=140&section=footer" width="100%"/>

</div>
