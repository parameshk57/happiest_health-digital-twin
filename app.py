import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(
    page_title="ChronoTwin | 3D Human Digital Twin",
    page_icon="🧬",
    layout="wide"
)


# ==========================================
# SELF-CONTAINED DIGITAL TWIN ENGINE (10 FEATURES)
# ==========================================
class DigitalTwinEngine:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=180, learning_rate=0.05, max_depth=4, random_state=42
        )
        self.feature_names = [
            'age', 'bmi', 'hba1c', 'sys_bp', 'ldl',
            'resting_hr', 'hrv', 'daily_steps', 'sleep_hrs', 'spo2'
        ]
        self._train_baseline_model()

    def _generate_synthetic_cohort(self, n=1500):
        np.random.seed(42)
        age = np.random.randint(22, 82, n)
        bmi = np.random.normal(26.5, 4.5, n).clip(17.5, 45.0)
        hba1c = np.random.normal(5.8, 1.1, n).clip(4.5, 11.5)
        sys_bp = np.random.normal(126, 18, n).clip(90, 195)
        ldl = np.random.normal(118, 32, n).clip(50, 230)

        resting_hr = np.random.normal(72, 11, n).clip(45, 115)
        hrv = np.random.normal(46, 16, n).clip(12, 105)
        daily_steps = np.random.normal(6400, 3100, n).clip(800, 20000)
        sleep_hrs = np.random.normal(6.8, 1.2, n).clip(3.5, 9.5)
        spo2 = np.random.normal(97.2, 1.5, n).clip(88.0, 100.0)

        risk = (
            0.24 * (age - 25) +
            0.85 * (bmi - 22) +
            6.60 * (hba1c - 5.0) +
            0.22 * (sys_bp - 110) +
            0.08 * (ldl - 90) +
            0.32 * (resting_hr - 60) -
            0.26 * (hrv - 50) -
            0.0018 * (daily_steps - 5000) -
            1.85 * (sleep_hrs - 7.0) -
            1.40 * (spo2 - 97.0) +
            np.random.normal(0, 1.8, n)
        )
        risk_prob = (1 / (1 + np.exp(-0.08 * (risk - 18)))) * 100

        return pd.DataFrame({
            'age': age, 'bmi': bmi, 'hba1c': hba1c, 'sys_bp': sys_bp, 'ldl': ldl,
            'resting_hr': resting_hr, 'hrv': hrv, 'daily_steps': daily_steps,
            'sleep_hrs': sleep_hrs, 'spo2': spo2, 'risk_score': risk_prob.clip(1.0, 99.0)
        })

    def _train_baseline_model(self):
        df = self._generate_synthetic_cohort(1500)
        X = df[self.feature_names]
        y = df['risk_score']
        self.model.fit(X, y)

    def predict_risk(self, patient_dict):
        clean_dict = {k: float(patient_dict.get(k, 0.0)) for k in self.feature_names}
        df_input = pd.DataFrame([clean_dict])[self.feature_names]
        return float(np.clip(self.model.predict(df_input)[0], 1.0, 99.0))

    def compute_organ_scores(self, p):
        cardiac = np.clip(100 - (max(0, p['resting_hr'] - 60) * 1.4 + max(0, p['sys_bp'] - 115) * 0.8), 10, 99)
        metabolic = np.clip(100 - (max(0, p['hba1c'] - 5.0) * 14.0 + max(0, p['bmi'] - 22.0) * 2.2), 10, 99)
        autonomic = np.clip(35 + (p['hrv'] * 0.75) - max(0, 7.0 - p['sleep_hrs']) * 6.0, 10, 99)
        vascular = np.clip(100 - (max(0, p['ldl'] - 85) * 0.45 + max(0, p['sys_bp'] - 115) * 0.6), 10, 99)
        respiratory_sleep = np.clip((p['spo2'] - 85) * 5.5 + (p['sleep_hrs'] / 8.0) * 18.0, 10, 99)
        return {
            'Cardiac Efficiency': round(float(cardiac), 1),
            'Glycemic & Metabolic': round(float(metabolic), 1),
            'Autonomic (HRV) Tone': round(float(autonomic), 1),
            'Vascular Health': round(float(vascular), 1),
            'Sleep & Oxygenation': round(float(respiratory_sleep), 1)
        }

    def simulate_trajectory(self, baseline_patient, target_patient, months=6):
        timeline = list(range(months + 1))
        baseline_trend, intervention_trend, hr_trend, hba1c_trend = [], [], [], []

        base_risk = self.predict_risk(baseline_patient)
        for m in timeline:
            baseline_trend.append(round(min(99.0, base_risk * (1 + 0.015 * m)), 2))

            alpha = 1 - np.exp(-0.55 * m)
            state = {
                k: baseline_patient[k] + alpha * (target_patient[k] - baseline_patient[k])
                for k in self.feature_names
            }
            step_delta = state['daily_steps'] - baseline_patient['daily_steps']
            sleep_delta = state['sleep_hrs'] - baseline_patient['sleep_hrs']

            state['resting_hr'] = np.clip(state['resting_hr'] - (step_delta / 1000.0) * 1.15 * alpha, 45, 115)
            state['hrv'] = np.clip(state['hrv'] + (step_delta / 1000.0) * 1.6 * alpha + max(0, sleep_delta) * 2.2 * alpha, 12, 105)
            state['hba1c'] = np.clip(state['hba1c'] - (step_delta / 2200.0) * 0.14 * alpha, 4.5, 11.5)

            intervention_trend.append(round(self.predict_risk(state), 2))
            hr_trend.append(round(float(state['resting_hr']), 1))
            hba1c_trend.append(round(float(state['hba1c']), 2))

        return pd.DataFrame({
            'Month': timeline,
            'Status Quo Trajectory': baseline_trend,
            'Digital Twin Guided Care': intervention_trend,
            'Predicted Resting HR (bpm)': hr_trend,
            'Predicted HbA1c (%)': hba1c_trend
        })

    def generate_24h_watch_stream(self, resting_hr, hrv, steps, spo2):
        np.random.seed(int(resting_hr + steps) % 1000)
        hours = [f"{h:02d}:00" for h in range(24)]
        circadian = np.array([
            -8, -10, -11, -12, -11, -9, -2, 6, 14, 18, 12, 10,
            15, 11, 9, 13, 22, 28, 16, 8, 3, -2, -5, -7
        ])
        activity_boost = (steps / 8000.0) * 8.0
        hr_series = np.clip(resting_hr + circadian * (activity_boost / 8.0) + np.random.normal(0, 2.5, 24), 45, 165)
        spo2_series = np.clip(spo2 + np.random.normal(0, 0.4, 24), 90.0, 100.0)
        return pd.DataFrame({
            'Hour': hours,
            'Heart Rate (bpm)': np.round(hr_series, 1),
            'SpO2 (%)': np.round(spo2_series, 1)
        })


@st.cache_resource
def load_engine_v2():
    return DigitalTwinEngine()


engine = load_engine_v2()


def build_3d_human_analyzer(organ_scores, patient_state, watch_source):
    """Generates an interactive 3D Holographic Human Body & Organ Risk Analyzer."""
    np.random.seed(42)
    body_x, body_y, body_z = [], [], []

    def add_cylinder(x_c, y_c, z_min, z_max, rx, ry, n_pts=220):
        theta = np.random.uniform(0, 2 * np.pi, n_pts)
        z = np.random.uniform(z_min, z_max, n_pts)
        x = x_c + rx * np.cos(theta)
        y = y_c + ry * np.sin(theta)
        body_x.extend(x)
        body_y.extend(y)
        body_z.extend(z)

    # 1. Head & Neck
    u = np.random.uniform(0, 2 * np.pi, 250)
    v = np.random.uniform(0, np.pi, 250)
    body_x.extend(0.13 * np.cos(u) * np.sin(v))
    body_y.extend(0.15 * np.sin(u) * np.sin(v))
    body_z.extend(1.68 + 0.14 * np.cos(v))
    add_cylinder(0, 0, 1.48, 1.55, 0.06, 0.06, 60)

    # 2. Torso
    add_cylinder(0, 0, 0.92, 1.48, 0.23, 0.14, 650)

    # 3. Left & Right Arms
    add_cylinder(-0.31, 0, 0.78, 1.44, 0.055, 0.055, 180)
    add_cylinder(0.31, 0, 0.78, 1.44, 0.055, 0.055, 180)

    # 4. Left & Right Legs
    add_cylinder(-0.11, 0, 0.05, 0.92, 0.08, 0.08, 280)
    add_cylinder(0.11, 0, 0.05, 0.92, 0.08, 0.08, 280)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=body_x, y=body_y, z=body_z,
        mode='markers',
        name='3D Human Bio-Mesh',
        marker=dict(size=2.2, color='#00D4FF', opacity=0.22),
        hoverinfo='skip'
    ))

    def score_to_color(score):
        if score >= 75:
            return '#00CC96', 'OPTIMAL'
        elif score >= 50:
            return '#FFA15A', 'MODERATE STRAIN'
        return '#EF553B', 'HIGH RISK ALERT'

    organs_3d = [
        {
            'name': '🧠 Brain & Autonomic Core',
            'coords': (0.0, 0.0, 1.70),
            'score': organ_scores['Autonomic (HRV) Tone'],
            'metric': f"HRV: {patient_state['hrv']} ms | Sleep: {patient_state['sleep_hrs']} hrs",
            'size': 18
        },
        {
            'name': '🫀 Heart (Cardiac Core)',
            'coords': (-0.05, 0.03, 1.30),
            'score': organ_scores['Cardiac Efficiency'],
            'metric': f"Resting HR: {patient_state['resting_hr']} bpm | Sys BP: {patient_state['sys_bp']} mmHg",
            'size': 22
        },
        {
            'name': '🫁 Left & Right Lungs',
            'coords': (0.08, 0.02, 1.32),
            'score': organ_scores['Sleep & Oxygenation'],
            'metric': f"SpO2: {patient_state['spo2']}% | Sleep: {patient_state['sleep_hrs']} hrs",
            'size': 19
        },
        {
            'name': '🧬 Liver & Pancreas (Metabolic)',
            'coords': (-0.07, 0.02, 1.08),
            'score': organ_scores['Glycemic & Metabolic'],
            'metric': f"HbA1c: {patient_state['hba1c']}% | BMI: {patient_state['bmi']} kg/m²",
            'size': 20
        },
        {
            'name': '🩸 Aortic & Vascular Tree',
            'coords': (0.02, 0.0, 1.18),
            'score': organ_scores['Vascular Health'],
            'metric': f"LDL: {patient_state['ldl']} mg/dL | Sys BP: {patient_state['sys_bp']} mmHg",
            'size': 17
        }
    ]

    for org in organs_3d:
        color, status = score_to_color(org['score'])
        fig.add_trace(go.Scatter3d(
            x=[org['coords'][0]],
            y=[org['coords'][1]],
            z=[org['coords'][2]],
            mode='markers+text',
            name=f"{org['name']} ({org['score']}/100)",
            text=[org['name'].split()[0]],
            textposition="top center",
            marker=dict(size=org['size'], color=color, opacity=0.95, line=dict(color='white', width=2)),
            hovertemplate=(
                f"<b>{org['name']}</b><br>"
                f"Resilience Score: <b>{org['score']}/100 ({status})</b><br>"
                f"{org['metric']}<extra></extra>"
            )
        ))

    # Smartwatch Sensor Node on Left Wrist
    fig.add_trace(go.Scatter3d(
        x=[-0.31], y=[0.04], z=[0.83],
        mode='markers+text',
        name='⌚ Connected Smartwatch Sensor',
        text=['⌚ Watch'],
        textposition="bottom center",
        marker=dict(size=14, color='#AB63FA', symbol='diamond', line=dict(color='white', width=2)),
        hovertemplate=(
            f"<b>⌚ {watch_source}</b><br>"
            f"Steps: {patient_state['daily_steps']:,}/day<br>"
            f"HR: {patient_state['resting_hr']} bpm | SpO2: {patient_state['spo2']}%<extra></extra>"
        )
    ))

    # Biometric Telemetry Conduit
    fig.add_trace(go.Scatter3d(
        x=[0.0, -0.05, 0.02, -0.07, -0.31],
        y=[0.0, 0.03, 0.0, 0.02, 0.04],
        z=[1.70, 1.30, 1.18, 1.08, 0.83],
        mode='lines',
        name='Biometric Telemetry Conduit',
        line=dict(color='#00FFCC', width=3, dash='dot'),
        hoverinfo='skip'
    ))

    fig.update_layout(
        title="Interactive 3D Human Digital Twin Analyzer (Drag to Rotate | Scroll to Zoom)",
        scene=dict(
            xaxis=dict(visible=False, range=[-0.6, 0.6]),
            yaxis=dict(visible=False, range=[-0.6, 0.6]),
            zaxis=dict(visible=False, range=[0.0, 1.95]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=0.8, z=2.1),
            camera=dict(eye=dict(x=1.45, y=1.45, z=0.8)),
            bgcolor='rgba(14, 17, 23, 0.95)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.01)
    )
    return fig


# Header
st.title("🧬 ChronoTwin: 3D Human Digital Twin & Smartwatch Analyzer")
st.markdown(
    "**Happiest Health Digital Twin Challenge** | *Interactive 3D Anatomy + Smartwatch Telemetry + EHR Predictive Engine*"
)

default_watch = {
    'resting_hr': 80,
    'hrv': 34,
    'daily_steps': 4200,
    'sleep_hrs': 5.8,
    'spo2': 96.2,
    'watch_source': 'Manual / Default Baseline'
}
for k, v in default_watch.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================
# SMARTWATCH CONNECTIVITY HUB (TOP EXPANDER)
# ==========================================
with st.expander("⌚ Connect Smartwatch / Wearable Device (BLE Stream, Health Connect CSV, or Cloud API)", expanded=False):
    sync_col1, sync_col2 = st.columns([1.3, 1.7])

    with sync_col1:
        sync_mode = st.radio(
            "Select Smartwatch Integration Method:",
            [
                "📡 Live Smartwatch BLE Preset Stream",
                "📂 Upload Smartwatch Export (CSV / JSON)",
                "🌐 Cloud Wearable API Webhook (Google Fit / Fitbit / Terra)"
            ]
        )

    with sync_col2:
        if sync_mode == "📡 Live Smartwatch BLE Preset Stream":
            profile = st.selectbox(
                "Select Connected Watch Telemetry Stream:",
                [
                    "Apple Watch Series 10 — High Stress / Sedentary Alert",
                    "Samsung Galaxy Watch 7 — Post-Workout Recovery",
                    "Garmin Forerunner — Endurance Athlete Profile",
                    "Fitbit Sense 2 — Sleep Deprivation / Elevated HR"
                ]
            )
            if st.button("🔄 Sync Live BLE GATT Telemetry Packet", type="primary"):
                presets = {
                    "Apple Watch Series 10 — High Stress / Sedentary Alert": (84, 28, 3100, 5.5, 95.8),
                    "Samsung Galaxy Watch 7 — Post-Workout Recovery": (64, 58, 9400, 7.6, 98.2),
                    "Garmin Forerunner — Endurance Athlete Profile": (52, 78, 13500, 8.2, 98.8),
                    "Fitbit Sense 2 — Sleep Deprivation / Elevated HR": (88, 24, 4600, 4.8, 95.1)
                }
                r_hr, hrv_v, stp, slp, ox = presets[profile]
                st.session_state.update({
                    'resting_hr': r_hr, 'hrv': hrv_v, 'daily_steps': stp,
                    'sleep_hrs': slp, 'spo2': ox, 'watch_source': f"Live BLE: {profile.split('—')[0].strip()}"
                })
                st.success(f"Synced live BLE packet from {profile.split('—')[0].strip()}!")

        elif sync_mode == "📂 Upload Smartwatch Export (CSV / JSON)":
            uploaded = st.file_uploader("Upload Wearable Data File", type=["csv", "json"])
            if uploaded is not None:
                try:
                    if uploaded.name.endswith(".csv"):
                        df_up = pd.read_csv(uploaded)
                        row = df_up.iloc[-1].to_dict()
                    else:
                        row = json.load(uploaded)
                    st.session_state['resting_hr'] = int(row.get('resting_hr', row.get('heart_rate', 75)))
                    st.session_state['hrv'] = int(row.get('hrv', 42))
                    st.session_state['daily_steps'] = int(row.get('daily_steps', row.get('steps', 6000)))
                    st.session_state['sleep_hrs'] = float(row.get('sleep_hrs', row.get('sleep', 6.8)))
                    st.session_state['spo2'] = float(row.get('spo2', 97.0))
                    st.session_state['watch_source'] = f"Uploaded File ({uploaded.name})"
                    st.success("Smartwatch telemetry parsed and ingested into Digital Twin!")
                except Exception as e:
                    st.error(f"Could not parse file: {e}")

        else:
            api_payload = st.text_area(
                "Incoming Wearable Webhook JSON Payload:",
                value=json.dumps({
                    "device": "Pixel Watch 3 (Health Connect API)",
                    "resting_hr": 76, "hrv": 41, "daily_steps": 7250, "sleep_hrs": 6.9, "spo2": 97.4
                }, indent=2),
                height=120
            )
            if st.button("⚡ Ingest Cloud API Payload"):
                try:
                    data = json.loads(api_payload)
                    st.session_state['resting_hr'] = int(data.get('resting_hr', 76))
                    st.session_state['hrv'] = int(data.get('hrv', 41))
                    st.session_state['daily_steps'] = int(data.get('daily_steps', 7250))
                    st.session_state['sleep_hrs'] = float(data.get('sleep_hrs', 6.9))
                    st.session_state['spo2'] = float(data.get('spo2', 97.4))
                    st.session_state['watch_source'] = data.get('device', 'Cloud Wearable API')
                    st.success("Cloud API telemetry synchronized!")
                except Exception as e:
                    st.error(f"Invalid JSON payload: {e}")

# ==========================================
# SIDEBAR: EHR + SMARTWATCH TELEMETRY INPUTS
# ==========================================
st.sidebar.header("📋 1. Clinical EHR Baseline")
age = st.sidebar.slider("Age (Years)", 22, 80, 52)
bmi = st.sidebar.slider("BMI (kg/m²)", 18.0, 42.0, 29.2, step=0.2)
hba1c = st.sidebar.slider("HbA1c (%)", 4.5, 11.0, 6.7, step=0.1)
sys_bp = st.sidebar.slider("Systolic BP (mmHg)", 90, 190, 138)
ldl = st.sidebar.slider("LDL Cholesterol (mg/dL)", 60, 220, 144)

st.sidebar.markdown("---")
st.sidebar.subheader("⌚ 2. Smartwatch Stream")
st.sidebar.caption(f"Source: **{st.session_state['watch_source']}**")

resting_hr = st.sidebar.slider("Resting Heart Rate (bpm)", 45, 110, int(st.session_state['resting_hr']))
hrv = st.sidebar.slider("Heart Rate Variability - HRV (ms)", 15, 100, int(st.session_state['hrv']))
daily_steps = st.sidebar.slider("Daily Step Count", 1000, 18000, int(st.session_state['daily_steps']), step=250)
sleep_hrs = st.sidebar.slider("Sleep Duration (hrs/night)", 4.0, 9.5, float(st.session_state['sleep_hrs']), step=0.25)
spo2 = st.sidebar.slider("Blood Oxygen - SpO2 (%)", 90.0, 100.0, float(st.session_state['spo2']), step=0.2)

baseline_state = {
    'age': age, 'bmi': bmi, 'hba1c': hba1c, 'sys_bp': sys_bp, 'ldl': ldl,
    'resting_hr': resting_hr, 'hrv': hrv, 'daily_steps': daily_steps,
    'sleep_hrs': sleep_hrs, 'spo2': spo2
}

# ==========================================
# DIGITAL TWIN SIMULATION & 3D HUMAN ANALYZER
# ==========================================
col_sim, col_viz = st.columns([1.0, 2.0])

with col_sim:
    st.subheader("🧪 3. Predictive Care Simulator")
    st.caption("Adjust interventions to see the 3D Human Twin respond:")
    view_state_mode = st.radio(
        "3D Human Body Inspection Mode:",
        ["Current Baseline Patient", "6-Month Post-Intervention Twin"],
        horizontal=True
    )
    sim_steps = st.slider("Prescribed Daily Steps (Watch Target)", 1000, 18000, max(daily_steps, 9500), step=500)
    sim_sleep = st.slider("Prescribed Sleep Target (hrs/night)", 4.0, 9.5, max(sleep_hrs, 7.5), step=0.25)
    sim_bmi = st.slider("Target BMI (Nutrition Plan)", 18.0, 42.0, round(max(18.5, bmi * 0.90), 1), step=0.2)
    sim_bp = st.slider("Target Systolic BP (Therapy/Diet)", 90, 190, min(sys_bp, 120))

target_state = baseline_state.copy()
target_state.update({
    'daily_steps': sim_steps,
    'sleep_hrs': sim_sleep,
    'bmi': sim_bmi,
    'sys_bp': sim_bp
})

base_risk = engine.predict_risk(baseline_state)
trajectory_df = engine.simulate_trajectory(baseline_state, target_state, months=6)
final_twin_risk = trajectory_df['Digital Twin Guided Care'].iloc[-1]
risk_delta = final_twin_risk - base_risk

final_state_approx = target_state.copy()
final_state_approx['resting_hr'] = trajectory_df['Predicted Resting HR (bpm)'].iloc[-1]
final_state_approx['hba1c'] = trajectory_df['Predicted HbA1c (%)'].iloc[-1]
final_state_approx['hrv'] = round(min(100, hrv + ((sim_steps - daily_steps) / 1000.0) * 1.8), 1)

base_organs = engine.compute_organ_scores(baseline_state)
target_organs = engine.compute_organ_scores(final_state_approx)

with col_viz:
    st.subheader("📊 4. 3D Human Analyzer & AI Outcome Forecast")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current Adverse Risk", f"{base_risk:.1f}%")
    k2.metric("6-Month Twin Risk", f"{final_twin_risk:.1f}%", f"{risk_delta:.1f}%", delta_color="inverse")
