import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(
    page_title="ChronoTwin | 3D Plexus Human Digital Twin",
    page_icon="🧬",
    layout="wide"
)


# ==========================================
# BUILT-IN DIGITAL TWIN ENGINE (ERROR-PROOF)
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
def load_engine_v3():
    return DigitalTwinEngine()

engine = load_engine_v3()


# =========================================================
# 3D LOW-POLY PLEXUS HUMAN MESH GENERATOR (LIKE IMAGE)
# =========================================================
def build_3d_human_analyzer(organ_scores, patient_state, watch_source):
    """Generates an anatomical 3D Low-Poly Cyan Wireframe Plexus Human Model."""
    edge_x, edge_y, edge_z = [], [], []
    node_x, node_y, node_z = [], [] , []

    def add_anatomical_limb(slices, n_angles=12):
        """
        Builds a triangulated 3D plexus mesh tube along a list of anatomical cross-sections:
        Each slice is (cx, cy, z, rx, ry).
        """
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        rings = []
        for idx, (cx, cy, z, rx, ry) in enumerate(slices):
            # Twist half-step on odd rings for triangulated low-poly plexus look
            twist = (np.pi / n_angles) if (idx % 2 == 1) else 0.0
            pts = []
            for a in angles + twist:
                px = cx + rx * np.cos(a)
                py = cy + ry * np.sin(a)
                pts.append((px, py, z))
                node_x.append(px)
                node_y.append(py)
                node_z.append(z)
            rings.append(pts)

        # Connect rings with horizontal, vertical, and diagonal plexus edges
        for i in range(len(rings) - 1):
            r1, r2 = rings[i], rings[i + 1]
            for j in range(n_angles):
                jn = (j + 1) % n_angles
                # Horizontal ring edge
                edge_x.extend([r1[j][0], r1[jn][0], None])
                edge_y.extend([r1[j][1], r1[jn][1], None])
                edge_z.extend([r1[j][2], r1[jn][2], None])
                # Vertical / diagonal strut 1
                edge_x.extend([r1[j][0], r2[j][0], None])
                edge_y.extend([r1[j][1], r2[j][1], None])
                edge_z.extend([r1[j][2], r2[j][2], None])
                # Diagonal triangulation strut 2
                edge_x.extend([r1[jn][0], r2[j][0], None])
                edge_y.extend([r1[jn][1], r2[j][1], None])
                edge_z.extend([r1[jn][2], r2[j][2], None])

        # Close top ring
        top = rings[-1]
        for j in range(n_angles):
            jn = (j + 1) % n_angles
            edge_x.extend([top[j][0], top[jn][0], None])
            edge_y.extend([top[j][1], top[jn][1], None])
            edge_z.extend([top[j][2], top[jn][2], None])

    # 1. Head, Cranium, Jaw & Neck ( sculpted proportions )
    head_neck_slices = [
        (0.0, 0.0, 1.46, 0.065, 0.060),  # Base of neck
        (0.0, 0.0, 1.52, 0.058, 0.055),  # Mid neck
        (0.0, 0.01, 1.57, 0.075, 0.085), # Jawline / Chin
        (0.0, 0.01, 1.63, 0.095, 0.110), # Cheekbones
        (0.0, 0.0, 1.70, 0.105, 0.120),  # Cranium / Brow
        (0.0, -0.01, 1.77, 0.090, 0.105),# Upper skull
        (0.0, -0.01, 1.82, 0.045, 0.055) # Crown
    ]
    add_anatomical_limb(head_neck_slices, n_angles=12)

    # 2. Torso (Shoulders -> Chest -> Waist -> Pelvis)
    torso_slices = [
        (0.0, 0.0, 0.88, 0.155, 0.100),  # Pelvis / Hips
        (0.0, 0.0, 0.96, 0.165, 0.105),  # Iliac crest
        (0.0, 0.0, 1.05, 0.145, 0.095),  # Narrow waist
        (0.0, 0.01, 1.15, 0.165, 0.110), # Lower ribcage
        (0.0, 0.015, 1.26, 0.195, 0.125),# Pectorals / Chest
        (0.0, 0.01, 1.36, 0.235, 0.120), # Broad shoulders / Deltoid bridge
        (0.0, 0.0, 1.44, 0.130, 0.080)   # Trapezius / Collarbone
    ]
    add_anatomical_limb(torso_slices, n_angles=16)

    # 3. Left & Right Arms in Anatomical A-Pose (angled outward like uploaded image)
    for side in [-1, 1]:
        arm_slices = [
            (side * 0.22, 0.0, 1.37, 0.062, 0.060),  # Deltoid / Shoulder
            (side * 0.27, 0.0, 1.25, 0.052, 0.050),  # Bicep / Tricep
            (side * 0.32, 0.01, 1.13, 0.045, 0.044), # Elbow joint
            (side * 0.37, 0.02, 1.01, 0.042, 0.038), # Forearm
            (side * 0.42, 0.03, 0.90, 0.032, 0.025), # Wrist
            (side * 0.46, 0.04, 0.82, 0.042, 0.018), # Palm / Hand
            (side * 0.49, 0.045, 0.76, 0.028, 0.012) # Fingers
        ]
        add_anatomical_limb(arm_slices, n_angles=9)

    # 4. Left & Right Legs (Thigh -> Knee -> Calf -> Ankle -> Foot)
    for side in [-1, 1]:
        leg_slices = [
            (side * 0.095, 0.0, 0.88, 0.082, 0.082),  # Upper Thigh / Hip
            (side * 0.100, 0.0, 0.72, 0.074, 0.074),  # Mid Quadriceps
            (side * 0.102, 0.0, 0.56, 0.058, 0.058),  # Above Knee
            (side * 0.105, -0.005, 0.47, 0.052, 0.054),# Knee joint
            (side * 0.108, -0.01, 0.34, 0.058, 0.060),# Calf muscle
            (side * 0.110, 0.0, 0.19, 0.042, 0.044),  # Lower Shin
            (side * 0.112, 0.0, 0.08, 0.036, 0.038),  # Ankle
            (side * 0.120, 0.05, 0.02, 0.045, 0.075)  # Foot
        ]
        add_anatomical_limb(leg_slices, n_angles=10)

    fig = go.Figure()

    # Outer Holographic Bounding Frame (like the second reference image)
    box_x = [-0.55, 0.55, 0.55, -0.55, -0.55, None, -0.55, 0.55, 0.55, -0.55, -0.55, None,
             -0.55, -0.55, None, 0.55, 0.55, None, 0.55, 0.55, None, -0.55, -0.55]
    box_y = [-0.30, -0.30, 0.30, 0.30, -0.30, None, -0.30, -0.30, 0.30, 0.30, -0.30, None,
             -0.30, -0.30, None, -0.30, -0.30, None, 0.30, 0.30, None, 0.30, 0.30]
    box_z = [0.0, 0.0, 0.0, 0.0, 0.0, None, 1.88, 1.88, 1.88, 1.88, 1.88, None,
             0.0, 1.88, None, 0.0, 1.88, None, 0.0, 1.88, None, 0.0, 1.88]
    fig.add_trace(go.Scatter3d(
        x=box_x, y=box_y, z=box_z,
        mode='lines',
        name='Holographic Scan Frame',
        line=dict(color='rgba(0, 212, 255, 0.22)', width=1.5),
        hoverinfo='skip',
        showlegend=False
    ))

    # Trace 1: Glowing Cyan Wireframe Plexus Lines
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        name='3D Plexus Wireframe',
        line=dict(color='rgba(0, 225, 255, 0.42)', width=1.6),
        hoverinfo='skip'
    ))

    # Trace 2: Glowing Cyan Constellation Vertex Dots
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        name='Bio-Mesh Nodes',
        marker=dict(size=2.3, color='#70F3FF', opacity=0.75),
        hoverinfo='skip',
        showlegend=False
    ))

    def score_to_color(score):
        if score >= 75:
            return '#00FF9D', 'OPTIMAL'
        elif score >= 50:
            return '#FFB020', 'MODERATE STRAIN'
        return '#FF3B5C', 'HIGH RISK ALERT'

    # Glowing Internal Organ Cores inside the Wireframe Body
    organs_3d = [
        {
            'name': '🧠 Brain & Autonomic Core',
            'coords': (0.0, 0.01, 1.71),
            'score': organ_scores['Autonomic (HRV) Tone'],
            'metric': f"HRV: {patient_state['hrv']} ms | Sleep: {patient_state['sleep_hrs']} hrs",
            'size': 18
        },
        {
            'name': '🫀 Heart (Cardiac Core)',
            'coords': (-0.045, 0.03, 1.26),
            'score': organ_scores['Cardiac Efficiency'],
            'metric': f"Resting HR: {patient_state['resting_hr']} bpm | Sys BP: {patient_state['sys_bp']} mmHg",
            'size': 22
        },
        {
            'name': '🫁 Left & Right Lungs',
            'coords': (0.075, 0.02, 1.28),
            'score': organ_scores['Sleep & Oxygenation'],
            'metric': f"SpO2: {patient_state['spo2']}% | Sleep: {patient_state['sleep_hrs']} hrs",
            'size': 18
        },
        {
            'name': '🧬 Liver & Pancreas (Metabolic)',
            'coords': (-0.06, 0.02, 1.07),
            'score': organ_scores['Glycemic & Metabolic'],
            'metric': f"HbA1c: {patient_state['hba1c']}% | BMI: {patient_state['bmi']} kg/m²",
            'size': 19
        },
        {
            'name': '🩸 Aortic & Vascular Tree',
            'coords': (0.01, 0.01, 1.17),
            'score': organ_scores['Vascular Health'],
            'metric': f"LDL: {patient_state['ldl']} mg/dL | Sys BP: {patient_state['sys_bp']} mmHg",
            'size': 16
        }
    ]

    for org in organs_3d:
        color, status = score_to_color(org['score'])
        # Outer Halo Glow
        fig.add_trace(go.Scatter3d(
            x=[org['coords'][0]], y=[org['coords'][1]], z=[org['coords'][2]],
            mode='markers',
            marker=dict(size=org['size'] * 1.65, color=color, opacity=0.28),
            hoverinfo='skip',
            showlegend=False
        ))
        # Inner Core Node
        fig.add_trace(go.Scatter3d(
            x=[org['coords'][0]], y=[org['coords'][1]], z=[org['coords'][2]],
            mode='markers+text',
            name=f"{org['name']} ({org['score']}/100)",
            text=[org['name'].split()[0]],
            textposition="top center",
            marker=dict(size=org['size'] * 0.75, color=color, opacity=1.0, line=dict(color='white', width=2)),
            hovertemplate=(
                f"<b>{org['name']}</b><br>"
                f"Resilience Score: <b>{org['score']}/100 ({status})</b><br>"
                f"{org['metric']}<extra></extra>"
            )
        ))

    # Smartwatch Sensor Node on Wrist (-0.42, 0.03, 0.90)
    fig.add_trace(go.Scatter3d(
        x=[-0.42], y=[0.03], z=[0.90],
        mode='markers+text',
        name='⌚ Connected Smartwatch Sensor',
        text=['⌚ Watch'],
        textposition="bottom center",
        marker=dict(size=13, color='#D668FF', symbol='diamond', line=dict(color='white', width=2)),
        hovertemplate=(
            f"<b>⌚ {watch_source}</b><br>"
            f"Steps: {patient_state['daily_steps']:,}/day<br>"
            f"HR: {patient_state['resting_hr']} bpm | SpO2: {patient_state['spo2']}%<extra></extra>"
        )
    ))

    # Glowing Telemetry Pathway from Wrist Watch -> Shoulder -> Heart -> Brain
    fig.add_trace(go.Scatter3d(
        x=[-0.42, -0.32, -0.22, -0.045, 0.01, -0.06, 0.0],
        y=[0.03, 0.01, 0.0, 0.03, 0.01, 0.02, 0.01],
        z=[0.90, 1.13, 1.37, 1.26, 1.17, 1.07, 1.71],
        mode='lines',
        name='Biometric Neural Conduit',
        line=dict(color='#00FFCC', width=3.5, dash='dot'),
        hoverinfo='skip'
    ))

    fig.update_layout(
        title="ANATOMY | Interactive 3D Low-Poly Plexus Digital Twin (Drag to Rotate | Scroll to Zoom)",
        scene=dict(
            xaxis=dict(visible=False, range=[-0.65, 0.65]),
            yaxis=dict(visible=False, range=[-0.55, 0.55]),
            zaxis=dict(visible=False, range=[-0.02, 1.92]),
            aspectmode='manual',
            aspectratio=dict(x=1.15, y=0.85, z=2.0),
            camera=dict(eye=dict(x=0.0, y=1.85, z=0.25)), # Front-facing anatomical view like reference image
            bgcolor='#04142C' # Deep navy holographic background matching uploaded image
        ),
        paper_bgcolor='#04142C',
        font=dict(color='#70F3FF'),
        height=600,
        margin=dict(l=0, r=0, t=45, b=0),
        legend=dict(
            yanchor="top", y=0.95, xanchor="left", x=0.01,
            bgcolor="rgba(4, 20, 44, 0.75)",
            bordercolor="rgba(0, 212, 255, 0.3)",
            borderwidth=1
        )
    )
    return fig


# Header
st.title("🧬 ChronoTwin: 3D Plexus Human Digital Twin & Smartwatch Analyzer")
st.markdown(
    "**Happiest Health Digital Twin Challenge** | *3D Holographic Anatomy + Smartwatch Telemetry + EHR Predictive Engine*"
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

# Smartwatch Hub
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
                                                        
