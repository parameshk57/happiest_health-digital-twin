import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="ChronoTwin | 3D Digital Twin", page_icon="🧬", layout="wide")


class DigitalTwinEngine:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
        self.feature_names = [
            'age', 'bmi', 'hba1c', 'sys_bp', 'ldl',
            'resting_hr', 'hrv', 'daily_steps', 'sleep_hrs', 'spo2'
        ]
        self._train_baseline_model()

    def _generate_synthetic_cohort(self, n=1200):
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
            0.24 * (age - 25) + 0.85 * (bmi - 22) + 6.60 * (hba1c - 5.0) +
            0.22 * (sys_bp - 110) + 0.08 * (ldl - 90) + 0.32 * (resting_hr - 60) -
            0.26 * (hrv - 50) - 0.0018 * (daily_steps - 5000) -
            1.85 * (sleep_hrs - 7.0) - 1.40 * (spo2 - 97.0) + np.random.normal(0, 1.8, n)
        )
        risk_prob = (1 / (1 + np.exp(-0.08 * (risk - 18)))) * 100
        return pd.DataFrame({
            'age': age, 'bmi': bmi, 'hba1c': hba1c, 'sys_bp': sys_bp, 'ldl': ldl,
            'resting_hr': resting_hr, 'hrv': hrv, 'daily_steps': daily_steps,
            'sleep_hrs': sleep_hrs, 'spo2': spo2, 'risk_score': risk_prob.clip(1.0, 99.0)
        })

    def _train_baseline_model(self):
        df = self._generate_synthetic_cohort(1200)
        self.model.fit(df[self.feature_names], df['risk_score'])

    def predict_risk(self, patient_dict):
        clean_dict = {k: float(patient_dict.get(k, 0.0)) for k in self.feature_names}
        df_input = pd.DataFrame([clean_dict])[self.feature_names]
        return float(np.clip(self.model.predict(df_input)[0], 1.0, 99.0))

    def compute_organ_scores(self, p):
        cardiac = np.clip(100 - (max(0, p['resting_hr'] - 60) * 1.4 + max(0, p['sys_bp'] - 115) * 0.8), 10, 99)
        metabolic = np.clip(100 - (max(0, p['hba1c'] - 5.0) * 14.0 + max(0, p['bmi'] - 22.0) * 2.2), 10, 99)
        autonomic = np.clip(35 + (p['hrv'] * 0.75) - max(0, 7.0 - p['sleep_hrs']) * 6.0, 10, 99)
        vascular = np.clip(100 - (max(0, p['ldl'] - 85) * 0.45 + max(0, p['sys_bp'] - 115) * 0.6), 10, 99)
        respiratory = np.clip((p['spo2'] - 85) * 5.5 + (p['sleep_hrs'] / 8.0) * 18.0, 10, 99)
        return {
            'Cardiac Efficiency': round(float(cardiac), 1),
            'Glycemic & Metabolic': round(float(metabolic), 1),
            'Autonomic (HRV) Tone': round(float(autonomic), 1),
            'Vascular Health': round(float(vascular), 1),
            'Sleep & Oxygenation': round(float(respiratory), 1)
        }

    def simulate_trajectory(self, base_p, target_p, months=6):
        timeline = list(range(months + 1))
        b_trend, i_trend, hr_trend, a1c_trend = [], [], [], []
        base_risk = self.predict_risk(base_p)
        for m in timeline:
            b_trend.append(round(min(99.0, base_risk * (1 + 0.015 * m)), 2))
            alpha = 1 - np.exp(-0.55 * m)
            state = {k: base_p[k] + alpha * (target_p[k] - base_p[k]) for k in self.feature_names}
            s_delta = state['daily_steps'] - base_p['daily_steps']
            slp_delta = state['sleep_hrs'] - base_p['sleep_hrs']
            state['resting_hr'] = np.clip(state['resting_hr'] - (s_delta / 1000.0) * 1.15 * alpha, 45, 115)
            state['hrv'] = np.clip(state['hrv'] + (s_delta / 1000.0) * 1.6 * alpha + max(0, slp_delta) * 2.2 * alpha, 12, 105)
            state['hba1c'] = np.clip(state['hba1c'] - (s_delta / 2200.0) * 0.14 * alpha, 4.5, 11.5)
            i_trend.append(round(self.predict_risk(state), 2))
            hr_trend.append(round(float(state['resting_hr']), 1))
            a1c_trend.append(round(float(state['hba1c']), 2))
        return pd.DataFrame({
            'Month': timeline, 'Status Quo Trajectory': b_trend,
            'Digital Twin Guided Care': i_trend,
            'Predicted Resting HR (bpm)': hr_trend, 'Predicted HbA1c (%)': a1c_trend
        })

    def generate_24h_watch_stream(self, resting_hr, hrv, steps, spo2):
        np.random.seed(int(resting_hr + steps) % 1000)
        hours = [f"{h:02d}:00" for h in range(24)]
        circ = np.array([-8, -10, -11, -12, -11, -9, -2, 6, 14, 18, 12, 10, 15, 11, 9, 13, 22, 28, 16, 8, 3, -2, -5, -7])
        hr_s = np.clip(resting_hr + circ * (steps / 8000.0) + np.random.normal(0, 2.5, 24), 45, 165)
        ox_s = np.clip(spo2 + np.random.normal(0, 0.4, 24), 90.0, 100.0)
        return pd.DataFrame({'Hour': hours, 'Heart Rate (bpm)': np.round(hr_s, 1), 'SpO2 (%)': np.round(ox_s, 1)})


@st.cache_resource
def get_engine():
    return DigitalTwinEngine()

engine = get_engine()


def build_3d_human_analyzer(organ_scores, patient_state, watch_source):
    edge_x, edge_y, edge_z = [], [], []
    node_x, node_y, node_z = [], [], []

    def add_limb(slices, n_angles=12):
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        rings = []
        for idx, (cx, cy, z, rx, ry) in enumerate(slices):
            twist = (np.pi / n_angles) if (idx % 2 == 1) else 0.0
            pts = []
            for a in angles + twist:
                px, py = cx + rx * np.cos(a), cy + ry * np.sin(a)
                pts.append((px, py, z))
                node_x.append(px)
                node_y.append(py)
                node_z.append(z)
            rings.append(pts)
        for i in range(len(rings) - 1):
            r1, r2 = rings[i], rings[i + 1]
            for j in range(n_angles):
                jn = (j + 1) % n_angles
                edge_x.extend([r1[j][0], r1[jn][0], None, r1[j][0], r2[j][0], None, r1[jn][0], r2[j][0], None])
                edge_y.extend([r1[j][1], r1[jn][1], None, r1[j][1], r2[j][1], None, r1[jn][1], r2[j][1], None])
                edge_z.extend([r1[j][2], r1[jn][2], None, r1[j][2], r2[j][2], None, r1[jn][2], r2[j][2], None])

    add_limb([
        (0, 0, 1.46, 0.065, 0.06), (0, 0, 1.52, 0.058, 0.055), (0, 0.01, 1.57, 0.075, 0.085),
        (0, 0.01, 1.63, 0.095, 0.11), (0, 0, 1.70, 0.105, 0.12), (0, -0.01, 1.77, 0.09, 0.105), (0, -0.01, 1.82, 0.045, 0.055)
    ], 12)
    add_limb([
        (0, 0, 0.88, 0.155, 0.10), (0, 0, 0.96, 0.165, 0.105), (0, 0, 1.05, 0.145, 0.095),
        (0, 0.01, 1.15, 0.165, 0.11), (0, 0.015, 1.26, 0.195, 0.125), (0, 0.01, 1.36, 0.235, 0.12), (0, 0, 1.44, 0.13, 0.08)
    ], 16)
    for s in [-1, 1]:
        add_limb([
            (s * 0.22, 0, 1.37, 0.062, 0.06), (s * 0.27, 0, 1.25, 0.052, 0.05), (s * 0.32, 0.01, 1.13, 0.045, 0.044),
            (s * 0.37, 0.02, 1.01, 0.042, 0.038), (s * 0.42, 0.03, 0.90, 0.032, 0.025), (s * 0.46, 0.04, 0.82, 0.042, 0.018)
        ], 9)
        add_limb([
            (s * 0.095, 0, 0.88, 0.082, 0.082), (s * 0.10, 0, 0.72, 0.074, 0.074), (s * 0.102, 0, 0.56, 0.058, 0.058),
            (s * 0.105, -0.005, 0.47, 0.052, 0.054), (s * 0.108, -0.01, 0.34, 0.058, 0.06),
            (s * 0.11, 0, 0.19, 0.042, 0.044), (s * 0.112, 0, 0.08, 0.036, 0.038), (s * 0.12, 0.05, 0.02, 0.045, 0.075)
        ], 10)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode='lines', name='3D Plexus Mesh',
        line=dict(color='rgba(0, 225, 255, 0.42)', width=1.6), hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z, mode='markers', name='Bio-Nodes',
        marker=dict(size=2.2, color='#70F3FF', opacity=0.75), hoverinfo='skip', showlegend=False
    ))

    def s_color(sc):
        if sc >= 75:
            return '#00FF9D', 'OPTIMAL'
        if sc >= 50:
            return '#FFB020', 'MODERATE STRAIN'
        return '#FF3B5C', 'HIGH RISK ALERT'

    organs = [
        ('🧠 Brain & Autonomic', (0, 0.01, 1.71), organ_scores['Autonomic (HRV) Tone'], f"HRV: {patient_state['hrv']}ms", 18),
        ('🫀 Heart (Cardiac)', (-0.045, 0.03, 1.26), organ_scores['Cardiac Efficiency'], f"HR: {patient_state['resting_hr']}bpm", 22),
        ('🫁 Lungs (Pulmonary)', (0.075, 0.02, 1.28), organ_scores['Sleep & Oxygenation'], f"SpO2: {patient_state['spo2']}%", 18),
        ('🧬 Liver & Pancreas', (-0.06, 0.02, 1.07), organ_scores['Glycemic & Metabolic'], f"HbA1c: {patient_state['hba1c']}%", 19),
        ('🩸 Vascular Tree', (0.01, 0.01, 1.17), organ_scores['Vascular Health'], f"BP: {patient_state['sys_bp']}mmHg", 16)
    ]
    for name, (ox, oy, oz), sc, metric, sz in organs:
        col, st_txt = s_color(sc)
        fig.add_trace(go.Scatter3d(
            x=[ox], y=[oy], z=[oz], mode='markers',
            marker=dict(size=sz * 1.6, color=col, opacity=0.28), hoverinfo='skip', showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[ox], y=[oy], z=[oz], mode='markers+text', name=f"{name} ({sc}/100)",
            text=[name.split()[0]], textposition="top center",
            marker=dict(size=sz * 0.75, color=col, opacity=1.0, line=dict(color='white', width=2)),
            hovertemplate=f"<b>{name}</b><br>Score: {sc}/100 ({st_txt})<br>{metric}<extra></extra>"
        ))

    fig.add_trace(go.Scatter3d(
        x=[-0.42], y=[0.03], z=[0.90], mode='markers+text', name='⌚ Smartwatch Node',
        text=['⌚ Watch'], textposition="bottom center",
        marker=dict(size=13, color='#D668FF', symbol='diamond', line=dict(color='white', width=2)),
        hovertemplate=f"<b>⌚ {watch_source}</b><br>Steps: {patient_state['daily_steps']:,}<extra></extra>"
    ))
    fig.add_trace(go.Scatter3d(
        x=[-0.42, -0.32, -0.22, -0.045, 0.01, -0.06, 0.0],
        y=[0.03, 0.01, 0.0, 0.03, 0.01, 0.02, 0.01],
        z=[0.90, 1.13, 1.37, 1.26, 1.17, 1.07, 1.71],
        mode='lines', name='Neural Conduit', line=dict(color='#00FFCC', width=3, dash='dot'), hoverinfo='skip'
    ))
    fig.update_layout(
        title="ANATOMY | Interactive 3D Low-Poly Plexus Digital Twin",
        scene=dict(
            xaxis=dict(visible=False, range=[-0.65, 0.65]),
            yaxis=dict(visible=False, range=[-0.55, 0.55]),
            zaxis=dict(visible=False, range=[-0.02, 1.92]),
            aspectmode='manual', aspectratio=dict(x=1.15, y=0.85, z=2.0),
            camera=dict(eye=dict(x=0.0, y=1.85, z=0.25)), bgcolor='#04142C'
        ),
        paper_bgcolor='#04142C', font=dict(color='#70F3FF'), height=580,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.01, bgcolor="rgba(4,20,44,0.75)")
    )
    return fig


st.title("🧬 ChronoTwin: 3D Plexus Human Digital Twin & Smartwatch Analyzer")
st.markdown("**Happiest Health Digital Twin Challenge** | *3D Holographic Anatomy + Smartwatch Telemetry + EHR Engine*")

defaults = {'resting_hr': 80, 'hrv': 34, 'daily_steps': 4200, 'sleep_hrs': 5.8, 'spo2': 96.2, 'watch_source': 'Manual Baseline'}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

with st.expander("⌚ Connect Smartwatch / Wearable Device (BLE Stream, CSV, or Cloud API)", expanded=False):
    c1, c2 = st.columns([1.2, 1.8])
    with c1:
        mode = st.radio("Integration Mode:", ["📡 Live BLE Stream", "📂 Upload Watch CSV/JSON", "🌐 Cloud API Webhook"])
    with c2:
        if mode == "📡 Live BLE Stream":
            prof = st.selectbox("Select Watch Stream:", [
                "Apple Watch Series 10 — High Stress Alert",
                "Samsung Galaxy Watch 7 — Optimal Recovery",
                "Garmin Forerunner — Endurance Athlete",
                "Fitbit Sense 2 — Sleep Deprivation"
            ])
            if st.button("🔄 Sync Live BLE Packet", type="primary"):
                p_map = {
                    "Apple Watch Series 10 — High Stress Alert": (84, 28, 3100, 5.5, 95.8),
                    "Samsung Galaxy Watch 7 — Optimal Recovery": (64, 58, 9400, 7.6, 98.2),
                    "Garmin Forerunner — Endurance Athlete": (52, 78, 13500, 8.2, 98.8),
                    "Fitbit Sense 2 — Sleep Deprivation": (88, 24, 4600, 4.8, 95.1)
                }
                r_hr, hrv_v, stp, slp, ox = p_map[prof]
                st.session_state.update({
                    'resting_hr': r_hr, 'hrv': hrv_v, 'daily_steps': stp,
                    'sleep_hrs': slp, 'spo2': ox, 'watch_source': prof.split('—')[0].strip()
                })
                st.success("Synced live BLE packet!")
        elif mode == "📂 Upload Watch CSV/JSON":
            up = st.file_uploader("Upload CSV/JSON", type=["csv", "json"])
            if up is not None:
                try:
                    row = pd.read_csv(up).iloc[-1].to_dict() if up.name.endswith(".csv") else json.load(up)
                    st.session_state['resting_hr'] = int(row.get('resting_hr', 75))
                    st.session_state['hrv'] = int(row.get('hrv', 42))
                    st.session_state['daily_steps'] = int(row.get('daily_steps', 6000))
                    st.session_state['sleep_hrs'] = float(row.get('sleep_hrs', 6.8))
                    st.session_state['spo2'] = float(row.get('spo2', 97.0))
                    st.session_state['watch_source'] = f"File ({up.name})"
                    st.success("Watch data loaded!")
                except Exception as e:
                    st.error(f"Format error: {e}")
        else:
            payload = st.text_area("Webhook JSON:", value='{"device":"Pixel Watch 3","resting_hr":74,"hrv":45,"daily_steps":7800,"sleep_hrs":7.1,"spo2":97.6}')
            if st.button("⚡ Ingest Cloud API"):
                data = json.loads(payload)
                st.session_state.update({
                    'resting_hr': int(data.get('resting_hr', 74)), 'hrv': int(data.get('hrv', 45)),
                    'daily_steps': int(data.get('daily_steps', 7800)), 'sleep_hrs': float(data.get('sleep_hrs', 7.1)),
                    'spo2': float(data.get('spo2', 97.6)), 'watch_source': data.get('device', 'Cloud API')
                })
                st.success("Cloud API synced!")

st.sidebar.header("📋 1. Clinical EHR Baseline")
age = st.sidebar.slider("Age (Years)", 22, 80, 52)
bmi = st.sidebar.slider("BMI (kg/m²)", 18.0, 42.0, 29.2, step=0.2)
hba1c = st.sidebar.slider("HbA1c (%)", 4.5, 11.0, 6.7, step=0.1)
sys_bp = st.sidebar.slider("Systolic BP (mmHg)", 90, 190, 138)
ldl = st.sidebar.slider("LDL Cholesterol (mg/dL)", 60, 220, 144)

st.sidebar.markdown("---")
st.sidebar.subheader("⌚ 2. Smartwatch Stream")
st.sidebar.caption(f"Source: **{st.session_state['watch_source']}**")
resting_hr = st.sidebar.slider("Resting HR (bpm)", 45, 110, int(st.session_state['resting_hr']))
hrv = st.sidebar.slider("HRV (ms)", 15, 100, int(st.session_state['hrv']))
daily_steps = st.sidebar.slider("Daily Steps", 1000, 18000, int(st.session_state['daily_steps']), step=250)
sleep_hrs = st.sidebar.slider("Sleep (hrs/night)", 4.0, 9.5, float(st.session_state['sleep_hrs']), step=0.25)
spo2 = st.sidebar.slider("SpO2 (%)", 90.0, 100.0, float(st.session_state['spo2']), step=0.2)

baseline_state = {
    'age': age, 'bmi': bmi, 'hba1c': hba1c, 'sys_bp': sys_bp, 'ldl': ldl,
    'resting_hr': resting_hr, 'hrv': hrv, 'daily_steps': daily_steps, 'sleep_hrs': sleep_hrs, 'spo2': spo2
}

col_sim, col_viz = st.columns([0.95, 2.05])
with col_sim:
    st.subheader("🧪 3. Predictive Care Simulator")
    view_mode = st.radio("3D Body Mode:", ["Current Baseline Patient", "6-Month Post-Intervention Twin"])
    sim_steps = st.slider("Prescribed Daily Steps", 1000, 18000, max(daily_steps, 9500), step=500)
    sim_sleep = st.slider("Prescribed Sleep (hrs)", 4.0, 9.5, max(sleep_hrs, 7.5), step=0.25)
    sim_bmi = st.slider("Target BMI", 18.0, 42.0, round(max(18.5, bmi * 0.90), 1), step=0.2)
    sim_bp = st.slider("Target Systolic BP", 90, 190, min(sys_bp, 120))

target_state = baseline_state.copy()
target_state.update({'daily_steps': sim_steps, 'sleep_hrs': sim_sleep, 'bmi': sim_bmi, 'sys_bp': sim_bp})

base_risk = engine.predict_risk(baseline_state)
traj_df = engine.simulate_trajectory(baseline_state, target_state, months=6)
final_risk = traj_df['Digital Twin Guided Care'].iloc[-1]

final_approx = target_state.copy()
final_approx['resting_hr'] = traj_df['Predicted Resting HR (bpm)'].iloc[-1]
final_approx['hba1c'] = traj_df['Predicted HbA1c (%)'].iloc[-1]
final_approx['hrv'] = round(min(100, hrv + ((sim_steps - daily_steps) / 1000.0) * 1.8), 1)

base_organs = engine.compute_organ_scores(baseline_state)
target_organs = engine.compute_organ_scores(final_approx)

with col_viz:
    st.subheader("📊 4. 3D Holographic Anatomy & AI Outcome Forecast")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current Risk", f"{base_risk:.1f}%")
    k2.metric("6-Month Twin Risk", f"{final_risk:.1f}%", f"{final_risk - base_risk:.1f}%", delta_color="inverse")
    k3.metric("Predicted HR", f"{final_approx['resting_hr']} bpm", f"{round(final_approx['resting_hr'] - resting_hr, 1)} bpm", delta_color="inverse")
    k4.metric("Predicted HbA1c", f"{final_approx['hba1c']}%", f"{round(final_approx['hba1c'] - hba1c, 2)}%", delta_color="inverse")

    t_3d, t_1, t_2, t_3 = st.tabs(["🧍 3D Plexus Human Analyzer", "📈 6-Month Risk Trajectory", "🫀 Organ Radar", "⌚ 24h Watch Stream"])
    with t_3d:
        act_org = base_organs if view_mode == "Current Baseline Patient" else target_organs
        act_pat = baseline_state if view_mode == "Current Baseline Patient" else final_approx
        st.plotly_chart(build_3d_human_analyzer(act_org, act_pat, st.session_state['watch_source']), use_container_width=True)
    with t_1:
        f1 = go.Figure()
        f1.add_trace(go.Scatter(x=traj_df['Month'], y=traj_df['Status Quo Trajectory'], mode='lines+markers', name='Status Quo', line=dict(color='#EF553B', width=3, dash='dash')))
        f1.add_trace(go.Scatter(x=traj_df['Month'], y=traj_df['Digital Twin Guided Care'], mode='lines+markers', name='Digital Twin Care', line=dict(color='#00CC96', width=4)))
        f1.update_layout(title="6-Month Cardiometabolic Risk Projection", yaxis=dict(range=[0, 100]), height=440)
        st.plotly_chart(f1, use_container_width=True)
    with t_2:
        cats = list(base_organs.keys())
        f2 = go.Figure()
        f2.add_trace(go.Scatterpolar(r=list(base_organs.values()) + [list(base_organs.values())[0]], theta=cats + [cats[0]], fill='toself', name='Baseline', line_color='#EF553B'))
        f2.add_trace(go.Scatterpolar(r=list(target_organs.values()) + [list(target_organs.values())[0]], theta=cats + [cats[0]], fill='toself', name='6-Month Twin', line_color='#00CC96'))
        f2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=440)
        st.plotly_chart(f2, use_container_width=True)
    with t_3:
        s_df = engine.generate_24h_watch_stream(resting_hr, hrv, daily_steps, spo2)
        f3 = go.Figure()
        f3.add_trace(go.Scatter(x=s_df['Hour'], y=s_df['Heart Rate (bpm)'], mode='lines+markers', name='HR (bpm)', line=dict(color='#636EFA', width=3)))
        f3.add_trace(go.Scatter(x=s_df['Hour'], y=s_df['SpO2 (%)'], mode='lines', name='SpO2 (%)', line=dict(color='#00CC96', width=2, dash='dot'), yaxis='y2'))
        f3.update_layout(yaxis=dict(title="HR (bpm)"), yaxis2=dict(title="SpO2 (%)", overlaying='y', side='right', range=[88, 100]), height=440)
        st.plotly_chart(f3, use_container_width=True)

st.markdown("---")
st.subheader("🩺 5. AI Clinical Action Plan & Export")
rc1, rc2 = st.columns([2, 1])
with rc1:
    st.info(
