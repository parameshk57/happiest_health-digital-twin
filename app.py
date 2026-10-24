import streamlit as st
import plotly.graph_objects as go
from twin_engine import DigitalTwinEngine

st.set_page_config(page_title="Happiest Health | Digital Twin PoC", layout="wide")

@st.cache_resource
def load_engine():
    return DigitalTwinEngine()

engine = load_engine()

st.title("🧬 Virtual Patient Digital Twin")
st.markdown("**Build. Predict. Reimagine Healthcare.** | *EHR + Wearable Predictive Care Engine*")

# Sidebar: Baseline Patient Profile (EHR + Wearable Stream)
st.sidebar.header("📋 1. Patient Baseline (EHR + Wearable)")
age = st.sidebar.slider("Age (Years)", 25, 80, 52)
bmi = st.sidebar.slider("BMI (kg/m²)", 18.0, 42.0, 29.5)
hba1c = st.sidebar.slider("HbA1c (%)", 4.5, 10.5, 6.6)
sys_bp = st.sidebar.slider("Systolic BP (mmHg)", 90, 185, 138)
ldl = st.sidebar.slider("LDL Cholesterol (mg/dL)", 60, 210, 142)

st.sidebar.subheader("⌚ Live Wearable Telemetry (30d Avg)")
resting_hr = st.sidebar.slider("Resting Heart Rate (bpm)", 50, 105, 81)
hrv = st.sidebar.slider("HRV - Heart Rate Variability (ms)", 15, 90, 32)
daily_steps = st.sidebar.slider("Daily Steps", 1000, 15000, 3800, step=500)
sleep_hrs = st.sidebar.slider("Average Sleep (hrs/night)", 4.0, 9.0, 5.8, step=0.5)

baseline_state = {
    'age': age, 'bmi': bmi, 'hba1c': hba1c, 'sys_bp': sys_bp, 'ldl': ldl,
    'resting_hr': resting_hr, 'hrv': hrv, 'daily_steps': daily_steps, 'sleep_hrs': sleep_hrs
}

# Main Area: Digital Twin Simulation Controls
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🧪 2. Simulate Predictive Care")
    st.caption("Adjust lifestyle & clinical interventions on the Digital Twin:")
    sim_steps = st.slider("Target Daily Steps", 1000, 15000, max(daily_steps, 8500), step=500)
    sim_sleep = st.slider("Target Sleep (hrs/night)", 4.0, 9.0, max(sleep_hrs, 7.5), step=0.5)
    sim_bmi = st.slider("Target BMI (via Nutrition Plan)", 18.0, 42.0, round(bmi * 0.93, 1))
    sim_bp = st.slider("Target Systolic BP (Medication/Diet)", 90, 185, min(sys_bp, 122))

target_state = baseline_state.copy()
target_state.update({
    'daily_steps': sim_steps,
    'sleep_hrs': sim_sleep,
    'bmi': sim_bmi,
    'sys_bp': sim_bp
})

base_risk = engine.predict_risk(baseline_state)
trajectory_df = engine.simulate_trajectory(baseline_state, target_state, months=6)
final_twin_risk = trajectory_df['Digital Twin Intervention Trajectory'].iloc[-1]
risk_delta = final_twin_risk - base_risk

with col2:
    st.subheader("📊 3. AI Insights & 6-Month Outcome Prediction")
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Baseline Risk", f"{base_risk:.1f}%")
    m2.metric("6-Month Twin Predicted Risk", f"{final_twin_risk:.1f}%", f"{risk_delta:.1f}%", delta_color="inverse")
    m3.metric("Predicted Physiological State", "Optimized" if final_twin_risk < 35 else "Moderate Alert")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trajectory_df['Month'], y=trajectory_df['Status Quo Trajectory'],
        mode='lines+markers', name='Status Quo (Unintervened)', line=dict(color='#EF553B', width=3, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=trajectory_df['Month'], y=trajectory_df['Digital Twin Intervention Trajectory'],
        mode='lines+markers', name='Digital Twin Guided Care', line=dict(color='#00CC96', width=4)
    ))
    fig.update_layout(
        title="6-Month Cardiometabolic Adverse Event Risk Trajectory",
        xaxis_title="Months from Baseline",
        yaxis_title="Predicted Risk Score (%)",
        yaxis=dict(range=[0, 100]),
        height=380
    )
    st.plotly_chart(fig, use_container_width=True)
