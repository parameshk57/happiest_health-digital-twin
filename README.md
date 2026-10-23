# 🧬 ChronoTwin: Predictive Virtual Patient Model
**Submission for the Happiest Health Digital Twin Challenge**

## 1. Problem Statement & Healthcare Impact
Clinical care typically relies on episodic Electronic Health Record (EHR) snapshots, missing continuous physiological degradation between hospital visits. **ChronoTwin** creates a dynamic, living virtual representation of a patient by fusing static EHR biomarkers (HbA1c, LDL, BP, BMI) with continuous wearable telemetry (Resting HR, HRV, Daily Steps, Sleep Architecture).

## 2. System Architecture
1. **Multimodal Data Ingestion Layer**: Harmonizes clinical lab records with 30-day rolling wearable biometric streams.
2. **Coupled Physiological Twin Engine**: Models secondary biological feedback loops (e.g., how increased daily step volume and sleep restoration autonomously down-regulate resting heart rate, boost HRV, and improve glycemic control over time).
3. **Counterfactual Simulation ("What-If" Engine)**: Allows clinicians and patients to test lifestyle and pharmacological interventions virtually before prescribing them.

## 3. Installation & Local Execution
```bash
git clone [https://github.com/](https://github.com/)<your-username>/happiest-health-digital-twin.git
cd happiest-health-digital-twin
pip install -r requirements.txt
streamlit run app.py
