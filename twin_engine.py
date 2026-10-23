import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

class DigitalTwinEngine:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
        self._train_baseline_model()

    def _generate_synthetic_cohort(self, n=1000):
        np.random.seed(42)
        # EHR Data
        age = np.random.randint(25, 80, n)
        bmi = np.random.normal(26.5, 4.5, n).clip(18, 45)
        hba1c = np.random.normal(5.8, 1.1, n).clip(4.5, 11.0)
        sys_bp = np.random.normal(125, 18, n).clip(90, 190)
        ldl = np.random.normal(115, 30, n).clip(50, 220)

        # Wearable Telemetry Data (30-day rolling averages)
        resting_hr = np.random.normal(72, 10, n).clip(48, 110)
        hrv = np.random.normal(45, 15, n).clip(15, 95) # Heart Rate Variability (ms)
        daily_steps = np.random.normal(6500, 3000, n).clip(1000, 18000)
        sleep_hrs = np.random.normal(6.8, 1.2, n).clip(4.0, 9.5)

        # Non-linear physiological risk score (0 to 100 scale)
        risk = (
            0.25 * (age - 25) +
            0.85 * (bmi - 22) +
            6.50 * (hba1c - 5.0) +
            0.22 * (sys_bp - 110) +
            0.08 * (ldl - 90) +
            0.30 * (resting_hr - 60) -
            0.25 * (hrv - 50) -
            0.0018 * (daily_steps - 5000) -
            1.80 * (sleep_hrs - 7.0) +
            np.random.normal(0, 2.0, n)
        )
        risk_prob = (1 / (1 + np.exp(-0.08 * (risk - 18)))) * 100

        return pd.DataFrame({
            'age': age, 'bmi': bmi, 'hba1c': hba1c, 'sys_bp': sys_bp, 'ldl': ldl,
            'resting_hr': resting_hr, 'hrv': hrv, 'daily_steps': daily_steps,
            'sleep_hrs': sleep_hrs, 'risk_score': risk_prob.clip(1, 99)
        })

    def _train_baseline_model(self):
        df = self._generate_synthetic_cohort(1200)
        X = df.drop(columns=['risk_score'])
        y = df['risk_score']
        self.model.fit(X, y)

    def predict_risk(self, patient_dict):
        df_input = pd.DataFrame([patient_dict])
        return float(self.model.predict(df_input)[0])

    def simulate_trajectory(self, baseline_patient, target_patient, months=6):
        """Simulates physiological transition over N months from baseline to intervention state."""
        timeline = list(range(months + 1))
        baseline_trend = []
        intervention_trend = []

        base_risk = self.predict_risk(baseline_patient)
        for m in timeline:
            # Without intervention, chronic strain slightly increases risk over time
            baseline_trend.append(min(99.0, base_risk * (1 + 0.012 * m)))

            # Physiological adaptation curve (gradual transition to target state)
            alpha = 1 - np.exp(-0.6 * m)
            interpolated_state = {
                k: baseline_patient[k] + alpha * (target_patient[k] - baseline_patient[k])
                for k in baseline_patient
            }
            # Secondary physiological coupling: more steps & sleep improve BMI, HR, HRV, and HbA1c
            step_delta = interpolated_state['daily_steps'] - baseline_patient['daily_steps']
            sleep_delta = interpolated_state['sleep_hrs'] - baseline_patient['sleep_hrs']
            
            interpolated_state['resting_hr'] -= (step_delta / 1000) * 1.1 * alpha
            interpolated_state['hrv'] += (step_delta / 1000) * 1.5 * alpha + max(0, sleep_delta) * 2.0 * alpha
            interpolated_state['hba1c'] -= (step_delta / 2000) * 0.12 * alpha

            intervention_trend.append(self.predict_risk(interpolated_state))

        return pd.DataFrame({
            'Month': timeline,
            'Status Quo Trajectory': baseline_trend,
            'Digital Twin Intervention Trajectory': intervention_trend
        })
