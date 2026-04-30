from datetime import datetime, timezone
import unittest

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from Scripts import model_utils


class ModelUtilsTests(unittest.TestCase):
    def test_apply_feature_engineering_adds_expected_columns(self):
        df = pd.DataFrame(
            [
                {
                    'tenure': 10,
                    'MonthlyCharges': 50.0,
                    'Data_Usage_GB': 20.0,
                    'OnlineSecurity': 'Yes',
                    'OnlineBackup': 'No',
                    'DeviceProtection': 'Yes',
                    'StreamingTV': 'No',
                    'StreamingMovies': 'Yes',
                    'SeniorCitizen': 'No',
                }
            ]
        )

        engineered = model_utils.apply_feature_engineering(df)

        self.assertIn('TotalServices', engineered.columns)
        self.assertIn('Total_Revenue', engineered.columns)
        self.assertIn('Monthly_per_Tenure', engineered.columns)
        self.assertIn('Is_First_Year', engineered.columns)
        self.assertIn('Usage_per_Month', engineered.columns)
        self.assertEqual(engineered.loc[0, 'TotalServices'], 3)
        self.assertEqual(engineered.loc[0, 'Total_Revenue'], 500.0)

    def test_prepare_customer_features_returns_scaled_row(self):
        training_rows = [
            {
                'tenure': 6,
                'MonthlyCharges': 65.0,
                'Data_Usage_GB': 30.0,
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'No',
                'DeviceProtection': 'Yes',
                'StreamingTV': 'No',
                'StreamingMovies': 'No',
                'SeniorCitizen': 'No',
                'Support_Tickets': 2,
                'App_Logins': 12,
            },
            {
                'tenure': 24,
                'MonthlyCharges': 90.0,
                'Data_Usage_GB': 80.0,
                'OnlineSecurity': 'No',
                'OnlineBackup': 'Yes',
                'DeviceProtection': 'Yes',
                'StreamingTV': 'Yes',
                'StreamingMovies': 'Yes',
                'SeniorCitizen': 'Yes',
                'Support_Tickets': 5,
                'App_Logins': 8,
            },
        ]
        train_df = model_utils.drop_unused_columns(
            model_utils.apply_feature_engineering(pd.DataFrame(training_rows))
        )
        dv = DictVectorizer(sparse=False)
        X = dv.fit_transform(train_df.to_dict(orient='records'))
        scaler = StandardScaler().fit(X)

        customer = training_rows[0]
        X_scaled, final_df = model_utils.prepare_customer_features(customer, dv, scaler)

        self.assertEqual(X_scaled.shape, (1, len(dv.get_feature_names_out())))
        self.assertNotIn('Support_Tickets', final_df.columns)
        self.assertNotIn('App_Logins', final_df.columns)
        self.assertIn('Total_Revenue', final_df.columns)

    def test_calculate_live_metrics_handles_empty_ground_truth(self):
        logs = pd.DataFrame(
            [
                {
                    'predicted_label': 'Churn',
                    'predicted_probability': 0.8,
                    'actual_label': None,
                }
            ]
        )

        metrics = model_utils.calculate_live_metrics(logs)

        self.assertEqual(metrics['coverage'], 0.0)
        self.assertEqual(metrics['sample_size'], 0)

    def test_detect_monitoring_alerts_flags_low_volume_and_limited_sample(self):
        base_time = datetime(2026, 4, 30, tzinfo=timezone.utc)
        logs = pd.DataFrame(
            [
                {
                    'created_at': base_time,
                    'predicted_probability': 0.81,
                    'predicted_risk': 'HIGH RISK',
                    'predicted_label': 'Churn',
                    'actual_label': 'Churn',
                },
                {
                    'created_at': base_time,
                    'predicted_probability': 0.77,
                    'predicted_risk': 'HIGH RISK',
                    'predicted_label': 'Churn',
                    'actual_label': None,
                },
                {
                    'created_at': base_time,
                    'predicted_probability': 0.32,
                    'predicted_risk': 'LOW RISK',
                    'predicted_label': 'No churn',
                    'actual_label': None,
                },
                {
                    'created_at': base_time,
                    'predicted_probability': 0.68,
                    'predicted_risk': 'MEDIUM RISK',
                    'predicted_label': 'Churn',
                    'actual_label': None,
                },
                {
                    'created_at': base_time,
                    'predicted_probability': 0.41,
                    'predicted_risk': 'LOW RISK',
                    'predicted_label': 'No churn',
                    'actual_label': None,
                },
            ]
        )

        alerts = model_utils.detect_monitoring_alerts(logs)
        alert_types = {alert['type'] for alert in alerts}

        self.assertIn('low_prediction_volume', alert_types)
        self.assertIn('ground_truth_gap', alert_types)
        self.assertIn('limited_ground_truth_sample', alert_types)


if __name__ == '__main__':
    unittest.main()
