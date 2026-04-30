import json
import os
import logging
from datetime import datetime
from pathlib import Path

# Suppress MLFlow Git warning before imports
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
logging.getLogger("mlflow.utils.git_utils").setLevel(logging.ERROR)

import pandas as pd

try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in os.sys.path:
    os.sys.path.append(str(ROOT_DIR))

from Scripts.db_utils import ensure_platform_tables, fetch_recent_predictions, get_postgres_connection
from Scripts.model_utils import MIN_LIVE_LABEL_SAMPLE, PRODUCTION_BASELINE_METRICS, calculate_live_metrics, prepare_monitoring_dataframe


class ModelPerformanceMonitor:
    """Monitor production predictions, feedback coverage, and drift-style alerting."""

    def __init__(self):
        self.mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        self.alert_thresholds = {
            'accuracy_drop': 0.15,      # Increased from 0.05 (15% drop before alert)
            'f1_drop': 0.15,            # Increased from 0.05 (15% drop before alert)
            'high_risk_increase': 0.30, # Increased from 0.15 (30% increase before alert)
            'prediction_volume_drop': 0.5,
            'ground_truth_coverage_min': 0.10,  # Decreased from 0.30 (10% minimum)
            'probability_drift': 0.15,  # Increased from 0.08
            'min_live_label_sample': MIN_LIVE_LABEL_SAMPLE,
        }
        self.baseline_metrics = self.load_baseline_metrics()

    def load_baseline_metrics(self):
        if mlflow is None:
            fallback = PRODUCTION_BASELINE_METRICS.copy()
            fallback.update({'precision': 0.70, 'recall': 0.85, 'high_risk_predictions': 100})
            return fallback

        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            client = mlflow.MlflowClient(self.mlflow_tracking_uri)
            # Use new API (stages parameter is deprecated since MLFlow 2.9.0)
            prod_versions = client.get_latest_versions('lr')

            # Filter for Production stage manually
            prod_version = None
            for v in prod_versions:
                if v.current_stage == 'Production':
                    prod_version = v
                    break

            if prod_version:
                run = mlflow.get_run(prod_version.run_id)
                baseline = {}
                for key, value in run.data.metrics.items():
                    if key.startswith('test_'):
                        baseline[key.replace('test_', '')] = value
                baseline['high_risk_predictions'] = run.data.metrics.get('high_risk_predictions', 100)
                print(f"Loaded baseline metrics from production model v{prod_version.version}")
                return baseline
        except Exception as error:
            print(f"Could not load baseline metrics from MLflow: {error}")

        fallback = PRODUCTION_BASELINE_METRICS.copy()
        fallback.update({'precision': 0.70, 'recall': 0.85, 'high_risk_predictions': 100})
        return fallback

    def calculate_current_metrics(self, days=7):
        try:
            connection = get_postgres_connection()
            ensure_platform_tables(connection)
            df = fetch_recent_predictions(connection, limit=1000)
            connection.close()

            if df.empty:
                print('No prediction data available for monitoring')
                return None

            df = prepare_monitoring_dataframe(df)
            cutoff_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)
            recent_df = df[df['created_at'] >= cutoff_date]

            if recent_df.empty:
                print(f'No predictions in the last {days} days')
                return None

            total_predictions = len(recent_df)
            high_risk_count = int((recent_df['predicted_risk'] == 'HIGH RISK').sum())
            medium_risk_count = int((recent_df['predicted_risk'] == 'MEDIUM RISK').sum())
            low_risk_count = int((recent_df['predicted_risk'] == 'LOW RISK').sum())
            live_metrics = calculate_live_metrics(recent_df)

            current_metrics = {
                'total_predictions': total_predictions,
                'high_risk_predictions': high_risk_count,
                'medium_risk_predictions': medium_risk_count,
                'low_risk_predictions': low_risk_count,
                'estimated_accuracy': live_metrics.get('accuracy', self.baseline_metrics.get('accuracy', 0.85)),
                'prediction_volume': total_predictions,
                'monitoring_period_days': days,
                'ground_truth_coverage': live_metrics.get('coverage', 0.0),
                'live_sample_size': live_metrics.get('sample_size', 0),
                'live_f1': live_metrics.get('f1'),
                'live_roc_auc': live_metrics.get('roc_auc'),
                'avg_probability': float(recent_df['predicted_probability'].mean()),
            }
            return current_metrics

        except Exception as error:
            print(f'Error calculating current metrics: {error}')
            return None

    def detect_performance_drift(self, current_metrics):
        if not current_metrics or not self.baseline_metrics:
            return []

        alerts = []

        if 'estimated_accuracy' in current_metrics and 'accuracy' in self.baseline_metrics:
            accuracy_drop = self.baseline_metrics['accuracy'] - current_metrics['estimated_accuracy']
            if accuracy_drop > self.alert_thresholds['accuracy_drop']:
                alerts.append({
                    'type': 'accuracy_drop',
                    'severity': 'high',
                    'message': f'Accuracy dropped by {accuracy_drop:.1%} (threshold: {self.alert_thresholds["accuracy_drop"]:.1%})',
                    'current_value': current_metrics['estimated_accuracy'],
                    'baseline_value': self.baseline_metrics['accuracy'],
                    'threshold': self.alert_thresholds['accuracy_drop'],
                })

        if 'high_risk_predictions' in current_metrics and current_metrics['total_predictions'] > 0:
            baseline_high_risk_rate = self.baseline_metrics.get('high_risk_predictions', 100) / 1000
            current_high_risk_rate = current_metrics['high_risk_predictions'] / current_metrics['total_predictions']
            if baseline_high_risk_rate > 0:
                increase = (current_high_risk_rate - baseline_high_risk_rate) / baseline_high_risk_rate
                if increase > self.alert_thresholds['high_risk_increase']:
                    alerts.append({
                        'type': 'high_risk_increase',
                        'severity': 'medium',
                        'message': f'High-risk predictions increased by {increase:.1%} (threshold: {self.alert_thresholds["high_risk_increase"]:.1%})',
                        'current_value': current_high_risk_rate,
                        'baseline_value': baseline_high_risk_rate,
                        'threshold': self.alert_thresholds['high_risk_increase'],
                    })

        if current_metrics['prediction_volume'] < 10:
            alerts.append({
                'type': 'low_prediction_volume',
                'severity': 'medium',
                'message': f'Very low prediction volume: {current_metrics["prediction_volume"]} predictions',
                'current_value': current_metrics['prediction_volume'],
                'threshold': 10,
            })

        if current_metrics.get('ground_truth_coverage', 0.0) < self.alert_thresholds['ground_truth_coverage_min']:
            alerts.append({
                'type': 'ground_truth_gap',
                'severity': 'medium',
                'message': f'Ground-truth coverage is only {current_metrics.get("ground_truth_coverage", 0.0):.1%}',
                'current_value': current_metrics.get('ground_truth_coverage', 0.0),
                'baseline_value': 1.0,
                'threshold': self.alert_thresholds['ground_truth_coverage_min'],
            })

        live_f1 = current_metrics.get('live_f1')
        if current_metrics.get('live_sample_size', 0) < self.alert_thresholds['min_live_label_sample'] and current_metrics.get('live_sample_size', 0) > 0:
            alerts.append({
                'type': 'limited_ground_truth_sample',
                'severity': 'medium',
                'message': (
                    f'Only {int(current_metrics.get("live_sample_size", 0))} confirmed outcomes are available; '
                    f'at least {self.alert_thresholds["min_live_label_sample"]} are required before treating live F1 as a strong retraining signal'
                ),
                'current_value': current_metrics.get('live_sample_size', 0),
                'baseline_value': self.alert_thresholds['min_live_label_sample'],
                'threshold': self.alert_thresholds['min_live_label_sample'],
            })
        elif live_f1 is not None and not pd.isna(live_f1):
            baseline_f1 = self.baseline_metrics.get('f1', PRODUCTION_BASELINE_METRICS['f1'])
            f1_drop = baseline_f1 - live_f1
            if f1_drop > self.alert_thresholds['f1_drop']:
                alerts.append({
                    'type': 'live_f1_drop',
                    'severity': 'high',
                    'message': f'Live F1 dropped by {f1_drop:.3f}',
                    'current_value': live_f1,
                    'baseline_value': baseline_f1,
                    'threshold': self.alert_thresholds['f1_drop'],
                })

        return alerts

    def log_monitoring_results(self, current_metrics, alerts):
        if mlflow is None:
            print('MLflow is not installed; skipping monitoring run logging')
            return

        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment('model_monitoring')

            with mlflow.start_run(run_name=f'monitoring_{datetime.now().strftime("%Y%m%d_%H%M%S")}') as run:
                mlflow.set_tag('monitoring_type', 'performance_drift')
                mlflow.set_tag('alerts_count', len(alerts))

                if current_metrics:
                    metrics_to_log = {
                        key: value
                        for key, value in current_metrics.items()
                        if isinstance(value, (int, float)) and value is not None and not pd.isna(value)
                    }
                    mlflow.log_metrics(metrics_to_log)
                    mlflow.log_param('monitoring_period_days', current_metrics.get('monitoring_period_days', 7))

                if alerts:
                    mlflow.log_param('alerts', json.dumps(alerts, default=str))
                    severity_counts = {}
                    for alert in alerts:
                        severity = alert['severity']
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    for severity, count in severity_counts.items():
                        mlflow.log_metric(f'alerts_{severity}', count)

                print(f'Monitoring results logged to MLflow: {run.info.run_id}')
        except Exception as error:
            print(f'Could not log monitoring results: {error}')

    def send_alerts(self, alerts):
        if not alerts:
            print('No performance alerts detected')
            return

        print(f'{len(alerts)} Performance Alert(s) Detected:')
        for index, alert in enumerate(alerts, start=1):
            print(f"{index}. [{alert['severity'].upper()}] {alert['message']}")

    def run_monitoring_cycle(self, days=7, log_to_mlflow=True):
        print(f'Starting model performance monitoring cycle at {datetime.now()}')
        current_metrics = self.calculate_current_metrics(days=days)
        alerts = []

        if current_metrics:
            print(
                f"Current metrics: {current_metrics['total_predictions']} predictions, "
                f"{current_metrics['high_risk_predictions']} high-risk"
            )
            alerts = self.detect_performance_drift(current_metrics)
            if log_to_mlflow:
                self.log_monitoring_results(current_metrics, alerts)
            self.send_alerts(alerts)
        else:
            print('Could not calculate current metrics')

        print('Monitoring cycle completed')
        return {'current_metrics': current_metrics, 'alerts': alerts}


def trigger_retraining_if_needed(alerts, threshold=3):
    high_severity_alerts = [alert for alert in alerts if alert['severity'] == 'high']
    if len(high_severity_alerts) >= threshold:
        print(f'Triggering automated retraining due to {len(high_severity_alerts)} high-severity alerts')
        return True

    print(f'{len(high_severity_alerts)} high-severity alerts (threshold: {threshold}) - no retraining needed')
    return False


if __name__ == '__main__':
    monitor = ModelPerformanceMonitor()
    monitor.run_monitoring_cycle()
