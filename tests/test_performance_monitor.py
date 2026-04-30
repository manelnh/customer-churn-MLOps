import sys
import types
import unittest
from unittest.mock import patch

mlflow_module = types.ModuleType('mlflow')
mlflow_module.MlflowClient = object
mlflow_module.get_run = lambda *args, **kwargs: None
mlflow_module.set_tracking_uri = lambda *args, **kwargs: None
mlflow_module.set_experiment = lambda *args, **kwargs: None
mlflow_module.start_run = lambda *args, **kwargs: None

mlflow_exceptions = types.ModuleType('mlflow.exceptions')
mlflow_exceptions.MlflowException = Exception
mlflow_module.exceptions = mlflow_exceptions

sys.modules.setdefault('mlflow', mlflow_module)
sys.modules.setdefault('mlflow.exceptions', mlflow_exceptions)

from Scripts.performance_monitor import ModelPerformanceMonitor, trigger_retraining_if_needed
from Scripts.run_monitoring import main as run_monitoring_main


class PerformanceMonitorTests(unittest.TestCase):
    @patch.object(
        ModelPerformanceMonitor,
        'load_baseline_metrics',
        return_value={'accuracy': 0.9, 'f1': 0.85, 'high_risk_predictions': 100},
    )
    def test_detect_performance_drift_flags_high_and_medium_alerts(self, _load_baseline_metrics):
        monitor = ModelPerformanceMonitor()

        alerts = monitor.detect_performance_drift(
            {
                'estimated_accuracy': 0.70,
                'high_risk_predictions': 5,
                'total_predictions': 10,
                'prediction_volume': 10,
                'ground_truth_coverage': 0.5,
                'live_sample_size': 25,
                'live_f1': 0.60,
            }
        )
        alert_types = {alert['type'] for alert in alerts}

        self.assertIn('accuracy_drop', alert_types)
        self.assertIn('high_risk_increase', alert_types)
        self.assertIn('live_f1_drop', alert_types)

    @patch.object(
        ModelPerformanceMonitor,
        'load_baseline_metrics',
        return_value={'accuracy': 0.9, 'f1': 0.85, 'high_risk_predictions': 100},
    )
    def test_run_monitoring_cycle_returns_current_metrics_and_alerts(self, _load_baseline_metrics):
        monitor = ModelPerformanceMonitor()

        with patch('builtins.print'), \
                patch.object(monitor, 'calculate_current_metrics', return_value={'total_predictions': 12, 'high_risk_predictions': 4}), \
                patch.object(monitor, 'detect_performance_drift', return_value=[{'type': 'sample', 'severity': 'medium'}]), \
                patch.object(monitor, 'log_monitoring_results') as log_results_mock, \
                patch.object(monitor, 'send_alerts') as send_alerts_mock:
            result = monitor.run_monitoring_cycle(days=14, log_to_mlflow=True)

        self.assertEqual(result['current_metrics']['total_predictions'], 12)
        self.assertEqual(len(result['alerts']), 1)
        log_results_mock.assert_called_once()
        send_alerts_mock.assert_called_once()

    def test_trigger_retraining_if_needed_uses_high_severity_threshold(self):
        alerts = [
            {'severity': 'high'},
            {'severity': 'high'},
            {'severity': 'medium'},
        ]

        with patch('builtins.print'):
            self.assertTrue(trigger_retraining_if_needed(alerts, threshold=2))
            self.assertFalse(trigger_retraining_if_needed(alerts, threshold=3))

    @patch('Scripts.run_monitoring.ModelPerformanceMonitor')
    @patch('Scripts.run_monitoring.trigger_retraining_if_needed', return_value=False)
    def test_run_monitoring_cli_returns_nonzero_for_high_alerts_when_requested(self, _trigger_retraining, monitor_cls):
        monitor = monitor_cls.return_value
        monitor.run_monitoring_cycle.return_value = {
            'current_metrics': {'total_predictions': 5},
            'alerts': [{'severity': 'high', 'type': 'live_f1_drop'}],
        }

        with patch('builtins.print'):
            exit_code = run_monitoring_main(['--fail-on-high-alerts', '--skip-mlflow-logging'])

        self.assertEqual(exit_code, 2)
        monitor.run_monitoring_cycle.assert_called_once_with(days=7, log_to_mlflow=False)


if __name__ == '__main__':
    unittest.main()
