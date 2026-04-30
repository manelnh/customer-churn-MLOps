#!/usr/bin/env python3
"""
Automated Model Performance Monitoring Script
Run this script periodically to monitor model performance and trigger alerts/retraining
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Suppress MLFlow Git warning before imports
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
logging.getLogger("mlflow.utils.git_utils").setLevel(logging.ERROR)

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from Scripts.performance_monitor import ModelPerformanceMonitor, trigger_retraining_if_needed


def build_parser():
    parser = argparse.ArgumentParser(description='Run the churn model monitoring cycle.')
    parser.add_argument('--days', type=int, default=7, help='Lookback window in days for production metrics.')
    parser.add_argument(
        '--threshold',
        type=int,
        default=2,
        help='High-severity alert count required before retraining is triggered.',
    )
    parser.add_argument(
        '--skip-mlflow-logging',
        action='store_true',
        help='Run monitoring without logging the monitoring run to MLflow.',
    )
    parser.add_argument(
        '--fail-on-high-alerts',
        action='store_true',
        help='Exit with status code 2 when one or more high-severity alerts are detected.',
    )
    return parser


def main(argv=None):
    """Run automated model performance monitoring."""
    args = build_parser().parse_args(argv)
    print(f"Starting automated model monitoring at {datetime.now()}")

    monitor = ModelPerformanceMonitor()
    result = monitor.run_monitoring_cycle(days=args.days, log_to_mlflow=not args.skip_mlflow_logging)
    current_metrics = result.get('current_metrics')
    alerts = result.get('alerts', [])

    if current_metrics:
        retraining_triggered = trigger_retraining_if_needed(alerts, threshold=args.threshold)

        if retraining_triggered:
            print("Retraining triggered - starting training pipeline...")
            # In production, you would call the training script here.
            # os.system("python Scripts/train.py")
            print("Training not automatically started (uncomment in production)")
        else:
            print("No retraining needed at this time")
    else:
        print("Could not check retraining needs due to missing metrics")

    print(f"Automated monitoring completed at {datetime.now()}")
    if args.fail_on_high_alerts and any(alert.get('severity') == 'high' for alert in alerts):
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
