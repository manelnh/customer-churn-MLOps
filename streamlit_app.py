"""
TelCo Churn Prediction - Managerial Decision Support System
===============================================================
A Streamlit MLOps dashboard for customer churn prediction with executive insights,
actionable recommendations, and technical model governance.
"""

import os
import sys
import json
import base64
import logging
import subprocess
from urllib import error as urllib_error
from urllib import request as urllib_request
import warnings
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import Scripts.db_utils as db_utils

# Suppress Git/Python warnings for cleaner output
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Import local modules
from Scripts.db_utils import (
    get_postgres_connection,
    bootstrap_platform_tables_if_enabled,
    insert_prediction_log,
    insert_governance_decision,
    update_prediction_ground_truth,
    replace_monitoring_alerts,
    fetch_recent_predictions as load_prediction_logs,
    fetch_recent_alerts as load_alerts,
    fetch_recent_governance_decisions as load_governance_decisions,
)
from Scripts.model_utils import (
    DEFAULT_DECISION_THRESHOLD,
    calculate_model_selection_score,
    classify_from_probability,
    filter_predictions_for_active_model,
    get_active_bundle_metadata,
    get_baseline_metrics_from_metadata,
    load_model_bundle,
    load_bundle,
    prepare_customer_features,
    get_risk_label,
    calculate_live_metrics,
    detect_monitoring_alerts,
    build_probability_drift_frame,
    build_lifecycle_frame,
    build_retraining_recommendation,
    calculate_model_trust_score,
    serialize_top_drivers,
    explain_top_drivers,
    run_lab_experiment_cached,
    log_lab_run_to_mlflow,
    build_lab_run_name,
    get_mlflow_tracking_uri,
    get_mlflow_experiment_name,
)
from Scripts.model_utils import get_production_baseline_metrics, get_production_baseline_params

# --- Configuration Constants ---
ROOT_DIR = Path(__file__).parent.resolve()
PAGE_ICON = '📊'
PRODUCTION_BASELINE_METRICS = get_production_baseline_metrics()
PRODUCTION_BASELINE_PARAMS = get_production_baseline_params()
FORM_SOURCE = 'streamlit_ui'


def get_bundle_metadata() -> dict:
    metadata = get_active_bundle_metadata()
    if metadata:
        return metadata

    candidate_paths = [ROOT_DIR / 'churn_production.pkl', ROOT_DIR / 'models' / 'churn_production_bundle.pkl']
    for path in candidate_paths:
        if path.exists():
            try:
                return load_bundle(path).get('metadata', {})
            except Exception:
                continue
    return {}

# --- Custom Styling ---
def inject_custom_style():
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.35rem;
            font-weight: 800;
            color: #0f3057;
            margin: 0;
            letter-spacing: -0.03em;
        }
        .sub-header {
            font-size: 1.4rem;
            font-weight: 600;
            color: #1d3557;
        }
        .hero-title-wrap {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 0.75rem;
            padding: 0.95rem 1.15rem;
            background: linear-gradient(135deg, rgba(240, 247, 255, 0.95) 0%, rgba(226, 239, 255, 0.9) 100%);
            border: 1px solid rgba(15, 98, 168, 0.14);
            border-radius: 16px;
            box-shadow: 0 10px 28px rgba(15, 48, 87, 0.08);
        }
        .hero-title-icon {
            width: 56px;
            height: 56px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.9rem;
            background: linear-gradient(135deg, #0f62a8 0%, #3aa0d8 100%);
            color: white;
            flex-shrink: 0;
            box-shadow: 0 10px 20px rgba(15, 98, 168, 0.22);
        }
        .hero-title-copy {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        .hero-title-subtitle {
            color: #5f6f82;
            font-size: 1rem;
            font-weight: 500;
            letter-spacing: 0.01em;
        }
        .metric-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f8f9fa;
            border-radius: 8px 8px 0px 0px;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0f62a8;
            color: white;
        }
        .manager-action-banner {
            padding: 1rem;
            border-radius: 10px;
            font-weight: 700;
            text-align: center;
            margin: 0.5rem 0;
        }
        .manager-action-banner.critical {
            background: linear-gradient(135deg, #fde2e4 0%, #f8d7da 100%);
            color: #9d0208;
            border: 2px solid #9d0208;
        }
        .manager-action-banner.warning {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
            color: #8d6e00;
            border: 2px solid #8d6e00;
        }
        .manager-action-banner.healthy {
            background: linear-gradient(135deg, #d8f3dc 0%, #b7e4c7 100%);
            color: #1b4332;
            border: 2px solid #1b4332;
        }
        .insight-box {
            background: #f0f7ff;
            border-left: 4px solid #0f62a8;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin: 0.5rem 0;
        }
        .kpi-container {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            text-align: center;
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
            color: #0f62a8;
        }
        .kpi-label {
            font-size: 0.85rem;
            color: #6c757d;
            margin-top: 0.25rem;
        }
        .status-healthy { color: #2a9d8f; font-weight: 700; }
        .status-warning { color: #f4a261; font-weight: 700; }
        .status-critical { color: #d62839; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- Data Loading Functions ---
@st.cache_data
def load_dataset() -> pd.DataFrame:
    dataset_path = ROOT_DIR / 'telco_churn_cleaned.csv'
    return pd.read_csv(dataset_path)


def load_platform_data():
    connection = get_postgres_connection()
    try:
        bootstrap_platform_tables_if_enabled(connection)
        predictions = load_prediction_logs(connection)
        alerts_df = load_alerts(connection)
        governance_df = load_governance_decisions(connection)
        return predictions, alerts_df, governance_df
    finally:
        connection.close()


def get_github_training_dispatch_config() -> dict:
    return {
        'token': os.getenv('GITHUB_ACTIONS_TOKEN', '').strip(),
        'repository': os.getenv('GITHUB_REPOSITORY', '').strip(),
        'workflow': os.getenv('GITHUB_TRAINING_WORKFLOW', 'training.yml').strip(),
        'ref': os.getenv('GITHUB_WORKFLOW_REF', 'main').strip(),
    }


def get_github_monitoring_dispatch_config() -> dict:
    config = get_github_training_dispatch_config().copy()
    config['workflow'] = os.getenv('GITHUB_MONITORING_WORKFLOW', 'monitoring.yml').strip()
    return config


def get_automation_execution_mode() -> str:
    mode = os.getenv('AUTOMATION_EXECUTION_MODE', 'auto').strip().lower()
    return mode if mode in {'auto', 'github', 'local'} else 'auto'


def is_github_api_ready() -> bool:
    config = get_github_training_dispatch_config()
    return bool(config['token'] and config['repository'])


def is_github_training_dispatch_ready() -> bool:
    config = get_github_training_dispatch_config()
    return all([config['token'], config['repository'], config['workflow'], config['ref']])


def is_github_monitoring_dispatch_ready() -> bool:
    config = get_github_monitoring_dispatch_config()
    return all([config['token'], config['repository'], config['workflow'], config['ref']])


def build_github_api_headers() -> dict:
    config = get_github_training_dispatch_config()
    return {
        'Accept': 'application/vnd.github+json',
        'Authorization': f'Bearer {config["token"]}',
        'User-Agent': 'TeleLink-Streamlit-MLOps-App',
    }


def fetch_latest_github_workflow_run(workflow_file: str) -> dict | None:
    if not is_github_api_ready():
        return None

    config = get_github_training_dispatch_config()
    url = (
        f'https://api.github.com/repos/{config["repository"]}/actions/workflows/'
        f'{workflow_file}/runs?per_page=1'
    )
    request = urllib_request.Request(
        url,
        method='GET',
        headers=build_github_api_headers(),
    )

    try:
        with urllib_request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode('utf-8'))
        runs = payload.get('workflow_runs', [])
        if not runs:
            return {
                'workflow': workflow_file,
                'status': 'no_runs',
                'conclusion': '',
                'created_at': '',
                'html_url': '',
            }
        run = runs[0]
        return {
            'workflow': workflow_file,
            'status': run.get('status', 'unknown'),
            'conclusion': run.get('conclusion') or '',
            'created_at': run.get('created_at') or '',
            'html_url': run.get('html_url') or '',
            'event': run.get('event') or '',
            'run_number': run.get('run_number') or '',
        }
    except Exception as error:
        return {
            'workflow': workflow_file,
            'status': 'api_error',
            'conclusion': '',
            'created_at': '',
            'html_url': '',
            'error': str(error),
        }


def get_recent_github_workflow_runs() -> list[dict]:
    workflow_files = ['ci.yml', 'cd.yml', 'training.yml', 'monitoring.yml']
    runs = []
    for workflow in workflow_files:
        run = fetch_latest_github_workflow_run(workflow)
        if run is not None:
            runs.append(run)
    return runs


def trigger_github_workflow_dispatch(config: dict, inputs: dict, workflow_label: str) -> tuple[bool, str]:
    required_values = [config.get('token'), config.get('repository'), config.get('workflow'), config.get('ref')]
    if not all(required_values):
        return False, (
            f'GitHub Actions dispatch is not configured for {workflow_label}. '
            'Set GITHUB_ACTIONS_TOKEN, GITHUB_REPOSITORY, the workflow filename, and GITHUB_WORKFLOW_REF.'
        )

    owner_repo = config['repository']
    workflow = config['workflow']
    url = f'https://api.github.com/repos/{owner_repo}/actions/workflows/{workflow}/dispatches'
    payload = json.dumps(
        {
            'ref': config['ref'],
            'inputs': inputs,
        }
    ).encode('utf-8')
    request = urllib_request.Request(
        url,
        data=payload,
        method='POST',
        headers={
            'Accept': 'application/vnd.github+json',
            'Authorization': f'Bearer {config["token"]}',
            'Content-Type': 'application/json',
            'User-Agent': 'TeleLink-Streamlit-MLOps-App',
        },
    )

    try:
        with urllib_request.urlopen(request, timeout=20) as response:
            status_code = getattr(response, 'status', response.getcode())
        if status_code in (200, 201, 204):
            return True, (
                f'GitHub Actions {workflow_label} workflow "{workflow}" was dispatched successfully '
                f'for repository "{owner_repo}" on ref "{config["ref"]}".'
            )
        return False, f'GitHub API returned unexpected status code {status_code}.'
    except urllib_error.HTTPError as error:
        body = error.read().decode('utf-8', errors='ignore')
        if error.code == 422 and 'Unexpected inputs provided' in body:
            return False, (
                f'GitHub {workflow_label} workflow dispatch failed with HTTP 422: {body} '
                f'This usually means the workflow file "{workflow}" on ref "{config["ref"]}" '
                f'in repository "{owner_repo}" does not declare the inputs this app is sending yet. '
                'Push the updated workflow file to that branch/ref, confirm the workflow filename matches, '
                'or change GITHUB_WORKFLOW_REF to the branch that already contains the new workflow_dispatch inputs.'
            )
        return False, f'GitHub {workflow_label} workflow dispatch failed with HTTP {error.code}: {body or error.reason}'
    except urllib_error.URLError as error:
        return False, f'Unable to reach GitHub Actions API: {error.reason}'
    except Exception as error:
        return False, f'Unexpected GitHub {workflow_label} workflow dispatch error: {error}'


def trigger_github_training_workflow(reason: str, profile: str = 'quick') -> tuple[bool, str]:
    return trigger_github_workflow_dispatch(
        config=get_github_training_dispatch_config(),
        inputs={
            'reason': reason,
            'profile': profile,
        },
        workflow_label='training',
    )


def trigger_github_monitoring_workflow(
    days: int = 14,
    threshold: int = 2,
    auto_retrain: bool = True,
    profile: str = 'quick',
) -> tuple[bool, str]:
    return trigger_github_workflow_dispatch(
        config=get_github_monitoring_dispatch_config(),
        inputs={
            'days': str(days),
            'threshold': str(threshold),
            'auto_retrain': 'true' if auto_retrain else 'false',
            'profile': profile,
        },
        workflow_label='monitoring',
    )


def launch_local_training_pipeline(reason: str, profile: str = 'quick') -> tuple[bool, str]:
    command = [
        sys.executable,
        str(ROOT_DIR / 'Scripts' / 'run_training.py'),
        '--reason',
        reason,
        '--profile',
        profile,
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    combined_output = '\n'.join(part for part in [completed.stdout, completed.stderr] if part).strip()
    return completed.returncode == 0, combined_output


def launch_training_pipeline(reason: str, profile: str = 'quick', execution_mode: str = 'auto') -> tuple[bool, str]:
    execution_mode = execution_mode or get_automation_execution_mode()
    if execution_mode == 'github':
        return trigger_github_training_workflow(reason=reason, profile=profile)
    if execution_mode == 'local':
        return launch_local_training_pipeline(reason=reason, profile=profile)
    if is_github_training_dispatch_ready():
        return trigger_github_training_workflow(reason=reason, profile=profile)
    return launch_local_training_pipeline(reason=reason, profile=profile)


def launch_monitoring_pipeline(
    days: int = 14,
    threshold: int = 2,
    auto_retrain: bool = True,
    profile: str = 'quick',
    execution_mode: str = 'auto',
) -> tuple[bool, str]:
    execution_mode = execution_mode or get_automation_execution_mode()
    if execution_mode == 'github':
        return trigger_github_monitoring_workflow(
            days=days,
            threshold=threshold,
            auto_retrain=auto_retrain,
            profile=profile,
        )
    if execution_mode == 'auto' and is_github_monitoring_dispatch_ready():
        return trigger_github_monitoring_workflow(
            days=days,
            threshold=threshold,
            auto_retrain=auto_retrain,
            profile=profile,
        )
    return False, 'Remote monitoring dispatch is disabled because the selected execution mode is not GitHub.'


def count_high_severity_alerts(alerts: list[dict]) -> int:
    return sum(1 for alert in alerts if alert.get('severity') == 'high')


def prepare_monitoring_dataframe(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    df = predictions.copy()
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    return df.sort_values('created_at', ascending=False)


# --- UI Helper Functions ---
def get_repo_health_snapshot() -> dict:
    workflow_dir = ROOT_DIR / '.github' / 'workflows'
    workflow_files = {
        'CI': workflow_dir / 'ci.yml',
        'CD': workflow_dir / 'cd.yml',
        'Monitoring': workflow_dir / 'monitoring.yml',
        'Training': workflow_dir / 'training.yml',
    }
    bundle_paths = [
        ROOT_DIR / 'models' / 'churn_production_bundle.pkl',
        ROOT_DIR / 'churn_production.pkl',
    ]
    available_bundles = [path for path in bundle_paths if path.exists()]
    latest_bundle = max(available_bundles, key=lambda path: path.stat().st_mtime) if available_bundles else None

    return {
        'git_repo_present': (ROOT_DIR / '.git').exists(),
        'workflows': {name: path.exists() for name, path in workflow_files.items()},
        'has_alembic': (ROOT_DIR / 'alembic.ini').exists() and (ROOT_DIR / 'alembic').exists(),
        'bundle_count': len(available_bundles),
        'latest_bundle_name': latest_bundle.name if latest_bundle else None,
    }


def get_runtime_configuration_snapshot() -> pd.DataFrame:
    github_dispatch_ready = is_github_training_dispatch_ready()
    github_monitoring_ready = is_github_monitoring_dispatch_ready()
    github_config = get_github_training_dispatch_config()
    monitoring_config = get_github_monitoring_dispatch_config()
    rows = [
        {'Setting': 'MLflow Tracking URI', 'Value': get_mlflow_tracking_uri()},
        {'Setting': 'MLflow Experiment', 'Value': get_mlflow_experiment_name()},
        {'Setting': 'Postgres Host', 'Value': os.getenv('POSTGRES_HOST', 'db')},
        {'Setting': 'Postgres Port', 'Value': os.getenv('POSTGRES_PORT', '5432')},
        {'Setting': 'Automation Execution Mode', 'Value': get_automation_execution_mode()},
        {'Setting': 'GitHub Training Dispatch', 'Value': 'Ready' if github_dispatch_ready else 'Not Configured'},
        {'Setting': 'GitHub Monitoring Dispatch', 'Value': 'Ready' if github_monitoring_ready else 'Not Configured'},
        {'Setting': 'GitHub Repository', 'Value': github_config['repository'] or 'Not set'},
        {'Setting': 'GitHub Training Workflow', 'Value': github_config['workflow'] or 'Not set'},
        {'Setting': 'GitHub Monitoring Workflow', 'Value': monitoring_config['workflow'] or 'Not set'},
        {'Setting': 'GitHub Workflow Ref', 'Value': github_config['ref'] or 'Not set'},
        {'Setting': 'Env Example Present', 'Value': 'Yes' if (ROOT_DIR / '.env.example').exists() else 'No'},
    ]
    return pd.DataFrame(rows)


def render_github_workflow_status_panel():
    st.markdown('### Recent GitHub Workflow Runs')
    if not is_github_api_ready():
        st.info('Configure `GITHUB_ACTIONS_TOKEN` and `GITHUB_REPOSITORY` to show recent CI/CD workflow run status inside the app.')
        return

    workflow_runs = get_recent_github_workflow_runs()
    if not workflow_runs:
        st.info('No GitHub workflow run information is available yet.')
        return

    run_rows = []
    for run in workflow_runs:
        status = run.get('status', 'unknown')
        conclusion = run.get('conclusion') or '-'
        created_at = run.get('created_at') or '-'
        run_rows.append(
            {
                'Workflow': run.get('workflow', '-'),
                'Status': status,
                'Conclusion': conclusion,
                'Created At': created_at,
                'Run Number': run.get('run_number') or '-',
                'Event': run.get('event') or '-',
            }
        )
        if run.get('status') == 'api_error':
            st.warning(f'Unable to read `{run.get("workflow")}` workflow status: {run.get("error", "unknown error")}')

    st.dataframe(pd.DataFrame(run_rows), width='stretch', hide_index=True)


def render_mlops_sidebar_status():
    snapshot = get_repo_health_snapshot()
    workflow_count = sum(snapshot['workflows'].values())
    st.markdown('### MLOps Status')
    st.metric('Automation Workflows', f'{workflow_count}/4')
    st.metric('Model Bundles Found', snapshot['bundle_count'])
    st.write(f"Alembic ready: `{'Yes' if snapshot['has_alembic'] else 'No'}`")
    st.write(f"Git repo detected: `{'Yes' if snapshot['git_repo_present'] else 'No'}`")
    with st.expander('Workflow file check'):
        for name, is_present in snapshot['workflows'].items():
            st.write(f"- `{name}`: {'Present' if is_present else 'Missing'}")


def render_delivery_readiness_panel():
    snapshot = get_repo_health_snapshot()
    runtime_df = get_runtime_configuration_snapshot()

    st.markdown('### CI/CD And Registry Readiness')
    readiness_cols = st.columns(5)
    readiness_cols[0].metric('CI', 'Ready' if snapshot['workflows'].get('CI') else 'Missing')
    readiness_cols[1].metric('CD', 'Ready' if snapshot['workflows'].get('CD') else 'Missing')
    readiness_cols[2].metric('Migrations', 'Ready' if snapshot['has_alembic'] else 'Missing')
    readiness_cols[3].metric('Bundles', snapshot['bundle_count'])
    readiness_cols[4].metric('GitHub Retrain', 'Ready' if is_github_training_dispatch_ready() else 'Local Only')

    if snapshot['latest_bundle_name']:
        st.caption(f"Latest local production bundle: `{snapshot['latest_bundle_name']}`")
    else:
        st.caption('No production bundle found yet. Run the training pipeline to generate one.')

    info_col, table_col = st.columns([0.95, 1.05])
    with info_col:
        st.info(
            'This panel helps you verify whether the repo contains the pieces expected for GitHub CI/CD, '
            'manual retraining, manual monitoring, and release readiness.'
        )
        if not snapshot['git_repo_present']:
            st.warning('No local `.git` directory was detected in this workspace. Push the project from a real Git repository to activate GitHub Actions.')
    with table_col:
        st.dataframe(runtime_df, width='stretch', hide_index=True)
    render_github_workflow_status_panel()


def render_cicd_story_panel():
    snapshot = get_repo_health_snapshot()
    github_ready = is_github_training_dispatch_ready()

    st.markdown('### CI/CD Overview')
    st.caption(
        'This section summarizes how validation, deployment, monitoring, and retraining are connected in the project workflow.'
    )

    summary_cols = st.columns(4)
    summary_cols[0].metric('CI', 'Active' if snapshot['workflows'].get('CI') else 'Missing')
    summary_cols[1].metric('CD', 'Active' if snapshot['workflows'].get('CD') else 'Missing')
    summary_cols[2].metric('Monitoring Job', 'Active' if snapshot['workflows'].get('Monitoring') else 'Missing')
    default_mode = get_automation_execution_mode()
    if default_mode == 'github':
        retraining_path = 'GitHub Only'
    elif github_ready:
        retraining_path = 'GitHub Preferred'
    else:
        retraining_path = 'Local Fallback'
    summary_cols[3].metric('Retraining Path', retraining_path)

    flow_cols = st.columns(4)
    flow_cols[0].info(
        '1. Build Quality\n\n'
        'Every push or pull request goes through CI checks: linting, test execution, coverage, and Docker build validation.'
    )
    flow_cols[1].info(
        '2. Delivery\n\n'
        'After validation, CD packages the application into a deployable container so the delivered version matches a reviewed Git commit.'
    )
    flow_cols[2].info(
        '3. Operations\n\n'
        'This Streamlit app logs production evidence to PostgreSQL, and the monitoring workflow turns that evidence into health alerts.'
    )
    flow_cols[3].info(
        '4. Continuous Improvement\n\n'
        'When monitoring shows enough risk, retraining can be launched on GitHub runners and tracked in MLflow without depending on your local machine.'
    )

    benefit_cols = st.columns(3)
    benefit_cols[0].success(
        'Reproducibility\n\n'
        'Training, testing, and packaging run the same way each time instead of depending on one machine.'
    )
    benefit_cols[1].success(
        'Auditability\n\n'
        'Model changes, monitoring runs, and release steps leave a visible execution trail for reviewers and teams.'
    )
    benefit_cols[2].success(
        'Production Credibility\n\n'
        'The project demonstrates an operational ML system, not only a prediction interface or notebook result.'
    )

    render_delivery_readiness_panel()


def render_automation_quick_guide():
    st.markdown('### Automation Flow')
    guide_cols = st.columns(3)
    guide_cols[0].info(
        '1. CI\n\nPush the repo to GitHub. Every push or pull request will run lint, tests, coverage, and Docker build validation.'
    )
    guide_cols[1].info(
        '2. Training\n\nUse the scheduled Training workflow or the in-app GitHub dispatch controls to retrain on GitHub runners and create updated MLflow runs and bundles.'
    )
    guide_cols[2].info(
        '3. Monitoring\n\nUse the scheduled Monitoring workflow or the in-app GitHub dispatch controls to evaluate production evidence and trigger remote retraining when needed.'
    )


def render_header():
    hero_image_path = ROOT_DIR / 'assets' / 'telco_churn_prediction.png'
    if hero_image_path.exists():
        hero_image_base64 = base64.b64encode(hero_image_path.read_bytes()).decode('utf-8')
        st.markdown(
            f"""
            <div style="width:100%;margin-bottom:1rem;">
                <img
                    src="data:image/png;base64,{hero_image_base64}"
                    alt="TelCo Churn dashboard hero"
                    style="width:100%;max-height:500px;object-fit:cover;object-position:center 46%;display:block;border-radius:18px;"
                />
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <div class="hero-title-wrap">
            <div class="hero-title-icon">&#128202;</div>
            <div class="hero-title-copy">
                <div class="main-header">TelCo Churn Prediction</div>
                <div class="hero-title-subtitle">Managerial Decision Support System</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


def format_feature_name(feature: str) -> str:
    replacements = {
        'tenure': 'Tenure',
        'MonthlyCharges': 'Monthly Charges',
        'Data_Usage_GB': 'Data Usage',
        'TotalServices': 'Total Services',
        'Total_Revenue': 'Total Revenue',
        'Usage_per_Month': 'Usage/Month',
        'gender': 'Gender',
        'SeniorCitizen': 'Senior Citizen',
        'Contract': 'Contract Type',
        'InternetService': 'Internet Service',
        'PaymentMethod': 'Payment Method',
        'PaperlessBilling': 'Paperless Billing',
        'Gouvernorat': 'Governorate',
        'Dependents': 'Dependents',
        'OnlineSecurity': 'Online Security',
        'OnlineBackup': 'Online Backup',
        'DeviceProtection': 'Device Protection',
        'StreamingTV': 'Streaming TV',
        'StreamingMovies': 'Streaming Movies',
        'Support_Tickets': 'Support Tickets',
        'App_Logins': 'App Logins',
    }
    return replacements.get(feature, feature.replace('_', ' ').title())


def format_monitoring_alert(alert: dict) -> str:
    alert_type = alert.get('alert_type', 'unknown')
    if alert_type == 'accuracy_drop':
        return f"Accuracy dropped by {alert.get('value', 0):.1%} (threshold: {alert.get('threshold', 0):.1%})"
    if alert_type == 'f1_drop':
        return f"F1-score dropped by {alert.get('value', 0):.1%} (threshold: {alert.get('threshold', 0):.1%})"
    if alert_type == 'probability_drift':
        return f"Probability drift detected: {alert.get('value', 0):.3f} shift"
    if alert_type == 'high_risk_increase':
        return f"High-risk customers increased by {alert.get('value', 0):.1%}"
    if alert_type == 'low_ground_truth_coverage':
        return f"Ground-truth coverage too low: {alert.get('value', 0):.1%} (minimum: {alert.get('threshold', 0):.1%})"
    return f"{alert_type}: {alert.get('value', 'N/A')}"


def evaluate_governance_candidate(candidate_metrics: dict, baseline_metrics: dict) -> dict:
    accuracy_delta = candidate_metrics['accuracy'] - baseline_metrics['accuracy']
    f1_delta = candidate_metrics['f1'] - baseline_metrics['f1']
    roc_auc_delta = candidate_metrics['roc_auc'] - baseline_metrics['roc_auc']

    if accuracy_delta >= 0 and f1_delta >= 0 and roc_auc_delta >= 0:
        decision = 'APPROVED'
        rationale = 'Candidate outperforms or matches production baseline across all metrics.'
    elif accuracy_delta >= -0.02 and f1_delta >= -0.02 and roc_auc_delta >= -0.02:
        decision = 'CONDITIONAL APPROVAL'
        rationale = 'Candidate within 2% of baseline. Consider deployment if F1 improves.'
    else:
        decision = 'REJECTED'
        rationale = 'Candidate underperforms production baseline. Retrain with different hyperparameters.'

    return {'decision': decision, 'rationale': rationale}


# --- Customer Input Functions ---
def build_customer_inputs(prefix: str = 'customer') -> dict:
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox('Gender', ['Female', 'Male'], key=f'{prefix}_gender')
        senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'], key=f'{prefix}_senior')
        tenure = st.number_input('Tenure (months)', min_value=0, max_value=72, value=12, key=f'{prefix}_tenure')
        monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, max_value=200.0, value=70.0, key=f'{prefix}_charges')
    with col2:
        contract = st.selectbox('Contract', ['month-to-month', 'one year', 'two year'], key=f'{prefix}_contract')
        data_usage = st.number_input('Data Usage (GB)', min_value=0.0, max_value=100.0, value=10.0, key=f'{prefix}_data')
        support_tickets = st.number_input('Support Tickets', min_value=0, max_value=20, value=1, key=f'{prefix}_tickets')
        app_logins = st.number_input('App Logins', min_value=0, max_value=50, value=5, key=f'{prefix}_logins')
    with col3:
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'None'], key=f'{prefix}_internet')
        payment_method = st.selectbox(
            'Payment Method',
            ['Electronic Check', 'Mailed Check', 'Bank Transfer (automatic)', 'Credit Card (automatic)'],
            key=f'{prefix}_payment',
        )
        paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'], key=f'{prefix}_paperless')
        gouvernorat = st.selectbox(
            'Governorate',
            ['Tunis', 'Sfax', 'Sousse', 'Ariana', 'Nabeul', 'Ben Arous', 'Monastir', 'Bizerte', 'Kairouan', 'Gabes'],
            key=f'{prefix}_gov',
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        dependents = st.selectbox('Dependents', ['No', 'Yes'], key=f'{prefix}_dependents')
        online_security = st.selectbox('Online Security', ['No', 'Yes'], key=f'{prefix}_security')
    with col5:
        online_backup = st.selectbox('Online Backup', ['No', 'Yes'], key=f'{prefix}_backup')
        device_protection = st.selectbox('Device Protection', ['No', 'Yes'], key=f'{prefix}_device')
    with col6:
        streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes'], key=f'{prefix}_tv')
        streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes'], key=f'{prefix}_movies')

    return {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'Contract': contract,
        'Data_Usage_GB': data_usage,
        'Support_Tickets': support_tickets,
        'App_Logins': app_logins,
        'PaymentMethod': payment_method,
        'InternetService': internet_service,
        'PaperlessBilling': paperless_billing,
        'Gouvernorat': gouvernorat,
        'Dependents': dependents,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
    }


# --- Next Best Action Logic ---
def get_next_best_action(risk_label: str) -> str:
    config_path = ROOT_DIR / 'next_best_action_config.json'
    default_actions = {
        'HIGH RISK': 'Phone Call - Priority 1',
        'MEDIUM RISK': 'SMS Discount Offer',
        'LOW RISK': 'No Action Needed',
    }
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get(risk_label, default_actions.get(risk_label, 'No Action Needed'))
        except Exception:
            pass
    return default_actions.get(risk_label, 'No Action Needed')


def get_manager_action_options(risk_label: str) -> list[str]:
    options_map = {
        'HIGH RISK': [
            'Phone Call - Priority 1',
            'Immediate Retention Call + 15% Discount Offer',
            'Retention Call + 10% Discount Offer',
            'Escalate to Account Manager',
        ],
        'MEDIUM RISK': [
            'SMS Discount Offer',
            'Nurture Campaign + Check-in Call',
            'Personalized Email Offer',
            'Schedule Follow-up Review',
        ],
        'LOW RISK': [
            'No Action Needed',
            'Monitor Only',
            'Loyalty Thank-You Message',
        ],
    }
    return options_map.get(risk_label, ['No Action Needed'])


def save_manager_action(connection, prediction_id: int, manager_action: str):
    update_fn = getattr(db_utils, 'update_prediction_manager_action', None)
    if update_fn is not None:
        return update_fn(connection, prediction_id, manager_action)

    with connection.cursor() as cursor:
        cursor.execute(
            '''
            ALTER TABLE prediction_logs
            ADD COLUMN IF NOT EXISTS manager_action TEXT,
            ADD COLUMN IF NOT EXISTS manager_action_at TIMESTAMPTZ;
            '''
        )
        cursor.execute(
            '''
            UPDATE prediction_logs
            SET manager_action = %s,
                manager_action_at = NOW()
            WHERE id = %s;
            ''',
            (manager_action, prediction_id),
        )
    connection.commit()


def load_manager_actions() -> pd.DataFrame:
    connection = get_postgres_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                '''
                ALTER TABLE prediction_logs
                ADD COLUMN IF NOT EXISTS manager_action TEXT,
                ADD COLUMN IF NOT EXISTS manager_action_at TIMESTAMPTZ;
                '''
            )
            cursor.execute(
                '''
                SELECT id, manager_action, manager_action_at
                FROM prediction_logs;
                '''
            )
            rows = cursor.fetchall()
        connection.commit()
    finally:
        connection.close()

    return pd.DataFrame(rows, columns=['id', 'manager_action', 'manager_action_at'])


def build_decision_support_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    df = prepare_monitoring_dataframe(predictions).copy()
    if df.empty:
        return df

    manager_actions_df = load_manager_actions()
    if not manager_actions_df.empty:
        df = df.drop(columns=['manager_action', 'manager_action_at'], errors='ignore').merge(
            manager_actions_df,
            on='id',
            how='left',
        )

    df['Next Best Action'] = df['predicted_risk'].map(get_next_best_action)
    df['Manager Action'] = df.get('manager_action', pd.Series(index=df.index, dtype='object')).fillna('Not selected')
    df['Action Status'] = np.where(df['Manager Action'].eq('Not selected'), 'Pending Action', 'Action Taken')
    df['Outcome Status'] = np.where(df['actual_label'].isna(), 'Outcome Pending', 'Outcome Known')
    df['MonthlyCharges'] = pd.to_numeric(df.get('MonthlyCharges'), errors='coerce').fillna(0.0)
    df['Revenue at Risk'] = np.where(df['predicted_risk'].eq('HIGH RISK'), df['MonthlyCharges'], 0.0)
    return df


def get_manager_recommended_action(alerts: list[dict], high_risk_count: int) -> str:
    alert_types = {alert.get('alert_type') for alert in alerts}
    has_high_severity = any(alert.get('severity') == 'high' for alert in alerts)

    if has_high_severity or 'probability_drift' in alert_types or 'live_f1_drop' in alert_types:
        return 'Hold - Model Drift Detected'
    if high_risk_count > 0:
        return 'Launch 10% Discount Campaign'
    return 'Maintain Current Retention Plan'


# --- Demo Data Functions ---
def build_demo_customer_profiles() -> list[dict]:
    return [
        {
            'gender': 'Female', 'SeniorCitizen': 'No', 'tenure': 62, 'MonthlyCharges': 58.0,
            'Contract': 'two year', 'Data_Usage_GB': 8.0, 'Support_Tickets': 0, 'App_Logins': 18,
            'PaymentMethod': 'Bank Transfer (automatic)', 'InternetService': 'DSL', 'PaperlessBilling': 'No',
            'Gouvernorat': 'Tunis', 'Dependents': 'Yes', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes', 'StreamingTV': 'No', 'StreamingMovies': 'No',
        },
        {
            'gender': 'Male', 'SeniorCitizen': 'No', 'tenure': 48, 'MonthlyCharges': 64.0,
            'Contract': 'one year', 'Data_Usage_GB': 11.0, 'Support_Tickets': 1, 'App_Logins': 14,
            'PaymentMethod': 'Credit Card (automatic)', 'InternetService': 'DSL', 'PaperlessBilling': 'No',
            'Gouvernorat': 'Sfax', 'Dependents': 'Yes', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes',
            'DeviceProtection': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
        },
        {
            'gender': 'Female', 'SeniorCitizen': 'No', 'tenure': 72, 'MonthlyCharges': 72.0,
            'Contract': 'two year', 'Data_Usage_GB': 15.0, 'Support_Tickets': 0, 'App_Logins': 20,
            'PaymentMethod': 'Bank Transfer (automatic)', 'InternetService': 'Fiber optic', 'PaperlessBilling': 'Yes',
            'Gouvernorat': 'Sousse', 'Dependents': 'Yes', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes', 'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
        },
        {
            'gender': 'Male', 'SeniorCitizen': 'No', 'tenure': 18, 'MonthlyCharges': 82.0,
            'Contract': 'month-to-month', 'Data_Usage_GB': 18.0, 'Support_Tickets': 2, 'App_Logins': 7,
            'PaymentMethod': 'Electronic Check', 'InternetService': 'Fiber optic', 'PaperlessBilling': 'Yes',
            'Gouvernorat': 'Tunis', 'Dependents': 'No', 'OnlineSecurity': 'No', 'OnlineBackup': 'Yes',
            'DeviceProtection': 'No', 'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
        },
        {
            'gender': 'Female', 'SeniorCitizen': 'Yes', 'tenure': 14, 'MonthlyCharges': 91.0,
            'Contract': 'month-to-month', 'Data_Usage_GB': 14.0, 'Support_Tickets': 3, 'App_Logins': 4,
            'PaymentMethod': 'Electronic Check', 'InternetService': 'Fiber optic', 'PaperlessBilling': 'Yes',
            'Gouvernorat': 'Nabeul', 'Dependents': 'No', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
            'DeviceProtection': 'No', 'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
        },
        {
            'gender': 'Male', 'SeniorCitizen': 'No', 'tenure': 9, 'MonthlyCharges': 86.0,
            'Contract': 'month-to-month', 'Data_Usage_GB': 21.0, 'Support_Tickets': 4, 'App_Logins': 3,
            'PaymentMethod': 'Electronic Check', 'InternetService': 'Fiber optic', 'PaperlessBilling': 'Yes',
            'Gouvernorat': 'Monastir', 'Dependents': 'No', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
            'DeviceProtection': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'Yes',
        },
        {
            'gender': 'Female', 'SeniorCitizen': 'Yes', 'tenure': 4, 'MonthlyCharges': 104.0,
            'Contract': 'month-to-month', 'Data_Usage_GB': 24.0, 'Support_Tickets': 5, 'App_Logins': 1,
            'PaymentMethod': 'Electronic Check', 'InternetService': 'Fiber optic', 'PaperlessBilling': 'Yes',
            'Gouvernorat': 'Tunis', 'Dependents': 'No', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
            'DeviceProtection': 'No', 'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
        },
        {
            'gender': 'Male', 'SeniorCitizen': 'No', 'tenure': 27, 'MonthlyCharges': 77.0,
            'Contract': 'one year', 'Data_Usage_GB': 12.0, 'Support_Tickets': 1, 'App_Logins': 10,
            'PaymentMethod': 'Credit Card (automatic)', 'InternetService': 'DSL', 'PaperlessBilling': 'Yes',
            'Gouvernorat': 'Ariana', 'Dependents': 'No', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'No',
            'DeviceProtection': 'Yes', 'StreamingTV': 'No', 'StreamingMovies': 'No',
        },
        {
            'gender': 'Female', 'SeniorCitizen': 'No', 'tenure': 33, 'MonthlyCharges': 69.0,
            'Contract': 'one year', 'Data_Usage_GB': 10.0, 'Support_Tickets': 1, 'App_Logins': 12,
            'PaymentMethod': 'Mailed Check', 'InternetService': 'DSL', 'PaperlessBilling': 'No',
            'Gouvernorat': 'Ben Arous', 'Dependents': 'Yes', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes',
            'DeviceProtection': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
        },
    ]


def seed_demo_predictions(connection, model, dv, scaler) -> int:
    bundle_metadata = get_bundle_metadata()
    decision_threshold = float(bundle_metadata.get('decision_threshold', DEFAULT_DECISION_THRESHOLD))
    inserted_count = 0
    for customer_data in build_demo_customer_profiles():
        X_scaled, _ = prepare_customer_features(customer_data, dv, scaler)
        probability = float(model.predict_proba(X_scaled)[0, 1])
        risk_label, _ = get_risk_label(probability)
        predicted_label = classify_from_probability(probability, threshold=decision_threshold)
        top_drivers = serialize_top_drivers(explain_top_drivers(X_scaled, model, dv))
        insert_prediction_log(
            connection,
            input_features=customer_data,
            predicted_probability=probability,
            predicted_label=predicted_label,
            predicted_risk=risk_label,
            top_drivers=top_drivers,
            model_name=str(bundle_metadata.get('model_family', 'logistic_regression')),
            model_version=str(bundle_metadata.get('variant_name', 'logistic_regression')),
            model_stage='Production',
            source='demo_seed',
            mlflow_run_id=bundle_metadata.get('mlflow_run_id') or bundle_metadata.get('run_id'),
        )
        inserted_count += 1
    return inserted_count


def seed_demo_ground_truth(connection, monitoring_df: pd.DataFrame) -> int:
    unlabeled = monitoring_df[monitoring_df['actual_label'].isna()].copy().sort_values('created_at')
    if unlabeled.empty:
        return 0

    updated_count = 0
    for index, row in unlabeled.head(24).iterrows():
        risk_label = row['predicted_risk']
        if risk_label == 'HIGH RISK':
            actual_label = 'Churn' if updated_count % 4 != 0 else 'No churn'
        elif risk_label == 'MEDIUM RISK':
            actual_label = 'Churn' if updated_count % 2 == 0 else 'No churn'
        else:
            actual_label = 'No churn' if updated_count % 5 != 0 else 'Churn'

        update_prediction_ground_truth(
            connection,
            prediction_id=int(row['id']),
            actual_label=actual_label,
            ground_truth_source='demo_seed',
            feedback_notes='Demo seeded outcome for balanced monitoring behavior.',
        )
        updated_count += 1

    return updated_count


# --- Styling Functions ---
def style_next_best_action(value: str) -> str:
    if value == 'Phone Call - Priority 1':
        return 'background-color: #fde2e4; color: #9d0208; font-weight: 700;'
    if value == 'SMS Discount Offer':
        return 'background-color: #fff3cd; color: #8d6e00; font-weight: 700;'
    if value == 'No Action Needed':
        return 'background-color: #d8f3dc; color: #1b4332; font-weight: 700;'
    return ''


def get_action_banner_class(action: str) -> str:
    if action in {'Hold - Model Drift Detected', 'Phone Call - Priority 1'}:
        return 'critical'
    if action in {'Launch 10% Discount Campaign', 'SMS Discount Offer'}:
        return 'warning'
    return 'healthy'


# --- Chart Building Functions ---
def build_metric_comparison_chart(selected_metrics: dict, baseline_metrics: dict):
    comparison_df = pd.DataFrame([
        {'Metric': 'Accuracy', 'Model': 'Production', 'Score': baseline_metrics['accuracy']},
        {'Metric': 'Accuracy', 'Model': 'Candidate', 'Score': selected_metrics['accuracy']},
        {'Metric': 'F1-Score', 'Model': 'Production', 'Score': baseline_metrics['f1']},
        {'Metric': 'F1-Score', 'Model': 'Candidate', 'Score': selected_metrics['f1']},
        {'Metric': 'ROC-AUC', 'Model': 'Production', 'Score': baseline_metrics['roc_auc']},
        {'Metric': 'ROC-AUC', 'Model': 'Candidate', 'Score': selected_metrics['roc_auc']},
    ])
    figure = px.bar(
        comparison_df, x='Metric', y='Score', color='Model', barmode='group',
        text='Score', color_discrete_sequence=['#8da9c4', '#0f62a8'],
    )
    figure.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    figure.update_layout(height=420, legend_title_text='', margin={'l': 20, 'r': 20, 't': 30, 'b': 20})
    return figure


def build_precision_recall_figure(selected_result: dict, baseline_result: dict):
    figure = go.Figure()
    for label, result, color in [('Production', baseline_result, '#8da9c4'), ('Candidate', selected_result, '#0f62a8')]:
        pr_df = result['precision_recall']
        figure.add_trace(go.Scatter(
            x=pr_df['recall'], y=pr_df['precision'], mode='lines',
            name=f"{label} (PR AUC={float(pr_df['pr_auc'].iloc[0]):.3f})",
            line={'width': 3, 'color': color},
        ))
    figure.update_layout(height=420, xaxis_title='Recall', yaxis_title='Precision', legend_title_text='')
    return figure


def build_risk_distribution_chart(logs: pd.DataFrame):
    risk_counts = logs['predicted_risk'].value_counts().rename_axis('risk_level').reset_index(name='count')
    figure = px.pie(
        risk_counts, values='count', names='risk_level', hole=0.60,
        color='risk_level',
        color_discrete_map={'HIGH RISK': '#d62839', 'MEDIUM RISK': '#f4a261', 'LOW RISK': '#2a9d8f'},
    )
    figure.update_traces(textposition='inside', textinfo='percent+label')
    figure.update_layout(height=410, legend_title_text='')
    return figure


def build_probability_drift_chart(drift_df: pd.DataFrame):
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=drift_df['created_at'], y=drift_df['avg_probability'],
        mode='lines+markers', name='Avg probability', line={'width': 3, 'color': '#0f62a8'},
    ))
    figure.add_trace(go.Scatter(
        x=drift_df['created_at'], y=drift_df['high_risk_share'],
        mode='lines+markers', name='High-risk share', line={'width': 3, 'color': '#d62839'}, yaxis='y2',
    ))
    figure.update_layout(
        height=410, xaxis_title='Prediction time', yaxis={'title': 'Avg churn probability'},
        yaxis2={'title': 'High-risk share', 'overlaying': 'y', 'side': 'right', 'tickformat': '.0%'},
        legend_title_text='',
    )
    return figure


def build_lifecycle_figure(lifecycle_df: pd.DataFrame):
    color_map = {
        'Completed': '#2a9d8f', 'Active': '#0f62a8', 'Pending': '#adb5bd',
        'approve_candidate': '#2a9d8f', 'review_candidate': '#f4a261', 'reject_candidate': '#d62839',
        'Healthy': '#2a9d8f', 'Monitor Closely': '#f4a261', 'Retraining Recommended': '#d62839',
    }
    statuses = lifecycle_df['status'].tolist()
    colors = [color_map.get(status, '#0f62a8') for status in statuses]

    figure = go.Figure(data=[go.Scatter(
        x=list(range(len(lifecycle_df))), y=[1] * len(lifecycle_df),
        mode='markers+lines+text', marker={'size': 26, 'color': colors},
        line={'color': '#8da9c4', 'width': 4}, text=lifecycle_df['step'],
        textposition='top center',
        customdata=lifecycle_df[['status', 'description']].values,
        hovertemplate='<b>%{text}</b><br>Status: %{customdata[0]}<br>%{customdata[1]}<extra></extra>',
    )])
    figure.update_layout(
        height=230, margin={'l': 20, 'r': 20, 't': 30, 'b': 20},
        yaxis={'visible': False}, xaxis={'visible': False}, showlegend=False,
    )
    return figure


# --- Analytics Functions ---
def prepare_analytics_dataset(df: pd.DataFrame) -> pd.DataFrame:
    analytics_df = df.copy()
    analytics_df['Churn_Binary'] = (analytics_df['Churn'] == 'Yes').astype(int)
    analytics_df['TenureBucket'] = pd.cut(
        analytics_df['tenure'], bins=[0, 6, 12, 24, 48, 72],
        labels=['0-6', '7-12', '13-24', '25-48', '49+'], include_lowest=True,
    )
    return analytics_df


def build_churn_rate_chart(df: pd.DataFrame, group_col: str, title: str, color_sequence: list[str]):
    churn_rate_df = (
        df.groupby(group_col, dropna=False)['Churn_Binary'].mean()
        .reset_index(name='churn_rate').sort_values('churn_rate', ascending=False)
    )
    figure = px.bar(churn_rate_df, x=group_col, y='churn_rate', title=title,
                    color=group_col, color_discrete_sequence=color_sequence)
    figure.update_layout(height=360, showlegend=False, yaxis_tickformat='.0%')
    return figure


def build_distribution_chart(df: pd.DataFrame, x_col: str, title: str):
    figure = px.histogram(
        df, x=x_col, color='Churn', barmode='overlay', opacity=0.72, title=title,
        color_discrete_map={'Yes': '#d62839', 'No': '#2a9d8f'},
    )
    figure.update_layout(height=360)
    return figure


def build_geo_churn_chart(df: pd.DataFrame):
    geo_df = (
        df.groupby('Gouvernorat', dropna=False)['Churn_Binary'].mean()
        .reset_index(name='churn_rate').sort_values('churn_rate', ascending=False)
    )
    figure = px.bar(geo_df, x='Gouvernorat', y='churn_rate', color='churn_rate',
                    title='Churn Rate by Governorate', color_continuous_scale=['#2a9d8f', '#f4a261', '#d62839'])
    figure.update_layout(height=380, xaxis_tickangle=-35, yaxis_tickformat='.0%')
    return figure


def build_correlation_heatmap(df: pd.DataFrame):
    correlation_cols = [
        column for column in ['tenure', 'MonthlyCharges', 'Data_Usage_GB', 'TotalServices', 'Total_Revenue', 'Usage_per_Month', 'Churn_Binary']
        if column in df.columns
    ]
    correlation_df = df[correlation_cols].corr(numeric_only=True)
    figure = px.imshow(correlation_df, title='Feature Correlation Matrix',
                       color_continuous_scale=['#2a9d8f', '#f5f8fc', '#d62839'], aspect='auto')
    figure.update_layout(height=430)
    return figure


# --- Analytics Render Functions ---
def render_demography_analysis(df: pd.DataFrame):
    st.markdown('### 🌍 Demography')
    col1, col2 = st.columns(2)
    with col1:
        if 'gender' in df.columns:
            st.plotly_chart(build_churn_rate_chart(df, 'gender', 'Churn Rate by Gender', ['#0f62a8', '#8da9c4']), width='stretch')
        elif 'Gender' in df.columns:
            st.plotly_chart(build_churn_rate_chart(df, 'Gender', 'Churn Rate by Gender', ['#0f62a8', '#8da9c4']), width='stretch')
    with col2:
        senior_col = 'SeniorCitizen' if 'SeniorCitizen' in df.columns else None
        if senior_col:
            senior_df = df.copy()
            senior_df[senior_col] = senior_df[senior_col].replace({1: 'Yes', 0: 'No'})
            st.plotly_chart(build_churn_rate_chart(senior_df, senior_col, 'Churn Rate by Senior Citizen Status', ['#f4a261', '#2a9d8f']), width='stretch')
    if 'Dependents' in df.columns:
        st.plotly_chart(build_churn_rate_chart(df, 'Dependents', 'Churn Rate by Dependents', ['#457b9d', '#a8dadc']), width='stretch')


def render_services_analysis(df: pd.DataFrame):
    st.markdown('### 💳 Services')
    col1, col2 = st.columns(2)
    with col1:
        if 'InternetService' in df.columns:
            st.plotly_chart(build_churn_rate_chart(df, 'InternetService', 'Churn Rate by Internet Service', ['#f4a261', '#e76f51', '#2a9d8f']), width='stretch')
        service_columns = [col for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection'] if col in df.columns]
        if service_columns:
            melted = df.melt(id_vars=['Churn'], value_vars=service_columns, var_name='service', value_name='enabled')
            service_chart = px.histogram(melted, x='service', color='enabled', barmode='group',
                                         title='Adoption of Core Protection Services', color_discrete_map={'Yes': '#0f62a8', 'No': '#d62839'})
            service_chart.update_layout(height=360)
            st.plotly_chart(service_chart, width='stretch')
    with col2:
        stream_columns = [col for col in ['StreamingTV', 'StreamingMovies'] if col in df.columns]
        if stream_columns:
            stream_df = df.copy()
            stream_df['StreamingBundle'] = stream_df[stream_columns].apply(lambda row: 'Enabled' if 'Yes' in row.values else 'Disabled', axis=1)
            st.plotly_chart(build_churn_rate_chart(stream_df, 'StreamingBundle', 'Churn Rate by Streaming Bundle', ['#1d3557', '#a8dadc']), width='stretch')
        if 'PaperlessBilling' in df.columns:
            st.plotly_chart(build_churn_rate_chart(df, 'PaperlessBilling', 'Churn Rate by Paperless Billing', ['#577590', '#90be6d']), width='stretch')


def render_billing_analysis(df: pd.DataFrame):
    st.markdown('### 📡 Bills And Contracts')
    col1, col2 = st.columns(2)
    with col1:
        if 'Contract' in df.columns:
            st.plotly_chart(build_churn_rate_chart(df, 'Contract', 'Churn Rate by Contract Type', ['#0f62a8', '#4ea8de', '#8da9c4']), width='stretch')
        if 'PaymentMethod' in df.columns:
            payment_chart = build_churn_rate_chart(df, 'PaymentMethod', 'Churn Rate by Payment Method', ['#1d3557', '#457b9d', '#a8dadc', '#e9c46a'])
            payment_chart.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(payment_chart, width='stretch')
    with col2:
        if 'MonthlyCharges' in df.columns:
            st.plotly_chart(build_distribution_chart(df, 'MonthlyCharges', 'Monthly Charges Distribution by Churn'), width='stretch')
        if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            scatter_plot = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn', title='Tenure vs Monthly Charges',
                                      color_discrete_map={'Yes': '#d62839', 'No': '#2a9d8f'}, opacity=0.65)
            scatter_plot.update_layout(height=360)
            st.plotly_chart(scatter_plot, width='stretch')


def render_tenure_analysis(df: pd.DataFrame):
    st.markdown('### ⏳ Tenure And Usage')
    col1, col2 = st.columns(2)
    with col1:
        if 'tenure' in df.columns:
            st.plotly_chart(build_distribution_chart(df, 'tenure', 'Tenure Distribution by Churn'), width='stretch')
        tenure_bucket_df = df.groupby('TenureBucket', dropna=False)['Churn_Binary'].mean().reset_index(name='churn_rate')
        tenure_bucket_chart = px.line(tenure_bucket_df, x='TenureBucket', y='churn_rate', markers=True,
                                       title='Churn Rate by Tenure Bucket', color_discrete_sequence=['#0f62a8'])
        tenure_bucket_chart.update_layout(height=360, yaxis_tickformat='.0%')
        st.plotly_chart(tenure_bucket_chart, width='stretch')
    with col2:
        if 'Data_Usage_GB' in df.columns:
            usage_chart = px.histogram(df, x='Data_Usage_GB', color='Churn', barmode='overlay', opacity=0.72,
                                       title='Data Usage Distribution by Churn', color_discrete_map={'Yes': '#d62839', 'No': '#2a9d8f'})
            usage_chart.update_layout(height=360)
            st.plotly_chart(usage_chart, width='stretch')
        if 'Usage_per_Month' in df.columns:
            usage_tenure_chart = px.scatter(df, x='tenure', y='Usage_per_Month', color='Churn', title='Tenure vs Usage Per Month',
                                            color_discrete_map={'Yes': '#d62839', 'No': '#2a9d8f'}, opacity=0.65)
            usage_tenure_chart.update_layout(height=360)
            st.plotly_chart(usage_tenure_chart, width='stretch')


def render_geography_analysis(df: pd.DataFrame):
    st.markdown('### 🗺️ Geography')
    if 'Gouvernorat' in df.columns:
        st.plotly_chart(build_geo_churn_chart(df), width='stretch')


def render_correlation_analysis(df: pd.DataFrame):
    st.markdown('### 🛠️ Feature Correlations')
    st.plotly_chart(build_correlation_heatmap(df), width='stretch')


# ============================================================
# TAB RENDER FUNCTIONS - Managerial Decision Support System
# ============================================================

def render_single_prediction_tab(model, dv, scaler):
    """Single Prediction Tab - Run production model on customer data"""
    st.subheader('📋 Predictions')
    st.caption('Run the production logistic regression model, explain the result with top drivers, and log the inference event to PostgreSQL.')

    with st.form('single_prediction_form', clear_on_submit=False):
        customer_data = build_customer_inputs('single_prediction')
        submitted = st.form_submit_button('Run Production Prediction', type='primary')

    if not submitted:
        return

    current_model, current_dv, current_scaler = load_model_bundle()
    bundle_metadata = get_bundle_metadata()
    decision_threshold = float(bundle_metadata.get('decision_threshold', DEFAULT_DECISION_THRESHOLD))
    X_scaled, _ = prepare_customer_features(customer_data, current_dv, current_scaler)
    probability = float(current_model.predict_proba(X_scaled)[0, 1])
    risk_label, risk_tag = get_risk_label(probability)
    predicted_label = classify_from_probability(probability, threshold=decision_threshold)
    top_drivers_df = explain_top_drivers(X_scaled, current_model, current_dv)
    top_drivers = serialize_top_drivers(top_drivers_df)

    st.divider()
    metric_cols = st.columns(3)
    metric_cols[0].metric('Churn Probability', f'{probability:.1%}')
    metric_cols[1].metric('Predicted Label', predicted_label)
    metric_cols[2].metric('Risk Tier', risk_label)
    st.caption(f'Decision threshold in use: {decision_threshold:.2f}')

    left_col, right_col = st.columns([0.9, 1.1])
    with left_col:
        st.markdown(f'### {risk_tag}')
        if risk_label == 'HIGH RISK':
            st.error('This customer should be prioritized for retention action.')
        elif risk_label == 'MEDIUM RISK':
            st.warning('This customer deserves closer review and targeted intervention.')
        else:
            st.success('This customer currently appears stable.')
    with right_col:
        st.markdown('### Top Drivers')
        for driver in top_drivers:
            st.write(f"- **{format_feature_name(driver['feature'])}**: {driver['impact']:.3f}")

    connection = get_postgres_connection()
    try:
        bootstrap_platform_tables_if_enabled(connection)
        prediction_id = insert_prediction_log(
            connection, input_features=customer_data, predicted_probability=probability,
            predicted_label=predicted_label, predicted_risk=risk_label, top_drivers=top_drivers,
            model_name=str(bundle_metadata.get('model_family', 'logistic_regression')),
            model_version=str(bundle_metadata.get('variant_name', 'logistic_regression')),
            model_stage='Production', source=FORM_SOURCE,
            mlflow_run_id=bundle_metadata.get('mlflow_run_id') or bundle_metadata.get('run_id'),
        )
        st.success(f'Prediction stored successfully with record ID `{prediction_id}`.')
    except Exception as error:
        st.error(f'Unable to save the prediction to PostgreSQL: {error}')
    finally:
        connection.close()

    with st.expander('Inference payload'):
        st.json(customer_data)


def render_manager_insights_tab(predictions: pd.DataFrame, alerts_df: pd.DataFrame, governance_df: pd.DataFrame):
    """Manager Insights Tab - Executive KPIs and system status"""
    st.subheader('📊 Manager Insights')
    st.caption('Executive overview of churn risk, revenue exposure, and system health status.')

    active_bundle_metadata = get_bundle_metadata()
    active_baseline_metrics = get_baseline_metrics_from_metadata(active_bundle_metadata)
    monitoring_df = build_decision_support_frame(
        filter_predictions_for_active_model(predictions, active_bundle_metadata)
    )
    full_history_monitoring_df = build_decision_support_frame(predictions)

    # Calculate KPIs
    if monitoring_df.empty:
        st.info('No prediction data available yet. Run some predictions first.')
        return

    live_metrics = calculate_live_metrics(monitoring_df)
    computed_alerts = detect_monitoring_alerts(monitoring_df, active_baseline_metrics)
    kpi_df = full_history_monitoring_df.copy()
    high_risk_count = int((kpi_df['predicted_risk'] == 'HIGH RISK').sum())
    medium_risk_count = int((kpi_df['predicted_risk'] == 'MEDIUM RISK').sum())
    low_risk_count = int((kpi_df['predicted_risk'] == 'LOW RISK').sum())

    avg_monthly_bill = float(kpi_df['MonthlyCharges'].mean()) if 'MonthlyCharges' in kpi_df.columns else 0.0
    revenue_at_risk = high_risk_count * avg_monthly_bill
    model_trust_score = calculate_model_trust_score(computed_alerts)
    recommended_action = get_manager_recommended_action(computed_alerts, high_risk_count)

    # Top Governorate
    top_governorate = 'N/A'
    top_gov_churn = 0.0
    if 'Gouvernorat' in kpi_df.columns:
        gov_stats = kpi_df.groupby('Gouvernorat')['predicted_probability'].mean().sort_values(ascending=False)
        if not gov_stats.empty:
            top_governorate = gov_stats.index[0]
            top_gov_churn = gov_stats.iloc[0]

    # System Status
    system_status = 'Healthy'
    status_class = 'healthy'
    status_color = '#2a9d8f'
    if any(alert.get('severity') == 'high' for alert in computed_alerts):
        system_status = 'Critical'
        status_class = 'critical'
        status_color = '#d62839'
    elif any(alert.get('severity') == 'medium' for alert in computed_alerts):
        system_status = 'Warning'
        status_class = 'warning'
        status_color = '#f4a261'

    # --- KPI Cards with Containers ---
    st.markdown('### Key Performance Indicators')

    kpi_row1 = st.columns(4)
    with kpi_row1[0]:
        with st.container(border=True):
            st.markdown(f'<div class="kpi-value">${revenue_at_risk:,.0f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">Revenue at Risk</div>', unsafe_allow_html=True)
    with kpi_row1[1]:
        with st.container(border=True):
            status_html = f'<div class="kpi-value" style="color:{status_color};">{system_status}</div>'
            st.markdown(status_html, unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">System Status</div>', unsafe_allow_html=True)
    with kpi_row1[2]:
        with st.container(border=True):
            st.markdown(f'<div class="kpi-value">{top_governorate}</div>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">Top Governorate (Risk)</div>', unsafe_allow_html=True)
    with kpi_row1[3]:
        with st.container(border=True):
            st.markdown(f'<div class="kpi-value">{model_trust_score}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="kpi-label">Model Trust Score</div>', unsafe_allow_html=True)

    # Risk Distribution Donut Chart
    st.markdown('### Risk Distribution')
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.plotly_chart(
            build_risk_distribution_chart(full_history_monitoring_df),
            width='stretch',
            key='monitoring_manager_risk_distribution',
        )
    with col_chart2:
        # Risk breakdown table
        risk_summary = pd.DataFrame([
            {'Risk Level': 'HIGH RISK', 'Count': high_risk_count, 'Revenue Impact': f'${high_risk_count * avg_monthly_bill:,.0f}'},
            {'Risk Level': 'MEDIUM RISK', 'Count': medium_risk_count, 'Revenue Impact': f'${medium_risk_count * avg_monthly_bill * 0.5:,.0f}'},
            {'Risk Level': 'LOW RISK', 'Count': low_risk_count, 'Revenue Impact': '$0'},
        ])
        st.dataframe(risk_summary, width='stretch', hide_index=True)

    # Recommended Action Banner
    st.markdown('### Recommended Action')
    st.markdown(
        f'<div class="manager-action-banner {status_class}">{recommended_action}</div>',
        unsafe_allow_html=True,
    )

    # Active Alerts Summary
    st.markdown('### Active Alerts')
    if not computed_alerts:
        st.success('✅ No active alerts - system is operating normally.')
    else:
        alert_cols = st.columns(3)
        high_alerts = [a for a in computed_alerts if a.get('severity') == 'high']
        medium_alerts = [a for a in computed_alerts if a.get('severity') == 'medium']
        alert_cols[0].metric('🔴 High Severity', len(high_alerts))
        alert_cols[1].metric('🟡 Medium Severity', len(medium_alerts))
        alert_cols[2].metric('📊 Total Alerts', len(computed_alerts))

        with st.expander('View Alert Details'):
            for alert in computed_alerts:
                severity_icon = '🔴' if alert.get('severity') == 'high' else '🟡'
                st.write(f"{severity_icon} {format_monitoring_alert(alert)}")

    # Live Performance Metrics
    st.markdown('### Live Performance')
    perf_cols = st.columns(4)
    perf_cols[0].metric('Total Inferences', f'{len(monitoring_df)}')
    perf_cols[1].metric('Avg Churn Probability', f"{monitoring_df['predicted_probability'].mean():.1%}")
    perf_cols[2].metric('Ground-Truth Coverage', f"{live_metrics.get('coverage', 0.0):.1%}")
    perf_cols[3].metric('Live F1-Score', f"{live_metrics.get('f1', 0.0):.3f}" if live_metrics.get('sample_size', 0) > 0 else 'N/A')

    st.divider()
    st.markdown('### Decision Operations')
    ops_df = full_history_monitoring_df.copy()
    ops_cols = st.columns(4)
    pending_actions = int((ops_df['Action Status'] == 'Pending Action').sum())
    actioned_cases = int((ops_df['Action Status'] == 'Action Taken').sum())
    known_outcomes = int((ops_df['Outcome Status'] == 'Outcome Known').sum())
    ops_cols[0].metric('Pending Action', pending_actions)
    ops_cols[1].metric('Action Taken', actioned_cases)
    ops_cols[2].metric('Outcome Known', known_outcomes)
    ops_cols[3].metric(
        'Action Coverage',
        f"{(actioned_cases / max(len(ops_df), 1)):.1%}",
    )

    action_status_df = (
        ops_df.groupby(['predicted_risk', 'Action Status'])
        .size()
        .reset_index(name='count')
    )
    status_col, action_col = st.columns(2)
    with status_col:
        status_chart = px.bar(
            action_status_df,
            x='predicted_risk',
            y='count',
            color='Action Status',
            barmode='group',
            title='Pending vs Completed Actions by Risk Tier',
            category_orders={'predicted_risk': ['HIGH RISK', 'MEDIUM RISK', 'LOW RISK']},
            color_discrete_map={'Pending Action': '#d62839', 'Action Taken': '#2a9d8f'},
        )
        status_chart.update_layout(height=360)
        st.plotly_chart(status_chart, width='stretch', key='manager_action_status_chart')
    with action_col:
        action_mix_df = (
            ops_df['Manager Action']
            .value_counts()
            .rename_axis('Manager Action')
            .reset_index(name='count')
        )
        action_mix_chart = px.pie(
            action_mix_df,
            names='Manager Action',
            values='count',
            hole=0.45,
            title='Manager Action Mix',
        )
        action_mix_chart.update_layout(height=360)
        st.plotly_chart(action_mix_chart, width='stretch', key='manager_action_mix_chart')


def render_action_center_tab(predictions: pd.DataFrame, model, dv, scaler):
    """Action Center Tab - Manager's actionable customer list"""
    st.subheader('📞 Action Center')
    st.caption('Actionable customer list with next best actions. Every prediction is translated into a recommended action.')

    monitoring_df = build_decision_support_frame(predictions)

    if monitoring_df.empty:
        st.info('No saved customer predictions are available yet. Run predictions first.')
        return

    saved_df = monitoring_df.copy()
    saved_df["Manager's Action"] = saved_df['Manager Action']

    st.markdown('### Decision Queue')
    queue_cols = st.columns(4)
    queue_cols[0].metric('Pending Action', int((saved_df['Action Status'] == 'Pending Action').sum()))
    queue_cols[1].metric('Action Taken', int((saved_df['Action Status'] == 'Action Taken').sum()))
    queue_cols[2].metric('Outcome Known', int((saved_df['Outcome Status'] == 'Outcome Known').sum()))
    queue_cols[3].metric('Revenue at Risk', f"${saved_df['Revenue at Risk'].sum():,.0f}")

    pending_df = saved_df[saved_df['Action Status'] == 'Pending Action'].copy()
    if not pending_df.empty:
        st.warning(f"{len(pending_df)} customer(s) are still waiting for a manager decision.")
    else:
        st.success('All current customer predictions already have a manager action assigned.')

    st.markdown('### Assign Manager Action')
    selected_prediction_id = st.selectbox(
        'Choose a saved prediction',
        options=saved_df['id'].tolist(),
        format_func=lambda record_id: (
            f"#{record_id} | "
            f"{saved_df.loc[saved_df['id'] == record_id, 'predicted_risk'].iloc[0]} | "
            f"{saved_df.loc[saved_df['id'] == record_id, 'predicted_probability'].iloc[0]:.1%}"
        ),
        key='action_center_prediction_selector',
    )
    selected_row = saved_df.loc[saved_df['id'] == selected_prediction_id].iloc[0]
    manager_options = get_manager_action_options(selected_row['predicted_risk'])
    saved_action = selected_row.get('manager_action')
    default_action = saved_action if saved_action in manager_options else manager_options[0]

    choice_col, context_col = st.columns([0.9, 1.1])
    with choice_col:
        chosen_action = st.selectbox(
            'Select the manager action',
            options=manager_options,
            index=manager_options.index(default_action),
            key=f"manager_action_choice_{selected_prediction_id}",
        )
        if st.button('Save Manager Action', type='primary', key=f"manager_action_save_{selected_prediction_id}"):
            connection = get_postgres_connection()
            try:
                bootstrap_platform_tables_if_enabled(connection)
                save_manager_action(connection, int(selected_prediction_id), chosen_action)
                st.success(f"Manager action saved for prediction #{selected_prediction_id}.")
                st.rerun()
            except Exception as error:
                st.error(f'Unable to save manager action: {error}')
            finally:
                connection.close()
    with context_col:
        st.markdown('**Selected Customer Context**')
        st.write(f"Risk Tier: `{selected_row['predicted_risk']}`")
        st.write(f"Churn Probability: `{selected_row['predicted_probability']:.1%}`")
        st.write(f"Next Best Action: `{selected_row['Next Best Action']}`")
        st.write(f"Saved Manager Action: `{selected_row.get('manager_action') or 'Not selected yet'}`")
        st.write(f"Outcome Status: `{selected_row['Outcome Status']}`")

    # Export button for high-risk customers
    high_risk_list = saved_df[saved_df['predicted_risk'] == 'HIGH RISK'].copy()
    if not high_risk_list.empty:
        export_columns = [
            column for column in [
                'id', 'created_at', 'predicted_probability', 'predicted_risk',
                'Next Best Action', "Manager's Action", 'MonthlyCharges', 'Contract',
                'InternetService', 'PaymentMethod', 'Gouvernorat', 'tenure'
            ]
            if column in high_risk_list.columns
        ]
        st.download_button(
            '📥 Export Priority Call List for Marketing',
            data=high_risk_list[export_columns].to_csv(index=False),
            file_name='priority_call_list.csv',
            mime='text/csv',
            type='primary',
        )
    else:
        st.info('No high-risk customers are currently available for the priority call list.')

    # Display columns
    display_columns = [
        column for column in [
            'id', 'created_at', 'predicted_probability', 'predicted_label', 'predicted_risk',
            'Next Best Action', "Manager's Action", 'MonthlyCharges', 'Contract',
            'InternetService', 'PaymentMethod', 'Gouvernorat', 'tenure'
        ]
        if column in saved_df.columns
    ]

    # Style function for Manager's Action
    def style_managers_action(value: str) -> str:
        if 'Immediate' in value or '15%' in value or 'Priority 1' in value or 'Escalate' in value:
            return 'background-color: #fde2e4; color: #9d0208; font-weight: 700;'
        if 'Retention Call' in value:
            return 'background-color: #fff3cd; color: #8d6e00; font-weight: 700;'
        if 'SMS' in value or 'Nurture' in value or 'Email' in value or 'Follow-up' in value:
            return 'background-color: #e8f4f8; color: #0f62a8; font-weight: 600;'
        if 'No Action' in value or 'Monitor' in value or 'Loyalty' in value:
            return 'background-color: #d8f3dc; color: #1b4332; font-weight: 700;'
        if 'Not selected' in value:
            return 'background-color: #f1f3f5; color: #495057; font-weight: 600;'
        return ''

    styled_df = (
        saved_df[display_columns]
        .sort_values('created_at', ascending=False)
        .style
        .map(style_next_best_action, subset=['Next Best Action'])
        .map(style_managers_action, subset=["Manager's Action"])
    )
    st.dataframe(styled_df, width='stretch')

    # Action Summary
    st.markdown('### Action Summary')
    summary_col, effectiveness_col = st.columns(2)
    with summary_col:
        action_counts = saved_df["Manager's Action"].value_counts().to_dict()
        summary_df = pd.DataFrame(list(action_counts.items()), columns=['Action', 'Count'])
        st.dataframe(summary_df, width='stretch', hide_index=True)
    with effectiveness_col:
        effectiveness_df = saved_df[saved_df['actual_label'].notna() & saved_df['manager_action'].notna()].copy()
        if effectiveness_df.empty:
            st.info('Action effectiveness will appear here once manager actions and confirmed outcomes are both available.')
        else:
            effectiveness_df['retained_customer'] = effectiveness_df['actual_label'].eq('No churn')
            action_effectiveness = (
                effectiveness_df.groupby('manager_action')
                .agg(customers=('id', 'count'), retention_rate=('retained_customer', 'mean'))
                .reset_index()
                .sort_values(['retention_rate', 'customers'], ascending=[False, False])
            )
            action_effectiveness['retention_rate'] = action_effectiveness['retention_rate'].map(lambda value: f'{value:.1%}')
            st.dataframe(action_effectiveness, width='stretch', hide_index=True)


def render_technical_lab_tab():
    """Technical Lab Tab - Model experimentation and governance"""
    st.subheader('🧪 Technical Lab')
    st.caption('Tune hyperparameters offline, compare candidate with production baseline, log experiments to MLflow, and record governance decisions.')
    render_delivery_readiness_panel()
    st.divider()

    control_col, summary_col = st.columns([0.95, 1.05])
    with control_col:
        c_exponent = st.slider('log10(C)', min_value=-3.0, max_value=2.0, value=2.0, step=0.1)
        selected_c = round(10 ** c_exponent, 6)
        selected_solver = st.selectbox('Solver', ['lbfgs', 'liblinear', 'saga'])
        if selected_solver == 'lbfgs':
            selected_penalty = 'l2'
        else:
            penalty_options = ['l1', 'l2']
            if selected_solver == 'saga':
                penalty_options.append('elasticnet')
            selected_penalty = st.selectbox('Penalty', penalty_options)
        selected_l1_ratio = None
        if selected_solver == 'saga' and selected_penalty == 'elasticnet':
            selected_l1_ratio = st.slider('Elastic-Net Mix (l1_ratio)', min_value=0.05, max_value=0.95, value=0.50, step=0.05)
        class_weight_options = {
            'balanced': 'balanced',
            'None': None,
            'Positive x2': {0: 1, 1: 2},
            'Positive x3': {0: 1, 1: 3},
            'Positive x4': {0: 1, 1: 4},
        }
        class_weight_label = st.selectbox('Class Weight', list(class_weight_options.keys()))
        selected_class_weight = class_weight_options[class_weight_label]
        candidate_params = {
            'C': selected_c,
            'solver': selected_solver,
            'penalty': selected_penalty,
            'class_weight': selected_class_weight,
            'l1_ratio': selected_l1_ratio,
        }
    with summary_col:
        st.info(
            'MLflow is the offline experimentation layer. A candidate only reaches production after governance review. '
            'PostgreSQL then becomes the online evidence layer, capturing how that production model behaves on real inferences.'
        )
        st.caption('Selection rule: validation F1 is prioritized, then ROC-AUC and accuracy contribute positively, while log loss slightly penalizes unstable candidates.')

    baseline_result = run_lab_experiment_cached({
        'C': PRODUCTION_BASELINE_PARAMS['C'],
        'solver': PRODUCTION_BASELINE_PARAMS['solver'],
        'penalty': PRODUCTION_BASELINE_PARAMS['penalty'],
        'class_weight': PRODUCTION_BASELINE_PARAMS['class_weight'],
    })
    candidate_result = run_lab_experiment_cached(candidate_params)

    baseline_metrics = baseline_result['metrics']['test']
    candidate_metrics = candidate_result['metrics']['test']
    governance = evaluate_governance_candidate(candidate_metrics, PRODUCTION_BASELINE_METRICS)
    baseline_selection_score = calculate_model_selection_score(baseline_result['metrics']['val'])
    candidate_selection_score = calculate_model_selection_score(candidate_result['metrics']['val'])

    st.divider()
    metric_cols = st.columns(4)
    for column, metric_name, label in zip(metric_cols[:3], ['accuracy', 'f1', 'roc_auc'], ['Accuracy', 'F1-Score', 'ROC-AUC']):
        delta_value = candidate_metrics[metric_name] - baseline_metrics[metric_name]
        column.metric(label, f"{candidate_metrics[metric_name]:.3f}", delta=f'{delta_value:+.3f}')
    metric_cols[3].metric('Selection Score', f'{candidate_selection_score:.3f}', delta=f'{candidate_selection_score - baseline_selection_score:+.3f}')

    chart_col, detail_col = st.columns([1.25, 0.75])
    with chart_col:
        st.plotly_chart(build_metric_comparison_chart(candidate_metrics, baseline_metrics), width='stretch')
    with detail_col:
        st.markdown('### Governance Verdict')
        st.write(f"Decision: `{governance['decision']}`")
        st.write(governance['rationale'])
        st.write(f"Production baseline: `C={PRODUCTION_BASELINE_PARAMS['C']:.0f}`, `solver={PRODUCTION_BASELINE_PARAMS['solver']}`, `penalty={PRODUCTION_BASELINE_PARAMS['penalty']}`")
        st.write(f"Candidate threshold: `{candidate_result['params']['decision_threshold']:.2f}`")
        st.write(f"MLflow tracking URI: `{get_mlflow_tracking_uri()}`")
        st.write(f"MLflow experiment: `{get_mlflow_experiment_name()}`")

    st.plotly_chart(build_precision_recall_figure(candidate_result, baseline_result), width='stretch')

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button('Log Candidate To MLflow', type='primary'):
            try:
                run_id = log_lab_run_to_mlflow(candidate_result)
                st.session_state.latest_candidate_run_id = run_id
                st.success(f'Candidate logged to MLflow with run ID `{run_id}`.')
            except Exception as error:
                st.error(str(error))
    with action_col2:
        if st.button('Record Governance Decision'):
            connection = get_postgres_connection()
            try:
                bootstrap_platform_tables_if_enabled(connection)
                insert_governance_decision(
                    connection, candidate_name=build_lab_run_name(candidate_result['params']),
                    mlflow_run_id=st.session_state.get('latest_candidate_run_id'),
                    candidate_source='technical_lab', decision=governance['decision'],
                    rationale=governance['rationale'], candidate_params=candidate_result['params'],
                    candidate_metrics=candidate_metrics, production_metrics=PRODUCTION_BASELINE_METRICS,
                )
                st.success('Governance decision recorded in PostgreSQL.')
            except Exception as error:
                st.error(f'Unable to record governance decision: {error}')
            finally:
                connection.close()


def render_deep_dive_analytics_tab():
    """Deep-Dive Analytics Tab - Business intelligence with icons"""
    st.subheader('🔬 Deep-Dive Analytics')
    st.caption('Business intelligence views on the churn dataset with comprehensive analysis blocks.')

    if px is None:
        st.error('Plotly is required for the analytics interface.')
        return

    analytics_df = prepare_analytics_dataset(load_dataset())
    analytics_df = analytics_df.fillna({'Gouvernorat': 'Unknown'})

    # KPI Row
    kpi_cols = st.columns(4)
    kpi_cols[0].metric('Dataset Rows', f'{len(analytics_df)}')
    kpi_cols[1].metric('Overall Churn Rate', f"{analytics_df['Churn_Binary'].mean():.1%}")
    kpi_cols[2].metric('Avg Monthly Charges', f"${analytics_df['MonthlyCharges'].mean():.2f}")
    kpi_cols[3].metric('Avg Tenure', f"{analytics_df['tenure'].mean():.1f} mo")

    # Analytics blocks with icons
    analytics_blocks = st.tabs(
        ['🌍 Demography', '💳 Services', '📡 Bills & Contracts', '⏳ Tenure & Usage', '🗺️ Geography', '🛠️ Correlations']
    )

    with analytics_blocks[0]:
        render_demography_analysis(analytics_df)
    with analytics_blocks[1]:
        render_services_analysis(analytics_df)
    with analytics_blocks[2]:
        render_billing_analysis(analytics_df)
    with analytics_blocks[3]:
        render_tenure_analysis(analytics_df)
    with analytics_blocks[4]:
        render_geography_analysis(analytics_df)
    with analytics_blocks[5]:
        render_correlation_analysis(analytics_df)

    with st.expander('Analytics dataset preview'):
        preview_columns = [
            column for column in ['Churn', 'Contract', 'InternetService', 'PaymentMethod', 'Gouvernorat', 'tenure', 'MonthlyCharges']
            if column in analytics_df.columns
        ]
        st.dataframe(analytics_df[preview_columns].head(20), width='stretch')


def render_ground_truth_form(monitoring_df: pd.DataFrame):
    """Ground Truth Form - Confirm real outcomes"""
    unlabeled = monitoring_df[monitoring_df['actual_label'].isna()].copy()
    labeled = monitoring_df[monitoring_df['actual_label'].notna()].copy()

    status_cols = st.columns(3)
    status_cols[0].metric('Pending Ground Truth', int(len(unlabeled)))
    status_cols[1].metric('Confirmed Outcomes', int(len(labeled)))
    status_cols[2].metric(
        'Ground Truth Coverage',
        f"{(len(labeled) / max(len(monitoring_df), 1)):.1%}",
    )

    st.markdown('### Confirm Real Outcomes')
    st.caption('Use this block after some time has passed and you know what really happened to a customer.')
    st.info(
        'Select a saved prediction, choose what actually happened to that customer, and optionally document the retention action or business context.'
    )

    selection_scope = st.radio(
        'Prediction scope',
        ['Pending outcomes only', 'All recent predictions'],
        horizontal=True,
        index=0 if not unlabeled.empty else 1,
        help='Use "All recent predictions" if you want to review or correct a prediction that already has an outcome saved.',
    )

    selection_df = unlabeled.copy() if selection_scope == 'Pending outcomes only' else monitoring_df.copy()
    if selection_df.empty:
        st.success('All recent predictions already have a confirmed outcome.')
        if not labeled.empty:
            preview_columns = [
                column
                for column in ['id', 'created_at', 'predicted_label', 'actual_label', 'ground_truth_source', 'feedback_notes']
                if column in labeled.columns
            ]
            with st.expander('Preview confirmed outcomes'):
                st.dataframe(
                    labeled[preview_columns].sort_values('created_at', ascending=False).head(10),
                    width='stretch',
                )
        return

    selection_df = selection_df.sort_values('created_at', ascending=False).copy()
    selection_df['selection_label'] = selection_df.apply(
        lambda row: f"ID {row['id']} | {row['created_at']} | predicted={row['predicted_label']} | prob={row['predicted_probability']:.1%}",
        axis=1,
    )
    selection_map = dict(zip(selection_df['selection_label'], selection_df['id']))

    with st.form('ground_truth_form', clear_on_submit=False):
        selected_label = st.selectbox('Choose a past prediction', list(selection_map.keys()))
        selected_row = selection_df.loc[selection_df['selection_label'] == selected_label].iloc[0]
        actual_options = ['Churn', 'No churn']
        current_actual_label = selected_row['actual_label'] if pd.notna(selected_row.get('actual_label')) else 'Churn'
        current_actual_label = current_actual_label if current_actual_label in actual_options else 'Churn'
        actual_label = st.selectbox(
            'What really happened?',
            actual_options,
            index=actual_options.index(current_actual_label),
        )
        feedback_notes = st.text_area(
            'Optional note',
            value='' if pd.isna(selected_row.get('feedback_notes')) else str(selected_row.get('feedback_notes')),
            placeholder='Example: customer renewed contract after retention call',
        )
        submitted = st.form_submit_button('Save Real Outcome')

    if submitted:
        connection = get_postgres_connection()
        try:
            bootstrap_platform_tables_if_enabled(connection)
            update_prediction_ground_truth(
                connection, prediction_id=selection_map[selected_label], actual_label=actual_label,
                ground_truth_source='manual_review', feedback_notes=feedback_notes,
            )
            st.success('Ground-truth outcome stored successfully.')
            st.rerun()
        except Exception as error:
            st.error(f'Unable to store ground truth: {error}')
        finally:
            connection.close()


def render_monitoring_dashboard_tab(predictions: pd.DataFrame, alerts_df: pd.DataFrame, governance_df: pd.DataFrame, model, dv, scaler):
    """Monitoring Dashboard Tab - Production evidence and alerts"""
    st.subheader('📈 Monitoring Dashboard')
    render_cicd_story_panel()
    st.divider()
    render_automation_quick_guide()
    st.divider()

    active_bundle_metadata = get_bundle_metadata()
    active_baseline_metrics = get_baseline_metrics_from_metadata(active_bundle_metadata)
    filtered_predictions = filter_predictions_for_active_model(predictions, active_bundle_metadata)
    monitoring_df = build_decision_support_frame(filtered_predictions)
    full_history_monitoring_df = build_decision_support_frame(predictions)
    if monitoring_df.empty:
        st.info('No prediction logs are available yet. Run a few production inferences first.')
        return

    live_metrics = calculate_live_metrics(monitoring_df)
    computed_alerts = detect_monitoring_alerts(monitoring_df, active_baseline_metrics)
    latest_governance = governance_df.iloc[0].to_dict() if not governance_df.empty else None
    lifecycle_df = build_lifecycle_frame(
        alerts=computed_alerts,
        governance_decision=latest_governance['decision'] if latest_governance else None,
        feedback_coverage=live_metrics.get('coverage', 0.0),
    )
    retraining = build_retraining_recommendation(computed_alerts)
    high_risk_count = int((monitoring_df['predicted_risk'] == 'HIGH RISK').sum())
    avg_monthly_bill = float(monitoring_df['MonthlyCharges'].mean()) if 'MonthlyCharges' in monitoring_df.columns else 0.0
    revenue_at_risk = high_risk_count * avg_monthly_bill
    model_trust_score = calculate_model_trust_score(computed_alerts)
    recommended_action = get_manager_recommended_action(computed_alerts, high_risk_count)
    active_variant_name = active_bundle_metadata.get('variant_name', 'Unknown')
    active_run_id = active_bundle_metadata.get('mlflow_run_id') or active_bundle_metadata.get('run_id') or 'Unknown'
    active_model_version = active_bundle_metadata.get('mlflow_model_version', 'Unknown')
    filtered_from_full_history = len(filtered_predictions) != len(predictions)

    # Summary Status
    summary_status = "Healthy"
    summary_color = "#d8f3dc"
    if any(alert.get('severity') == 'high' for alert in computed_alerts):
        summary_status = "Critical: Immediate Attention Required"
        summary_color = "#fde2e4"
    elif any(alert.get('severity') == 'medium' for alert in computed_alerts):
        summary_status = "Warning: Monitor Closely"
        summary_color = "#fff3cd"
    st.markdown(f'<div style="border-radius:14px;padding:0.8rem 1rem;margin-bottom:0.7rem;background:{summary_color};font-weight:600;">Monitoring Summary: {summary_status}</div>', unsafe_allow_html=True)
    if filtered_from_full_history:
        st.info(
            f'Monitoring health is currently computed from predictions produced by the active deployed model '
            f'`{active_variant_name}` (MLflow version `{active_model_version}`, run `{active_run_id}`).'
        )
        st.caption('Historical charts below still use the full prediction history for business visibility.')
    else:
        st.caption(
            'Monitoring is using all available prediction logs because no narrower active-model match '
            'was found in stored prediction history yet.'
        )

    # Manager Command Center
    st.markdown('### Manager Command Center')
    manager_cols = st.columns(3)
    manager_cols[0].metric('Revenue at Risk', f"${revenue_at_risk:,.0f}", help="Sum of monthly revenue for all high-risk customers.")
    manager_cols[1].metric('Model Trust Score', f'{model_trust_score}%', help="Score based on active alerts and model health.")
    manager_cols[2].metric('Recommended Action', recommended_action, help="System-suggested next step based on current risk and alerts.")
    st.markdown(f'<div class="manager-action-banner {get_action_banner_class(recommended_action)}">{recommended_action}</div>', unsafe_allow_html=True)

    # Demo Seed Buttons
    demo_col1, demo_col2 = st.columns(2)
    with demo_col1:
        if st.button('Seed Balanced Demo Predictions'):
            connection = get_postgres_connection()
            try:
                bootstrap_platform_tables_if_enabled(connection)
                inserted_count = seed_demo_predictions(connection, model, dv, scaler)
                st.success(f'Inserted {inserted_count} demo prediction records.')
                st.rerun()
            except Exception as error:
                st.error(f'Unable to seed demo predictions: {error}')
            finally:
                connection.close()
    with demo_col2:
        if st.button('Seed Demo Ground Truth Outcomes'):
            connection = get_postgres_connection()
            try:
                bootstrap_platform_tables_if_enabled(connection)
                updated_count = seed_demo_ground_truth(connection, monitoring_df)
                if updated_count == 0:
                    st.info('No unlabeled prediction records are available for demo outcomes.')
                else:
                    st.success(f'Attached demo outcomes to {updated_count} prediction record(s).')
                    st.rerun()
            except Exception as error:
                st.error(f'Unable to seed demo ground truth: {error}')
            finally:
                connection.close()

    # KPIs
    kpi_cols = st.columns(4)
    kpi_cols[0].metric('Total Inferences', f'{len(monitoring_df)}')
    kpi_cols[1].metric('Avg Churn Probability', f"{monitoring_df['predicted_probability'].mean():.1%}")
    kpi_cols[2].metric('Open Alerts', f"{len(computed_alerts)}")
    kpi_cols[3].metric('Ground-Truth Coverage', f"{live_metrics.get('coverage', 0.0):.1%}")

    decision_cols = st.columns(4)
    decision_cols[0].metric('Pending Action', int((monitoring_df['Action Status'] == 'Pending Action').sum()))
    decision_cols[1].metric('Action Taken', int((monitoring_df['Action Status'] == 'Action Taken').sum()))
    decision_cols[2].metric('Outcome Known', int((monitoring_df['Outcome Status'] == 'Outcome Known').sum()))
    decision_cols[3].metric(
        'Action Completion',
        f"{((monitoring_df['Action Status'] == 'Action Taken').sum() / max(len(monitoring_df), 1)):.1%}",
    )

    st.divider()
    left_col, right_col = st.columns([1.15, 0.85])
    with left_col:
        st.markdown('### Lifecycle View')
        st.plotly_chart(
            build_lifecycle_figure(lifecycle_df),
            width='stretch',
            key='monitoring_lifecycle_view',
        )
    with right_col:
        st.markdown('### Retraining Status')
        if retraining['status'] == 'Retraining Recommended':
            st.error(retraining['status'])
        elif retraining['status'] == 'Monitor Closely':
            st.warning(retraining['status'])
        else:
            st.success(retraining['status'])
        st.write(retraining['reason'])
        retrain_profile = st.selectbox(
            'Training profile',
            ['quick', 'balanced', 'full'],
            index=0,
            help='Quick is fastest for operational retraining. Full is the most exhaustive search.',
            key='monitoring_retrain_profile',
        )
        default_execution_mode = get_automation_execution_mode()
        execution_mode_options = ['auto', 'github', 'local']
        if default_execution_mode == 'github':
            execution_mode_options = ['github']
        elif default_execution_mode == 'local':
            execution_mode_options = ['local', 'auto', 'github']
        retrain_execution_mode = st.selectbox(
            'Retraining execution mode',
            execution_mode_options,
            index=0,
            help='Use GitHub to keep retraining off the local machine. Auto follows the configured default and falls back only when allowed.',
            key='monitoring_retrain_execution_mode',
        )
        if retrain_execution_mode == 'github' and not is_github_training_dispatch_ready():
            st.warning(
                'GitHub Actions retraining is not configured yet. '
                'Set the GitHub dispatch environment variables to enable CI/CD-triggered retraining.'
            )
        monitoring_execution_mode_options = ['github'] if default_execution_mode == 'github' else ['github', 'auto']
        monitoring_execution_mode = st.selectbox(
            'Monitoring execution mode',
            monitoring_execution_mode_options,
            index=0,
            help='GitHub mode runs monitoring on GitHub-hosted runners instead of inside the Streamlit container.',
            key='monitoring_execution_mode',
        )
        if not is_github_monitoring_dispatch_ready():
            st.warning(
                'GitHub Actions monitoring is not configured yet. '
                'Set GITHUB_ACTIONS_TOKEN, GITHUB_REPOSITORY, GITHUB_MONITORING_WORKFLOW, and GITHUB_WORKFLOW_REF.'
            )
        auto_retrain_enabled = st.checkbox(
            'Auto-retrain when monitoring threshold is reached',
            value=True,
            help='In GitHub mode, the remote monitoring workflow will launch remote retraining when the threshold is reached.',
            key='monitoring_auto_retrain_enabled',
        )
        auto_retrain_threshold = st.number_input(
            'Auto-retrain threshold',
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help='Number of high-severity alerts required before retraining starts automatically.',
            key='monitoring_auto_retrain_threshold',
        )
        st.caption(
            f'Current high-severity alerts: {count_high_severity_alerts(computed_alerts)} '
            f'of {int(auto_retrain_threshold)} required.'
        )

    action_col1, action_col2 = st.columns(2)
    if action_col1.button('Run Monitoring Cycle', type='primary'):
        if monitoring_execution_mode in {'github', 'auto'}:
            with st.spinner('Dispatching remote monitoring workflow on GitHub Actions...'):
                success, output = launch_monitoring_pipeline(
                    days=14,
                    threshold=int(auto_retrain_threshold),
                    auto_retrain=auto_retrain_enabled,
                    profile=retrain_profile,
                    execution_mode='github' if monitoring_execution_mode == 'github' else get_automation_execution_mode(),
                )
            if success:
                st.success(
                    'Remote monitoring workflow launched successfully on GitHub Actions. '
                    'It will evaluate production evidence and optionally launch remote retraining based on the threshold.'
                )
                if output:
                    st.info(output)
            else:
                st.error('Remote monitoring workflow could not be launched.')
                if output:
                    st.code(output)
        else:
            connection = get_postgres_connection()
            monitoring_stored = False
            try:
                bootstrap_platform_tables_if_enabled(connection)
                replace_monitoring_alerts(connection, computed_alerts)
                monitoring_stored = True
            except Exception as error:
                st.error(f'Unable to store monitoring alerts: {error}')
            finally:
                connection.close()
            if monitoring_stored:
                high_alert_count = count_high_severity_alerts(computed_alerts)
                should_auto_retrain = auto_retrain_enabled and high_alert_count >= int(auto_retrain_threshold)
                if should_auto_retrain:
                    with st.spinner(f'Monitoring threshold reached. Launching {retrain_profile} retraining pipeline...'):
                        success, output = launch_training_pipeline(
                            reason='streamlit_monitoring_auto_retrain',
                            profile=retrain_profile,
                            execution_mode=retrain_execution_mode,
                        )
                    if success:
                        st.cache_data.clear()
                        st.success(
                            f'Monitoring alerts were stored and retraining was launched successfully via '
                            f'`{retrain_execution_mode}` mode. The app is reloading its state now.'
                        )
                        if output:
                            st.info(output)
                        st.rerun()
                    st.error('Monitoring alerts were stored, but automatic retraining failed.')
                    if output:
                        st.code(output)
                else:
                    st.success('Monitoring alerts refreshed and stored in PostgreSQL.')
                    if auto_retrain_enabled:
                        st.info(
                            f'Automatic retraining did not start because only {high_alert_count} high-severity '
                            f'alert(s) are active and the threshold is {int(auto_retrain_threshold)}.'
                        )
                    st.rerun()
    retrain_label = 'Retrain Model Now' if retraining['status'] == 'Retraining Recommended' else 'Run Retraining Now'
    if action_col2.button(retrain_label):
        with st.spinner(f'Launching {retrain_profile} training pipeline...'):
            success, output = launch_training_pipeline(
                reason='streamlit_manual_retrain',
                profile=retrain_profile,
                execution_mode=retrain_execution_mode,
            )
        if success:
            st.cache_data.clear()
            st.success(
                f'Training pipeline launched successfully with the "{retrain_profile}" profile '
                f'via `{retrain_execution_mode}` mode.'
            )
            if output:
                st.info(output)
            st.rerun()
        else:
            st.error('Training pipeline failed.')
            if output:
                st.code(output)

    st.divider()
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown('### Risk Distribution')
        st.plotly_chart(
            build_risk_distribution_chart(full_history_monitoring_df),
            width='stretch',
            key='monitoring_operational_risk_distribution',
        )
    with chart_col2:
        st.markdown('### Probability Drift')
        drift_df = build_probability_drift_frame(full_history_monitoring_df)
        if drift_df.empty:
            st.info('Not enough monitoring data to compute drift yet.')
        else:
            st.plotly_chart(
                build_probability_drift_chart(drift_df),
                width='stretch',
                key='monitoring_probability_drift',
            )

    st.divider()
    perf_col, gov_col = st.columns([1.0, 1.0])
    with perf_col:
        st.markdown('### Live Performance From Confirmed Outcomes')
        if live_metrics.get('sample_size', 0) == 0:
            st.info('No confirmed outcomes yet. Attach ground truth below to unlock true post-deployment performance measurement.')
        else:
            if live_metrics['sample_size'] < 20:
                st.warning(f"These live metrics are based on only {live_metrics['sample_size']} confirmed outcome(s).")
            metric_cols = st.columns(4)
            metric_cols[0].metric('Labeled Records', f"{live_metrics['sample_size']}")
            metric_cols[1].metric('Live Accuracy', f"{live_metrics['accuracy']:.3f}")
            metric_cols[2].metric('Live F1', f"{live_metrics['f1']:.3f}", delta=f"{live_metrics['f1'] - active_baseline_metrics['f1']:+.3f}")
            roc_auc = live_metrics.get('roc_auc')
            metric_cols[3].metric('Live ROC-AUC', f"{roc_auc:.3f}" if pd.notna(roc_auc) else 'N/A')
    with gov_col:
        st.markdown('### Latest Governance Decision')
        if latest_governance is None:
            st.info('No governance decision has been recorded yet.')
        else:
            st.write(f"Candidate: `{latest_governance['candidate_name']}`")
            st.write(f"Decision: `{latest_governance['decision']}`")
            st.write(latest_governance['rationale'])

    st.divider()
    render_ground_truth_form(monitoring_df)

    st.divider()
    alerts_col, records_col = st.columns([0.95, 1.05])
    with alerts_col:
        st.markdown('### Active Alerts')
        if not computed_alerts:
            st.success('No active alerts at the moment for the currently deployed model.')
        else:
            for alert in computed_alerts:
                st.write(f"- `{alert['severity']}`: {format_monitoring_alert(alert)}")
        st.markdown('### Stored Alert History')
        if alerts_df.empty:
            st.info('No monitoring alerts have been stored in PostgreSQL yet.')
        else:
            alert_history_columns = [
                column for column in ['created_at', 'alert_type', 'severity', 'status', 'message']
                if column in alerts_df.columns
            ]
            st.dataframe(
                alerts_df[alert_history_columns].sort_values('created_at', ascending=False).head(10),
                width='stretch',
                hide_index=True,
            )
    with records_col:
        st.markdown('### Recent Production Records')
        display_columns = ['id', 'created_at', 'predicted_probability', 'predicted_label', 'predicted_risk', 'actual_label', 'model_version', 'Contract', 'InternetService']
        available_columns = [column for column in display_columns if column in monitoring_df.columns]
        st.dataframe(monitoring_df[available_columns].sort_values('created_at', ascending=False), width='stretch')


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    # Sidebar configuration
    with st.sidebar:
        st.markdown('### Next Best Action Logic')
        config_path = ROOT_DIR / 'next_best_action_config.json'
        default_config = {
            'HIGH RISK': 'Phone Call - Priority 1',
            'MEDIUM RISK': 'SMS Discount Offer',
            'LOW RISK': 'No Action Needed'
        }
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except Exception:
                config = default_config.copy()
        else:
            config = default_config.copy()
        st.write('Edit the action for each risk tier:')
        for tier in ['HIGH RISK', 'MEDIUM RISK', 'LOW RISK']:
            config[tier] = st.text_input(f"{tier} Action", value=config.get(tier, default_config[tier]), key=f"nba_{tier}")
        if st.button('Save Action Logic'):
            try:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                st.success('Next Best Action logic saved!')
            except Exception as e:
                st.error(f'Failed to save config: {e}')
        st.divider()
        render_mlops_sidebar_status()

    # Page configuration
    st.set_page_config(page_title='TelCo Churn Prediction', page_icon=PAGE_ICON, layout='wide')
    inject_custom_style()

    # Load data
    with st.spinner('Loading production model and platform state...'):
        model, dv, scaler = load_model_bundle()
        predictions, alerts_df, governance_df = load_platform_data()

    # Render header
    render_header()

    # New tab structure: Managerial Decision Support System
    tabs = st.tabs([
        '📋 Predictions',
        '📊 Manager Insights',
        '📞 Action Center',
        '🧪 Technical Lab',
        '🔬 Deep-Dive Analytics',
        '📈 Monitoring Dashboard',
    ])

    with tabs[0]:
        render_single_prediction_tab(model, dv, scaler)
    with tabs[1]:
        render_manager_insights_tab(predictions, alerts_df, governance_df)
    with tabs[2]:
        render_action_center_tab(predictions, model, dv, scaler)
    with tabs[3]:
        render_technical_lab_tab()
    with tabs[4]:
        render_deep_dive_analytics_tab()
    with tabs[5]:
        render_monitoring_dashboard_tab(predictions, alerts_df, governance_df, model, dv, scaler)


if __name__ == '__main__':
    main()
