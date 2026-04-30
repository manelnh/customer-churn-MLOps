import os
import sys
import logging
from pathlib import Path

# Suppress MLFlow Git warning before imports
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
logging.getLogger("mlflow.utils.git_utils").setLevel(logging.ERROR)

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from Scripts.model_utils import (
    LEGACY_MODEL_BUNDLE_PATH,
    MODEL_BUNDLE_PATH,
    apply_feature_engineering,
    encode_target,
    evaluate_model,
    drop_unused_columns,
    load_data,
    prepare_features,
    scale_features,
    save_bundle,
    split_data,
)

DATA_PATH = ROOT_DIR / 'telco_churn_cleaned.csv'
BUNDLE_PATH = MODEL_BUNDLE_PATH
LEGACY_BUNDLE_PATH = LEGACY_MODEL_BUNDLE_PATH
DROP_COLUMNS = ['Support_Tickets', 'App_Logins']


def get_latest_versions_by_stage(client, model_name, stage=None):
    """Fetch latest model versions and optionally filter by registry stage."""
    versions = client.get_latest_versions(model_name)
    if stage is None:
        return versions
    return [version for version in versions if getattr(version, 'current_stage', None) == stage]


def generate_logistic_variants():
    variants = []
    for solver in ['liblinear', 'lbfgs']:
        for C in [0.01, 0.1, 1, 10]:
            for class_weight in [None, 'balanced']:
                variant_name = f'lr_{solver}_C{C}_cw{class_weight or "none"}'
                variants.append({
                    'variant_name': variant_name,
                    'solver': solver,
                    'C': C,
                    'class_weight': class_weight,
                    'max_iter': 1000,
                })
    return variants


def calculate_business_metrics(model, X_test, y_test, df_test):
    """Calculate business-specific KPIs beyond standard ML metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Business metrics
    high_risk_predictions = (y_pred_proba >= 0.7).sum()
    medium_risk_predictions = ((y_pred_proba >= 0.5) & (y_pred_proba < 0.7)).sum()
    low_risk_predictions = (y_pred_proba < 0.5).sum()

    # Revenue impact (assuming average customer value)
    avg_customer_value = 1000  # Monthly revenue per customer
    potential_loss_prevented = high_risk_predictions * avg_customer_value * 0.3  # 30% retention success

    # Customer segments by tenure
    tenure_groups = pd.cut(df_test['tenure'], bins=[0, 12, 24, 48, 72], labels=['New', 'Growing', 'Established', 'Loyal'])
    churn_by_segment = {}
    for segment in tenure_groups.unique():
        segment_mask = tenure_groups == segment
        if segment_mask.sum() > 0:
            churn_rate = y_test[segment_mask].mean()
            churn_by_segment[f'{segment}_churn_rate'] = churn_rate

    return {
        'high_risk_predictions': high_risk_predictions,
        'medium_risk_predictions': medium_risk_predictions,
        'low_risk_predictions': low_risk_predictions,
        'potential_loss_prevented': potential_loss_prevented,
        'prediction_coverage': len(y_pred) / len(df_test) * 100,
        **churn_by_segment
    }


def register_best_model(best_model_record):
    """Register the best model in MLflow Model Registry with production tag"""
    try:
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        client = mlflow.MlflowClient()

        # Get the latest version of the model
        versions = get_latest_versions_by_stage(client, 'lr', stage='None')
        if versions:
            latest_version = max(versions, key=lambda v: int(v.version))
            current_version = int(latest_version.version)
        else:
            current_version = 0

        # Create new version
        new_version = current_version + 1

        # Log the best model with additional metadata
        with mlflow.start_run(run_name=f'production_v{new_version}') as run:
            mlflow.set_tag('model_type', 'logistic_regression')
            mlflow.set_tag('stage', 'production')
            mlflow.set_tag('best_model', 'true')
            mlflow.log_param('production_version', new_version)
            mlflow.log_param('selected_by', 'validation_f1')
            mlflow.log_param('validation_f1', best_model_record['val_metrics']['f1'])
            mlflow.log_metrics(best_model_record['business_metrics'])

            # Register the model
            model_uri = f'runs:/{best_model_record["run_id"]}/model'
            mv = mlflow.register_model(model_uri, 'lr')

            # Transition to production if this is the first version or better than current production
            if new_version == 1:
                client.transition_model_version_stage('lr', mv.version, 'Production')
            else:
                # Compare with current production model
                try:
                    prod_versions = get_latest_versions_by_stage(client, 'lr', stage='Production')
                    if prod_versions:
                        prod_version = prod_versions[0]
                        prod_run = mlflow.get_run(prod_version.run_id)
                        prod_f1 = prod_run.data.metrics.get('val_f1', 0)

                        if best_model_record['val_metrics']['f1'] > prod_f1:
                            client.transition_model_version_stage('lr', mv.version, 'Production')
                            print(f'✅ New model version {mv.version} promoted to production (F1: {best_model_record["val_metrics"]["f1"]:.4f} > {prod_f1:.4f})')
                        else:
                            client.transition_model_version_stage('lr', mv.version, 'Staging')
                            print(f'📋 New model version {mv.version} kept in staging (F1: {best_model_record["val_metrics"]["f1"]:.4f} <= {prod_f1:.4f})')
                    else:
                        client.transition_model_version_stage('lr', mv.version, 'Production')
                except Exception as e:
                    print(f'⚠️ Could not compare with production model: {e}')
                    client.transition_model_version_stage('lr', mv.version, 'Staging')

            print(f'✅ Model registered as version {mv.version} of "lr"')

    except Exception as e:
        print(f'⚠️ Could not register model: {e}')


def main():
    df = load_data(DATA_PATH)
    df_train, df_val, df_test = split_data(df)

    y_train = encode_target(df_train['Churn'])
    y_val = encode_target(df_val['Churn'])
    y_test = encode_target(df_test['Churn'])

    df_train = df_train.drop(columns=['Churn'])
    df_val = df_val.drop(columns=['Churn'])
    df_test = df_test.drop(columns=['Churn'])

    df_train_fe = apply_feature_engineering(df_train)
    df_val_fe = apply_feature_engineering(df_val)
    df_test_fe = apply_feature_engineering(df_test)

    df_train_final = drop_unused_columns(df_train_fe, DROP_COLUMNS)
    df_val_final = drop_unused_columns(df_val_fe, DROP_COLUMNS)
    df_test_final = drop_unused_columns(df_test_fe, DROP_COLUMNS)

    X_train, dv = prepare_features(df_train_final, fit_dv=True)
    X_val, _ = prepare_features(df_val_final, dv=dv)
    X_test, _ = prepare_features(df_test_final, dv=dv)

    X_train_scaled, scaler = scale_features(X_train)
    X_val_scaled, _ = scale_features(X_val, scaler=scaler)
    X_test_scaled, _ = scale_features(X_test, scaler=scaler)

    variants = generate_logistic_variants()

    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow_experiment = os.getenv('MLFLOW_EXPERIMENT_NAME', 'churn_prediction')
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    all_results = []
    for variant in variants:
        params = {k: v for k, v in variant.items() if k not in ['variant_name']}
        model = LogisticRegression(random_state=42, **{k: v for k, v in params.items() if v is not None})
        model.fit(X_train_scaled, y_train)

        train_metrics = evaluate_model(model, X_train_scaled, y_train)
        val_metrics = evaluate_model(model, X_val_scaled, y_val)
        test_metrics = evaluate_model(model, X_test_scaled, y_test)

        with mlflow.start_run(run_name=variant['variant_name']) as run:
            mlflow.set_tag('model_type', 'logistic_regression')
            mlflow.log_param('model_name', 'logistic_regression')
            mlflow.log_param('variant_name', variant['variant_name'])
            mlflow.log_params(params)
            mlflow.log_metrics({f'train_{k}': v for k, v in train_metrics.items()})
            mlflow.log_metrics({f'val_{k}': v for k, v in val_metrics.items()})
            mlflow.log_metrics({f'test_{k}': v for k, v in test_metrics.items()})
            mlflow.log_param('experiment_name', mlflow_experiment)
            mlflow.sklearn.log_model(model, 'model', registered_model_name='lr')

            # Log custom business metrics
            business_metrics = calculate_business_metrics(model, X_test_scaled, y_test, df_test_final)
            mlflow.log_metrics(business_metrics)

            all_results.append({
                'name': variant['variant_name'],
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'business_metrics': business_metrics,
                'model': model,
                'run_id': run.info.run_id,
            })
            print(f'Logged MLflow run for {variant["variant_name"]}: {run.info.run_id}')

    best_model_record = max(all_results, key=lambda record: record['val_metrics']['f1'])
    save_bundle(BUNDLE_PATH, best_model_record['model'], dv, scaler)
    save_bundle(LEGACY_BUNDLE_PATH, best_model_record['model'], dv, scaler)

    # Register the best model as production version
    register_best_model(best_model_record)

    print(f'✅ Best model saved to: {BUNDLE_PATH} ({best_model_record["name"]})')


    print(f'Legacy compatibility bundle saved to: {LEGACY_BUNDLE_PATH}')


if __name__ == '__main__':
    main()
