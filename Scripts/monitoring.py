import json
from pathlib import Path

import pandas as pd
from Scripts.db_utils import get_postgres_connection
from Scripts.model_utils import load_data, split_data

BASELINE_LOG_COLUMNS = ['tenure', 'MonthlyCharges', 'Data_Usage_GB', 'Support_Tickets', 'App_Logins']


def load_baseline_data():
    df = load_data()
    df_train, _, _ = split_data(df)
    return df_train.drop(columns=['Churn'], errors='ignore')


def load_recent_logs(limit: int = 500):
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                '''
                SELECT input_features, predicted_probability, predicted_label, predicted_risk, created_at
                FROM prediction_logs
                ORDER BY created_at DESC
                LIMIT %s;
                ''',
                (limit,),
            )
            rows = cursor.fetchall()

    if not rows:
        return pd.DataFrame()

    records = []
    for row in rows:
        input_features = row[0]
        records.append({**input_features, 'predicted_probability': row[1], 'predicted_label': row[2], 'predicted_risk': row[3], 'created_at': row[4]})

    return pd.DataFrame(records)


def detect_numeric_drift(baseline: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = baseline.select_dtypes(include='number').columns.intersection(recent.select_dtypes(include='number').columns)
    drift_rows = []

    for col in numeric_cols:
        baseline_mean = baseline[col].mean()
        recent_mean = recent[col].mean()
        baseline_std = baseline[col].std(ddof=0) or 1.0
        drift_score = abs(recent_mean - baseline_mean) / baseline_std
        drift_rows.append({
            'feature': col,
            'baseline_mean': baseline_mean,
            'recent_mean': recent_mean,
            'drift_score': drift_score,
        })

    return pd.DataFrame(drift_rows).sort_values(by='drift_score', ascending=False)


def format_alerts(drift_df: pd.DataFrame) -> list[str]:
    alerts = []
    for _, row in drift_df.iterrows():
        if row['drift_score'] >= 0.5:
            alerts.append(
                f"Feature '{row['feature']}' drift score {row['drift_score']:.2f}: baseline mean={row['baseline_mean']:.3f}, recent mean={row['recent_mean']:.3f}."
            )
    return alerts


def main():
    print('Loading baseline training data...')
    baseline_df = load_baseline_data()
    print('Connecting to Postgres and reading recent prediction logs...')
    recent_df = load_recent_logs()

    if recent_df.empty:
        print('No recent prediction logs found. Run the Streamlit app and make a prediction first.')
        return

    print(f'Loaded {len(recent_df)} recent prediction records.')
    drift_df = detect_numeric_drift(baseline_df, recent_df)
    alerts = format_alerts(drift_df)

    print('\n=== Prediction Monitoring Summary ===')
    print(f'Recent mean predicted probability: {recent_df["predicted_probability"].mean():.3f}')
    print(f'Recent churn prediction share: {(recent_df["predicted_label"] == "Churn").mean():.3%}')

    if not drift_df.empty:
        print('\nNumeric drift scores:')
        print(drift_df.head(10).to_string(index=False))

    if alerts:
        print('\nAlerts:')
        for alert in alerts:
            print('- ' + alert)
    else:
        print('\nNo drift alerts found for numeric features.')

    print('\nRecent prediction label counts:')
    print(recent_df['predicted_label'].value_counts().to_string())


if __name__ == '__main__':
    main()
