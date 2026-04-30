"""create platform tables"""

from alembic import op


revision = '20260430_01'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            model_name TEXT DEFAULT 'logistic_regression',
            model_version TEXT DEFAULT 'C100_lbfgs_l2',
            model_stage TEXT DEFAULT 'Production',
            source TEXT,
            mlflow_run_id TEXT,
            input_features JSONB,
            top_drivers JSONB,
            predicted_probability DOUBLE PRECISION,
            predicted_label TEXT,
            predicted_risk TEXT,
            actual_label TEXT,
            actual_risk TEXT,
            actual_label_at TIMESTAMPTZ,
            ground_truth_source TEXT,
            feedback_notes TEXT,
            manager_action TEXT,
            manager_action_at TIMESTAMPTZ
        );
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS monitoring_alerts (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            current_value DOUBLE PRECISION,
            baseline_value DOUBLE PRECISION,
            threshold_value DOUBLE PRECISION,
            status TEXT NOT NULL DEFAULT 'open',
            resolved_at TIMESTAMPTZ,
            details JSONB
        );
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS governance_decisions (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            candidate_name TEXT NOT NULL,
            mlflow_run_id TEXT,
            candidate_source TEXT NOT NULL,
            decision TEXT NOT NULL,
            rationale TEXT NOT NULL,
            approver TEXT,
            candidate_params JSONB,
            candidate_metrics JSONB,
            production_metrics JSONB
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS governance_decisions;")
    op.execute("DROP TABLE IF EXISTS monitoring_alerts;")
    op.execute("DROP TABLE IF EXISTS prediction_logs;")
