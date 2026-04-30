# Mini Technical Report: TeleLink Churn Prediction System

## Executive Summary

The TeleLink Churn Prediction System represents a production-ready MLOps pipeline deployed for a major Tunisian telecommunications provider. In the highly competitive telecom sector, customer churn represents a critical revenue threat—acquiring a new customer costs 5–25× more than retaining an existing one. This system addresses this business imperative by providing real-time churn predictions with an **F1-score of 0.785**, enabling proactive customer retention interventions.

The F1-score metric is particularly crucial for imbalanced classification problems like churn prediction, where the positive class (churners) typically represents only 20–30% of the customer base. The harmonic mean of precision and recall ensures that the model neither over-predicts churn (causing unnecessary retention costs) nor under-predicts it (missing at-risk customers). An F1-score of 0.785 indicates that TeleLink can correctly identify approximately 78.5% of true churners while maintaining acceptable precision, translating directly into measurable ROI through targeted retention campaigns.

---

## System Architecture: The Predictive Trifecta

The TeleLink system employs a three-tier architecture that separates concerns across inference, storage, and experimentation:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit UI (Port 8501)                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ Single       │ │ Model        │ │ Monitoring               │ │
│  │ Prediction   │ │ Laboratory   │ │ Dashboard                │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              PostgreSQL Database (Port 5432)                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ predictions  │ │ alerts       │ │ governance               │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              MLflow Registry (Port 5000)                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ Experiments  │ │ Models       │ │ Metrics                  │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Streamlit UI** serves as the front-end application, providing interactive interfaces for single predictions, model experimentation, analytics visualization, and monitoring dashboards. **PostgreSQL** maintains the operational data store—storing inference records, alerts, and governance decisions—enabling auditability and historical analysis. **MLflow** provides the experiment tracking registry, managing model versions, parameters, and metrics throughout the development lifecycle.

---

## Block-by-Block Analysis

### 1. Single Prediction & Analytics

The **Single Prediction** module enables real-time churn inference for individual customers. The core model is a **Logistic Regression** with regularization parameter **$C = 100$**:

```python
LogisticRegression(C=100, max_iter=1000, solver='lbfgs', random_state=42)
```

The high value of $C = 100$ (contrasting with the default $C = 1.0$) reduces regularization strength, allowing the model to fit the training data more closely. This is appropriate when the feature set has been carefully engineered and the risk of overfitting is mitigated through feature selection.

**Explainability via Top 3 Drivers:** The system extracts feature coefficients from the trained logistic regression model to identify the **Top 3 Drivers** of churn for each prediction. Since logistic regression coefficients represent log-odds, the absolute values indicate feature importance:

$$\text{Feature Importance}_i = |\beta_i|$$

where $\beta_i$ is the coefficient for feature $i$. Customers receive personalized insights into which factors most influence their churn probability, enabling targeted retention strategies.

The **Analytics** tab provides aggregate visualizations including:
- Demography analysis (customer segments by region, age group)
- Services analysis (subscription patterns)
- Billing analysis (payment behavior)
- Tenure analysis (customer lifetime)
- Correlation analysis (feature relationships)

### 2. Model Laboratory

The **Model Laboratory** serves as the **Hyperparameter Sandbox** for systematic experimentation. This module addresses a critical MLOps challenge: preventing manual errors in hyperparameter tuning through automated experiment tracking.

Key capabilities include:
- **Grid Search over Hyperparameters:** Testing combinations of $C$, solver, and class weights
- **MLflow Integration:** Every experiment run is logged with:
  - Parameters (C, solver, penalty, class_weight)
  - Metrics (accuracy, precision, recall, F1, ROC-AUC)
  - Artifacts (serialized model, feature engineering pipeline)

```python
with mlflow.start_run(run_name=experiment_name):
    mlflow.log_param("C", C)
    mlflow.log_param("solver", solver)
    mlflow.log_metric("f1", f1_score)
    mlflow.sklearn.log_model(model, "model")
```

This systematic approach eliminates "notebook archaeology"—the problematic practice of manually recording results in spreadsheets—and ensures reproducibility. When a candidate model outperforms the current production model, it can be promoted through the governance workflow.

### 3. Monitoring Dashboard

The **Monitoring Dashboard** provides operational observability by tracking model performance in production. Two critical components enable this:

**PostgreSQL as Inference Evidence Store:** Every prediction is persisted to the `predictions` table:

```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    predicted_probability FLOAT,
    predicted_class INTEGER,
    actual_churn BOOLEAN,
    prediction_timestamp TIMESTAMP DEFAULT NOW()
);
```

This creates an **evidence chain**—a complete audit trail from prediction to eventual ground truth (when customer feedback is collected). The `actual_churn` field is populated through the ground truth form, enabling continuous performance monitoring.

**Plotly Visualizations:** The dashboard renders:
- Risk Distribution histogram (probability buckets)
- Alerts timeline (severity over time)
- Governance decisions log

The monitoring system calculates live metrics including:
- **Precision@K**: Of customers flagged as high-risk, what fraction actually churned
- **Recall@K**: Of all actual churners, what fraction were flagged
- **Coverage**: Proportion of predictions with confirmed outcomes

### 4. Managerial Command Center

The header section of the Streamlit app functions as the **Managerial Command Center**, providing business-oriented metrics:

**Revenue at Risk Calculation:**

$$\text{Revenue at Risk} = \sum_{i \in \text{High Risk}} \text{ARPU}_i \times \text{Churn Probability}_i$$

where ARPU is the Average Revenue Per User. This metric translates model outputs into financial terms, enabling executives to justify retention investment budgets.

**Next Best Action (NBA) Logic:** The system implements a configurable action matrix:

| Risk Tier | Action | Rationale |
|-----------|--------|-----------|
| HIGH RISK | Phone Call - Priority 1 | Personal outreach for highest-value customers |
| MEDIUM RISK | SMS Discount Offer | Cost-effective automated retention |
| LOW RISK | No Action Needed | Avoid unnecessary retention costs |

The NBA configuration is persisted to `next_best_action_config.json`, allowing business users to adjust strategies without code changes.

---

## The MLOps Lifecycle: Closed Loop

The TeleLink system implements a **Closed Loop** MLOps lifecycle that automates the feedback cycle from production monitoring to retraining decisions:

```
┌──────────────────────────────────────────────────────────────────┐
│                     PRODUCTION MONITORING                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 1. Collect predictions + ground truth in PostgreSQL       │ │
│  │ 2. Calculate live metrics (precision, recall, F1)         │ │
│  │ 3. Compare against baseline (MLflow or fallback)          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    ALERT GENERATION                        │ │
│  │  • accuracy_drop > 15%  → HIGH severity                   │ │
│  │  • f1_drop > 15%        → HIGH severity                   │ │
│  │  • ground_truth < 10%   → MEDIUM severity                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              RETRAINING DECISION (threshold=3)             │ │
│  │  If ≥ 3 HIGH severity alerts → Trigger retraining         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  MODEL LABORATORY                          │ │
│  │  • Train new candidate models                              │ │
│  │  • Log experiments to MLflow                               │ │
│  │  • Compare against production baseline                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   GOVERNANCE LAYER                         │ │
│  │  Human-in-the-loop: Review candidates, approve promotion   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│                     (Back to Production)                         │
└──────────────────────────────────────────────────────────────────┘
```

This architecture ensures that the model never "runs blind"—performance degradation is detected automatically, and retraining is triggered based on quantitative thresholds rather than subjective judgment.

---

## Conclusion

The TeleLink Churn Prediction System demonstrates **MLOps maturity** by implementing the critical separation between:

1. **Offline Experimentation** (Model Laboratory): Systematic hyperparameter search with full experiment tracking, ensuring reproducibility and eliminating manual errors
2. **Online Observability** (Monitoring Dashboard): Continuous performance tracking with automated alert generation and closed-loop retraining

The system achieves production-grade reliability through:
- **Auditability**: Complete prediction history in PostgreSQL
- **Reproducibility**: MLflow experiment tracking for all model versions
- **Governance**: Human-in-the-loop approval for model promotion
- **Automation**: Closed-loop feedback from monitoring to retraining

With an F1-score of 0.785 and a fully automated MLOps pipeline, TeleLink is positioned to reduce customer churn through data-driven retention interventions while maintaining operational excellence in model management.

---

*Report generated for TeleLink Tunisia — MLOps Technical Documentation*