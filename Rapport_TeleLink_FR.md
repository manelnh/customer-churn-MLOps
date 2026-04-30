# Rapport Technique : Système de Prédiction de Désabonnement TeleLink

## Résumé Exécutif

Le système de prédiction de désabonnement TeleLink représente un pipeline MLOps de production déployé pour un opérateur télécom majeur en Tunisie. Dans le secteur hautement compétitif des télécommunications, le désabonnement des clients représente une menace critique pour les revenus—l'acquisition d'un nouveau client coûte 5 à 25 fois plus que la rétention d'un client existant. Ce système répond à cet impératif commercial en fournissant des prédictions de désabonnement en temps réel avec un **score F1 de 0,785**, permettant des interventions proactives de rétention des clients.

La métrique F1 est particulièrement cruciale pour les problèmes de classification déséquilibrés comme la prédiction de désabonnement, où la classe positive (désabonnes) représente typiquement seulement 20 à 30% de la base clients. La moyenne harmonique de la précision et du rappel assure que le modèle ne sur-prédit pas le désabonnement (causant des coûts de rétention inutiles) ni ne le sous-prédit (manquant des clients à risque). Un score F1 de 0,785 indique que TeleLink peut identifier correctement environ 78,5% des vrais désabonnes tout en maintenant une précision acceptable, se traduisant directement en ROI mesurable grâce à des campagnes de rétention ciblées.

---

## Architecture Système : La Trinité Prédictive

Le système TeleLink emploie une architecture à trois niveaux qui sépare les préoccupations à travers l'inférence, le stockage et l'expérimentation :

```
┌─────────────────────────────────────────────────────────────────┐
│                    Interface Streamlit (Port 8501)              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ Prédiction   │ │ Laboratoire  │ │ Tableau de Bord          │ │
│  │ Simple       │ │ Modèle       │ │ Monitoring               │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Base de données PostgreSQL (Port 5432)             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ prédictions  │ │ alertes      │ │ gouvernance              │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Registre MLflow (Port 5000)                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ Expériences  │ │ Modèles      │ │ Métriques                │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Streamlit UI** sert d'application frontale, fournissant des interfaces interactives pour les prédictions simples, l'expérimentation de modèles, la visualisation analytique et les tableaux de bord de monitoring. **PostgreSQL** maintient le magasin de données opérationnel—stockant les enregistrements d'inférence, les alertes et les décisions de gouvernance—permettant l'auditabilité et l'analyse historique. **MLflow** fournit le registre de suivi des expériences, gérant les versions de modèles, les paramètres et les métriques tout au long du cycle de vie du développement.

---

## Analyse Bloc par Bloc

### 1. Prédiction Simple et Analytique

Le module de **Prédiction Simple** permet l'inférence de désabonnement en temps réel pour les clients individuels. Le modèle central est une **Régression Logistique** avec le paramètre de régularisation **$C = 100$** :

```python
LogisticRegression(C=100, max_iter=1000, solver='lbfgs', random_state=42)
```

La valeur élevée de $C = 100$ (contrastant avec la valeur par défaut $C = 1,0$) réduit la force de régularisation, permettant au modèle de mieux ajuster les données d'entraînement. Ceci est approprié lorsque l'ensemble des fonctionnalités a été soigneusement ingénieré et que le risque de surapprentissage est atténué par la sélection des fonctionnalités.

**Explicabilité via les 3 Principaux Facteurs :** Le système extrait les coefficients des fonctionnalités du modèle de régression logistique entraîné pour identifier les **3 Principaux Facteurs** de désabonnement pour chaque prédiction. Comme les coefficients de régression logistique représentent les cotes logarithmiques, les valeurs absolues indiquent l'importance des fonctionnalités :

$$\text{Importance de la fonctionnalité}_i = |\beta_i|$$

où $\beta_i$ est le coefficient de la fonctionnalité $i$. Les clients reçoivent des informations personnalisées sur les facteurs qui influencent le plus leur probabilité de désabonnement, permettant des stratégies de rétention ciblées.

L'onglet **Analytique** fournit des visualisations agrégées incluant :
- Analyse démographique (segments clients par région, groupe d'âge)
- Analyse des services (modèles d'abonnement)
- Analyse de la facturation (comportement de paiement)
- Analyse de l'ancienneté (durée de vie client)
- Analyse de corrélation (relations entre fonctionnalités)

### 2. Laboratoire Modèle

Le **Laboratoire Modèle** sert de **Bac à Sable d'Hyperparamètres** pour l'expérimentation systématique. Ce module répond à un défi MLOps critique : prévenir les erreurs manuelles dans l'ajustement des hyperparamètres grâce au suivi automatisé des expériences.

Les capacités clés incluent :
- **Recherche sur Grille des Hyperparamètres :** Test des combinaisons de $C$, solveur et poids des classes
- **Intégration MLflow :** Chaque exécution d'expérience est enregistrée avec :
  - Paramètres (C, solver, penalty, class_weight)
  - Métriques (accuracy, precision, recall, F1, ROC-AUC)
  - Artefacts (modèle sérialisé, pipeline de feature engineering)

```python
with mlflow.start_run(run_name=experiment_name):
    mlflow.log_param("C", C)
    mlflow.log_param("solver", solver)
    mlflow.log_metric("f1", f1_score)
    mlflow.sklearn.log_model(model, "model")
```

Cette approche systématique élimine l'« archéologie de notebook »—la pratique problématique d'enregistrer manuellement les résultats dans des tableurs—et assure la reproductibilité. Lorsqu'un modèle candidat surpasse le modèle de production actuel, il peut être promu à travers le flux de gouvernance.

### 3. Tableau de Bord de Monitoring

Le **Tableau de Bord de Monitoring** fournit l'observabilité opérationnelle en suivant les performances du modèle en production. Deux composants critiques permettent cela :

**PostgreSQL comme Magasin de Preuves d'Inférence :** Chaque prédiction est persistée dans la table `predictions` :

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

Ceci crée une **chaîne de preuves**—une piste d'audit complète de la prédiction jusqu'à la vérité terrain finale (lorsque le feedback client est collecté). Le champ `actual_churn` est populé à travers le formulaire de vérité terrain, permettant le suivi continu des performances.

**Visualisations Plotly :** Le tableau de bord rend :
- Histogramme de distribution des risques (seuils de probabilité)
- Chronologie des alertes (sévérité dans le temps)
- Journal des décisions de gouvernance

Le système de monitoring calcule les métriques en direct incluant :
- **Précision@K** : Parmi les clients identifiés comme à haut risque, quelle fraction a réellement désabonné
- **Rappel@K** : Parmi tous les désabonnes réels, quelle fraction a été identifiée
- **Couverture** : Proportion de prédictions avec des résultats confirmés

### 4. Centre de Commande Manageriel

La section d'en-tête de l'application Streamlit fonctionne comme le **Centre de Commande Manageriel**, fournissant des métriques orientées business :

**Calcul du Revenu à Risque :**

$$\text{Revenue à Risque} = \sum_{i \in \text{Haut Risque}} \text{ARPU}_i \times \text{Probabilité de Désabonnement}_i$$

où ARPU est le Revenu Moyen Par Utilisateur. Cette métrique traduit les sorties du modèle en termes financiers, permettant aux dirigeants de justifier les budgets d'investissement de rétention.

**Logique de Meilleure Prochaine Action (NBA) :** Le système implémente une matrice d'actions configurable :

| Niveau de Risque | Action | Justification |
|------------------|--------|---------------|
| HAUT RISQUE | Appel Téléphonique - Priorité 1 | Approche personnelle pour les clients à haute valeur |
| RISQUE MOYEN | Offre SMS avec Réduction | Rétention automatisée rentable |
| FAIBLE RISQUE | Aucune Action Nécessaire | Éviter les coûts de rétention inutiles |

La configuration NBA est persistée dans `next_best_action_config.json`, permettant aux utilisateurs métier d'ajuster les stratégies sans modification de code.

---

## Le Cycle de Vie MLOps : Boucle Fermée

Le système TeleLink implémente un cycle de vie MLOps en **Boucle Fermée** qui automatise le cycle de feedback du monitoring de production vers les décisions de réentraînement :

```
┌──────────────────────────────────────────────────────────────────┐
│                     MONITORING DE PRODUCTION                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 1. Collecter prédictions + vérité terrain dans PostgreSQL │ │
│  │ 2. Calculer les métriques en direct (précision, rappel, F1)│ │
│  │ 3. Comparer avec la ligne de base (MLflow ou repli)        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    GÉNÉRATION D'ALERTES                    │ │
│  │  • accuracy_drop > 15%  → Sévérité HAUTE                   │ │
│  │  • f1_drop > 15%        → Sévérité HAUTE                   │ │
│  │  • ground_truth < 10%   → Sévérité MOYENNE                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              DÉCISION DE RÉENTRAÎNEMENT (seuil=3)          │ │
│  │  Si ≥ 3 alertes de sévérité HAUTE → Déclencher réentraînement│
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   LABORATOIRE MODÈLE                       │ │
│  │  • Entraîner de nouveaux modèles candidats                 │ │
│  │  • Enregistrer les expériences dans MLflow                 │ │
│  │  • Comparer avec la ligne de base de production            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    COUCHE DE GOUVERNANCE                   │ │
│  │  Humain dans la boucle : Examiner les candidats,           │ │
│  │  approuver la promotion                                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│                     (Retour en Production)                       │
└──────────────────────────────────────────────────────────────────┘
```

Cette architecture assure que le modèle ne fonctionne jamais « à l'aveugle »—la dégradation des performances est détectée automatiquement, et le réentraînement est déclenché basé sur des seuils quantitatifs plutôt que sur un jugement subjectif.

---

## Conclusion

Le Système de Prédiction de Désabonnement TeleLink démontre une **maturité MLOps** en implémentant la séparation critique entre :

1. **Expérimentation Hors Ligne** (Laboratoire Modèle) : Recherche systématique d'hyperparamètres avec suivi complet des expériences, assurant la reproductibilité et éliminant les erreurs manuelles
2. **Observabilité En Ligne** (Tableau de Bord de Monitoring) : Suivi continu des performances avec génération automatique d'alertes et réentraînement en boucle fermée

Le système atteint une fiabilité de niveau production à travers :
- **Auditabilité** : Historique complet des prédictions dans PostgreSQL
- **Reproductibilité** : Suivi des expériences MLflow pour toutes les versions de modèles
- **Gouvernance** : Approbation humain-dans-la-boucle pour la promotion des modèles
- **Automatisation** : Boucle fermée de feedback du monitoring vers le réentraînement

Avec un score F1 de 0,785 et un pipeline MLOps entièrement automatisé, TeleLink est positionné pour réduire le désabonnement des clients grâce à des interventions de rétention basées sur les données tout en maintenant l'excellence opérationnelle dans la gestion des modèles.

---

*Rapport généré pour TeleLink Tunisie — Documentation Technique MLOps*