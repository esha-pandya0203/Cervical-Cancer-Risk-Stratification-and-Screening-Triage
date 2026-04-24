# CxRisk — Cervical Cancer Risk Stratification and Screening Triage DSS

CMU 94-706 Healthcare Information Systems · Spring 2026  
Team: Abigail Torbatian · Esha Pandya · Devavrath Sandeep

---

## Table of Contents

1. [What This Is](#1-what-this-is)
2. [Clinical Context & Motivation](#2-clinical-context--motivation)
3. [Data](#3-data)
4. [Model](#4-model)
5. [DSS Architecture](#5-dss-architecture)
6. [Application Features](#6-application-features)
7. [Local Setup](#7-local-setup)
8. [Running the App](#8-running-the-app)
9. [Limitations](#9-limitations)
10. [Deployment Recommendations](#10-deployment-recommendations)

---

## 1. What This Is

CxRisk is a clinician-facing Decision Support System (DSS) for cervical cancer screening triage. Given a patient's risk factor profile, it produces:

- A predicted probability of an abnormal screening or diagnostic outcome
- A risk level classification: **Low**, **Medium**, or **High**
- A recommended next action: routine recall, expedited HPV/Pap testing, or referral
- Top contributing risk drivers with a natural language explanation
- A confidence score indicating how much to trust the prediction
- Clinical warnings when the model is uncertain or data is borderline
- Clinician override support — every decision is auditable

The system is **advisory only**. Clinicians can accept or override any recommendation, and all overrides are logged with a required reason.

---

## 2. Clinical Context & Motivation

Cervical cancer is largely preventable through HPV vaccination and screening, but screening and follow-up remain uneven across populations. Clinics face a structural problem: they must triage patients with incomplete histories, finite HPV/Pap testing capacity, and competing referral demand — and current tools (like ASCCP guidelines) require a prior lab result to run, leaving out patients who may be at risk based on behavioral factors alone.

**What this DSS adds:**

- Aggregates structured risk factors into a consistent patient-level score
- Surfaces top drivers and missing-data flags rather than a black-box recommendation
- Standardizes triage logic so follow-up decisions are explicit and auditable
- Supports population-level monitoring: high-risk cohorts, referral volume, override rates

**Who it is for:**

| Role | What they use it for |
|---|---|
| Physicians / OB-GYNs / PCPs | Risk estimate at point of care, override capability, documentation |
| Nurses | Follow-up status, scheduling, outreach visibility |
| Administrators / Managers | Population dashboard, referral volume, equity monitoring |

---

## 3. Data

**Source:** Cervical Cancer Risk Factors dataset, UCI Machine Learning Repository (Fernandes et al., 2017)

- 858 patient records × 32 features
- 4 binary outcome labels: Hinselmann, Schiller, Citology, Biopsy
- Behavioral predictors: smoking, pregnancies, number of sexual partners, hormonal contraceptive use, IUD use, STI history, prior diagnosis flags

**Preprocessing pipeline:**

1. Normalize column names to snake_case; remove exact duplicates
2. Add derived fields: `abnormal_any` (positive if any of the 4 outcomes is 1) and `missing_count_predictors`
3. Preserve NULLs in the cleaned dataset for database staging
4. For the analysis-ready file: create missingness indicators, mode-impute binary variables, median-impute continuous variables
5. Drop `stds_time_since_first_diagnosis` and `stds_time_since_last_diagnosis` (>90% missing) from the imputed modeling file only

**Output files:**

| File | Rows × Cols | Purpose |
|---|---|---|
| `cervical-cancer_csv (1).csv` | 858 × 32 | Raw source |
| `cervical_cancer_clean_nullable.csv` | 835 × 38 | Cleaned, NULLs preserved |
| `cervical_cancer_analysis_ready_imputed.csv` | 835 × 67 | Imputed, ready for modeling |

**Key dataset facts:**

- Class imbalanced: 12.1% any abnormal outcome
- Strong missingness in STI timing fields — motivates explicit missing-data handling
- Train/holdout split: 835 training patients, 172 holdout patients, 6.4% positive rate in holdout

---

## 4. Model

### Architecture

Four base learners combined in a weighted ensemble, trained with 5-fold stratified cross-validation, SMOTE for class imbalance, and isotonic calibration to convert raw scores to true probabilities.

| Model | Role | OOF AUC | Ensemble Weight |
|---|---|---|---|
| Logistic Regression (LR) | Linear baseline, interpretable coefficients | 0.543 | 0.15 |
| Random Forest (RF) | Best performer, handles non-linear interactions | 0.593 | 0.75 |
| XGBoost (XGB) | Gradient boosted trees, complex interactions | 0.523 | 0.01 |
| LightGBM (LGB) | Fast gradient boosting, low memory | 0.480 | 0.09 |
| **Weighted Ensemble** | **Final prediction** | **0.684** | — |

**Features used (9 total):**

Age, Number of sexual partners, Number of pregnancies, Smokes (years), Hormonal Contraceptives (years), IUD (years), STD burden score, Risk cluster (K-means inferred), Pregnancies × Age interaction

### Performance

| Metric | Value |
|---|---|
| Holdout AUC | **0.768** |
| Holdout PR-AUC | 0.177 |
| OOF Ensemble AUC | 0.684 |
| Operating threshold | 0.096 |
| Recall at threshold | 73% (8 of 11 holdout positives flagged) |
| Specificity | 89% |
| Flag rate | ≤20% of patients |

**Threshold selection:** The threshold of 0.096 was chosen to keep the flag rate at or below 20%, optimizing the F2 score which weights recall 2× over precision — appropriate because missing a high-risk patient is costlier than an unnecessary referral.

### Feature Importance (SHAP)

Top drivers are consistent across all four models, which builds clinician trust:

| Feature | LR Rank | RF Rank | XGB Rank | LGB Rank |
|---|---|---|---|---|
| Num Pregnancies | #1 | #1 | #4 | #2 |
| Sexual Partners | #7 | #2 | #1 | #1 |
| Hormonal Contraceptives | #4 | #3 | #2 | #3 |
| Age | #3 | #5 | #3 | #4 |
| Preg × Age interaction | #2 | #4 | #5 | #5 |
| STD burden | #5 | #7 | #6 | #6 |
| Smoking | #9 | #9 | #9 | #9 |

### Confidence Score

Each prediction includes a composite confidence score (0–1) built from:

- Ensemble probability (distance from threshold)
- Model agreement fraction (how many of 4 models agree)
- Demographic profile match (cosine similarity to known-risk cluster centroids)
- Certainty bonus (1 – entropy across model probabilities)
- Confusion risk penalty (pull toward the opposite class)

At confidence ≥0.70, wrong predictions drop to ~31% of output. The separation gap of 0.581 between correct and incorrect predictions means the model reliably signals its own uncertainty.

### Fairness & Subgroup Performance

| Subgroup | AUC | Note |
|---|---|---|
| Age < 20 | 0.35 | Below chance — teen patients need a separate model or feature set |
| Age 20–29 | 0.67 | Solid performance in the largest screening cohort |
| Age 40–49 | 0.77 | Best subgroup — richer clinical histories |
| Risk Cluster 2 | 0.40 | Behaviorally distinct cluster — may need cluster-specific calibration |

Equity monitoring is a live DSS feature: override rates and AUC are tracked by subgroup in the population dashboard.

---

## 5. DSS Architecture

```
┌─────────────────────────────────────────────────────┐
│                   R Shiny Frontend                   │
│  Patient Search · Risk Assessment · Model Insights   │
│  Population Dashboard · Clinician Override           │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP (httr)
                       ▼
┌─────────────────────────────────────────────────────┐
│              Python FastAPI Backend (api.py)         │
│  POST /predict  ·  GET /health  ·  GET /schema       │
│  Ensemble inference · Confidence scoring             │
│  NL explanation · Chart generation (base64 PNG)      │
└──────────────────────┬──────────────────────────────┘
                       │ joblib
                       ▼
┌─────────────────────────────────────────────────────┐
│         cervical_model_bundle_v2.joblib              │
│  LR + RF + XGB + LGB + ensemble weights              │
│  PCA · K-means · scalers · calibrators               │
└─────────────────────────────────────────────────────┘
                       
┌─────────────────────────────────────────────────────┐
│           PostgreSQL (Supabase)                      │
│  patient · encounter · risk_assessment               │
│  dss_recommendation · app_user                       │
└─────────────────────────────────────────────────────┘
```

**Fallback behavior:** If the Python API is unreachable, the Shiny app falls back to a local logistic regression model fit in R. The app continues to function — it just uses simpler scoring without confidence scores, NL explanations, or population charts.

---

## 6. Application Features

### Patient Search
Search by name or patient ID. Select a patient to preview demographics and current risk level.

### Patient Risk Assessment
Enter or confirm clinical inputs. Run the model. View:
- Risk level pill (Low / Medium / High) with confidence badge
- Predicted probability
- Clinical warnings (borderline threshold, low confidence, missing data flags)
- Top risk drivers as tags
- Recommended next action
- Clinician override panel with required rationale field
- Downloadable HTML patient summary

### Model Insights *(requires API)*
- **Natural language explanation** — plain English paragraph explaining the prediction
- **Clinical alerts** — warnings surfaced by the API
- **Per-model probabilities** — bar chart showing each of the 4 models' individual predictions; disagreement is immediately visible
- **Population Risk Map** — PCA scatter showing where this patient sits relative to all 858 patients
- **Confidence Waterfall** — what is driving model confidence or uncertainty
- **Model Contribution Waterfall** — how much each model contributed to the ensemble

### Population Dashboard
- Summary metrics: total patients, abnormal outcome rate, high-risk count, override count
- Risk level distribution chart
- Common risk patterns in high-risk group
- Age vs. pregnancies scatter by risk level
- Full override activity log

---

## 7. Local Setup

### Requirements

- macOS, Linux, or Windows
- R 4.1 or higher
- Python 3.9 or higher

### Install R and Python

**macOS:**
```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install r
# Python is usually pre-installed; check with: python3 --version
```

**Ubuntu/Debian:**
```bash
sudo apt-get install r-base python3 python3-pip
```

**Windows:** Download from https://cran.r-project.org and https://python.org

### Project files needed

```
dss/
├── app.R
├── api.py
├── .Renviron                                  ← get from a team member
├── cervical-cancer_csv (1).csv
├── cervical_cancer_clean_nullable.csv
├── cervical_cancer_analysis_ready_imputed.csv
└── cervical_model_bundle_v2.joblib            ← generate with 02_model_v2.ipynb
```

### Configure environment variables

Create `.Renviron` in the project folder (get values from a team member):

```bash
cat > .Renviron << 'EOF'
SUPABASE_HOST="your-supabase-host"
SUPABASE_PASSWORD="your-supabase-password"
CERVICAL_API_URL="http://localhost:8001/predict"
EOF
```

### Install R packages (once)

```bash
R --vanilla -e "install.packages(c('shiny','DT','dplyr','ggplot2','scales','shinymanager','DBI','RPostgres','pool','httr','jsonlite'), repos='https://cloud.r-project.org')"
```

### Install Python packages (once)

```bash
pip install fastapi uvicorn joblib matplotlib numpy pandas scipy \
            scikit-learn pydantic imbalanced-learn xgboost lightgbm
```

### Add yourself to the database

```bash
psql "postgresql://YOUR_CONNECTION_STRING"
```

```sql
INSERT INTO app_user (user_id, first_name, last_name, email, password_hash, role, is_active)
VALUES (gen_random_uuid(), 'Your', 'Name', 'you@email.com', 'yourpassword', 'physician', TRUE);
\q
```

Passwords are plain text in this prototype.

---

## 8. Running the App

Open two terminals in the project folder. **Start Terminal 1 first.**

### Terminal 1 — Python API

```bash
cd path/to/dss
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

Expected output:
```
Uvicorn running on http://0.0.0.0:8001
Bundle loaded. Models: ['LR', 'RF', 'XGB', 'LGB'] | Threshold: 0.096 | Holdout AUC: 0.7685
Application startup complete.
```

Test it: `curl http://localhost:8001/health`

### Terminal 2 — Shiny App

```bash
cd path/to/dss
R --vanilla -e "readRenviron('.Renviron'); shiny::runApp('app.R', port=3838, launch.browser=FALSE)"
```

Open **http://localhost:3838** and log in.

### Stopping

| What | How |
|---|---|
| API | `Ctrl+C` |
| Shiny | `Ctrl+C`, or if stuck: `Ctrl+Z` then `kill %1` |
| R session | `q()` → `n` |
| psql | `\q` |

### Troubleshooting

| Symptom | Fix |
|---|---|
| `socket ... No such file or directory` | `.Renviron` not loaded — use `R --vanilla -e "readRenviron('.Renviron'); ..."` |
| `ModuleNotFoundError` on API start | Run the pip install command above |
| `FileNotFoundError: cervical_model_bundle_v2.joblib` | Run `02_model_v2.ipynb` to generate the model bundle |
| Model Insights shows "local model used" | API not running — check Terminal 1 |
| Save patient fails | Your email not in `app_user` — add via psql |
| API predictions return 500 | Check `grep -n "cosine" api.py` — line 35 should read `from sklearn.metrics.pairwise import cosine_similarity` |

---

## 9. Limitations

| Limitation | Detail |
|---|---|
| Small holdout positives | Only 11 positive cases in holdout — AUC and sensitivity carry wide confidence intervals |
| Teen cohort gap | AUC 0.35 for patients under 20 (below chance) — current features don't capture relevant risk for this group |
| No temporal features | STI timing and screening history were >90% missing and dropped — longitudinal data would improve the model |
| Calibration at extremes | After isotonic calibration, predicted probabilities above 0.6 are sparse — the model correctly signals low confidence at extremes |
| Plain text passwords | Prototype only — production deployment requires hashed passwords and 2FA |

---

## 10. Deployment Recommendations

### Short-term (1–2 years)
- End-to-end encryption for all patient data transmissions
- Two-factor authentication for all DSS users
- Implement strict role permissions: clinicians limited to patient dashboard; administrators to population dashboard; dedicated IT admin role for user management
- Automated lab result integration: parse HPV genotyping results, cytology reports, and biopsy findings directly into risk factor inputs

### Long-term
- Patient-facing interface with simplified risk status, screening history, and personalised HPV education
- Retrain on diverse datasets to improve equity across racial, ethnic, and demographic groups — particularly the under-20 cohort
- Epic BPA integration to auto-populate features from structured chart fields
- Monthly AUC drift monitoring, override rate tracking, and equity metrics; quarterly retraining with new outcome data

### Policy Considerations
- HIPAA compliance: cervical screening records carry heightened sensitivity — strict consent standards for collection, processing, and storage
- Cybersecurity: end-to-end encryption, access controls, audit logging required
- Clinical accountability: DSS outputs should be classified as advisory recommendations, not strict clinical pathways — clinician override must remain available and logged
- Cervical cancer prevention strategy: positions the DSS within national screening improvement efforts and early-stage detection via risk-stratified triage

---

## References

- Fernandes, K. et al. "Cervical Cancer (Risk Factors)." UCI Machine Learning Repository, 2017. https://doi.org/10.24432/C5Z310
- Perkins, R.B. et al. "2019 ASCCP Risk-Based Management Consensus Guidelines." Journal of Lower Genital Tract Disease, 2020.
- "Cancer of the Cervix Uteri." SEER Cancer Stat Facts, 2025.
- Zhang, S. et al. "Cervical Cancer: Epidemiology, Risk Factors and Screening." Chinese Journal of Cancer Research, 2020.
- World Health Organization. "Cervical Cancer." WHO Fact Sheet, Dec 2025.
