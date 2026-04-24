# Cervical Cancer DSS — Setup Guide

CMU 47-775 Healthcare Information Systems
Team: Abigail Torbatian · Esha Pandya · Devavrath Sandeep

---

## What this is

A clinician-facing decision support system for cervical cancer screening triage. Built with a R Shiny frontend and a Python FastAPI backend. Requires access to the team's Supabase database.

---

## System Requirements

- macOS or Linux (Windows works but commands may differ)
- R 4.1 or higher
- Python 3.9 or higher
- Git (to clone the repo)

---

## Step 1 — Install R and Python

### macOS

Install Homebrew if you don't have it:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install R:
```bash
brew install r
```

Python is usually pre-installed on macOS. Check:
```bash
python3 --version
```

If missing: `brew install python`

### Ubuntu / Debian
```bash
sudo apt-get install r-base python3 python3-pip
```

### Windows
Download and install from:
- R: https://cran.r-project.org
- Python: https://python.org

---

## Step 2 — Get the project files

Clone or download the repo and navigate into it:
```bash
cd path/to/dss
```

Make sure these files are present:
```
dss/
├── app.R
├── api.py
├── cervical-cancer_csv (1).csv
├── cervical_cancer_clean_nullable.csv
├── cervical_cancer_analysis_ready_imputed.csv
└── cervical_model_bundle_v2.joblib        <- generate with 02_model_v2.ipynb if missing
```

---

## Step 3 — Configure environment variables

Create a file called `.Renviron` in the project folder. Get the actual credential values from a team member — do not commit these to git.

```
SUPABASE_HOST="your-supabase-host"
SUPABASE_PASSWORD="your-supabase-password"
CERVICAL_API_URL="http://localhost:8001/predict"
```

To create it from the terminal:
```bash
cat > .Renviron << 'EOF'
SUPABASE_HOST="your-supabase-host"
SUPABASE_PASSWORD="your-supabase-password"
CERVICAL_API_URL="http://localhost:8001/predict"
EOF
```

---

## Step 4 — Install R packages (first time only)

```bash
R --vanilla -e "install.packages(c('shiny','DT','dplyr','ggplot2','scales','shinymanager','DBI','RPostgres','pool','httr','jsonlite'), repos='https://cloud.r-project.org')"
```

This takes a few minutes the first time.

---

## Step 5 — Install Python dependencies (first time only)

```bash
pip install fastapi uvicorn joblib matplotlib numpy pandas scipy \
            scikit-learn pydantic imbalanced-learn xgboost lightgbm
```

---

## Step 6 — Add yourself to the database

You need an account in the `app_user` table to log in. Get the psql connection string from a team member, then run:

```bash
psql "postgresql://YOUR_CONNECTION_STRING"
```

```sql
INSERT INTO app_user (user_id, first_name, last_name, email, password_hash, role, is_active)
VALUES (gen_random_uuid(), 'Your', 'Name', 'you@email.com', 'yourpassword', 'physician', TRUE);
\q
```

The password you set here is what you type at the login screen. Passwords are plain text in this prototype.

---

## Running the app

You need **two terminals** open in the project folder. **Always start Terminal 1 first.**

### Terminal 1 — Start the Python API

```bash
cd path/to/dss
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

You should see:
```
Uvicorn running on http://0.0.0.0:8001
Bundle loaded. Models: ['LR', 'RF', 'XGB', 'LGB']
Application startup complete.
```

Verify it's working:
```bash
curl http://localhost:8001/health
```

### Terminal 2 — Start the Shiny app

```bash
cd path/to/dss
R --vanilla -e "readRenviron('.Renviron'); shiny::runApp('app.R', port=3838, launch.browser=FALSE)"
```

Then open **http://localhost:3838** in your browser and log in with the email and password you inserted in Step 6.

---

## Stopping the app

| Terminal | How to stop |
|---|---|
| Terminal 1 (API) | `Ctrl+C` |
| Terminal 2 (Shiny) | `Ctrl+C` — if unresponsive, `Ctrl+Z` then `kill %1` |
| R interactive session | `q()` then `n` |
| psql session | `\q` |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `socket ... No such file or directory` on R startup | `.Renviron` not loaded | Use `R --vanilla -e "readRenviron('.Renviron'); ..."` as shown above |
| `ModuleNotFoundError` on API start | Missing Python package | Run the pip install command in Step 5 |
| `FileNotFoundError: cervical_model_bundle_v2.joblib` | Model bundle missing | Run `02_model_v2.ipynb` to generate it |
| Model Insights tab shows "local model used" | API not running | Start Terminal 1 first, confirm it shows "startup complete" |
| Save patient fails | Your email not in `app_user` | Add yourself via psql as shown in Step 6 |
| Login fails | Wrong email or password | Check the `app_user` table — passwords are plain text in this prototype |
| API predictions return 500 error | `cosine_similarity` import wrong | Run: `grep -n "cosine" api.py` and make sure line 35 reads `from sklearn.metrics.pairwise import cosine_similarity` |
