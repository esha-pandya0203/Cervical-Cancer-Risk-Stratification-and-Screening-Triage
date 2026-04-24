
# Cervical Cancer DSS Prototype (Shiny)
# Built from the uploaded project files, front-end design, and cleaned dataset outputs.
# This app is a prototype for clinician-facing cervical cancer screening triage.

library(shiny)
library(DT)
library(dplyr)
library(ggplot2)
library(scales)

`%||%` <- function(a, b) if (!is.null(a) && length(a) > 0) a else b

# ---------------------------
# Helpers
# ---------------------------
clean_names <- function(x) {
  x <- trimws(x)
  x <- gsub("Smokes \\(packs/year\\)", "Smokes packs per year", x)
  x <- gsub("First sexual intercourse", "First sexual intercourse age", x)
  x <- gsub("STDs \\(number\\)", "STDs number", x)
  x <- gsub("Hormonal Contraceptives \\(years\\)", "Hormonal Contraceptives years", x)
  x <- gsub("IUD \\(years\\)", "IUD years", x)
  x <- gsub("Smokes \\(years\\)", "Smokes years", x)
  x <- gsub(":", " ", x)
  x <- gsub("[()]", " ", x)
  x <- gsub("/", " per ", x)
  x <- gsub("[^0-9A-Za-z]+", "_", x)
  x <- gsub("_+", "_", x)
  x <- gsub("^_|_$", "", x)
  tolower(x)
}

first_existing <- function(paths) {
  for (p in paths) {
    if (!is.null(p) && file.exists(p)) return(p)
  }
  NULL
}

download_first_working <- function(urls, dest) {
  for (u in urls) {
    ok <- tryCatch({
      utils::download.file(u, destfile = dest, mode = "wb", quiet = TRUE)
      TRUE
    }, error = function(e) FALSE)
    if (ok && file.exists(dest) && file.info(dest)$size > 0) return(dest)
  }
  NULL
}

coalesce_num <- function(x, fallback = 0) {
  x <- suppressWarnings(as.numeric(x))
  ifelse(is.na(x), fallback, x)
}

binary_label <- function(x) ifelse(as.numeric(x) >= 1, "Yes", "No")

# ---------------------------
# Data loading (local files only)
# ---------------------------
RAW_CSV_PATH <- first_existing(c(
  "cervical-cancer_csv (1).csv",
  "cervical-cancer_csv.csv",
  "data/cervical-cancer_csv (1).csv",
  "data/cervical-cancer_csv.csv"
))
CLEAN_CSV_PATH <- first_existing(c(
  "cervical_cancer_clean_nullable(3).csv",
  "cervical_cancer_clean_nullable.csv",
  "data/cervical_cancer_clean_nullable.csv"
))
IMPUTED_CSV_PATH <- first_existing(c(
  "cervical_cancer_analysis_ready_imputed(3).csv",
  "cervical_cancer_analysis_ready_imputed.csv",
  "data/cervical_cancer_analysis_ready_imputed.csv"
))

load_project_data <- function() {
  required_file_labels <- c(
    raw = RAW_CSV_PATH %||% NA_character_,
    clean = CLEAN_CSV_PATH %||% NA_character_,
    imputed = IMPUTED_CSV_PATH %||% NA_character_
  )
  missing_files <- names(required_file_labels)[
    is.na(required_file_labels) | !file.exists(required_file_labels)
  ]
  
  if (length(missing_files) > 0) {
    stop(
      paste(
        "These required local data files were not found:",
        paste(missing_files, collapse = "\n"),
        sep = "\n"
      ),
      call. = FALSE
    )
  }
  
  raw_df <- read.csv(
    RAW_CSV_PATH,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  clean_df <- read.csv(
    CLEAN_CSV_PATH,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  analysis_df <- read.csv(
    IMPUTED_CSV_PATH,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  
  names(raw_df) <- clean_names(names(raw_df))
  names(clean_df) <- clean_names(names(clean_df))
  names(analysis_df) <- clean_names(names(analysis_df))
  
  outcome_cols_clean <- intersect(c("hinselmann", "schiller", "citology", "biopsy"), names(clean_df))
  outcome_cols_analysis <- intersect(c("hinselmann", "schiller", "citology", "biopsy"), names(analysis_df))
  
  if (!"abnormal_any" %in% names(clean_df) && length(outcome_cols_clean) > 0) {
    clean_df$abnormal_any <- do.call(pmax, c(clean_df[outcome_cols_clean], na.rm = TRUE))
  }
  
  if (!"missing_count_predictors" %in% names(clean_df) && length(outcome_cols_clean) > 0) {
    pred_cols <- setdiff(names(clean_df), outcome_cols_clean)
    clean_df$missing_count_predictors <- rowSums(is.na(clean_df[pred_cols]))
  }
  
  if (!"abnormal_any" %in% names(analysis_df) && length(outcome_cols_analysis) > 0) {
    analysis_df$abnormal_any <- do.call(pmax, c(analysis_df[outcome_cols_analysis], na.rm = TRUE))
  }
  
  list(raw = raw_df, clean = clean_df, analysis = analysis_df)
}

project_data <- load_project_data()
raw_df <- project_data$raw
clean_df <- project_data$clean
analysis_df <- project_data$analysis
patients_df <- analysis_df

# ---------------------------
# Feature engineering for the prototype model
# ---------------------------
feature_pool <- c(
  "age",
  "number_of_sexual_partners",
  "first_sexual_intercourse_age",
  "num_of_pregnancies",
  "smokes",
  "smokes_years",
  "smokes_packs_per_year",
  "hormonal_contraceptives",
  "hormonal_contraceptives_years",
  "iud",
  "iud_years",
  "stds",
  "stds_number",
  "dx_cancer",
  "dx_cin",
  "dx_hpv",
  "dx"
)

available_features <- intersect(feature_pool, names(analysis_df))
analysis_model <- analysis_df

for (nm in available_features) {
  analysis_model[[nm]] <- suppressWarnings(as.numeric(analysis_model[[nm]]))
}

feature_medians <- sapply(analysis_model[, available_features, drop = FALSE], function(x) {
  x <- suppressWarnings(as.numeric(x))
  med <- stats::median(x, na.rm = TRUE)
  ifelse(is.na(med), 0, med)
})

feature_labels <- c(
  age = "Age",
  number_of_sexual_partners = "Number of sexual partners",
  first_sexual_intercourse_age = "Age at first sexual intercourse",
  num_of_pregnancies = "Number of pregnancies",
  smokes = "Smoking status",
  smokes_years = "Smoking years",
  smokes_packs_per_year = "Smoking packs per year",
  hormonal_contraceptives = "Hormonal contraceptive use",
  hormonal_contraceptives_years = "Hormonal contraceptive years",
  iud = "IUD use",
  iud_years = "IUD years",
  stds = "Any STD history",
  stds_number = "Number of STDs",
  dx_cancer = "Prior cancer diagnosis",
  dx_cin = "Prior CIN diagnosis",
  dx_hpv = "Prior HPV diagnosis",
  dx = "Any prior diagnosis flag"
)

glm_formula <- as.formula(
  paste("abnormal_any ~", paste(available_features, collapse = " + "))
)

risk_model <- tryCatch({
  stats::glm(glm_formula, data = analysis_model, family = binomial())
}, warning = function(w) {
  suppressWarnings(stats::glm(glm_formula, data = analysis_model, family = binomial()))
}, error = function(e) NULL)

score_patient <- function(patient_row, low_thr = 0.15, high_thr = 0.35) {
  row <- patient_row[1, , drop = FALSE]
  
  for (nm in available_features) {
    if (!nm %in% names(row)) row[[nm]] <- feature_medians[[nm]]
    row[[nm]] <- coalesce_num(row[[nm]], feature_medians[[nm]])
  }
  
  if (!is.null(risk_model)) {
    prob <- as.numeric(stats::predict(risk_model, newdata = row, type = "response"))
    prob <- max(min(prob, 0.999), 0.001)
    
    coefs <- stats::coef(risk_model)
    coefs <- coefs[setdiff(names(coefs), "(Intercept)")]
    usable_coef_names <- intersect(names(coefs), available_features)
    vals <- sapply(usable_coef_names, function(nm) coalesce_num(row[[nm]], feature_medians[[nm]])[1])
    meds <- feature_medians[usable_coef_names]
    contrib <- coefs[usable_coef_names] * (vals - meds)
    
    top <- sort(contrib[contrib > 0], decreasing = TRUE)
    if (length(top) < 3) {
      top <- sort(abs(contrib), decreasing = TRUE)
    }
    driver_names <- names(utils::head(top, 3))
  } else {
    # Simple fallback if model cannot be fit
    linear_score <-
      0.02 * coalesce_num(row$age, 35) +
      0.25 * coalesce_num(row$smokes, 0) +
      0.20 * coalesce_num(row$stds, 0) +
      0.30 * coalesce_num(row$dx_hpv, 0) +
      0.20 * coalesce_num(row$dx_cancer, 0) +
      0.06 * coalesce_num(row$num_of_pregnancies, 0) +
      0.04 * coalesce_num(row$hormonal_contraceptives_years, 0)
    prob <- 1 / (1 + exp(-(-4 + linear_score)))
    driver_names <- c("age", "smokes", "stds")
  }
  
  risk_level <- if (prob < low_thr) {
    "Low"
  } else if (prob < high_thr) {
    "Medium"
  } else {
    "High"
  }
  
  recommendation <- dplyr::case_when(
    risk_level == "Low" ~ "Routine recall",
    risk_level == "Medium" ~ "Expedited HPV/Pap testing",
    TRUE ~ "Referral for further evaluation"
  )
  
  list(
    probability = prob,
    risk_level = risk_level,
    recommendation = recommendation,
    top_drivers = unname(feature_labels[driver_names]),
    driver_code = driver_names
  )
}

# ---------------------------
# Optional external API scoring
# ---------------------------
# Set these in .Renviron locally or in shinyapps.io environment variables:
# CERVICAL_API_URL=https://your-api-host/predict
# CERVICAL_API_KEY=your_key_if_required
# CERVICAL_API_TIMEOUT=10

api_is_configured <- function() {
  nzchar(Sys.getenv("CERVICAL_API_URL", unset = ""))
}

api_field <- function(x, names, default = NULL) {
  for (nm in names) {
    if (!is.null(x[[nm]]) && length(x[[nm]]) > 0) return(x[[nm]])
  }
  default
}

build_api_payload <- function(patient_row, low_thr = 0.15, high_thr = 0.35) {
  row <- patient_row[1, , drop = FALSE]
  
  # This payload matches the FastAPI PatientInput schema in api.py:
  # age, sexual_partners, pregnancies, smokes_years, hc_years, iud_years,
  # std_burden, and optional cluster. Do not nest these under "features";
  # FastAPI expects the fields at the top JSON level.
  list(
    age = unname(coalesce_num(row$age, feature_medians[["age"]] %||% 35)[1]),
    sexual_partners = unname(coalesce_num(row$number_of_sexual_partners, feature_medians[["number_of_sexual_partners"]] %||% 1)[1]),
    pregnancies = unname(coalesce_num(row$num_of_pregnancies, feature_medians[["num_of_pregnancies"]] %||% 0)[1]),
    smokes_years = unname(coalesce_num(row$smokes_years, 0)[1]),
    hc_years = unname(coalesce_num(row$hormonal_contraceptives_years, 0)[1]),
    iud_years = unname(coalesce_num(row$iud_years, 0)[1]),
    std_burden = unname(coalesce_num(row$stds_number, coalesce_num(row$stds, 0))[1])
  )
}

normalize_api_score <- function(api_body, local_score) {
  # The repo API returns a flat PredictionResponse object, not a nested object.
  response_obj <- api_body
  
  prob <- suppressWarnings(as.numeric(api_field(
    response_obj,
    c("probability", "predicted_probability", "risk_probability", "risk_score", "score"),
    local_score$probability
  )))
  prob <- max(min(prob, 0.999), 0.001)

  raw_prediction <- as.character(api_field(
    response_obj,
    c("prediction", "risk_level", "risk_category", "category", "triage_level"),
    local_score$risk_level
  ))
  raw_prediction_clean <- toupper(gsub("[_-]", " ", raw_prediction))
  
  risk_level <- if (grepl("HIGH", raw_prediction_clean)) {
    "High"
  } else if (grepl("LOW", raw_prediction_clean)) {
    "Low"
  } else if (grepl("MED", raw_prediction_clean)) {
    "Medium"
  } else {
    tools::toTitleCase(tolower(raw_prediction))
  }
  if (!risk_level %in% c("Low", "Medium", "High")) risk_level <- local_score$risk_level

  recommendation <- as.character(api_field(
    response_obj,
    c("recommendation", "recommended_action", "next_step", "triage_recommendation"),
    NA_character_
  ))
  if (!nzchar(recommendation) || is.na(recommendation)) {
    recommendation <- dplyr::case_when(
      risk_level == "Low" ~ "Routine recall",
      risk_level == "Medium" ~ "Expedited HPV/Pap testing",
      TRUE ~ "Referral for further evaluation"
    )
  }

  top_drivers <- api_field(
    response_obj,
    c("top_drivers", "drivers", "risk_drivers", "important_features"),
    local_score$top_drivers
  )
  if (is.data.frame(top_drivers)) top_drivers <- top_drivers[[1]]
  if (is.list(top_drivers)) top_drivers <- unlist(top_drivers, use.names = FALSE)
  top_drivers <- as.character(top_drivers)
  top_drivers <- top_drivers[nzchar(top_drivers)]
  if (length(top_drivers) == 0) top_drivers <- local_score$top_drivers
  
  confidence <- api_field(response_obj, c("confidence"), NULL)
  confidence_tier <- api_field(response_obj, c("confidence_tier"), NULL)
  warnings <- api_field(response_obj, c("warnings"), NULL)
  warning_text <- ""
  if (!is.null(warnings) && length(warnings) > 0) {
    warning_text <- paste(" Warnings:", paste(unlist(warnings), collapse = " | "))
  }
  confidence_text <- ""
  if (!is.null(confidence)) {
    confidence_text <- paste0(" Confidence: ", round(as.numeric(confidence), 3),
                              if (!is.null(confidence_tier)) paste0(" (", confidence_tier, ")") else "", ".")
  }

  list(
    probability = prob,
    risk_level = risk_level,
    recommendation = recommendation,
    top_drivers = top_drivers,
    driver_code = local_score$driver_code,
    source = "External FastAPI model",
    api_status = paste0("API scoring completed successfully.", confidence_text, warning_text)
  )
}

# ---------------------------
# Demo patient registry derived from the dataset
# ---------------------------
set.seed(706)

make_demo_patients <- function(df) {
  n <- nrow(df)
  first_names <- c("Ava", "Mia", "Sophia", "Isabella", "Emma", "Olivia", "Aria", "Layla", "Nora", "Camila",
                   "Grace", "Luna", "Chloe", "Zoey", "Maya", "Elena", "Jade", "Leah", "Naomi", "Ruby")
  last_names <- c("Johnson", "Smith", "Brown", "Davis", "Martinez", "Taylor", "Wilson", "Anderson", "Thomas",
                  "Moore", "Jackson", "Martin", "Lee", "Perez", "White", "Harris", "Clark", "Lewis", "Young", "Hall")
  streets <- c("Maple Ave", "Oak St", "Hillcrest Dr", "Pine Ln", "Cedar Ct", "Lakeview Rd", "Cherry St")
  cities <- c("Pittsburgh", "Monroeville", "Greensburg", "Bethel Park", "Cranberry", "McKeesport", "Washington")
  
  df$patient_id <- sprintf("P%04d", seq_len(n))
  df$first_name <- sample(first_names, n, replace = TRUE)
  df$last_name <- sample(last_names, n, replace = TRUE)
  df$patient_name <- paste(df$first_name, df$last_name)
  
  age_num <- coalesce_num(df$age, 35)
  age_num <- pmax(age_num, 18)
  df$dob <- format(Sys.Date() - round(age_num * 365.25), "%Y-%m-%d")
  df$address <- paste(sample(100:999, n, replace = TRUE), sample(streets, n, replace = TRUE))
  df$city <- sample(cities, n, replace = TRUE)
  df$state <- "PA"
  df$zip <- sample(15000:15299, n, replace = TRUE)
  df$phone <- paste0("(412) ", sample(200:999, n, replace = TRUE), "-", sample(1000:9999, n, replace = TRUE))
  df$email <- paste0(tolower(df$first_name), ".", tolower(df$last_name), seq_len(n), "@demohealth.org")
  
  if (!"abnormal_any" %in% names(df)) {
    outcome_cols <- intersect(c("hinselmann", "schiller", "citology", "biopsy"), names(df))
    if (length(outcome_cols) > 0) df$abnormal_any <- do.call(pmax, c(df[outcome_cols], na.rm = TRUE))
  }
  
  df
}

patients_df <- make_demo_patients(clean_df)

# Initialize baseline risk columns
baseline_scores <- lapply(seq_len(nrow(patients_df)), function(i) score_patient(patients_df[i, , drop = FALSE]))
patients_df$predicted_probability <- sapply(baseline_scores, function(x) x$probability)
patients_df$risk_level <- sapply(baseline_scores, function(x) x$risk_level)
patients_df$model_recommendation <- sapply(baseline_scores, function(x) x$recommendation)
patients_df$final_recommendation <- patients_df$model_recommendation
patients_df$top_drivers <- sapply(baseline_scores, function(x) paste(x$top_drivers, collapse = ", "))
patients_df$score_source <- "Local model"
patients_df$api_status <- "Baseline score generated locally."
patients_df$override_reason <- NA_character_
patients_df$last_reviewed <- NA_character_

# ---------------------------
# UI
# ---------------------------
ui <- navbarPage(
  title = "Cervical Cancer DSS",
  id = "main_navbar",
  header = tags$head(
    tags$style(HTML("
      body {
        background: #E5F6FF;
        color: #000000;
        font-family: 'Arial', 'Helvetica', sans-serif;
      }
      .navbar-default {
        background-color: #FFFFFF;
        border-color: #d7eaf6;
      }
      .navbar-default .navbar-brand,
      .navbar-default .navbar-nav > li > a {
        font-weight: 600;
        color: #000000 !important;
      }
      .hero-card, .panel-card, .metric-card, .mini-card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        margin-bottom: 18px;
      }
      .hero-card {
        padding: 30px;
      }
      .hero-title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 10px;
      }
      .hero-subtitle {
        font-size: 16px;
        line-height: 1.5;
      }
      .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 14px;
      }
      .soft-box {
        background: #E5F6FF;
        border-radius: 18px;
        padding: 16px;
      }
      .blue-btn, .btn-primary {
        background-color: #AFCBF8 !important;
        border-color: #AFCBF8 !important;
        color: #000000 !important;
        border-radius: 14px !important;
        font-weight: 700;
      }
      .red-btn, .btn-danger {
        background-color: #BC0E0E !important;
        border-color: #BC0E0E !important;
        color: #FFFFFF !important;
        border-radius: 14px !important;
        font-weight: 700;
      }
      .metric-value {
        font-size: 30px;
        font-weight: 700;
      }
      .metric-label {
        font-size: 13px;
        color: #333333;
      }
      .risk-pill {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        font-weight: 700;
        margin-right: 8px;
      }
      .risk-low { background: #d9f0e3; }
      .risk-medium { background: #fff2cc; }
      .risk-high { background: #f4cccc; }
      .driver-tag {
        display: inline-block;
        background: #E5F6FF;
        border-radius: 999px;
        padding: 7px 12px;
        margin: 4px 6px 4px 0;
        font-weight: 600;
      }
      .override-note {
        background: #fff3f3;
        border-left: 6px solid #BC0E0E;
        border-radius: 16px;
        padding: 14px 16px;
      }
      .dt-buttons, .dataTables_filter {
        margin-bottom: 8px;
      }
    "))
  ),
  
  tabPanel(
    "Home",
    fluidPage(
      br(),
      div(
        class = "hero-card",
        fluidRow(
          column(
            8,
            div(class = "hero-title", "Cervical Cancer Risk Stratification and Screening Triage"),
            div(
              class = "hero-subtitle",
              "This prototype clinical DSS combines a clinician-facing user interface, a patient risk scoring engine, and a structured data layer. It supports three operational decisions: routine recall, expedited HPV/Pap testing, or referral for further evaluation."
            ),
            br(),
            actionButton("go_patient_search", "Open patient search", class = "blue-btn"),
            HTML("&nbsp;"),
            actionButton("go_population_dashboard", "Open population dashboard", class = "blue-btn")
          ),
          column(
            4,
            div(
              class = "soft-box",
              h4(strong("DSS architecture")),
              tags$ul(
                tags$li(strong("User interface: "), "patient search, assessment, override, dashboard"),
                tags$li(strong("Problem processing: "), "logistic-risk scoring, top drivers, what-if thresholds"),
                tags$li(strong("Knowledge/data system: "), "raw, cleaned, and analysis-ready cervical cancer data")
              )
            )
          )
        )
      ),
      fluidRow(
        column(
          4,
          div(
            class = "mini-card",
            h4(strong("Opening screen")),
            p("Styled to match the front-end design deck: rounded cards, light blue background, white content panels, and clear black text.")
          )
        ),
        column(
          4,
          div(
            class = "mini-card",
            h4(strong("Point-of-care workflow")),
            p("Patient search, risk review, assessment, recommendation, override, and summary download are all included in the prototype.")
          )
        ),
        column(
          4,
          div(
            class = "mini-card",
            h4(strong("Population management")),
            p("The dashboard shows risk distribution, common high-risk patterns, and override monitoring to support operational decisions.")
          )
        )
      )
    )
  ),
  
  tabPanel(
    "Patient Search",
    fluidPage(
      br(),
      fluidRow(
        column(
          8,
          div(
            class = "panel-card",
            div(class = "section-title", "Patient Search"),
            textInput("patient_search", NULL, placeholder = "Search by patient name or ID"),
            DTOutput("patient_table"),
            br(),
            actionButton("open_selected_patient", "Open selected patient", class = "blue-btn"),
            HTML("&nbsp;"),
            actionButton("toggle_add_patient", "Add new patient", class = "blue-btn")
          )
        ),
        column(
          4,
          uiOutput("selected_patient_preview")
        )
      ),
      uiOutput("add_patient_panel")
    )
  ),
  
  tabPanel(
    "Patient Risk Assessment",
    fluidPage(
      br(),
      uiOutput("assessment_panel")
    )
  ),
  
  tabPanel(
    "Population Dashboard",
    fluidPage(
      br(),
      fluidRow(
        column(3, div(class = "metric-card", div(class = "metric-value", textOutput("metric_n_patients")), div(class = "metric-label", "Patients in prototype registry"))),
        column(3, div(class = "metric-card", div(class = "metric-value", textOutput("metric_abnormal")), div(class = "metric-label", "Any abnormal outcome in source data"))),
        column(3, div(class = "metric-card", div(class = "metric-value", textOutput("metric_high_risk")), div(class = "metric-label", "Current high-risk patients"))),
        column(3, div(class = "metric-card", div(class = "metric-value", textOutput("metric_overrides")), div(class = "metric-label", "Clinician overrides logged")))
    ),
    fluidRow(
      column(6, div(class = "panel-card", div(class = "section-title", "Risk level mix"), plotOutput("plot_risk_mix", height = 300))),
      column(6, div(class = "panel-card", div(class = "section-title", "Common patterns in high-risk group"), plotOutput("plot_top_drivers", height = 300)))
    ),
    fluidRow(
      column(6, div(class = "panel-card", div(class = "section-title", "Age vs. pregnancies"), plotOutput("plot_age_preg", height = 320))),
      column(6, div(class = "panel-card", div(class = "section-title", "Override activity"), DTOutput("override_table")))
    )
  )
)
)

# ---------------------------
# Server
# ---------------------------
server <- function(input, output, session) {
  
  rv <- reactiveValues(
    patients = patients_df,
    selected_patient_id = NULL,
    last_api_status = "API not called yet.",
    last_score_source = "Local model",
    show_add_patient = FALSE,
    show_override = FALSE,
    override_log = data.frame(
      timestamp = character(),
      patient_id = character(),
      patient_name = character(),
      model_recommendation = character(),
      final_recommendation = character(),
      reason = character(),
      stringsAsFactors = FALSE
    )
  )
  
  filtered_patients <- reactive({
    dat <- rv$patients
    q <- trimws(input$patient_search %||% "")
    if (nchar(q) == 0) return(dat)
    
    q_low <- tolower(q)
    keep <- grepl(q_low, tolower(dat$patient_name)) | grepl(q_low, tolower(dat$patient_id))
    dat[keep, , drop = FALSE]
  })
  
  output$patient_table <- renderDT({
    dat <- filtered_patients() %>%
      transmute(
        patient_id,
        patient_name,
        age = round(coalesce_num(age, 0), 0),
        smokes = binary_label(smokes),
        stds = binary_label(stds),
        risk_level,
        recommendation = final_recommendation
      )
    
    datatable(
      dat,
      rownames = FALSE,
      selection = "single",
      options = list(pageLength = 8, autoWidth = TRUE, dom = "tip")
    )
  })
  
  observeEvent(input$go_patient_search, {
    updateNavbarPage(session, "main_navbar", selected = "Patient Search")
  })
  
  observeEvent(input$go_population_dashboard, {
    updateNavbarPage(session, "main_navbar", selected = "Population Dashboard")
  })
  
  observeEvent(input$open_selected_patient, {
    idx <- input$patient_table_rows_selected
    if (length(idx) == 1) {
      chosen <- filtered_patients()[idx, , drop = FALSE]
      rv$selected_patient_id <- chosen$patient_id[1]
      updateNavbarPage(session, "main_navbar", selected = "Patient Risk Assessment")
    }
  })
  
  observeEvent(input$toggle_add_patient, {
    rv$show_add_patient <- !isTRUE(rv$show_add_patient)
  })
  
  current_patient <- reactive({
    req(rv$selected_patient_id)
    rv$patients %>% filter(patient_id == rv$selected_patient_id)
  })
  
  output$selected_patient_preview <- renderUI({
    idx <- input$patient_table_rows_selected
    if (length(idx) != 1) {
      return(
        div(
          class = "panel-card",
          div(class = "section-title", "Selected Patient"),
          p("Select a row in the table to preview a patient.")
        )
      )
    }
    
    pdat <- filtered_patients()[idx, , drop = FALSE]
    div(
      class = "panel-card",
      div(class = "section-title", "Selected Patient"),
      p(strong(pdat$patient_name)),
      p(paste("ID:", pdat$patient_id)),
      p(paste("DOB:", pdat$dob)),
      p(paste("Phone:", pdat$phone)),
      p(paste("Current risk level:", pdat$risk_level)),
      p(paste("Recommendation:", pdat$final_recommendation))
    )
  })
  
  output$add_patient_panel <- renderUI({
    if (!isTRUE(rv$show_add_patient)) return(NULL)
    
    div(
      class = "panel-card",
      div(class = "section-title", "Add New Patient"),
      fluidRow(
        column(4, textInput("new_name", "Full name", value = "")),
        column(4, dateInput("new_dob", "Date of birth", value = Sys.Date() - 35 * 365)),
        column(4, textInput("new_phone", "Phone", value = "(412) 555-0000"))
      ),
      fluidRow(
        column(6, textInput("new_address", "Address", value = "100 Clinic Way")),
        column(3, textInput("new_city", "City", value = "Pittsburgh")),
        column(3, textInput("new_zip", "ZIP", value = "15213"))
      ),
      fluidRow(
        column(3, numericInput("new_age", "Age", value = 35, min = 18, max = 100)),
        column(3, numericInput("new_partners", "Sexual partners", value = 1, min = 0)),
        column(3, numericInput("new_firstsex", "Age at first intercourse", value = 18, min = 0, max = 100)),
        column(3, numericInput("new_preg", "Pregnancies", value = 0, min = 0))
      ),
      fluidRow(
        column(3, selectInput("new_smokes", "Smokes", choices = c("No" = 0, "Yes" = 1), selected = 0)),
        column(3, numericInput("new_smokes_years", "Smoking years", value = 0, min = 0)),
        column(3, numericInput("new_smokes_packs", "Smoking packs/year", value = 0, min = 0, step = 0.1)),
        column(3, selectInput("new_hc", "Hormonal contraceptives", choices = c("No" = 0, "Yes" = 1), selected = 0))
      ),
      fluidRow(
        column(3, numericInput("new_hc_years", "Hormonal contraceptive years", value = 0, min = 0, step = 0.1)),
        column(3, selectInput("new_iud", "IUD", choices = c("No" = 0, "Yes" = 1), selected = 0)),
        column(3, numericInput("new_iud_years", "IUD years", value = 0, min = 0, step = 0.1)),
        column(3, selectInput("new_stds", "Any STDs", choices = c("No" = 0, "Yes" = 1), selected = 0))
      ),
      fluidRow(
        column(4, numericInput("new_stds_number", "Number of STDs", value = 0, min = 0)),
        column(4, selectInput("new_dx_hpv", "Prior HPV diagnosis", choices = c("No" = 0, "Yes" = 1), selected = 0)),
        column(4, selectInput("new_dx_cancer", "Prior cancer diagnosis", choices = c("No" = 0, "Yes" = 1), selected = 0))
      ),
      actionButton("save_new_patient", "Save patient and open assessment", class = "blue-btn")
    )
  })
  
  observeEvent(input$save_new_patient, {
    parts <- strsplit(trimws(input$new_name), "\\s+")[[1]]
    first_name <- if (length(parts) >= 1 && nzchar(parts[1])) parts[1] else "New"
    last_name <- if (length(parts) >= 2) paste(parts[-1], collapse = " ") else "Patient"
    
    new_id <- sprintf("P%04d", nrow(rv$patients) + 1)
    new_row <- rv$patients[1, , drop = FALSE]
    
    # Reset to NA then repopulate
    new_row[1, ] <- NA
    for (nm in names(new_row)) {
      if (nm %in% names(feature_medians)) new_row[[nm]] <- feature_medians[[nm]]
    }
    
    new_row$patient_id <- new_id
    new_row$first_name <- first_name
    new_row$last_name <- last_name
    new_row$patient_name <- paste(first_name, last_name)
    new_row$dob <- as.character(input$new_dob)
    new_row$address <- input$new_address
    new_row$city <- input$new_city
    new_row$state <- "PA"
    new_row$zip <- input$new_zip
    new_row$phone <- input$new_phone
    new_row$email <- paste0(tolower(first_name), ".", tolower(gsub("\\s+", "", last_name)), new_id, "@demohealth.org")
    
    new_row$age <- input$new_age
    new_row$number_of_sexual_partners <- input$new_partners
    new_row$first_sexual_intercourse_age <- input$new_firstsex
    new_row$num_of_pregnancies <- input$new_preg
    new_row$smokes <- as.numeric(input$new_smokes)
    new_row$smokes_years <- input$new_smokes_years
    new_row$smokes_packs_per_year <- input$new_smokes_packs
    new_row$hormonal_contraceptives <- as.numeric(input$new_hc)
    new_row$hormonal_contraceptives_years <- input$new_hc_years
    new_row$iud <- as.numeric(input$new_iud)
    new_row$iud_years <- input$new_iud_years
    new_row$stds <- as.numeric(input$new_stds)
    new_row$stds_number <- input$new_stds_number
    new_row$dx_hpv <- as.numeric(input$new_dx_hpv)
    new_row$dx_cancer <- as.numeric(input$new_dx_cancer)
    new_row$dx_cin <- 0
    new_row$dx <- as.numeric(input$new_dx_hpv) + as.numeric(input$new_dx_cancer) > 0
    
    sc <- call_risk_api(new_row)
    new_row$predicted_probability <- sc$probability
    new_row$risk_level <- sc$risk_level
    new_row$model_recommendation <- sc$recommendation
    new_row$final_recommendation <- sc$recommendation
    new_row$top_drivers <- paste(sc$top_drivers, collapse = ", ")
    new_row$score_source <- sc$source
    new_row$api_status <- sc$api_status
    new_row$override_reason <- NA_character_
    new_row$last_reviewed <- as.character(Sys.time())
    
    rv$patients <- bind_rows(rv$patients, new_row)
    rv$selected_patient_id <- new_id
    rv$show_add_patient <- FALSE
    updateNavbarPage(session, "main_navbar", selected = "Patient Risk Assessment")
  })
  
  output$assessment_panel <- renderUI({
    if (is.null(rv$selected_patient_id)) {
      return(
        div(
          class = "panel-card",
          div(class = "section-title", "Patient Risk Assessment"),
          p("Open a patient from the search screen to view and run an assessment.")
        )
      )
    }
    
    pdat <- current_patient()
    
    div(
      fluidRow(
        column(
          4,
          div(
            class = "panel-card",
            div(class = "section-title", "Patient Risk Assessment"),
            p(strong(pdat$patient_name)),
            p(paste("DOB:", pdat$dob)),
            p(paste("Address:", pdat$address, pdat$city, pdat$state, pdat$zip)),
            numericInput("assess_age", "Age", value = round(coalesce_num(pdat$age, 35), 0), min = 18, max = 100),
            numericInput("assess_partners", "Number of sexual partners", value = round(coalesce_num(pdat$number_of_sexual_partners, 1), 0), min = 0),
            numericInput("assess_firstsex", "Age at first sexual intercourse", value = round(coalesce_num(pdat$first_sexual_intercourse_age, 18), 0), min = 0, max = 100),
            numericInput("assess_preg", "Number of pregnancies", value = round(coalesce_num(pdat$num_of_pregnancies, 0), 0), min = 0),
            selectInput("assess_smokes", "Smokes", choices = c("No" = 0, "Yes" = 1), selected = as.character(round(coalesce_num(pdat$smokes, 0), 0))),
            numericInput("assess_smokes_years", "Smoking years", value = coalesce_num(pdat$smokes_years, 0), min = 0),
            numericInput("assess_smokes_packs", "Smoking packs/year", value = coalesce_num(pdat$smokes_packs_per_year, 0), min = 0, step = 0.1),
            selectInput("assess_hc", "Hormonal contraceptives", choices = c("No" = 0, "Yes" = 1), selected = as.character(round(coalesce_num(pdat$hormonal_contraceptives, 0), 0))),
            numericInput("assess_hc_years", "Hormonal contraceptive years", value = coalesce_num(pdat$hormonal_contraceptives_years, 0), min = 0, step = 0.1),
            selectInput("assess_iud", "IUD", choices = c("No" = 0, "Yes" = 1), selected = as.character(round(coalesce_num(pdat$iud, 0), 0))),
            numericInput("assess_iud_years", "IUD years", value = coalesce_num(pdat$iud_years, 0), min = 0, step = 0.1),
            selectInput("assess_stds", "Any STDs", choices = c("No" = 0, "Yes" = 1), selected = as.character(round(coalesce_num(pdat$stds, 0), 0))),
            numericInput("assess_stds_number", "Number of STDs", value = round(coalesce_num(pdat$stds_number, 0), 0), min = 0),
            selectInput("assess_dx_hpv", "Prior HPV diagnosis", choices = c("No" = 0, "Yes" = 1), selected = as.character(round(coalesce_num(pdat$dx_hpv, 0), 0))),
            selectInput("assess_dx_cancer", "Prior cancer diagnosis", choices = c("No" = 0, "Yes" = 1), selected = as.character(round(coalesce_num(pdat$dx_cancer, 0), 0))),
            sliderInput("low_threshold", "Routine / expedited threshold", min = 0.05, max = 0.40, value = 0.15, step = 0.01),
            sliderInput("high_threshold", "Expedited / referral threshold", min = 0.20, max = 0.80, value = 0.35, step = 0.01),
            actionButton("run_assessment", "Run risk assessment", class = "blue-btn"),
            br(), br(),
            helpText("To use an external API, set CERVICAL_API_URL before running or deploying the app. If no API is configured, the app automatically uses the local model.")
          )
        ),
        column(
          8,
          div(
            class = "panel-card",
            div(class = "section-title", "Assessment Output"),
            uiOutput("risk_summary_ui"),
            plotOutput("risk_bar_plot", height = 160),
            h4(strong("Top risk drivers")),
            uiOutput("driver_tags_ui"),
            br(),
            h4(strong("Recommended next step")),
            uiOutput("recommendation_ui"),
            br(),
            actionButton("toggle_override_panel", "Clinician override", class = "red-btn"),
            HTML("&nbsp;"),
            downloadButton("download_patient_summary", "Download patient summary")
          ),
          uiOutput("override_panel")
        )
      )
    )
  })
  
  observeEvent(input$toggle_override_panel, {
    rv$show_override <- !isTRUE(rv$show_override)
  })
  
  observeEvent(input$run_assessment, {
    req(rv$selected_patient_id)
    idx <- which(rv$patients$patient_id == rv$selected_patient_id)
    req(length(idx) == 1)
    
    rv$patients$age[idx] <- input$assess_age
    rv$patients$number_of_sexual_partners[idx] <- input$assess_partners
    rv$patients$first_sexual_intercourse_age[idx] <- input$assess_firstsex
    rv$patients$num_of_pregnancies[idx] <- input$assess_preg
    rv$patients$smokes[idx] <- as.numeric(input$assess_smokes)
    rv$patients$smokes_years[idx] <- input$assess_smokes_years
    rv$patients$smokes_packs_per_year[idx] <- input$assess_smokes_packs
    rv$patients$hormonal_contraceptives[idx] <- as.numeric(input$assess_hc)
    rv$patients$hormonal_contraceptives_years[idx] <- input$assess_hc_years
    rv$patients$iud[idx] <- as.numeric(input$assess_iud)
    rv$patients$iud_years[idx] <- input$assess_iud_years
    rv$patients$stds[idx] <- as.numeric(input$assess_stds)
    rv$patients$stds_number[idx] <- input$assess_stds_number
    rv$patients$dx_hpv[idx] <- as.numeric(input$assess_dx_hpv)
    rv$patients$dx_cancer[idx] <- as.numeric(input$assess_dx_cancer)
    rv$patients$dx[idx] <- ifelse(as.numeric(input$assess_dx_hpv) + as.numeric(input$assess_dx_cancer) > 0, 1, 0)
    
    sc <- call_risk_api(rv$patients[idx, , drop = FALSE], input$low_threshold, input$high_threshold)
    rv$patients$predicted_probability[idx] <- sc$probability
    rv$patients$risk_level[idx] <- sc$risk_level
    rv$patients$model_recommendation[idx] <- sc$recommendation
    # Only replace final recommendation if not overridden yet
    if (is.na(rv$patients$override_reason[idx]) || !nzchar(rv$patients$override_reason[idx])) {
      rv$patients$final_recommendation[idx] <- sc$recommendation
    }
    rv$patients$top_drivers[idx] <- paste(sc$top_drivers, collapse = ", ")
    rv$patients$score_source[idx] <- sc$source
    rv$patients$api_status[idx] <- sc$api_status
    rv$last_score_source <- sc$source
    rv$last_api_status <- sc$api_status
    rv$patients$last_reviewed[idx] <- as.character(Sys.time())
  })
  
  output$risk_summary_ui <- renderUI({
    req(rv$selected_patient_id)
    pdat <- current_patient()
    risk_level <- pdat$risk_level[1]
    pill_class <- if (risk_level == "Low") "risk-pill risk-low" else if (risk_level == "Medium") "risk-pill risk-medium" else "risk-pill risk-high"
    
    div(
      div(
        span(class = pill_class, risk_level),
        tags$span(style = "font-size: 20px; font-weight: 700;", percent(coalesce_num(pdat$predicted_probability[1], 0), accuracy = 0.1))
      ),
      p(style = "margin-top: 8px;", paste("Model recommendation:", pdat$model_recommendation[1])),
      p(paste("Final recommendation:", pdat$final_recommendation[1])),
      p(paste("Score source:", pdat$score_source[1] %||% "Local model")),
      p(style = "font-size: 12px; color: #444;", pdat$api_status[1] %||% rv$last_api_status)
    )
  })
  
  output$driver_tags_ui <- renderUI({
    req(rv$selected_patient_id)
    pdat <- current_patient()
    drivers <- unlist(strsplit(pdat$top_drivers[1], ",\\s*"))
    tagList(lapply(drivers, function(x) tags$span(class = "driver-tag", x)))
  })
  
  output$recommendation_ui <- renderUI({
    req(rv$selected_patient_id)
    pdat <- current_patient()
    
    if (!is.na(pdat$override_reason[1]) && nzchar(pdat$override_reason[1])) {
      div(
        class = "override-note",
        p(strong("Clinician override applied")),
        p(paste("Final recommendation:", pdat$final_recommendation[1])),
        p(paste("Reason:", pdat$override_reason[1]))
      )
    } else {
      div(
        class = "soft-box",
        p(strong(pdat$final_recommendation[1])),
        p("Recommendation is generated from the current risk estimate and threshold settings.")
      )
    }
  })
  
  output$risk_bar_plot <- renderPlot({
    req(rv$selected_patient_id)
    pdat <- current_patient()
    risk <- coalesce_num(pdat$predicted_probability[1], 0)
    
    ggplot(data.frame(label = "Risk", value = risk), aes(x = label, y = value, fill = label)) +
      geom_col(width = 0.55, show.legend = FALSE) +
      coord_flip() +
      scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
      labs(x = NULL, y = "Predicted probability") +
      theme_minimal(base_size = 13) +
      theme(
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white", colour = NA),
        panel.background = element_rect(fill = "white", colour = NA)
      )
  })
  
  output$override_panel <- renderUI({
    if (!isTRUE(rv$show_override) || is.null(rv$selected_patient_id)) return(NULL)
    
    pdat <- current_patient()
    div(
      class = "panel-card",
      div(class = "section-title", "Clinician Override"),
      selectInput(
        "override_recommendation",
        "Updated recommendation",
        choices = c("Routine recall", "Expedited HPV/Pap testing", "Referral for further evaluation"),
        selected = pdat$final_recommendation[1]
      ),
      textAreaInput("override_reason_text", "Override rationale", rows = 4, placeholder = "Explain the clinical reason for overriding the model output."),
      actionButton("save_override", "Save override", class = "red-btn")
    )
  })
  
  observeEvent(input$save_override, {
    req(rv$selected_patient_id)
    idx <- which(rv$patients$patient_id == rv$selected_patient_id)
    req(length(idx) == 1)
    
    rv$patients$final_recommendation[idx] <- input$override_recommendation
    rv$patients$override_reason[idx] <- input$override_reason_text
    rv$patients$last_reviewed[idx] <- as.character(Sys.time())
    
    rv$override_log <- bind_rows(
      rv$override_log,
      data.frame(
        timestamp = as.character(Sys.time()),
        patient_id = rv$patients$patient_id[idx],
        patient_name = rv$patients$patient_name[idx],
        model_recommendation = rv$patients$model_recommendation[idx],
        final_recommendation = input$override_recommendation,
        reason = input$override_reason_text,
        stringsAsFactors = FALSE
      )
    )
  })
  
  output$download_patient_summary <- downloadHandler(
    filename = function() {
      req(rv$selected_patient_id)
      paste0("patient_summary_", rv$selected_patient_id, ".html")
    },
    content = function(file) {
      req(rv$selected_patient_id)
      pdat <- current_patient()[1, , drop = FALSE]
      
      html <- paste0(
        "<html><head><meta charset='utf-8'><title>Patient Summary</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;padding:28px;} h1,h2{margin-bottom:8px;} .box{background:#f7fbff;border-radius:16px;padding:18px;margin-bottom:16px;} .pill{display:inline-block;padding:8px 14px;border-radius:999px;background:#e5f6ff;font-weight:700;}</style>",
        "</head><body>",
        "<h1>Cervical Cancer DSS Patient Summary</h1>",
        "<div class='box'><h2>", pdat$patient_name[1], " (", pdat$patient_id[1], ")</h2>",
        "<p><strong>DOB:</strong> ", pdat$dob[1], "<br>",
        "<strong>Address:</strong> ", pdat$address[1], ", ", pdat$city[1], ", ", pdat$state[1], " ", pdat$zip[1], "<br>",
        "<strong>Phone:</strong> ", pdat$phone[1], "</p></div>",
        "<div class='box'><h2>Clinical Inputs</h2>",
        "<p><strong>Age:</strong> ", pdat$age[1], "<br>",
        "<strong>Smoking:</strong> ", binary_label(pdat$smokes[1]), " (years: ", round(coalesce_num(pdat$smokes_years[1], 0), 1), ", packs/year: ", round(coalesce_num(pdat$smokes_packs_per_year[1], 0), 2), ")<br>",
        "<strong>Sexual partners:</strong> ", round(coalesce_num(pdat$number_of_sexual_partners[1], 0), 0), "<br>",
        "<strong>Pregnancies:</strong> ", round(coalesce_num(pdat$num_of_pregnancies[1], 0), 0), "<br>",
        "<strong>Hormonal contraceptives:</strong> ", binary_label(pdat$hormonal_contraceptives[1]), " (years: ", round(coalesce_num(pdat$hormonal_contraceptives_years[1], 0), 1), ")<br>",
        "<strong>IUD:</strong> ", binary_label(pdat$iud[1]), " (years: ", round(coalesce_num(pdat$iud_years[1], 0), 1), ")<br>",
        "<strong>STDs:</strong> ", binary_label(pdat$stds[1]), " (number: ", round(coalesce_num(pdat$stds_number[1], 0), 0), ")<br>",
        "<strong>Prior HPV diagnosis:</strong> ", binary_label(pdat$dx_hpv[1]), "<br>",
        "<strong>Prior cancer diagnosis:</strong> ", binary_label(pdat$dx_cancer[1]), "</p></div>",
        "<div class='box'><h2>Risk Assessment</h2>",
        "<p><span class='pill'>", pdat$risk_level[1], "</span></p>",
        "<p><strong>Predicted probability:</strong> ", percent(coalesce_num(pdat$predicted_probability[1], 0), accuracy = 0.1), "<br>",
        "<strong>Top risk drivers:</strong> ", pdat$top_drivers[1], "<br>",
        "<strong>Model recommendation:</strong> ", pdat$model_recommendation[1], "<br>",
        "<strong>Final recommendation:</strong> ", pdat$final_recommendation[1], "<br>",
        "<strong>Score source:</strong> ", pdat$score_source[1] %||% "Local model", "<br>",
        "<strong>API status:</strong> ", pdat$api_status[1] %||% "Not available", "</p>",
        if (!is.na(pdat$override_reason[1]) && nzchar(pdat$override_reason[1])) paste0("<p><strong>Override rationale:</strong> ", pdat$override_reason[1], "</p>") else "",
        "</div>",
        "<p><em>Note: this is a prototype clinician-facing summary generated for course project demonstration.</em></p>",
        "</body></html>"
      )
      writeLines(html, con = file)
    }
  )
  
  # Population dashboard outputs
  output$metric_n_patients <- renderText({
    format(nrow(rv$patients), big.mark = ",")
  })
  
  output$metric_abnormal <- renderText({
    if ("abnormal_any" %in% names(clean_df)) {
      percent(mean(coalesce_num(clean_df$abnormal_any, 0)), accuracy = 0.1)
    } else {
      "N/A"
    }
  })
  
  output$metric_high_risk <- renderText({
    sum(rv$patients$risk_level == "High", na.rm = TRUE)
  })
  
  output$metric_overrides <- renderText({
    nrow(rv$override_log)
  })
  
  output$plot_risk_mix <- renderPlot({
    plot_df <- rv$patients %>%
      count(risk_level) %>%
      mutate(risk_level = factor(risk_level, levels = c("Low", "Medium", "High")))
    
    ggplot(plot_df, aes(x = risk_level, y = n, fill = risk_level)) +
      geom_col(show.legend = FALSE) +
      labs(x = NULL, y = "Patients") +
      theme_minimal(base_size = 13) +
      theme(
        plot.background = element_rect(fill = "white", colour = NA),
        panel.background = element_rect(fill = "white", colour = NA)
      )
  })
  
  output$plot_top_drivers <- renderPlot({
    high_risk <- rv$patients %>% filter(risk_level == "High")
    if (nrow(high_risk) == 0) {
      plot.new()
      text(0.5, 0.5, "No high-risk patients in current view.")
      return()
    }
    
    summary_df <- data.frame(
      label = c("Smokes", "Any STD", "Prior HPV Dx", "Prior cancer Dx", "IUD use", "Hormonal contraceptive use"),
      value = c(
        mean(coalesce_num(high_risk$smokes, 0)),
        mean(coalesce_num(high_risk$stds, 0)),
        mean(coalesce_num(high_risk$dx_hpv, 0)),
        mean(coalesce_num(high_risk$dx_cancer, 0)),
        mean(coalesce_num(high_risk$iud, 0)),
        mean(coalesce_num(high_risk$hormonal_contraceptives, 0))
      )
    )
    
    ggplot(summary_df, aes(x = reorder(label, value), y = value, fill = label)) +
      geom_col(show.legend = FALSE) +
      coord_flip() +
      scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
      labs(x = NULL, y = "Prevalence within high-risk group") +
      theme_minimal(base_size = 13) +
      theme(
        plot.background = element_rect(fill = "white", colour = NA),
        panel.background = element_rect(fill = "white", colour = NA)
      )
  })
  
  output$plot_age_preg <- renderPlot({
    ggplot(rv$patients, aes(x = coalesce_num(age, 0), y = coalesce_num(num_of_pregnancies, 0), color = risk_level)) +
      geom_point(alpha = 0.65) +
      labs(x = "Age", y = "Number of pregnancies") +
      theme_minimal(base_size = 13) +
      theme(
        plot.background = element_rect(fill = "white", colour = NA),
        panel.background = element_rect(fill = "white", colour = NA)
      )
  })
  
  output$override_table <- renderDT({
    dat <- rv$override_log
    if (nrow(dat) == 0) {
      dat <- data.frame(
        timestamp = "No overrides recorded yet",
        patient_id = "",
        patient_name = "",
        model_recommendation = "",
        final_recommendation = "",
        reason = "",
        stringsAsFactors = FALSE
      )
    }
    datatable(dat, rownames = FALSE, options = list(pageLength = 5, autoWidth = TRUE, dom = "tip"))
  })
}

shinyApp(ui = ui, server = server)
