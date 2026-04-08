# Cervical-Cancer-Risk-Stratification-and-Screening-Triage

Cervical cancer is highly preventable when high-risk patients are identified early and connected to appropriate screening and follow-up care. However, clinics often face constraints in appointment availability, diagnostic capacity, and patient outreach resources, which can result in delayed testing for patients most likely to benefit. A data-driven decision support tool that estimates risk using patient risk-factor information can help prioritize screening and referral decisions while making the rationale transparent to clinicians.

We propose a prototype Decision Support System (DSS) for cervical cancer screening and triage that uses patient-level risk factors to generate an individualized risk estimate and a recommended next step. The DSS will combine (1) a patient risk-factor database, (2) a predictive model that estimates the probability of an abnormal screening or diagnostic outcome and provides interpretable drivers of risk, and (3) a workflow-oriented interface that supports decisions such as routine screening, expedited HPV/Pap testing, or referral for further evaluation (e.g., colposcopy), aligned with clinic capacity and guideline-based practice. Additionally, the interface will allow clinicians to view population-level data and identify the most common risk factors and mitigation strategies. 

We will use the UCI “Cervical Cancer (Risk Factors)” dataset, which includes demographic and behavioral/clinical variables (e.g., age, smoking indicators, contraceptive history, and STI-related features) as well as screening and diagnostic outcome labels. The modeling approach will compare logistic regression with gradient boosting and will explicitly address missingness and self-report uncertainty. Sensitivity and robustness analyses will test how predictions and triage recommendations change under plausible input uncertainty or missing data, and threshold selection will balance missed high-risk cases against over-referral under finite diagnostic capacity.

Example DSS questions:
1. Descriptive:
Which risk factors (age, smoking, contraceptive/IUD history, STI indicators) are most associated with abnormal screening or diagnostic outcomes?
2. Predictive/Analytical:
(a) What is the predicted probability that a patient will have an abnormal outcome given their risk-factor profile?
(b) Should this patient be recommended for routine screening, expedited testing, or referral for further evaluation given capacity constraints?
3. Sensitivity/Robustness:
(a) How sensitive is predicted risk to uncertainty in key inputs (e.g., smoking status, STI history)?
(b) How robust are recommendations to missing or noisy data, and which variables most strongly affect the triage category?
(c) What referral threshold best balances missed cases versus over-referral under clinic capacity and cost constraints?
