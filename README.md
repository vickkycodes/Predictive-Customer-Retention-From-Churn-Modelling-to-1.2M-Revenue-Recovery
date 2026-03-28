# Predictive Customer Retention: From Churn Modelling to $1.2M Revenue Recovery

# Customer Churn Analysis & Retention Intelligence
### Telecom Customer Data — IBM Sample Dataset

![Python](https://img.shields.io/badge/Python-3.13-blue)
![sklearn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-In_progress-yellow)

---

## Project Overview

This project builds a complete end-to-end customer churn analysis and retention intelligence system for a telecommunications company. Starting from raw customer event data, the analysis moves through exploratory analysis, statistical significance testing, machine learning modelling, and culminates in a quantified business case with a ranked retention action list worth over **$1.2 million in net annual value**.

The core business question: **Which customers are likely to churn, why, and what should the business do about it?**

---

## Table of Contents

1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Phase 1 — Data Cleaning](#phase-1--data-cleaning)
4. [Phase 2 — Exploratory Analysis](#phase-2--exploratory-analysis)
5. [Phase 3 — Statistical Analysis](#phase-3--statistical-analysis)
6. [Phase 4 — Machine Learning](#phase-4--machine-learning)
7. [Phase 5 — Revenue & Retention Business Case](#phase-5--revenue--retention-business-case)
8. [Key Findings](#key-findings)
9. [Business Recommendations](#business-recommendations)
10. [Next Steps](#next-steps)
11. [Dependencies](#dependencies)

---

## Dataset

**Source:** IBM Sample Dataset (available on Kaggle)  
**Records:** 7,043 customers | **Features:** 21 columns  
**Target variable:** `Churn` (Yes/No)  
**Overall churn rate:** 26.5%

| Feature Category | Columns |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Account Info | tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| Target | Churn |

---

## Project Structure

```
customer-churn-analysis/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── outputs/
│   ├── high_risk_customers.csv
│   └── retention_action_list.csv
│
├── README.md

```

---

## Phase 1 — Data Cleaning

**Objective:** Prepare the dataset for reliable analysis.

**Steps performed:**
- Loaded 7,043 customer records across 21 columns
- Identified and fixed `TotalCharges` stored as string (object dtype) instead of float — a known issue in this dataset
- Converted using `pd.to_numeric(errors='coerce')` which surfaced 11 null rows (customers with zero tenure — just joined, no charges yet)
- Dropped those 11 rows leaving 7,032 clean records
- Encoded `Churn` from Yes/No to binary 1/0
- Confirmed no other nulls or duplicates across remaining columns

**Key data quality note:**  
`TotalCharges` is largely redundant with `tenure` (longer customers naturally accumulate more charges). It was dropped from the predictive model to avoid multicollinearity.

---

## Phase 2 — Exploratory Analysis

**Objective:** Identify visible patterns in churn behaviour before any statistical testing.

### Overall Churn Rate
26.5% of customers churned — meaning roughly 1 in 4 customers left within the observation period.

### Churn by Contract Type
| Contract | Churn Rate |
|---|---|
| Month-to-month | 42.7% |
| One year | 11.3% |
| Two year | 2.8% |

Month-to-month customers churn at nearly 15x the rate of two-year customers. Contract type immediately emerged as the dominant visual signal.

### Churn by Tenure
| Tenure Group | Churn Rate |
|---|---|
| 0–12 months | 47.7% |
| 13–24 months | 28.7% |
| 25–36 months | 21.6% |
| 37–48 months | 19.0% |
| 49–60 months | 14.4% |
| 61–72 months | 6.6% |

Nearly half of all customers in their first year leave. Churn risk drops consistently with every additional year of tenure.

### Churn by Internet Service
| Service | Churn Rate |
|---|---|
| Fiber optic | 41.9% |
| DSL | 19.0% |
| No internet | 7.4% |

Fiber optic — the premium service — has the highest churn rate. A strong early signal of pricing dissatisfaction.

### Churn by Payment Method
| Payment Method | Churn Rate |
|---|---|
| Electronic check | 45.3% |
| Mailed check | 19.1% |
| Bank transfer (automatic) | 16.7% |
| Credit card (automatic) | 15.2% |

Electronic check customers churn at nearly 3x the rate of automatic payment customers.

### Monthly Charges
- Churned customers avg: **$74.44/month**
- Non-churned customers avg: **$61.27/month**

Higher-paying customers are more likely to leave — an early flag for pricing misalignment.

---

## Phase 3 — Statistical Analysis

**Objective:** Confirm that observed patterns are statistically real and not sampling noise.

### Methods Used

| Method | Purpose | Applied To |
|---|---|---|
| Chi-square test | Tests independence between two categorical variables | All categorical features vs Churn |
| Cramér's V | Effect size — how *strong* is the relationship (0 to 1) | All categorical features |
| Point-biserial correlation | Relationship between binary outcome and continuous variable | tenure, MonthlyCharges, TotalCharges |
| 95% Confidence Intervals | Range confirming churn rate estimates are reliable | Key group comparisons |

### Chi-Square Results

| Variable | Cramér's V | Effect Size | Significant |
|---|---|---|---|
| Contract | 0.410 | Strong | Yes |
| OnlineSecurity | 0.347 | Strong | Yes |
| TechSupport | 0.343 | Strong | Yes |
| InternetService | 0.323 | Strong | Yes |
| PaymentMethod | 0.303 | Strong | Yes |
| OnlineBackup | 0.292 | Moderate | Yes |
| DeviceProtection | 0.282 | Moderate | Yes |
| StreamingMovies | 0.231 | Moderate | Yes |
| StreamingTV | 0.231 | Moderate | Yes |
| PaperlessBilling | 0.192 | Moderate | Yes |
| gender | 0.008 | Weak | **No** |
| PhoneService | 0.011 | Weak | **No** |

**Gender and PhoneService are statistically insignificant** — they were dropped from the predictive model.

### Point-Biserial Correlation Results

| Variable | Correlation | Direction | Strength | Significant |
|---|---|---|---|---|
| tenure | -0.352 | Higher = less churn | Strong | Yes |
| MonthlyCharges | +0.193 | Higher = more churn | Moderate | Yes |
| TotalCharges | -0.198 | Higher = less churn | Moderate | Yes |

### Confidence Intervals (Key Groups)

| Group | Churn Rate | 95% CI | N |
|---|---|---|---|
| Month-to-month | 42.7% | 41.2% – 44.3% | 3,875 |
| Two year | 2.8% | 2.0% – 3.6% | 1,695 |
| Fiber optic | 41.9% | 40.2% – 43.6% | 3,096 |
| Electronic check | 45.3% | 43.3% – 47.3% | 2,365 |
| Auto payment | 16.0% | 14.7% – 17.3% | 3,066 |

Confidence intervals between key groups do not overlap — confirming these differences are statistically real.

---

## Phase 4 — Machine Learning

**Objective:** Build a model that scores every customer by their individual churn probability.

### Feature Engineering
- Dropped gender and PhoneService (statistically insignificant from Phase 3)
- Binary encoded Yes/No columns
- One-hot encoded Contract, InternetService, PaymentMethod
- Dropped TotalCharges (redundant with tenure)
- Final feature set: **23 features**

### Train/Test Split
- 80/20 split with stratification to preserve 26.5% churn rate
- Train: 5,634 customers | Test: 1,409 customers

### Models

**Logistic Regression** — chosen for interpretability. Coefficients directly explain *why* a customer is predicted to churn and by how much. Used `class_weight='balanced'` to handle the imbalanced dataset (74%/26% split).

**Random Forest** — chosen to validate findings from a completely different algorithmic approach. If both models agree on feature importance, the signals are genuine.

### Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Test AUC | **0.839** | 0.817 |
| CV AUC (5-fold) | **0.845 ± 0.012** | 0.816 ± 0.014 |
| Churn Recall | **0.78** | 0.49 |
| Overall Accuracy | 0.74 | 0.78 |

**Logistic Regression wins for this use case.** In a retention problem, recall on churners is the critical metric — you need to *catch* churners before they leave. LR identifies 78% of actual churners vs Random Forest's 49%.

Random Forest's higher accuracy is misleading — it achieves it by predicting "not churned" more conservatively, which looks accurate but misses too many at-risk customers.

### Feature Importance — Both Models Agree

Top drivers confirmed by both Logistic Regression (coefficients) and Random Forest (importance scores):

| Rank | Feature | Direction |
|---|---|---|
| 1 | tenure | Longer tenure → lower churn risk |
| 2 | MonthlyCharges | Higher charges → higher churn risk |
| 3 | Contract_Month-to-month | Increases churn risk |
| 4 | InternetService_Fiber optic | Increases churn risk |
| 5 | PaymentMethod_Electronic check | Increases churn risk |
| 6 | OnlineSecurity | Having it reduces churn risk |
| 7 | Contract_Two year | Reduces churn risk |

When two fundamentally different model types independently rank the same features, those features are genuinely driving the behaviour.

---

## Phase 5 — Revenue & Retention Business Case

**Objective:** Translate model outputs into quantified business impact.

### Customer Risk Segmentation

| Segment | Customers | Avg Monthly Charges | Avg Tenure | Actual Churn Rate |
|---|---|---|---|---|
| Low Risk (0–30%) | 2,989 | $56.00 | 49 months | 4.4% |
| Medium Risk (30–50%) | 1,183 | $58.39 | 27 months | 20.8% |
| High Risk (50–70%) | 1,107 | $68.99 | 24 months | 34.4% |
| Critical Risk (70–100%) | 1,764 | $81.22 | 13 months | 62.9% |

### The Pricing Paradox

**Customers who pay the most are the most likely to leave.** Critical risk customers pay $81/month on average vs $56/month for low risk customers. This is a structural business problem — premium customers are not experiencing value that justifies their price point.

### Revenue at Risk

| Segment | Monthly Revenue | Revenue at Risk | % at Risk |
|---|---|---|---|
| Low Risk | $167,398 | $20,968 | 12.5% |
| Medium Risk | $69,076 | $27,667 | 40.1% |
| High Risk | $76,370 | $45,900 | 60.1% |
| Critical Risk | $143,273 | $117,879 | 82.3% |
| **Total** | **$456,117** | **$212,414** | **46.6%** |

**$2,548,968 in annual revenue is at risk** — nearly half the business's entire annual revenue.

### Retention ROI

Assuming targeted interventions with segment-appropriate discounts:

| Segment | Customers | Save Rate | Revenue Saved | Discount Cost | Net Value |
|---|---|---|---|---|---|
| Critical Risk | 1,764 | 50% | $859,632 | $171,926 | $687,706 |
| High Risk | 1,107 | 40% | $366,585 | $54,988 | $311,597 |
| Medium Risk | 1,183 | 30% | $248,671 | $24,867 | $223,804 |
| **Total** | **4,054** | | **$1,474,889** | **$251,781** | **$1,223,108** |

**Overall ROI: 486%** — spend $251k, save $1.2 million net.

### Priority Action List

2,871 customers flagged for retention intervention:

| Action | Customers |
|---|---|
| URGENT: New month-to-month + high charges — offer contract switch + discount | 1,506 |
| Contract switch offer — loyalty discount for annual commitment | 1,249 |
| Pricing review — consider plan downgrade or value-add offer | 103 |
| General retention outreach | 13 |

---

## Key Findings

1. **Contract type is the single strongest predictor of churn** (Cramér's V = 0.41). Month-to-month customers churn at 42.7% vs 2.8% for two-year customers.

2. **The first 12 months is the highest-risk window.** 47.7% of customers churn in year one. Every critical risk customer in the top 10 had tenure of 1-4 months.

3. **Premium customers are leaving fastest.** Critical risk customers pay $81/month average — 45% more than stable low-risk customers at $56/month. The business is losing its highest-value customers.

4. **Fiber optic has a systemic problem.** 41.9% churn on the premium internet service signals pricing or quality misalignment.

5. **Electronic check is a behavioral flag.** 45.3% churn vs 15-16% for automatic payment methods. Auto-pay customers are more financially committed and engaged.

6. **Product data alone cannot fully explain churn.** Even with 23 features and an 84% AUC model, the model cannot perfectly predict churn — consistent with findings from other analyses that external factors (sales interactions, pricing perception, competitive offers) drive significant conversion and retention behaviour beyond what usage data captures.

7. **$2.5M annual revenue at risk, $1.2M recoverable** with a structured retention program delivering 486% ROI.

---

## Business Recommendations

**Immediate (This Week)**
- Export and action the 1,506 URGENT customers — new month-to-month + high charges
- Phone outreach is more effective than email for critical risk customers
- Offer: 20% discount on annual contract commitment or 3 months free add-on (OnlineSecurity or TechSupport)

**Short Term (This Month)**
- Design a 90-day onboarding program for all new customers paying over $70/month
- Structured check-ins at day 7, day 30, and day 60 before the first billing anxiety point
- Incentivise electronic check customers to switch to automatic payment

**Strategic (This Quarter)**
- Conduct a fiber optic service quality and pricing audit
- Investigate whether fiber churn is driven by service issues, competitor pricing, or unmet expectations
- Consider restructuring month-to-month pricing to close the gap with annual plan value

---


## Dependencies

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
scikit-learn>=1.3
```





*IBM Telco Customer Churn Dataset | Analysis completed March 2026*
