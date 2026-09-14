# Lab-4: Insurance Cost Analysis & Dashboard

**M.Sc. Data Science — Semester 1**

**Live App:** https://202618034rutujapolojwards602-6mfe7sky4jk7lby9wfmy6k.streamlit.app/

**Dataset:** Medical Cost Personal Datasets (Kaggle, [mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance))

---

## Project Structure

```
.
├── app.py              # Streamlit dashboard (3 tabs)
├── analysis.py         # EDA, hypothesis tests, regression, diagnostics
├── requirements.txt
├── data/
│   └── insurance.csv   # 1,338 records, 7 columns
├── results/             # Output plots and tables from analysis.py
└── README.md
```

## Dataset

| Column   | Type     | Description                                |
| -------- | -------- | ------------------------------------------ |
| age      | number   | Age of the person                          |
| sex      | category | male / female                              |
| bmi      | number   | Body Mass Index                            |
| children | number   | Number of dependents                       |
| smoker   | category | yes / no                                   |
| region   | category | northeast, northwest, southeast, southwest |
| charges  | number   | Medical cost billed (target variable)      |

1,338 rows, no missing values.

## How to Run

**Local setup:**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python analysis.py              # runs the statistical analysis
streamlit run app.py            # launches the dashboard
```

**Google Colab:** upload `insurance.csv`, `app.py`, and `analysis.py`, run `!pip install -r requirements.txt`, then run `analysis.py` directly.

## What We Found

**Smokers vs non-smokers:** Smokers pay about 3.8x more on average ($32,050 vs $8,434). We used a Mann-Whitney U test (since the data wasn't normally distributed) and the difference is statistically significant (p < 0.001).

**Charges across regions:** A one-way ANOVA shows charges differ significantly by region (p = 0.031). Southeast has the highest average charges, southwest the lowest.

**Smoking vs region:** A chi-square test shows smoking rates are similar across all regions (p = 0.062, not significant).

**Regression model:** `charges ~ age + bmi + children + sex + smoker + region`

| Predictor          | Effect on charges | Significant?                      |
| ------------------ | ----------------- | --------------------------------- |
| Age                | +$257 per year    | Yes                               |
| BMI                | +$339 per unit    | Yes                               |
| Children           | +$476 per child   | Yes                               |
| Sex (male)         | -$131             | No                                |
| Smoker (yes)       | +$23,849          | Yes (by far the strongest effect) |
| Region (southeast) | -$1,035           | Yes                               |
| Region (southwest) | -$960             | Yes                               |

The model explains about 75% of the variation in charges (R² = 0.751).

**Model checks:**

- Residuals show some fanning out (mild heteroscedasticity)
- Residuals are not perfectly normal (right-skewed, like the charges data itself)
- No multicollinearity problems (all VIF values under 2)

**Bottom line:** Smoking status is by far the biggest driver of medical costs, followed by BMI and age. Sex doesn't matter much. A log-transformed version of charges would likely fit the normality assumption better — that option is available to explore in the dashboard.

## Dashboard Tabs

1. **Data Exploration** — filters for age, BMI, region, smoker status; interactive charts and summary stats
2. **Hypothesis Testing Lab** — pick any metric and category, the app runs the right test automatically and shows the result
3. **Live Prediction & Diagnostics** — enter a person's details to get a predicted cost with confidence interval, plus residual plots and VIF table

## Notes

- Reference categories in the model: female, non-smoker, northeast
- All statistical tests use a significance level of 0.05
