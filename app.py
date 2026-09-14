"""
Lab-4: Interactive Statistical Dashboard — Medical Insurance Costs
====================================================================
Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st

# ---------------------------------------------------------------------------
# Page config & data loading
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Insurance Cost Analytics", layout="wide", page_icon="💊")

ALPHA = 0.05


@st.cache_data
def load_data():
    df = pd.read_csv("data/insurance.csv")
    return df


@st.cache_resource
def fit_model(data: pd.DataFrame):
    """Fit the full OLS model used across the app."""
    model = smf.ols(
        "charges ~ age + bmi + children + C(sex) + C(smoker) + C(region)",
        data=data,
    ).fit()
    return model


df = load_data()
model = fit_model(df)

st.title("💊 Medical Insurance Cost — Statistical Dashboard")
st.caption(
    "Dataset: Medical Cost Personal Datasets (Kaggle · mirichoi0218/insurance) — "
    f"{len(df):,} records"
)

tab1, tab2, tab3 = st.tabs(
    ["📊 Data Exploration", "🧪 Hypothesis Testing Lab", "🔮 Live Prediction & Diagnostics"]
)

# =============================================================================
# TAB 1 — DATA EXPLORATION
# =============================================================================
with tab1:
    st.header("Data Exploration")

    with st.sidebar:
        st.subheader("🔎 Filters")
        age_range = st.slider(
            "Age range", int(df.age.min()), int(df.age.max()),
            (int(df.age.min()), int(df.age.max())),
        )
        bmi_range = st.slider(
            "BMI range", float(df.bmi.min()), float(df.bmi.max()),
            (float(df.bmi.min()), float(df.bmi.max())),
        )
        regions = st.multiselect(
            "Region", sorted(df.region.unique()), default=sorted(df.region.unique())
        )
        smoker_filter = st.multiselect(
            "Smoker status", sorted(df.smoker.unique()), default=sorted(df.smoker.unique())
        )
        sex_filter = st.multiselect(
            "Sex", sorted(df.sex.unique()), default=sorted(df.sex.unique())
        )

    filtered = df[
        (df.age.between(*age_range))
        & (df.bmi.between(*bmi_range))
        & (df.region.isin(regions))
        & (df.smoker.isin(smoker_filter))
        & (df.sex.isin(sex_filter))
    ]

    st.markdown(f"**{len(filtered):,} records** match the current filters.")

    # Summary statistics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg. Charges", f"${filtered.charges.mean():,.0f}")
    c2.metric("Median Charges", f"${filtered.charges.median():,.0f}")
    c3.metric("Avg. BMI", f"{filtered.bmi.mean():.1f}")
    c4.metric("Avg. Age", f"{filtered.age.mean():.1f}")

    st.dataframe(filtered.describe().round(2), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox("Distribution of:", ["charges", "age", "bmi", "children"], index=0)
        fig_hist = px.histogram(
            filtered, x=metric, marginal="box", nbins=40, color="smoker",
            title=f"Distribution of {metric}", opacity=0.75,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        x_axis = st.selectbox("Scatter X-axis:", ["age", "bmi", "children"], index=1)
        fig_scatter = px.scatter(
            filtered, x=x_axis, y="charges", color="smoker", size="bmi",
            hover_data=["age", "sex", "region", "children"],
            title=f"{x_axis} vs charges",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Correlation matrix (numeric features)")
    corr = filtered[["age", "bmi", "children", "charges"]].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Average charges by region & smoking status")
    grp = filtered.groupby(["region", "smoker"])["charges"].mean().reset_index()
    fig_bar = px.bar(grp, x="region", y="charges", color="smoker", barmode="group")
    st.plotly_chart(fig_bar, use_container_width=True)

# =============================================================================
# TAB 2 — HYPOTHESIS TESTING LAB
# =============================================================================
with tab2:
    st.header("Hypothesis Testing Lab")
    st.write("Pick a numerical metric and a categorical factor, and the app will "
             "automatically choose and run the appropriate statistical test.")

    numeric_metrics = ["charges", "age", "bmi"]
    categorical_factors = ["smoker", "sex", "region"]

    colA, colB = st.columns(2)
    with colA:
        metric_sel = st.selectbox("Numerical metric:", numeric_metrics, key="ht_metric")
    with colB:
        factor_sel = st.selectbox("Categorical factor:", categorical_factors, key="ht_factor")

    groups_unique = df[factor_sel].unique()
    n_groups = len(groups_unique)

    st.markdown(f"**Groups in `{factor_sel}`:** {', '.join(map(str, groups_unique))} ({n_groups} groups)")

    if n_groups == 2:
        st.subheader(f"Two-Group Comparison: {metric_sel} by {factor_sel}")
        g1_label, g2_label = groups_unique
        g1 = df.loc[df[factor_sel] == g1_label, metric_sel]
        g2 = df.loc[df[factor_sel] == g2_label, metric_sel]

        st.latex(rf"H_0: \mu_{{{g1_label}}} = \mu_{{{g2_label}}} \qquad H_1: \mu_{{{g1_label}}} \neq \mu_{{{g2_label}}}")

        sh1 = stats.shapiro(g1.sample(min(500, len(g1)), random_state=1))
        sh2 = stats.shapiro(g2.sample(min(500, len(g2)), random_state=1))
        lev = stats.levene(g1, g2)

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.write("**Normality (Shapiro-Wilk)**")
            st.write(f"{g1_label}: W={sh1.statistic:.4f}, p={sh1.pvalue:.4g}")
            st.write(f"{g2_label}: W={sh2.statistic:.4f}, p={sh2.pvalue:.4g}")
        with res_col2:
            st.write("**Equal Variance (Levene)**")
            st.write(f"stat={lev.statistic:.4f}, p={lev.pvalue:.4g}")

        is_normal = (sh1.pvalue > ALPHA) and (sh2.pvalue > ALPHA)
        if is_normal:
            equal_var = lev.pvalue > ALPHA
            res = stats.ttest_ind(g1, g2, equal_var=equal_var)
            test_name = f"Two-Sample t-test (equal_var={equal_var})"
        else:
            res = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            test_name = "Mann-Whitney U test"

        decision = "Reject H0" if res.pvalue < ALPHA else "Fail to Reject H0"
        st.subheader(f"Result: {test_name}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Statistic", f"{res.statistic:.4f}")
        m2.metric("p-value", f"{res.pvalue:.4g}")
        m3.metric("Decision (α=0.05)", decision)

        fig_box = px.box(df, x=factor_sel, y=metric_sel, color=factor_sel, points="all")
        st.plotly_chart(fig_box, use_container_width=True)

    else:
        st.subheader(f"Multi-Group Comparison (ANOVA): {metric_sel} by {factor_sel}")
        st.latex(r"H_0: \mu_1 = \mu_2 = \dots = \mu_k \qquad H_1: \text{at least one mean differs}")

        groups = [df.loc[df[factor_sel] == g, metric_sel] for g in groups_unique]
        anova = stats.f_oneway(*groups)
        decision = "Reject H0" if anova.pvalue < ALPHA else "Fail to Reject H0"

        m1, m2, m3 = st.columns(3)
        m1.metric("F-statistic", f"{anova.statistic:.4f}")
        m2.metric("p-value", f"{anova.pvalue:.4g}")
        m3.metric("Decision (α=0.05)", decision)

        fig_box = px.box(df, x=factor_sel, y=metric_sel, color=factor_sel, points="all")
        st.plotly_chart(fig_box, use_container_width=True)

    st.divider()
    st.subheader("Bonus: Chi-Square Test of Independence (categorical vs categorical)")
    cat_a = st.selectbox("Factor A:", categorical_factors, index=0, key="chi_a")
    cat_b = st.selectbox("Factor B:", [c for c in categorical_factors if c != cat_a], index=0, key="chi_b")

    ctab = pd.crosstab(df[cat_a], df[cat_b])
    chi2, p_chi, dof, expected = stats.chi2_contingency(ctab)
    decision_chi = "Reject H0" if p_chi < ALPHA else "Fail to Reject H0"

    st.dataframe(ctab, use_container_width=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Chi-square", f"{chi2:.4f}")
    m2.metric("p-value", f"{p_chi:.4g}")
    m3.metric("Decision (α=0.05)", decision_chi)

# =============================================================================
# TAB 3 — LIVE PREDICTION & DIAGNOSTICS
# =============================================================================
with tab3:
    st.header("Live Prediction & Model Diagnostics")
    st.write("Model: `charges ~ age + bmi + children + sex + smoker + region` (OLS)")

    st.subheader("Model fit summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{model.rsquared:.4f}")
    c2.metric("Adjusted R²", f"{model.rsquared_adj:.4f}")
    c3.metric("N observations", f"{int(model.nobs)}")

    with st.expander("Full statsmodels summary table"):
        st.text(str(model.summary()))

    st.divider()
    st.subheader("🔮 Predict charges for a new individual")

    p1, p2, p3 = st.columns(3)
    with p1:
        in_age = st.slider("Age", 18, 64, 35)
        in_bmi = st.slider("BMI", 15.0, 55.0, 28.0, step=0.1)
    with p2:
        in_children = st.number_input("Children", 0, 5, 0)
        in_sex = st.radio("Sex", ["female", "male"], horizontal=True)
    with p3:
        in_smoker = st.radio("Smoker", ["no", "yes"], horizontal=True)
        in_region = st.selectbox("Region", sorted(df.region.unique()))

    new_point = pd.DataFrame({
        "age": [in_age], "bmi": [in_bmi], "children": [in_children],
        "sex": [in_sex], "smoker": [in_smoker], "region": [in_region],
    })

    pred = model.get_prediction(new_point)
    pred_summary = pred.summary_frame(alpha=ALPHA)

    point_pred = pred_summary["mean"].iloc[0]
    ci_low, ci_high = pred_summary["mean_ci_lower"].iloc[0], pred_summary["mean_ci_upper"].iloc[0]
    pi_low, pi_high = pred_summary["obs_ci_lower"].iloc[0], pred_summary["obs_ci_upper"].iloc[0]

    st.success(f"**Predicted charges: ${point_pred:,.2f}**")
    m1, m2 = st.columns(2)
    m1.info(f"95% Confidence Interval (mean charges): ${ci_low:,.2f} — ${ci_high:,.2f}")
    m2.warning(f"95% Prediction Interval (individual charges): ${pi_low:,.2f} — ${pi_high:,.2f}")

    st.divider()
    st.subheader("Gauss-Markov Diagnostic Plots")

    fitted = model.fittedvalues
    resid = model.resid

    d1, d2 = st.columns(2)
    with d1:
        fig_rvf = go.Figure()
        fig_rvf.add_trace(go.Scatter(x=fitted, y=resid, mode="markers",
                                      marker=dict(opacity=0.4, size=6), name="residuals"))
        fig_rvf.add_hline(y=0, line_dash="dash", line_color="red")
        fig_rvf.update_layout(title="Residuals vs Fitted Values",
                               xaxis_title="Fitted values", yaxis_title="Residuals")
        st.plotly_chart(fig_rvf, use_container_width=True)

    with d2:
        qq = sm.ProbPlot(resid)
        theo_q = qq.theoretical_quantiles
        samp_q = qq.sample_quantiles
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=theo_q, y=samp_q, mode="markers",
                                     marker=dict(opacity=0.5, size=6), name="residuals"))
        line_x = np.array([theo_q.min(), theo_q.max()])
        slope, intercept = np.polyfit(theo_q, samp_q, 1)
        fig_qq.add_trace(go.Scatter(x=line_x, y=slope * line_x + intercept,
                                     mode="lines", line=dict(color="red", dash="dash"), name="fit"))
        fig_qq.update_layout(title="Q-Q Plot of Residuals",
                              xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles")
        st.plotly_chart(fig_qq, use_container_width=True)

    from statsmodels.stats.stattools import jarque_bera, omni_normtest
    jb_stat, jb_p, _, _ = jarque_bera(resid)
    omni_stat, omni_p = omni_normtest(resid)

    m1, m2 = st.columns(2)
    m1.metric("Jarque-Bera p-value", f"{jb_p:.4g}")
    m2.metric("Omnibus p-value", f"{omni_p:.4g}")

    st.subheader("Multicollinearity — Variance Inflation Factor (VIF)")
    X_design = model.model.exog
    vif_data = pd.DataFrame({
        "feature": model.model.exog_names,
        "VIF": [variance_inflation_factor(X_design, i) for i in range(X_design.shape[1])],
    })
    st.dataframe(vif_data.round(3), use_container_width=True)
    st.caption("VIF > 5–10 typically signals problematic multicollinearity.")
