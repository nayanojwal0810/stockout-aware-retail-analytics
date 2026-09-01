
from pathlib import Path
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# STOCKOUT-AWARE RETAIL ANALYTICS
# Professional, story-first dashboard
# =========================================================

st.set_page_config(
    page_title="Stockout-Aware Retail Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# PROJECT PATHS
# =========================================================
# =========================================================
# PROJECT PATHS
# =========================================================
THIS_DIR = Path(__file__).resolve().parent

possible_roots = [
    THIS_DIR,
    THIS_DIR.parent,
]

ROOT = next(
    (
        p for p in possible_roots
        if (p / "data" / "processed" / "dashboard").exists()
    ),
    None,
)

if ROOT is None:
    raise FileNotFoundError(
        "Could not locate project root containing data/processed/dashboard."
    )

PROCESSED = (
    ROOT
    / "data"
    / "processed"
    / "dashboard"
)

asset_dirs = [
    THIS_DIR / "dashboard_assets",
    ROOT / "dashboard_assets",
]

ASSETS = next(
    (
        p for p in asset_dirs
        if p.exists()
    ),
    None,
)


# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    .topbar {
        padding: 1.35rem 1.5rem;
        border-radius: 18px;
        background:
            linear-gradient(135deg,#0D2338 0%,#183D5A 62%,#245E7C 100%);
        color: white;
        margin-bottom: 1rem;
    }

    .topbar h1 {
        margin: 0 0 .35rem 0;
        font-size: 2.1rem;
        letter-spacing: -.03em;
    }

    .topbar p {
        margin: 0;
        max-width: 1120px;
        font-size: 1rem;
        line-height: 1.55;
        opacity: .94;
    }

    .story-strip {
        padding: .9rem 1rem;
        border: 1px solid rgba(120,130,145,.22);
        border-radius: 14px;
        background: rgba(255,255,255,.03);
        height: 100%;
    }

    .story-strip .kicker {
        font-size: .70rem;
        font-weight: 800;
        letter-spacing: .08em;
        color: #667085;
    }

    .story-strip .title {
        font-size: 1rem;
        font-weight: 750;
        margin-top: .22rem;
    }

    .story-strip .body {
        font-size: .88rem;
        line-height: 1.42;
        color: #667085;
        margin-top: .35rem;
    }

    .takeaway {
        border-left: 5px solid #2E90FA;
        background: rgba(46,144,250,.08);
        border-radius: 10px;
        padding: .85rem 1rem;
        margin: .6rem 0 1rem 0;
    }

    .caveat {
        border-left: 4px solid #F79009;
        background: rgba(247,144,9,.08);
        border-radius: 9px;
        padding: .7rem .85rem;
        color: #7A2E0E;
        margin-top: .5rem;
    }

    .section-kicker {
        font-size: .72rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 800;
        color: #667085;
        margin-top: .3rem;
    }

    .mini-note {
        font-size: .78rem;
        color: #667085;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(120,130,145,.16);
        border-radius: 12px;
        padding: .75rem .8rem;
        background: rgba(255,255,255,.015);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data
def load_csv(filename, **kwargs):
    path = PROCESSED / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


@st.cache_data
def load_data():
    return {
        "exec": load_csv(
            "final_executive_summary.csv"
        ),
        "models": load_csv(
            "final_holdout_baseline_comparison.csv"
        ),
        "horizons": load_csv(
            "final_holdout_baseline_horizon_comparison.csv"
        ),
        "predictions": load_csv(
            "final_holdout_predictions.csv",
            parse_dates=["forecast_origin", "dt"],
        ),
        "recon_variants": load_csv(
            "demand_reconstruction_variant_summary.csv"
        ),
        "recon_rank": load_csv(
            "demand_reconstruction_rank_correlation.csv"
        ),
        "recon_full": load_csv(
            "demand_reconstruction_full_stockout_sensitivity.csv"
        ),
        "inventory_impact": load_csv(
            "corrected_inventory_policy_impact.csv"
        ),
        "inventory_summary": load_csv(
            "corrected_inventory_policy_summary.csv"
        ),
        "risk_band_summary": load_csv(
            "policy_risk_band_summary.csv"
        ),
        "risk_band_impact": load_csv(
            "policy_risk_band_impact.csv"
        ),
        "risk": load_csv(
            "operational_risk_prioritization.csv"
        ),
    }


def find_asset(stem):
    extensions = [
        ".png",
        ".svg",
        ".jpg",
        ".jpeg",
        ".webp",
    ]
    for ext in extensions:
        p = ASSETS / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def show_asset(stem):
    p = find_asset(stem)
    if p is None:
        st.info(
            f"Add `{stem}.png` or `{stem}.svg` to {ASSETS}"
        )
        return

    data = p.read_bytes()

    if data[:8] == bytes.fromhex(
        "89504E470D0A1A0A"
    ):
        st.image(
            data,
            use_container_width=True
        )
        return

    if data[:3] == bytes.fromhex("FFD8FF"):
        st.image(
            data,
            use_container_width=True
        )
        return

    if (
        len(data) >= 12
        and data[:4] == b"RIFF"
        and data[8:12] == b"WEBP"
    ):
        st.image(
            data,
            use_container_width=True
        )
        return

    # Valid SVG
    text = data.decode(
        "utf-8",
        errors="strict",
    ).lstrip()

    if (
        text.startswith("<svg")
        or (
            text.startswith("<?xml")
            and "<svg" in text[:2000]
        )
    ):
        encoded = base64.b64encode(
            data
        ).decode("ascii")

        st.markdown(
            f"""
            <div style="width:100%;overflow:hidden;border-radius:14px;">
                <img
                    src="data:image/svg+xml;base64,{encoded}"
                    style="width:100%;height:auto;display:block;"
                />
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    raise ValueError(
        f"Unsupported visual format: {p}"
    )


def first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# =========================================================
# LOAD + CORE METRICS
# =========================================================
try:
    D = load_data()
except Exception as exc:
    st.error("Dashboard data could not be loaded.")
    st.code(str(exc))
    st.stop()

E = D["exec"].iloc[0]
models = D["models"].copy()
horizons = D["horizons"].copy()
preds = D["predictions"].copy()
risk = D["risk"].copy()

final_wape = float(E["final_holdout_wape"])
baseline_wape = float(E["best_baseline_wape"])
improvement = float(E["wape_improvement_pct"])
horizon_wins = int(E["horizon_wins"])

top5_censored = float(
    E["top5_censored_demand_share_pct"]
)
top1_censored = float(
    E["top1_censored_demand_share_pct"]
)

critical = int(E["critical_series"])
high_priority = int(E["high_priority_series"])
monitor = int(E["monitor_series"])
standard = int(E["standard_series"])


# =========================================================
# NAVIGATION
# =========================================================
st.sidebar.title("Retail decision center")

page = st.sidebar.radio(
    "Explore",
    [
        "01 · The Bottom Line",
        "02 · Why Sales Mislead",
        "03 · Can We Forecast Better?",
        "04 · Where Is Demand Hidden?",
        "05 · Where Should Inventory Go?",
        "06 · Who Needs Attention?",
        "07 · Method & Reliability",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "Final holdout: 26 Jun 2024 → 2 Jul 2024"
)
st.sidebar.caption(
    "All dashboard metrics come from frozen processed artifacts."
)


# =========================================================
# PAGE 1 — THE BOTTOM LINE
# =========================================================
if page == "01 · The Bottom Line":

    st.markdown(
        """
        <div class="topbar">
            <h1>Stockout-Aware Retail Analytics</h1>
            <p>
            A retail item can look weak because customers did not want it,
            or because customers wanted it when it was unavailable.
            This project separates those two effects, forecasts the next seven days,
            and identifies where additional inventory protection is most valuable.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)

    story = [
        (
            "THE PROBLEM",
            "Sales are constrained by availability",
            "When a product is out of stock, recorded sales can stop even if customer demand remains."
        ),
        (
            "THE TEST",
            "Forecast on a future week we never touched",
            "The finalized model is evaluated on June 26–July 2, after every development choice was frozen."
        ),
        (
            "THE DECISION",
            "Protect the risky items first",
            "The inventory scenarios show much larger simulated benefit for high stockout-risk products."
        ),
    ]

    for col, (k, title, body) in zip(cols, story):
        with col:
            st.markdown(
                f"""
                <div class="story-strip">
                    <div class="kicker">{k}</div>
                    <div class="title">{title}</div>
                    <div class="body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Final-week forecast error",
        f"{final_wape:.4f}",
        help="WAPE: forecast error relative to total observed sales. Lower is better.",
    )
    k2.metric(
        "Better than best simple forecast",
        f"{improvement:.2f}%",
    )
    k3.metric(
        "Forecast days won",
        f"{horizon_wins}/7",
    )
    k4.metric(
        "Holdout observations with stockout",
        "40.98%",
    )

    st.markdown(
        '<div class="section-kicker">THE PROBLEM</div>',
        unsafe_allow_html=True,
    )
    st.subheader(
        "Why a retail forecast needs to understand stockouts"
    )

    show_asset("01_stockout_censoring")

    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="section-kicker">FORECAST TEST</div>',
            unsafe_allow_html=True,
        )
        st.subheader(
            "The model beats the strongest simple forecast"
        )

        plot_df = models[
            ["model", "WAPE"]
        ].copy()

        plot_df["label"] = (
            plot_df["model"]
            .replace(
                {
                    "Direct_Gradient_Boosting":
                        "Final model",
                    "Moving_Average_7":
                        "7-day moving average",
                    "Seasonal_Naive_7":
                        "Seasonal 7-day baseline",
                    "Naive_1":
                        "Previous-day baseline",
                }
            )
        )

        fig = px.bar(
            plot_df.sort_values("WAPE"),
            x="WAPE",
            y="label",
            orientation="h",
            text="WAPE",
        )

        fig.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside",
            cliponaxis=False,
        )

        fig.update_layout(
            title="Lower error is better",
            xaxis_title="WAPE",
            yaxis_title="",
            height=360,
            margin=dict(
                l=20,
                r=55,
                t=60,
                b=25,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        st.markdown(
            '<div class="section-kicker">INVENTORY TEST</div>',
            unsafe_allow_html=True,
        )
        st.subheader(
            "The benefit of extra protection rises with risk"
        )

        rb = D["risk_band_impact"].copy()

        rb = rb[
            (
                rb["service_policy"] == "SL90"
            )
            &
            (
                np.isclose(
                    rb["risk_multiplier"],
                    0.20,
                )
            )
        ].copy()

        risk_order = {
            "LOW": 0,
            "MEDIUM": 1,
            "HIGH": 2,
            "VERY_HIGH": 3,
        }

        rb["sort_order"] = (
            rb["risk_band"].map(
                risk_order
            )
        )

        rb = rb.sort_values("sort_order")

        fig = px.bar(
            rb,
            x="risk_band",
            y="shortage_reduction_pct",
            text="shortage_reduction_pct",
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            cliponaxis=False,
        )

        fig.update_layout(
            title="Same +20% protection scenario",
            xaxis_title="Pre-forecast stockout risk",
            yaxis_title="Simulated shortage reduction",
            height=360,
            margin=dict(
                l=20,
                r=35,
                t=60,
                b=25,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown(
        f"""
        <div class="takeaway">
        <b>What this says:</b> the strongest story is not “add more inventory.”
        It is “use better demand information to decide <i>where</i> protection is worth testing.”
        The final forecast improves WAPE by <b>{improvement:.2f}%</b>, while the
        risk-policy analysis shows a much larger simulated shortage reduction as
        stockout risk rises.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# PAGE 2 — WHY SALES MISLEAD
# =========================================================
elif page == "02 · Why Sales Mislead":

    st.markdown(
        """
        <div class="topbar">
            <h1>Why Sales Can Be Misleading</h1>
            <p>
            The first analytical question is not “Which model wins?”
            It is “Is the sales signal itself trustworthy when products are unavailable?”
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_asset("02_why_sales_mislead")

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Holdout rows with any stockout",
        "40.98%",
    )
    c2.metric(
        "Full-stockout holdout rows",
        "12,541",
    )
    c3.metric(
        "Full-stockout observed sales",
        "0.0341",
    )

    st.divider()

    st.markdown(
        """
        <div class="section-kicker">THE KEY DISTINCTION</div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            ### What the database records

            **Observed sales**

            The number of units that actually left the shelf.

            ### What planning needs

            **Underlying demand**

            The amount customers may have wanted to buy if the product had been available.
            """
        )

    with right:
        full = D["recon_full"]
        primary = full[
            full["variant"] == "primary"
        ].iloc[0]

        example = pd.DataFrame(
            {
                "Measure": [
                    "Observed sales",
                    "Estimated underlying demand",
                ],
                "Units": [
                    float(
                        primary[
                            "observed_sales_mean"
                        ]
                    ),
                    float(
                        primary[
                            "estimated_demand_mean"
                        ]
                    ),
                ],
            }
        )

        fig = px.bar(
            example,
            x="Measure",
            y="Units",
            text="Units",
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
        )

        fig.update_layout(
            title="Full-stockout example",
            xaxis_title="",
            yaxis_title="Units",
            height=360,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown(
        """
        <div class="takeaway">
        <b>Interpretation:</b> during a full stockout, observed sales can collapse
        even though the model estimates materially higher underlying demand.
        That is why the project treats stockouts as a demand-censoring problem.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Estimated underlying demand is a counterfactual model output, not directly observed lost sales."
    )


# =========================================================
# PAGE 3 — CAN WE FORECAST BETTER?
# =========================================================
elif page == "03 · Can We Forecast Better?":

    st.markdown(
        """
        <div class="topbar">
            <h1>Can We Forecast Better?</h1>
            <p>
            After accounting for the data-generating problem, the final test is simple:
            does the chosen model beat straightforward forecasting methods on a future
            week that was never used for tuning?
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Final model WAPE",
        f"{final_wape:.4f}",
    )
    c2.metric(
        "Best baseline WAPE",
        f"{baseline_wape:.4f}",
    )
    c3.metric(
        "Relative improvement",
        f"{improvement:.2f}%",
    )
    c4.metric(
        "Days with best WAPE",
        f"{horizon_wins}/7",
    )

    st.markdown(
        '<div class="section-kicker">ONE FAIR COMPARISON</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        models[
            [
                "model",
                "MAE",
                "WAPE",
                "RMSE",
            ]
        ]
        .sort_values("WAPE")
        .round(4),
        hide_index=True,
        use_container_width=True,
    )

    h = horizons.copy()

    fig = go.Figure()

    for col, label in [
        (
            "Naive_1_WAPE",
            "Previous-day baseline",
        ),
        (
            "Seasonal_Naive_7_WAPE",
            "Seasonal 7-day baseline",
        ),
        (
            "Moving_Average_7_WAPE",
            "7-day moving average",
        ),
        (
            "Direct_Gradient_Boosting_WAPE",
            "Final model",
        ),
    ]:
        fig.add_trace(
            go.Scatter(
                x=h["horizon"],
                y=h[col],
                mode="lines+markers",
                name=label,
            )
        )

    fig.update_layout(
        title="Forecast error stays lower across all seven horizons",
        xaxis_title="Days ahead",
        yaxis_title="WAPE — lower is better",
        height=470,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # Actual error distribution
    preds_local = preds.copy()
    preds_local["error"] = (
        preds_local["sale_amount"]
        -
        preds_local["prediction"]
    )

    err = preds_local["error"]

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            err,
            x="error",
            nbins=70,
            title="Final holdout forecast errors",
        )

        fig.update_layout(
            xaxis_title="Actual − forecast",
            yaxis_title="Observations",
            height=350,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        st.markdown(
            """
            <div class="section-kicker">WHY THIS MATTERS</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            ### The final model adds value

            On the untouched final week:

            **{improvement:.2f}% lower WAPE**

            than the strongest simple baseline.

            It also wins on **all seven forecast days**.

            That means the machine-learning layer is not merely more complicated;
            it demonstrated measurable incremental forecasting value on future data.
            """
        )

    st.markdown(
        """
        <div class="caveat">
        WAPE is a measure of observed-sales forecast error. It should not be
        interpreted as direct measurement of forecast quality for true demand
        during full stockouts, where observed sales are availability-constrained.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# PAGE 4 — WHERE IS DEMAND HIDDEN?
# =========================================================
elif page == "04 · Where Is Demand Hidden?":

    st.markdown(
        """
        <div class="topbar">
            <h1>Where Is Demand Hidden?</h1>
            <p>
            The reconstruction layer estimates the demand signal that may have been
            suppressed when stockouts constrained recorded sales.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_asset("02_demand_reconstruction")

    primary = D["recon_variants"][
        D["recon_variants"]["variant"] == "primary"
    ].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Stockout observations used",
        f"{int(primary['stockout_rows']):,}",
    )
    c2.metric(
        "Mean observed sales",
        f"{float(primary['mean_observed_sales']):.3f}",
    )
    c3.metric(
        "Mean estimated demand",
        f"{float(primary['mean_estimated_demand']):.3f}",
    )

    st.markdown(
        '<div class="section-kicker">HOW STABLE IS THE ESTIMATE?</div>',
        unsafe_allow_html=True,
    )

    variants = D["recon_variants"][
        [
            "variant",
            "mean_estimated_demand",
            "mean_estimated_gap",
            "total_estimated_gap",
            "p95_estimated_demand",
            "p99_estimated_demand",
        ]
    ].copy()

    st.dataframe(
        variants.round(4),
        hide_index=True,
        use_container_width=True,
    )

    # Sensitivity chart
    plot_var = variants[
        [
            "variant",
            "mean_estimated_demand",
        ]
    ].copy()

    fig = px.bar(
        plot_var,
        x="variant",
        y="mean_estimated_demand",
        text="mean_estimated_demand",
        title="Mean estimated demand under alternative assumptions",
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    fig.update_layout(
        xaxis_title="Reconstruction variant",
        yaxis_title="Mean estimated demand",
        height=390,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    r1, r2, r3 = st.columns(3)
    r1.metric(
        "Minimum rank correlation",
        f"{float(E['minimum_reconstruction_rank_corr']):.4f}",
    )
    r2.metric(
        "Minimum Top-5% overlap",
        f"{float(E['minimum_top5_overlap_pct']):.1f}%",
    )
    r3.metric(
        "Minimum Top-1% overlap",
        f"{float(E['minimum_top1_overlap_pct']):.1f}%",
    )

    st.markdown(
        """
        <div class="takeaway">
        <b>What survives the stress test:</b> the broad prioritization is robust,
        while the exact extreme Top-1% is more sensitive. The right interpretation is
        “directionally reliable for planning,” not “perfectly recovered lost sales.”
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "True counterfactual demand during genuine stockouts is unobserved."
    )


# =========================================================
# PAGE 5 — WHERE SHOULD INVENTORY GO?
# =========================================================
elif page == "05 · Where Should Inventory Go?":

    st.markdown(
        """
        <div class="topbar">
            <h1>Where Should Inventory Go?</h1>
            <p>
            The analysis does not recommend adding inventory everywhere.
            It tests how much additional protection changes simulated shortage,
            and whether the benefit is concentrated in high-risk segments.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_asset("03_targeted_inventory")

    service = st.selectbox(
        "Planning service-level scenario",
        [
            "SL80",
            "SL90",
            "SL95",
        ],
        index=1,
        help=(
            "Service-level target used in the empirical safety-stock scenario. "
            "This is a planning assumption, not a guarantee."
        ),
    )

    impact = D["inventory_impact"].copy()

    s = impact[
        impact["service_policy"] == service
    ].sort_values("risk_multiplier")

    scenario = s[
        np.isclose(
            s["risk_multiplier"],
            0.20,
        )
    ].iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Inventory increase",
        f"{float(scenario['inventory_increase_pct']):.2f}%",
    )
    c2.metric(
        "Simulated shortage reduction",
        f"{float(scenario['shortage_reduction_pct']):.2f}%",
    )
    c3.metric(
        "Fill-rate improvement",
        f"{float(scenario['fill_rate_improvement_pp']):.3f} pp",
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=s["inventory_increase_pct"],
            y=s["shortage_reduction_pct"],
            mode="lines+markers+text",
            text=[
                f"{x:.0f}%"
                for x in s["risk_multiplier"]
            ],
            textposition="top center",
            name=service,
        )
    )

    fig.update_layout(
        title=f"{service}: more protection versus simulated shortage reduction",
        xaxis_title="Inventory increase vs base (%)",
        yaxis_title="Simulated shortage reduction (%)",
        height=430,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    rb = D["risk_band_impact"].copy()

    focus = rb[
        (
            rb["service_policy"] == service
        )
        &
        (
            np.isclose(
                rb["risk_multiplier"],
                0.20,
            )
        )
    ].copy()

    risk_order = {
        "LOW": 0,
        "MEDIUM": 1,
        "HIGH": 2,
        "VERY_HIGH": 3,
    }

    focus["sort_order"] = (
        focus["risk_band"]
        .map(risk_order)
    )

    focus = focus.sort_values(
        "sort_order"
    )

    fig = px.bar(
        focus,
        x="risk_band",
        y="shortage_reduction_pct",
        text="shortage_reduction_pct",
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_layout(
        title=f"{service}: same +20% protection, different risk bands",
        xaxis_title="Stockout-risk band before forecast",
        yaxis_title="Simulated shortage reduction (%)",
        height=420,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.dataframe(
        focus[
            [
                "risk_band",
                "inventory_increase_pct",
                "shortage_reduction_pct",
                "fill_rate_improvement_pp",
            ]
        ].round(4),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="takeaway">
        <b>Decision:</b> the same level of extra protection has dramatically different
        simulated value depending on pre-forecast stockout risk. That supports targeted
        protection rather than a uniform inventory increase.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Simulation only: no historical inventory position, lead time, holding cost or lost-sales cost is available."
    )


# =========================================================
# PAGE 6 — WHO NEEDS ATTENTION?
# =========================================================
elif page == "06 · Who Needs Attention?":

    st.markdown(
        """
        <div class="topbar">
            <h1>Who Needs Attention?</h1>
            <p>
            The final risk layer converts demand, stockout exposure and forecast risk
            into a ranked operational queue — so the analysis ends with specific
            store-product combinations rather than a generic recommendation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Critical", f"{critical:,}")
    c2.metric("High priority", f"{high_priority:,}")
    c3.metric("Monitor", f"{monitor:,}")
    c4.metric("Standard", f"{standard:,}")

    st.markdown(
        '<div class="section-kicker">FOCUS THE PORTFOLIO</div>',
        unsafe_allow_html=True,
    )

    segments = sorted(
        risk[
            "action_segment"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    defaults = [
        x for x in [
            "CRITICAL",
            "HIGH_PRIORITY",
            "MONITOR",
        ]
        if x in segments
    ]

    selected = st.multiselect(
        "Priority segment",
        segments,
        default=defaults,
    )

    c1, c2 = st.columns(2)

    with c1:
        store_id = st.number_input(
            "Store ID (0 = all)",
            min_value=0,
            value=0,
            step=1,
        )

    with c2:
        product_id = st.number_input(
            "Product ID (0 = all)",
            min_value=0,
            value=0,
            step=1,
        )

    filtered = risk[
        risk[
            "action_segment"
        ].isin(selected)
    ].copy()

    if store_id > 0 and "store_id" in filtered.columns:
        filtered = filtered[
            filtered["store_id"] == store_id
        ]

    if product_id > 0 and "product_id" in filtered.columns:
        filtered = filtered[
            filtered["product_id"] == product_id
        ]

    rank_col = (
        "operational_rank"
        if "operational_rank"
        in filtered.columns
        else "operational_risk_score"
    )

    filtered = filtered.sort_values(
        rank_col
    )

    cols = [
        "operational_rank",
        "store_id",
        "product_id",
        "recent_mean_adjusted_demand",
        "recent_stockout_day_rate",
        "recent_full_stockout_rate",
        "recent_estimated_censored_demand",
        "daily_censored_demand",
        "censored_demand_rate",
        "forecast_mae",
        "mean_underforecast",
        "p95_underforecast",
        "operational_risk_score",
        "action_segment",
    ]

    cols = [
        c for c in cols
        if c in filtered.columns
    ]

    st.write(
        f"Showing up to 500 rows from {len(filtered):,} filtered series."
    )

    st.dataframe(
        filtered[
            cols
        ].head(500).round(4),
        hide_index=True,
        use_container_width=True,
        height=560,
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.metric(
            "Top 1% share of estimated censored demand",
            f"{top1_censored:.2f}%",
        )

    with right:
        st.metric(
            "Top 5% share of estimated censored demand",
            f"{top5_censored:.2f}%",
        )

    concentration = pd.DataFrame(
        {
            "Portfolio slice": [
                "Top 1%",
                "Top 5%",
            ],
            "Estimated censored-demand share": [
                top1_censored,
                top5_censored,
            ],
        }
    )

    fig = px.bar(
        concentration,
        x="Portfolio slice",
        y="Estimated censored-demand share",
        text="Estimated censored-demand share",
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    fig.update_layout(
        title="The potential demand burden is concentrated",
        xaxis_title="",
        yaxis_title="Share (%)",
        height=360,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "These are investigation priorities based on historical evidence; they are not confirmed root causes of stockouts."
    )


# =========================================================
# PAGE 7 — METHOD & RELIABILITY
# =========================================================
elif page == "07 · Method & Reliability":

    st.markdown(
        """
        <div class="topbar">
            <h1>Method & Reliability</h1>
            <p>
            This page is intentionally separated from the business story.
            It shows how the conclusions were protected against leakage,
            overstatement and premature use of the final holdout.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.graphviz_chart(
        """
        digraph G {
            rankdir=LR;
            node [
                shape=box,
                style="rounded,filled",
                fontname="Arial",
                color="#CBD5E1",
                fillcolor="#F8FAFC"
            ];

            A [label="Retail sales + stockout history"];
            B [label="Data quality + leakage checks"];
            C [label="Stockout diagnosis"];
            D [label="Cross-fitted demand reconstruction"];
            E [label="7-day direct forecasting"];
            F [label="Empirical uncertainty calibration"];
            G [label="Inventory scenario analysis"];
            H [label="Operational risk queue"];
            I [label="Untouched final holdout"];

            A -> B -> C;
            C -> D;
            C -> E;
            D -> F;
            E -> F;
            F -> G -> H;
            E -> I;
        }
        """
    )

    st.subheader("Final evidence")

    evidence = pd.DataFrame(
        {
            "Measure": [
                "Final holdout WAPE",
                "Best baseline WAPE",
                "Improvement vs baseline",
                "Forecast horizons won",
                "Minimum reconstruction rank correlation",
                "Minimum Top-5% overlap",
                "Minimum Top-1% overlap",
            ],
            "Result": [
                f"{final_wape:.4f}",
                f"{baseline_wape:.4f}",
                f"{improvement:.2f}%",
                f"{horizon_wins}/7",
                f"{float(E['minimum_reconstruction_rank_corr']):.4f}",
                f"{float(E['minimum_top5_overlap_pct']):.2f}%",
                f"{float(E['minimum_top1_overlap_pct']):.2f}%",
            ],
        }
    )

    st.dataframe(
        evidence,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Reliability gates that passed")

    gates = pd.DataFrame(
        {
            "Check": [
                "Temporal leakage control",
                "Cross-fitted reconstruction",
                "Final holdout untouched during tuning",
                "Final holdout baseline comparison",
                "Reconstruction sensitivity stress test",
                "Empirical uncertainty calibration",
                "Risk-policy sensitivity",
            ],
            "Status": [
                "PASS",
                "PASS",
                "PASS",
                "PASS",
                "PASS WITH CAUTION",
                "PASS",
                "PASS",
            ],
        }
    )

    st.dataframe(
        gates,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("What we can and cannot conclude")

    left, right = st.columns(2)

    with left:
        st.markdown(
            """
            **Supported by the analysis**

            - The finalized forecast adds measurable predictive value on the untouched holdout.
            - Stockouts can censor the observed demand signal.
            - Cross-fitted reconstruction provides a directional counterfactual estimate.
            - Higher-risk segments show larger simulated policy benefit.
            - The risk layer produces a targeted operational queue.
            """
        )

    with right:
        st.markdown(
            """
            **Not supported by the data**

            - True lost sales were directly observed.
            - Reconstructed demand is ground truth.
            - A +20% inventory increase is globally optimal.
            - Financial savings were proven.
            - The policy causally reduces stockouts.
            """
        )

    st.caption(
        "The inventory results are scenario simulations because the dataset does not contain actual inventory positions, replenishment lead times or costs."
    )


# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "Stockout-Aware Retail Analytics · 2024-03-28 → 2024-07-02 data window · "
    "Final holdout: 2024-06-26 → 2024-07-02"
)
