"""
Dark Store Inventory Intelligence System
Run AFTER running the Jupyter notebook (which saves models to models/ folder):
    streamlit run app.py
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ── PATHS ─────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
MODELS_DIR   = os.path.join(BASE_DIR, "models")
METRICS_DIR  = os.path.join(BASE_DIR, "metrics")
DATA_PATH    = os.path.join(BASE_DIR, "Data", "dark_store_inventory_raw.csv")

# ── STORE AND SKU METADATA ────────────────────────────────────────
STORE_META = {
    "DS_MUM_01": {"city": "Mumbai",    "area": "Andheri",     "area_type": "Residential", "income_level": "High"},
    "DS_MUM_02": {"city": "Mumbai",    "area": "Dharavi",     "area_type": "Mixed",       "income_level": "Low"},
    "DS_PUN_01": {"city": "Pune",      "area": "Koregaon",    "area_type": "Commercial",  "income_level": "High"},
    "DS_PUN_02": {"city": "Pune",      "area": "Hadapsar",    "area_type": "Residential", "income_level": "Medium"},
    "DS_BLR_01": {"city": "Bengaluru", "area": "Koramangala", "area_type": "Commercial",  "income_level": "High"},
}

SKU_META = {
    "MILK_500ML":    {"category": "Dairy",       "price": 28},
    "BREAD_WH":      {"category": "Bakery",      "price": 45},
    "EGGS_12PK":     {"category": "Dairy",       "price": 72},
    "RICE_1KG":      {"category": "Staples",     "price": 60},
    "ATTA_5KG":      {"category": "Staples",     "price": 220},
    "TOMATO_500G":   {"category": "Vegetables",  "price": 30},
    "ONION_1KG":     {"category": "Vegetables",  "price": 40},
    "BANANA_DOZ":    {"category": "Fruits",      "price": 50},
    "CHIPS_LAYS":    {"category": "Snacks",      "price": 20},
    "BISCUIT_PK":    {"category": "Snacks",      "price": 30},
    "COKE_2L":       {"category": "Beverages",   "price": 95},
    "WATER_1L":      {"category": "Beverages",   "price": 20},
    "SHAMPOO_SM":    {"category": "PersonalCare","price": 99},
    "SOAP_BAR":      {"category": "PersonalCare","price": 35},
    "DIAPERS_SM":    {"category": "BabyCare",    "price": 350},
    "MAGGI_NOOD":    {"category": "Packaged",    "price": 14},
    "PANEER_200G":   {"category": "Dairy",       "price": 85},
    "BUTTER_100G":   {"category": "Dairy",       "price": 55},
    "DETERGENT_1KG": {"category": "Household",   "price": 110},
    "CURD_400G":     {"category": "Dairy",       "price": 42},
}

BLUE  = "#2563EB"
RED   = "#DC2626"
GREEN = "#16A34A"
NOMINAL_COLS = ["store_id", "city", "area", "area_type", "sku_id", "category", "weather", "season"]

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Dark Store Inventory Intelligence",
    page_icon="🏪",
    layout="wide",
)

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
.result-box  { border-radius: 10px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }
.high-risk   { background: #FEF2F2; border: 2px solid #DC2626; }
.low-risk    { background: #F0FDF4; border: 2px solid #16A34A; }
.risk-title  { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.3rem; }
.risk-sub    { font-size: 0.9rem; color: #475569; }
.mini-row    { display: flex; gap: 1rem; margin-top: 1rem; }
.mini-card   { flex:1; background:white; border:1px solid #E2E8F0; border-radius:8px; padding:0.8rem; text-align:center; }
.mini-label  { font-size:0.7rem; font-weight:600; color:#64748B; text-transform:uppercase; letter-spacing:0.04em; }
.mini-value  { font-size:1.5rem; font-weight:700; color:#0F172A; }
.info-box    { background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px; padding:0.7rem 1rem; font-size:0.84rem; color:#1E40AF; margin-bottom:1rem; }
.auto-badge  { background:#F0FDF4; border:1px solid #BBF7D0; border-radius:6px; padding:0.4rem 0.7rem; font-size:0.78rem; color:#166534; display:inline-block; margin-bottom:0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── LOAD MODELS ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        sm  = joblib.load(os.path.join(MODELS_DIR, "model_stockout.pkl"))
        rm  = joblib.load(os.path.join(MODELS_DIR, "model_reorder.pkl"))
        enc = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
        fc  = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
        im  = joblib.load(os.path.join(MODELS_DIR, "income_map.pkl"))
        return sm, rm, enc, fc, im, None
    except FileNotFoundError as e:
        return None, None, None, None, None, str(e)


# ── LOAD HISTORICAL DEMAND STATS (auto-fill for predictor) ────────
@st.cache_data
def load_demand_stats():
    """
    Pre-compute historical demand stats per store+SKU from raw data.
    Used to auto-fill rolling averages and std — user does not type these.
    """
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df['units_sold'] = df['units_sold'].abs()
    df['inventory_level'] = df['inventory_level'].replace(9999, np.nan)

    stats = df.groupby(['store_id', 'sku_id']).agg(
        avg_demand     = ('units_sold', 'mean'),
        demand_std     = ('units_sold', 'std'),
        last_inventory = ('inventory_level', 'last'),
        last_units_sold= ('units_sold', 'last'),
    ).round(2)
    return stats


# ── LOAD METRICS ──────────────────────────────────────────────────
@st.cache_data
def load_metrics():
    path = os.path.join(METRICS_DIR, "model_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


stockout_model, reorder_model, encoders, feature_cols, income_map, load_error = load_models()
demand_stats = load_demand_stats()
metrics      = load_metrics()

if load_error:
    st.error("**Models not found.** Run the Jupyter notebook first — it saves models to the `models/` folder.")
    st.stop()


# ── BUILD INPUT ROW ───────────────────────────────────────────────
def build_input_row(store_id, sku_id, inventory_level, units_sold_today,
                    lead_time, discount_pct, weather, season,
                    is_weekend, is_holiday,
                    rolling_avg_7d, rolling_avg_30d,
                    rolling_std_7d, rolling_std_30d):
    """
    Builds a single-row DataFrame with all features the model expects.
    Rolling averages and std come from historical data, not user input.
    """
    store = STORE_META[store_id]
    sku   = SKU_META[sku_id]

    season_to_month = {"Summer": 4, "Monsoon": 7, "Festive": 10, "Winter": 1}
    month       = season_to_month[season]
    is_ipl      = 1 if month in [4, 5] else 0

    row = {
        "store_id":          store_id,
        "city":              store["city"],
        "area":              store["area"],
        "area_type":         store["area_type"],
        "income_level":      income_map[store["income_level"]],
        "sku_id":            sku_id,
        "category":          sku["category"],
        "price":             sku["price"],
        "discount_pct":      discount_pct,
        "weather":           weather,
        "is_weekend":        int(is_weekend),
        "is_holiday":        int(is_holiday),
        "units_sold":        units_sold_today,
        "inventory_level":   inventory_level,
        "lead_time_days":    lead_time,
        "order_placed":      0,
        "day_of_week":       5 if is_weekend else 2,
        "month":             month,
        "is_month_start":    0,
        "is_month_end":      0,
        "is_ipl_season":     is_ipl,
        "season":            season,
        "rolling_avg_7d":    rolling_avg_7d,
        "rolling_avg_30d":   rolling_avg_30d,
        "rolling_std_7d":    rolling_std_7d,
        "rolling_std_30d":   rolling_std_30d,
        "lag_sales_1d":      units_sold_today,
        "lag_sales_7d":      rolling_avg_7d,
        "lag_sales_14d":     rolling_avg_7d,
        "lag_inventory_1d":  inventory_level + units_sold_today,
        "sales_spike_flag":  int(units_sold_today > 1.5 * rolling_avg_7d),
    }

    df_row = pd.DataFrame([row])

    for col in NOMINAL_COLS:
        if col in df_row.columns and col in encoders:
            le  = encoders[col]
            val = df_row[col].astype(str).values[0]
            df_row[col] = le.transform([val])[0] if val in le.classes_ else 0

    for col in feature_cols:
        if col not in df_row.columns:
            df_row[col] = 0

    return df_row[feature_cols]


# ── SIDEBAR NAVIGATION ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏪 Dark Store Intelligence")
    st.markdown("---")
    page = st.radio("", [
        "Stockout Predictor",
        "Batch Prediction",
        "Model Performance",
    ], label_visibility="collapsed")


# ═════════════════════════════════════════════════════════════════
# PAGE 1 — STOCKOUT PREDICTOR
# ═════════════════════════════════════════════════════════════════
if page == "Stockout Predictor":

    st.markdown("# Inventory Stockout Predictor")
    st.markdown("Predict whether a product will go out of stock within 48 hours and get a reorder recommendation.")
    st.markdown("---")

    with st.sidebar:
        st.markdown("### Input Inventory Data")

        store_id = st.selectbox("Store ID", list(STORE_META.keys()))
        sku_id   = st.selectbox("SKU ID",   list(SKU_META.keys()))

        s = STORE_META[store_id]
        st.caption(f"📍 {s['area']}, {s['city']} · {s['area_type']} · Income: {s['income_level']}")
        st.caption(f"🏷️ Category: {SKU_META[sku_id]['category']} · Price: ₹{SKU_META[sku_id]['price']}")
        st.markdown("---")

        # ── Auto-fill demand stats from historical data ───────────
        if (store_id, sku_id) in demand_stats.index:
            hist = demand_stats.loc[(store_id, sku_id)]
            default_inv   = max(0, int(hist['last_inventory']) if not np.isnan(hist['last_inventory']) else 80)
            default_sold  = max(0, int(hist['last_units_sold']))
            rolling_avg_7d  = round(hist['avg_demand'], 2)
            rolling_avg_30d = round(hist['avg_demand'], 2)
            rolling_std_7d  = round(hist['demand_std'], 2)
            rolling_std_30d = round(hist['demand_std'], 2)
        else:
            default_inv, default_sold = 80, 20
            rolling_avg_7d = rolling_avg_30d = 20.0
            rolling_std_7d = rolling_std_30d = 5.0

        # Only inputs the manager genuinely knows right now
        inventory_level  = st.number_input("Current Inventory Level (units)", min_value=0, value=default_inv, step=5)
        units_sold_today = st.number_input("Units Sold Today",                min_value=0, value=default_sold, step=1)
        lead_time        = st.slider("Lead Time (days)",  min_value=1, max_value=5, value=2)
        discount_pct     = st.slider("Discount %",        min_value=0, max_value=30, value=0)
        st.markdown("---")
        weather    = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Hot", "Partly Cloudy", "Foggy"])
        season     = st.selectbox("Season",  ["Summer", "Monsoon", "Festive", "Winter"])
        is_weekend = st.checkbox("Is Weekend?")
        is_holiday = st.checkbox("Is Holiday?")
        st.markdown("---")

        predict_btn = st.button("🔍 Predict Stockout Risk", use_container_width=True, type="primary")

    # ── MAIN AREA ─────────────────────────────────────────────────
    if predict_btn:

        X_input = build_input_row(
            store_id, sku_id, inventory_level, units_sold_today,
            lead_time, discount_pct, weather, season, is_weekend, is_holiday,
            rolling_avg_7d, rolling_avg_30d, rolling_std_7d, rolling_std_30d,
        )

        prob   = stockout_model.predict_proba(X_input)[0][1]
        pred   = int(prob >= 0.4)

        # Model B only runs if at risk
        reorder_pt = int(reorder_model.predict(X_input)[0]) if pred == 1 else None

        days_remaining = inventory_level / rolling_avg_7d if rolling_avg_7d > 0 else 99
        risk_level     = "High" if days_remaining < 2 else "Medium" if days_remaining < 4 else "Low"

        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            if pred == 1:
                st.markdown(f"""
                <div class="result-box high-risk">
                    <div class="risk-title">🔴 HIGH RISK</div>
                    <div class="risk-sub">Stockout likely within 48 hours</div>
                    <div class="mini-row">
                        <div class="mini-card">
                            <div class="mini-label">Stockout Probability</div>
                            <div class="mini-value" style="color:#DC2626">{prob*100:.1f}%</div>
                        </div>
                        <div class="mini-card">
                            <div class="mini-label">Reorder Point</div>
                            <div class="mini-value">{reorder_pt} units</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.error("**Suggested Action:** Place reorder immediately. Stock will run out before next delivery arrives.")
            else:
                st.markdown(f"""
                <div class="result-box low-risk">
                    <div class="risk-title">🟢 LOW RISK</div>
                    <div class="risk-sub">No stockout expected within 48 hours</div>
                    <div class="mini-row">
                        <div class="mini-card">
                            <div class="mini-label">Stockout Probability</div>
                            <div class="mini-value" style="color:#16A34A">{prob*100:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.success("**Suggested Action:** No immediate reorder required. Monitor inventory daily.")

            st.markdown("#### Stockout Probability")
            st.progress(float(prob))
            st.caption(f"{prob*100:.1f}% probability of stockout in the next 48 hours")

        with col_right:
            st.markdown("#### Inventory Summary")
            st.markdown(f'<div class="auto-badge">📊 Demand stats auto-filled from historical data for {sku_id} at {store_id}</div>', unsafe_allow_html=True)
            summary_df = pd.DataFrame({
                "Metric": [
                    "Current Inventory",
                    "Avg Daily Demand (historical)",
                    "Demand Std Dev (historical)",
                    "Lead Time",
                    "Estimated Days Remaining",
                    "Units Sold Today",
                    "Risk Level",
                ],
                "Value": [
                    f"{inventory_level} units",
                    f"{rolling_avg_7d} units/day",
                    f"{rolling_std_7d} units",
                    f"{lead_time} days",
                    f"{days_remaining:.1f} days",
                    f"{units_sold_today} units",
                    risk_level,
                ]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Top Factors Driving This Prediction")
        fi = pd.Series(stockout_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.barh(fi.index[::-1], fi.values[::-1], color=BLUE, alpha=0.85)
        ax.set_xlabel("Importance Score")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    else:
        st.info("👈 Select a store and SKU in the sidebar and click **Predict Stockout Risk**. Demand stats are filled automatically from historical data.")
        st.markdown("#### How It Works")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Step 1 — Select**")
            st.write("Choose store, SKU, current inventory level, and today's conditions. Demand stats are auto-filled from historical data — no manual entry needed.")
        with c2:
            st.markdown("**Step 2 — Model A**")
            st.write("XGBoost classifier uses rolling demand patterns, lag features, weather, season, and supply chain variables to predict stockout probability.")
        with c3:
            st.markdown("**Step 3 — Model B**")
            st.write("Only if risk is high: XGBoost regressor calculates the reorder point — the stock level at which you must place an order.")


# ═════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PREDICTION
# ═════════════════════════════════════════════════════════════════
elif page == "Batch Prediction":

    st.markdown("# Batch Prediction")
    st.markdown("Upload a CSV with multiple products. Reorder point is shown only for at-risk SKUs.")
    st.markdown("---")

    template = pd.DataFrame([{
        "store_id":         "DS_MUM_01",
        "sku_id":           "MILK_500ML",
        "inventory_level":  80,
        "units_sold_today": 28,
        "lead_time":        2,
        "discount_pct":     0,
        "weather":          "Sunny",
        "season":           "Summer",
        "is_weekend":       0,
        "is_holiday":       0,
    }])
    st.download_button(
        "📥 Download Input Template",
        template.to_csv(index=False).encode("utf-8"),
        "batch_template.csv", "text/csv"
    )
    st.caption("Note: avg_daily_demand and demand_std are looked up from historical data automatically — no need to include them in the CSV.")

    uploaded = st.file_uploader("Upload your filled CSV", type=["csv"])

    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.markdown(f"**{len(batch_df)} rows loaded.** Running predictions...")

        results = []
        for _, row in batch_df.iterrows():
            try:
                key = (row["store_id"], row["sku_id"])
                if key in demand_stats.index:
                    hist = demand_stats.loc[key]
                    r_avg_7  = round(hist['avg_demand'], 2)
                    r_avg_30 = round(hist['avg_demand'], 2)
                    r_std_7  = round(hist['demand_std'], 2)
                    r_std_30 = round(hist['demand_std'], 2)
                else:
                    r_avg_7 = r_avg_30 = r_std_7 = r_std_30 = 20.0

                X = build_input_row(
                    store_id         = row["store_id"],
                    sku_id           = row["sku_id"],
                    inventory_level  = row["inventory_level"],
                    units_sold_today = row["units_sold_today"],
                    lead_time        = int(row["lead_time"]),
                    discount_pct     = row["discount_pct"],
                    weather          = row["weather"],
                    season           = row["season"],
                    is_weekend       = bool(row["is_weekend"]),
                    is_holiday       = bool(row["is_holiday"]),
                    rolling_avg_7d   = r_avg_7,
                    rolling_avg_30d  = r_avg_30,
                    rolling_std_7d   = r_std_7,
                    rolling_std_30d  = r_std_30,
                )
                prob    = stockout_model.predict_proba(X)[0][1]
                pred    = int(prob >= 0.4)
                reorder = int(reorder_model.predict(X)[0]) if pred == 1 else 0
                days_left = row["inventory_level"] / r_avg_7 if r_avg_7 > 0 else 99

                results.append({
                    "Store":           row["store_id"],
                    "SKU":             row["sku_id"],
                    "Category":        SKU_META.get(row["sku_id"], {}).get("category", "—"),
                    "Current Stock":   int(row["inventory_level"]),
                    "Days Remaining":  round(days_left, 1),
                    "Stockout Risk":   "Yes" if pred == 1 else "No",
                    "Probability (%)": round(prob * 100, 1),
                    "Reorder Point":   reorder,
                    "Action":          "⚠️ Order Now" if pred == 1 else "✅ Monitor",
                })
            except Exception as e:
                results.append({"Store": row.get("store_id","?"), "SKU": row.get("sku_id","?"), "Error": str(e)})

        result_df = pd.DataFrame(results)
        at_risk   = (result_df.get("Stockout Risk", pd.Series()) == "Yes").sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total SKUs", len(result_df))
        c2.metric("At Risk",    at_risk)
        c3.metric("Stock OK",   len(result_df) - at_risk)

        def highlight(row):
            return ["background-color: #FEF2F2"] * len(row) if row.get("Stockout Risk") == "Yes" else [""] * len(row)

        st.markdown("#### Prediction Results")
        if "Stockout Risk" in result_df.columns:
            st.dataframe(result_df.style.apply(highlight, axis=1), use_container_width=True, hide_index=True)
        else:
            st.dataframe(result_df, use_container_width=True, hide_index=True)

        st.download_button(
            "📤 Download Predictions CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            "predictions.csv", "text/csv"
        )
    else:
        st.info("Download the template, fill it with your products, then upload it here.")


# ═════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════
elif page == "Model Performance":

    st.markdown("# Model Performance")
    st.markdown("Train: Jan 2022 – Sep 2023 · Test: Oct 2023 – Mar 2024 · Time-based split (not random)")
    st.markdown("---")

    if metrics is None:
        st.warning("Metrics file not found. Run the Jupyter notebook completely — the last cell saves metrics to the `metrics/` folder.")
        st.stop()

    ma = metrics["model_a"]
    mb = metrics["model_b"]

    tab1, tab2 = st.tabs(["Model A — Stockout Classifier", "Model B — Reorder Point Regressor"])

    # ── MODEL A ───────────────────────────────────────────────────
    with tab1:
        st.markdown("**XGBoost Classifier · Target: `stockout_in_48hrs`**")
        st.markdown('<div class="info-box">Recall is the primary metric — missing a real stockout costs more than a false alarm.</div>', unsafe_allow_html=True)

        # Metrics table: model vs baseline side by side
        st.markdown("#### Results vs Baseline")
        comparison = pd.DataFrame({
            "Metric":    ["Precision", "Recall", "F1 Score"],
            "Model":     [ma["precision"], ma["recall"],  ma["f1_score"]],
            "Baseline":  [ma["baseline_precision"], ma["baseline_recall"], ma["baseline_f1"]],
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        # Visual: grouped bar chart model vs baseline
        st.markdown("#### Visual Comparison")
        labels  = ["Precision", "Recall", "F1 Score"]
        model_v = [ma["precision"], ma["recall"], ma["f1_score"]]
        base_v  = [ma["baseline_precision"], ma["baseline_recall"], ma["baseline_f1"]]
        x       = np.arange(len(labels))
        width   = 0.35

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - width/2, model_v, width, label="XGBoost Model", color=BLUE,  alpha=0.85)
        ax.bar(x + width/2, base_v,  width, label="Baseline Rule",  color="#94A3B8", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Feature importance
        st.markdown("#### Feature Importance")
        fi = pd.Series(stockout_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        ax2.barh(fi.index[::-1], fi.values[::-1], color=BLUE, alpha=0.85)
        ax2.set_xlabel("Importance Score")
        ax2.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ── MODEL B ───────────────────────────────────────────────────
    with tab2:
        st.markdown("**XGBoost Regressor · Target: `reorder_point` (units)**")

        # Metrics table
        st.markdown("#### Results vs Baseline")
        comparison_b = pd.DataFrame({
            "Metric":   ["RMSE", "MAE", "R²"],
            "Model":    [mb["rmse"],          mb["mae"],          mb["r2_score"]],
            "Baseline": [mb["baseline_rmse"], mb["baseline_mae"], mb["baseline_r2"]],
        })
        st.dataframe(comparison_b, use_container_width=True, hide_index=True)

        # Visual: grouped bar chart
        st.markdown("#### Visual Comparison")
        labels_b = ["RMSE", "MAE"]
        model_b  = [mb["rmse"], mb["mae"]]
        base_b   = [mb["baseline_rmse"], mb["baseline_mae"]]
        x2       = np.arange(len(labels_b))

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.bar(x2 - width/2, model_b, width, label="XGBoost Model", color=GREEN,    alpha=0.85)
        ax3.bar(x2 + width/2, base_b,  width, label="Baseline Formula", color="#94A3B8", alpha=0.85)
        ax3.set_xticks(x2)
        ax3.set_xticklabels(labels_b)
        ax3.set_ylabel("Error (units)")
        ax3.legend()
        ax3.spines[["top", "right"]].set_visible(False)
        for bar in ax3.patches:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{bar.get_height():.1f}", ha="center", fontsize=9)
        st.pyplot(fig3, use_container_width=True)
        plt.close()

        # R² separately since scale is different
        st.markdown(f"**R² Score:** Model = `{mb['r2_score']}` · Baseline = `{mb['baseline_r2']}`")

        # Feature importance
        st.markdown("#### Feature Importance")
        fi_r = pd.Series(reorder_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
        fig4, ax4 = plt.subplots(figsize=(8, 3.5))
        ax4.barh(fi_r.index[::-1], fi_r.values[::-1], color=GREEN, alpha=0.85)
        ax4.set_xlabel("Importance Score")
        ax4.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig4, use_container_width=True)
        plt.close()

