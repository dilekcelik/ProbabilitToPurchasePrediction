# app.py
# ====================================================
# Streamlit Inference App for Propensity-to-Buy Model
# - Loads: xgb_model_deployment.pkl (model, scaler, feature_columns, threshold)
# - Builds features:
#     segment_a/segment_b/segment_c
#     recency_active/recency_dormant/recency_inactive
#     age_bin_18_25 / ... / age_bin_61_plus (bins = [18,25,35,45,60,100], right=True, include_lowest=True)
#     age_squared, age_log
# - Single & Batch scoring
# ====================================================

import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Propensity Scoring", page_icon="💷⚖️🛒", layout="wide")

# ---------------------------
# Load packaged artifacts
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_package(path="xgb_model_deployment.pkl"):
    with open(path, "rb") as f:
        pkg = pickle.load(f)
    return pkg

pkg = load_package()
model = pkg["model"]
scaler = pkg["scaler"]
feature_columns = pkg["feature_columns"]
default_threshold = float(pkg.get("threshold", 0.3))

# ---------------------------
# Feature engineering helper
# ---------------------------
def build_features(
    raw_df: pd.DataFrame,
    feature_columns: list[str],
    age_col="age",
    segment_col="segment",
    recency_col="recency",
    use_log1p=False,  # set True only if you trained with log1p
) -> pd.DataFrame:
    """
    From raw columns: age, priorinterest, segment (A/B/C), recency (Active/Dormant/Inactive)
    build the exact matrix the model expects.
    """
    df = raw_df.copy()
    # normalize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # ensure required bases exist
    for base in [age_col, "priorinterest", segment_col, recency_col]:
        if base not in df.columns:
            df[base] = np.nan

    # clean bases
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce").fillna(0).clip(lower=0)
    df["priorinterest"] = pd.to_numeric(df["priorinterest"], errors="coerce").fillna(0).astype(int)
    df[segment_col] = df[segment_col].astype(str).str.strip().str.lower()
    df[recency_col] = df[recency_col].astype(str).str.strip().str.lower()

    # start zero-frame with training schema (prevents column mismatch)
    X = pd.DataFrame(0.0, index=df.index, columns=feature_columns, dtype=float)

    # -------------------------
    # segment one-hots: A/B/C -> segment_a/b/c
    # -------------------------
    seg_map = {"a": "segment_a", "b": "segment_b", "c": "segment_c",
               "segment a": "segment_a", "segment b": "segment_b", "segment c": "segment_c"}
    seg_col = df[segment_col].map(seg_map).fillna("")
    for i, col in enumerate(seg_col):
        if col in X.columns and col != "":
            X.loc[X.index[i], col] = 1.0

    # -------------------------
    # recency one-hots: Active/Dormant/Inactive -> recency_*
    # -------------------------
    rec_map = {"active": "recency_active", "dormant": "recency_dormant", "inactive": "recency_inactive"}
    rec_col = df[recency_col].map(rec_map).fillna("")
    for i, col in enumerate(rec_col):
        if col in X.columns and col != "":
            X.loc[X.index[i], col] = 1.0

    # -------------------------
    # Age bins via pd.cut (exact spec)
    # bins = [18, 25, 35, 45, 60, 100], labels = ["18_25","26_35","36_45","46_60","61_plus"]
    # right=True, include_lowest=True
    # -------------------------
    age = df[age_col].astype(float)
    bins = [18, 25, 35, 45, 60, 100]
    labels = ["18_25", "26_35", "36_45", "46_60", "61_plus"]
    age_bin = pd.cut(age, bins=bins, labels=labels, right=True, include_lowest=True)
    age_dum = pd.get_dummies(age_bin, prefix="age_bin").astype(int)

    # Ensure all bin columns exist
    for col in ["age_bin_18_25","age_bin_26_35","age_bin_36_45","age_bin_46_60","age_bin_61_plus"]:
        if col not in age_dum.columns:
            age_dum[col] = 0

    # write age bin dummies into X if present
    for col in age_dum.columns:
        if col in X.columns:
            X[col] = age_dum[col].values

    # -------------------------
    # numeric bases & transforms
    # -------------------------
    if "age" in X.columns:
        X["age"] = age.values

    if "age_squared" in X.columns:
        X["age_squared"] = (age**2).astype(float).values

    if "age_log" in X.columns:
        if use_log1p:
            X["age_log"] = np.log1p(age).astype(float).values
        else:
            X["age_log"] = np.log(np.clip(age, 1e-8, None)).astype(float).values

    # priorinterest
    if "priorinterest" in X.columns:
        X["priorinterest"] = df["priorinterest"].astype(float).values

    # final column order
    X = X[feature_columns]
    return X

# ---------------------------
# UI
# ---------------------------
st.title("💷⚖️🛒 Propensity-to-Buy Scoring")
st.caption("Loads your packaged model & builds features: Segments, Recency, Age bins, Age transforms.")

with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Decision threshold (class=1 if prob ≥ threshold)",
                          min_value=0.05, max_value=0.95, value=default_threshold, step=0.01)
    show_segments = st.checkbox("Show High/Medium/Low labels", value=True)
    low_cut = st.number_input("Medium lower bound", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
    high_cut = st.number_input("High lower bound", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
    st.markdown("---")
    st.markdown(f"**Model features:** {len(feature_columns)}")
    st.code("\n".join(feature_columns), language="text")

def band_label(p, low=0.3, high=0.75):
    if p >= high: return "High"
    if p >= low:  return "Medium"
    return "Low"

tab1, tab2 = st.tabs(["🔎 Single Customer", "📄 Batch Scoring (CSV)"])

# ===========================
# Single
# ===========================
with tab1:
    st.subheader("Single Customer")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age", min_value=0, max_value=120, value=35, step=1)
    with c2:
        prior = st.selectbox("Prior Interest", options=[0,1], index=0)
    with c3:
        segment = st.selectbox("Segment", options=["A","B","C"], index=0)
    with c4:
        recency = st.selectbox("Recency", options=["Active","Dormant","Inactive"], index=0)

    if st.button("🔮 Predict"):
        raw = pd.DataFrame([{
            "age": age,
            "priorinterest": prior,
            "segment": segment,
            "recency": recency
        }])
        X = build_features(raw, feature_columns, use_log1p=False)
        X_scaled = scaler.transform(X)
        proba = float(model.predict_proba(X_scaled)[:,1][0])
        pred = int(proba >= threshold)

        colA, colB = st.columns([1,2])
        with colA:
            st.metric("Predicted Probability", f"{proba:.3f}")
            #st.metric("Decision Threshold", f"{threshold:.2f}")
            st.metric("Predicted Class", "1 (Likely) ✅" if pred==1 else "0 (Unlikely) ❌")
            if show_segments:
                st.metric("Band", band_label(proba, low_cut, high_cut))
        with colB:
            st.caption("Non-zero engineered features for this customer:")
            nz = pd.Series(X.iloc[0].values, index=X.columns)
            nz = nz[nz!=0].sort_values(ascending=False)
            st.dataframe(nz.to_frame("value"))

# ===========================
# Batch
# ===========================
with tab2:
    st.subheader("Batch Scoring")
    st.write("Upload CSV with columns: **age, priorinterest, segment (A/B/C), recency (Active/Dormant/Inactive)**")
    ex = pd.DataFrame({
        "age":[22,41,63],
        "priorinterest":[0,1,0],
        "segment":["B","A","C"],
        "recency":["Dormant","Active","Inactive"]
    })
    with st.expander("Example input"):
        st.dataframe(ex, use_container_width=True)

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        try:
            raw_df = pd.read_csv(file)
            Xb = build_features(raw_df, feature_columns, use_log1p=False)
            Xb_scaled = scaler.transform(Xb)
            proba = model.predict_proba(Xb_scaled)[:,1]
            pred = (proba >= threshold).astype(int)

            out = raw_df.copy()
            out["pred_proba"] = proba
            out["pred_class"] = pred
            if show_segments:
                out["band"] = [band_label(p, low_cut, high_cut) for p in proba]

            st.success(f"Scored {len(out)} rows.")
            st.dataframe(out.head(50), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Scored CSV",
                               data=csv_bytes, file_name="scored_customers.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Failed to score file: {e}")
