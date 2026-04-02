
import streamlit as st
import pandas as pd
import numpy as np
import os, glob
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Time Traffic Prediction (G1traffic)", layout="wide")

# -------------------- Data loading --------------------
@st.cache_data(show_spinner=False)
def load_data(path: str = "data/G1traffic.csv") -> pd.DataFrame:
    """Load local CSV, normalize headers to DateTime/Junction/Vehicles, parse DateTime, sort."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)

    # Normalize headers
    rename = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("datetime", "date_time", "timestamp"):
            rename[c] = "DateTime"
        elif cl in ("junction", "junction_id", "junctionid"):
            rename[c] = "Junction"
        elif cl in ("vehicles", "vehicle_count", "count", "volume"):
            rename[c] = "Vehicles"
    if rename:
        df = df.rename(columns=rename)

    # Basic checks
    for col in ("DateTime", "Junction", "Vehicles"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df.sort_values(["Junction", "DateTime"]).reset_index(drop=True)

# -------------------- Feature engineering --------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time features + per-junction lags/rolling; drop rows with NA after lag/rolling."""
    df = df.copy()
    df["hour"] = df["DateTime"].dt.hour
    df["day"] = df["DateTime"].dt.dayofweek

    parts = []
    for j, g in df.groupby("Junction", group_keys=False):
        g = g.sort_values("DateTime")
        g["lag1"] = g["Vehicles"].shift(1)
        g["lag24"] = g["Vehicles"].shift(24)
        g["roll3"] = g["Vehicles"].shift(1).rolling(3).mean()
        parts.append(g)
    out = pd.concat(parts, axis=0).sort_values(["Junction", "DateTime"]).dropna().reset_index(drop=True)
    return out

# -------------------- Label creation --------------------
def label_from_train(train_df: pd.DataFrame) -> dict:
    """Compute Vehicles tertiles (q33, q66) per junction from TRAIN slice only."""
    thr = {}
    for j, g in train_df.groupby("Junction"):
        q1 = g["Vehicles"].quantile(0.33)
        q2 = g["Vehicles"].quantile(0.66)
        thr[int(j)] = (float(q1), float(q2))
    return thr

def apply_labels(df: pd.DataFrame, thr: dict) -> pd.DataFrame:
    """Apply 0/1/2 labels using per-junction thresholds; assumes all junctions exist in thr."""
    def f(r):
        q1, q2 = thr[int(r["Junction"])]
        if r["Vehicles"] <= q1: return 0
        if r["Vehicles"] <= q2: return 1
        return 2
    out = df.copy()
    out["congestion"] = out.apply(f, axis=1)
    return out

# -------------------- Time-aware split per junction --------------------
def split_per_junction(df: pd.DataFrame, valid_ratio: float = 0.2):
    """Split each junction's timeline into train/test by time; concat all junctions back."""
    train_parts, test_parts = [], []
    for j, g in df.groupby("Junction", group_keys=False):
        n = len(g)
        cut = int((1 - valid_ratio) * n)
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(test_parts).reset_index(drop=True)

# -------------------- Sidebar --------------------
st.sidebar.title("🚦 Time Traffic Prediction")
section = st.sidebar.radio("Go to", ["📁 Dataset", "🛠️ Train", "🔮 Predict", "📊 Analyze"], index=0)

# -------------------- Sections --------------------
if section == "📁 Dataset":
    st.title("📁 Dataset")
    df = load_data()
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Junctions", df["Junction"].nunique())
    c3.metric("Date range", f"{df['DateTime'].min():%Y-%m-%d} → {df['DateTime'].max():%Y-%m-%d}")
    st.dataframe(df.head(20))

elif section == "🛠️ Train":
    st.title("🛠️ Train Model")
    # 1) features
    base = load_data()
    df = add_features(base)

    # 2) time-aware split PER JUNCTION
    train, test = split_per_junction(df, valid_ratio=0.2)

    # 3) thresholds from training only, then apply
    thr = label_from_train(train)
    train = apply_labels(train, thr)
    test  = apply_labels(test, thr)

    # 4) features & targets
    feat_cols = ["hour", "day", "lag1", "lag24", "roll3", "Junction"]
    Xtr = pd.get_dummies(train[feat_cols], columns=["Junction"])
    ytr = train["congestion"].astype(int)
    Xte = pd.get_dummies(test[feat_cols], columns=["Junction"])
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    yte = test["congestion"].astype(int)

    if st.button("Train"):
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)

        st.success("Model trained ✔️")
        st.code(classification_report(yte, yp, digits=4), language="text")

        cm = confusion_matrix(yte, yp, labels=[0,1,2])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Low","Medium","High"],
                    yticklabels=["Low","Medium","High"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)

        # save to session for Predict
        st.session_state["model"] = clf
        st.session_state["cols"]  = Xtr.columns.tolist()
        st.session_state["thr"]   = thr

elif section == "🔮 Predict":
    st.title("🔮 Live Prediction")
    if "model" not in st.session_state:
        st.info("Train the model first in 🛠️ Train")
    else:
        clf = st.session_state["model"]
        cols = st.session_state["cols"]

        # Build features for the latest hour, per selected junction
        df = add_features(load_data())
        juncs = sorted(df["Junction"].unique().tolist())
        j = st.selectbox("Junction", juncs, index=0)

        # Take the last available row for that junction as the current context
        row = df[df["Junction"] == j].iloc[-1]
        # Allow tweaking (optional)
        c1, c2, c3, c4, c5 = st.columns(5)
        hour  = c1.number_input("hour", 0, 23, int(row["hour"]))
        day   = c2.number_input("day (Mon=0)", 0, 6, int(row["day"]))
        lag1  = c3.number_input("lag1", value=float(row["lag1"]))
        lag24 = c4.number_input("lag24", value=float(row["lag24"]))
        roll3 = c5.number_input("roll3", value=float(row["roll3"]))

        x = pd.DataFrame([{
            "hour": hour, "day": day, "lag1": lag1, "lag24": lag24, "roll3": roll3, "Junction": j
        }])
        x = pd.get_dummies(x, columns=["Junction"])
        x = x.reindex(columns=cols, fill_value=0)

        if st.button("Predict"):
            y = int(clf.predict(x)[0])
            lbl = {0: "Low", 1: "Medium", 2: "High"}[y]
            col = {0: "#22c55e", 1: "#eab308", 2: "#ef4444"}[y]
            st.markdown(f"### Predicted congestion: <span style='color:{col}'><b>{lbl}</b></span>", unsafe_allow_html=True)

else:
    st.title("📊 Analysis")
    df = load_data()
    df["hour"] = df["DateTime"].dt.hour
    st.subheader("Hourly profile by junction")
    st.line_chart(df.groupby(["hour", "Junction"])["Vehicles"].mean().unstack())
