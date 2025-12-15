import os, json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Mall Clustering + Regression", layout="wide")
st.title("Mall Customers: K-Means Clustering + Regression")

DATA_PATH = "Mall_Customers.csv"
ART_DIR = "artifacts"

@st.cache_data
def load_data(path_or_file):
    return pd.read_csv(path_or_file)

@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(ART_DIR, "scaler.joblib"))
    kmeans = joblib.load(os.path.join(ART_DIR, "kmeans.joblib"))
    reg = joblib.load(os.path.join(ART_DIR, "reg_model.joblib"))

    with open(os.path.join(ART_DIR, "cluster_cols.json"), "r") as f:
        cluster_cols = json.load(f)

    with open(os.path.join(ART_DIR, "feature_cols.json"), "r") as f:
        feature_cols = json.load(f)

    metrics_path = os.path.join(ART_DIR, "metrics.json")
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    return scaler, kmeans, reg, cluster_cols, feature_cols, metrics

st.sidebar.header("Input Data")
uploaded = st.sidebar.file_uploader("Upload CSV (opsional)", type=["csv"])

if uploaded is not None:
    df = load_data(uploaded)
else:
    df = load_data(DATA_PATH)

st.subheader("Preview Dataset")
st.dataframe(df.head(), use_container_width=True)

scaler, kmeans, reg, cluster_cols, feature_cols, metrics = load_artifacts()

# Validasi kolom
missing = [c for c in cluster_cols if c not in df.columns]
if missing:
    st.error(f"Kolom wajib tidak ditemukan: {missing}")
    st.stop()

# Clustering
Xc = df[cluster_cols].copy()
Xc_scaled = scaler.transform(Xc)
df_out = df.copy()
df_out["Cluster"] = kmeans.predict(Xc_scaled)

if metrics:
    st.subheader("Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("K", metrics.get("k", "-"))
    c2.metric("Silhouette", f"{metrics.get('silhouette', 0):.4f}" if "silhouette" in metrics else "-")
    c3.metric("MAE", f"{metrics.get('mae', 0):.4f}" if "mae" in metrics else "-")
    c4.metric("R2", f"{metrics.get('r2', 0):.4f}" if "r2" in metrics else "-")

st.subheader("Ringkasan per Cluster (Mean)")
summary = df_out.groupby("Cluster")[cluster_cols].mean().round(2)
st.dataframe(summary, use_container_width=True)

st.subheader("Visualisasi Cluster: Income vs Spending")
fig = plt.figure()
for c in sorted(df_out["Cluster"].unique()):
    part = df_out[df_out["Cluster"] == c]
    plt.scatter(
        part["Annual Income (k$)"],
        part["Spending Score (1-100)"],
        label=f"Cluster {c}",
        alpha=0.8
    )
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
st.pyplot(fig)

# Regresi
st.subheader("Regresi: Prediksi Spending Score")
with st.form("pred_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    income = st.number_input("Annual Income (k$)", min_value=0.0, max_value=1000.0, value=60.0)
    cluster_in = st.number_input("Cluster", min_value=0, max_value=int(kmeans.n_clusters - 1), value=0)
    submitted = st.form_submit_button("Prediksi")

if submitted:
    X_in = pd.DataFrame([{
        "Age": age,
        "Annual Income (k$)": income,
        "Cluster": int(cluster_in)
    }])
    pred = reg.predict(X_in)[0]
    st.success(f"Prediksi Spending Score: {pred:.2f}")

st.subheader("Download hasil (dengan Cluster)")
csv_bytes = df_out.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv_bytes, file_name="mall_customers_clustered.csv", mime="text/csv")
