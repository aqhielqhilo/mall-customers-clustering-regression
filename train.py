# train.py
import os, json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATA_PATH = "Mall_Customers.csv"
ART_DIR = "artifacts"

def main():
    os.makedirs(ART_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    cluster_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    Xc = df[cluster_cols].copy()

    scaler = StandardScaler()
    Xc_scaled = scaler.fit_transform(Xc)

    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(Xc_scaled)

    df["Cluster"] = clusters
    sil = silhouette_score(Xc_scaled, clusters)

    feature_cols = ["Age", "Annual Income (k$)", "Cluster"]
    target_col = "Spending Score (1-100)"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = RandomForestRegressor(n_estimators=300, random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    joblib.dump(scaler, f"{ART_DIR}/scaler.joblib")
    joblib.dump(kmeans, f"{ART_DIR}/kmeans.joblib")
    joblib.dump(reg, f"{ART_DIR}/reg_model.joblib")

    with open(f"{ART_DIR}/cluster_cols.json", "w") as f:
        json.dump(cluster_cols, f)

    with open(f"{ART_DIR}/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    with open(f"{ART_DIR}/metrics.json", "w") as f:
        json.dump({"k": k, "silhouette": sil, "mae": mae, "rmse": rmse, "r2": r2}, f, indent=2)

    print("Done. Artifacts saved to artifacts/")

if __name__ == "__main__":
    main()