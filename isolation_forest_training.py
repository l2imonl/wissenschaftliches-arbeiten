import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. CSV einlesen
df = pd.read_csv("dataset/zboss.csv")

# 2. Zeitfenster (10-Sekunden-Bins) erstellen
df["epoch_int"] = df["frame.time_epoch"].astype(int)
df["time_bin"] = (df["epoch_int"] // 10).astype(int)

# 3. Feature-Engineering pro Zeitfenster
features = df.groupby("time_bin").agg(
    pkt_count    = ("frame.len",       "count"),
    pkt_len_mean = ("frame.len",       "mean"),
    pkt_len_std  = ("frame.len",       "std"),
    distinct_src = ("wpan.src16",     lambda x: x.nunique()),
    distinct_dst = ("wpan.dst16",     lambda x: x.nunique())
).reset_index()
features["pkt_len_std"].fillna(0, inplace=True)

# Speichere die Features in einer CSV-Datei
features.to_csv("features/zigbee_features.csv", index=False)
print("zigbee_features.csv erzeugt mit", len(features), "Zeilen.")

# 4. Isolation Forest initialisieren und trainieren
iso = IsolationForest(
    n_estimators=100,
    contamination=0.01,   # ca. 1 % der Bins als Anomalien erwarten
    random_state=42
)
X = features[["pkt_count", "pkt_len_mean", "pkt_len_std", "distinct_src", "distinct_dst"]]
iso.fit(X)

# 5. Anomalie-Erkennung
features["anomaly_score"] = iso.decision_function(X)   # höhere Werte = normaler
features["is_anomaly"]   = iso.predict(X) == -1        # -1 = Anomalie

# 5a. Label-Initialisierung und -Setzung
features["label"] = 0                                  # alle zunächst 0
features.loc[features["is_anomaly"], "label"] = 1      # Anomalien als 1

# Extrahiere nur die beiden Spalten und speichere sie
label_df = features[["time_bin", "label"]]
label_df.to_csv("labels/labels.csv", index=False)
print("labels.csv erzeugt mit", len(label_df), "Zeilen.")

# 6. Top-Anomalien anzeigen
anomalies = features[features["is_anomaly"]].sort_values("anomaly_score")
print(anomalies.head(20))
