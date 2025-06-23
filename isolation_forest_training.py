"""Feature extraction and anomaly detection for Zigbee traffic.

The script reads ``dataset/zboss.csv`` exported via ``tshark`` with the
columns ``frame.time_epoch``, ``wpan.frame_type``, ``wpan.seq_no``,
``wpan.src16``, ``wpan.dst16`` and ``frame.len``. It computes sliding-window
statistics and trains an Isolation Forest. The resulting features are written
to ``features/zigbee_features.csv`` and labels to ``labels/labels.csv``.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DATASET_CSV = "dataset/zboss.csv"

# Parameters for the sliding window
WINDOW_SIZE = 5   # seconds
STEP_SIZE = 1     # seconds

# 1. CSV einlesen mit erweiterten Spalten
# Erwartet: frame.time_epoch, wpan.frame_type, wpan.seq_no, wpan.src16, wpan.dst16, frame.len

df = pd.read_csv(DATASET_CSV)

# Stelle sicher, dass die Zeitspalte numerisch ist
if not np.issubdtype(df["frame.time_epoch"].dtype, np.number):
    df["frame.time_epoch"] = df["frame.time_epoch"].astype(float)
if not np.issubdtype(df["wpan.seq_no"].dtype, np.number):
    df["wpan.seq_no"] = pd.to_numeric(df["wpan.seq_no"], errors="coerce")
if not np.issubdtype(df["frame.len"].dtype, np.number):
    df["frame.len"] = pd.to_numeric(df["frame.len"], errors="coerce")

# Einmal global sortieren
df.sort_values("frame.time_epoch", inplace=True)

# Alle vorkommenden Frame-Typen bestimmen (z.B. 0=Beacon,1=Data,...)
frame_types = sorted(df["wpan.frame_type"].dropna().unique())

start_time = df["frame.time_epoch"].min()
end_time = df["frame.time_epoch"].max()

features_list = []
bin_id = 0

for current in np.arange(start_time, end_time + STEP_SIZE, STEP_SIZE):
    window_mask = (df["frame.time_epoch"] >= current) & (df["frame.time_epoch"] < current + WINDOW_SIZE)
    window_df = df[window_mask]

    feat = {"time_bin": bin_id}
    feat["pkt_count"] = len(window_df)
    if len(window_df) > 0:
        feat["pkt_len_mean"] = window_df["frame.len"].mean()
        feat["pkt_len_std"] = window_df["frame.len"].std() if window_df.shape[0] > 1 else 0
        feat["distinct_src"] = window_df["wpan.src16"].nunique()
        feat["distinct_dst"] = window_df["wpan.dst16"].nunique()

        # Frame-Type-Zählungen
        for ft in frame_types:
            col = f"ftype_{int(ft)}_count"
            feat[col] = int((window_df["wpan.frame_type"] == ft).sum())

        # Sequenznummer-Lücken
        seq_diffs = window_df.sort_values("frame.time_epoch")["wpan.seq_no"].diff().dropna().abs()
        feat["seq_gap_mean"] = seq_diffs.mean() if len(seq_diffs) > 0 else 0

        # Interarrival-Times
        time_diffs = window_df["frame.time_epoch"].diff().dropna()
        feat["interarrival_mean"] = time_diffs.mean() if len(time_diffs) > 0 else 0
        feat["interarrival_std"] = time_diffs.std() if len(time_diffs) > 0 else 0
    else:
        # Bei leeren Fenstern Nullen verwenden
        feat.update({
            "pkt_len_mean": 0,
            "pkt_len_std": 0,
            "distinct_src": 0,
            "distinct_dst": 0,
            "seq_gap_mean": 0,
            "interarrival_mean": 0,
            "interarrival_std": 0,
        })
        for ft in frame_types:
            feat[f"ftype_{int(ft)}_count"] = 0

    features_list.append(feat)
    bin_id += 1

features = pd.DataFrame(features_list)
features.fillna(0, inplace=True)

# Feature-CSV speichern
os.makedirs("features", exist_ok=True)
features.to_csv("features/zigbee_features.csv", index=False)
print("zigbee_features.csv erzeugt mit", len(features), "Zeilen.")

# Isolation Forest trainieren
iso = IsolationForest(
    n_estimators=100,
    contamination=0.01,
    random_state=42,
)
feature_cols = [c for c in features.columns if c != "time_bin"]
X = features[feature_cols]
iso.fit(X)

# Anomalien bestimmen
features["anomaly_score"] = iso.decision_function(X)
features["is_anomaly"] = iso.predict(X) == -1

features["label"] = 0
features.loc[features["is_anomaly"], "label"] = 1

# Labels speichern
os.makedirs("labels", exist_ok=True)
label_df = features[["time_bin", "label"]]
label_df.to_csv("labels/labels.csv", index=False)
print("labels.csv erzeugt mit", len(label_df), "Zeilen.")

# Top-Anomalien anzeigen
anomalies = features[features["is_anomaly"]].sort_values("anomaly_score")
print(anomalies.head(20))
