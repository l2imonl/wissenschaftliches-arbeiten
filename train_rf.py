import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1. Feature- und Labeldateien einlesen
features = pd.read_csv("features/zigbee_features.csv")
labels   = pd.read_csv("labels/labels.csv")

# 2. Merge über time_bin
data = features.merge(labels, on="time_bin", how="inner")

# 3. X und y definieren
feature_cols = ["pkt_count", "pkt_len_mean", "pkt_len_std", "distinct_src", "distinct_dst"]
X = data[feature_cols]
y = data["label"]

# 4. Trainings- und Test-Set aufteilen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Random Forest instanziieren und trainieren
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)
clf.fit(X_train, y_train)

# 6. Evaluation ausgeben
y_pred = clf.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Modell speichern
joblib.dump(clf, "rf_model.pkl")
print("✅ Model saved to rf_model.pkl")
