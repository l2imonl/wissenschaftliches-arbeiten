import pandas as pd
import joblib

# 1. Modell laden
clf = joblib.load("rf_model.pkl")

# 2. Ein Beispiel-Feature-Fenster einlesen
features = pd.read_csv("features/zigbee_features.csv")
# dieselben Feature-Spalten wie beim Training nutzen
feature_cols = [c for c in features.columns if c != "time_bin"]
sample = features.iloc[:400][feature_cols]

# 3. Vorhersagen
pred = clf.predict(sample)
print("Sample Predictions:", pred)
