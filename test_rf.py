import pandas as pd
import joblib

# 1. Modell laden
clf = joblib.load("rf_model.pkl")

# 2. Ein Beispiel-Feature-Fenster einlesen
features = pd.read_csv("features/zigbee_features.csv")
sample = features.iloc[:400][["pkt_count","pkt_len_mean","pkt_len_std","distinct_src","distinct_dst"]]

# 3. Vorhersagen
pred = clf.predict(sample)
print("Sample Predictions:", pred)
