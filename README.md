# Zigbee Anomaly-Labeling Pipeline

Dieses Repository enthält ein Python-Skript, das aus einem Zigbee-Traffic-Datensatz (PCAP → CSV) automatisch Features extrahiert, einen Isolation Forest trainiert und aus den Anomalien ein Label-File (`labels.csv`) erzeugt.

## Voraussetzungen

- Python 3.7 oder höher  
- `pip`  
- WLAN-Dongle oder Zigbee-Stick (für frühere Schritte) nicht erforderlich für dieses Skript

## Installation

1. Repository klonen (oder Skript & CSVs in ein Verzeichnis kopieren)  
2. Virtuelle Umgebung anlegen und aktivieren  
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # Linux/macOS
   # venv\Scripts\activate.bat    # Windows
   ```
3. Abhängigkeiten installieren
    ```bash
    pip install pandas scikit-learn
    ```

## Dateistruktur

```
├── dataset/
│   └── zboss.csv             # Roh-CSV mit Spalten: frame.time_epoch, wpan.frame_type, wpan.seq_no, wpan.src16, wpan.dst16, frame.len
├── features/
│   └── zigbee_features.csv   # (wird erzeugt) aggregierte Features pro 5‑Sekunden-Fenster (Schrittweite 1 Sekunde)
├── labels/
│   └── labels.csv            # (wird erzeugt) time_bin + label (0 = normal, 1 = Anomalie)
└── isolation_forest_training.py  # Hauptskript
```

## Usage

1. Rohdaten vorbereiten

    Aus Deiner PCAP-Datei mit tshark in dataset/zboss.csv exportieren, z. B.:
    ```bash
    tshark -r input.pcap \
    -Y "wpan" \
    -T fields \
    -e frame.time_epoch \
    -e wpan.frame_type \
    -e wpan.seq_no \
    -e wpan.src16 \
    -e wpan.dst16 \
    -e frame.len \
    -E header=y -E separator=, \
    > dataset/zboss.csv
    ```

2. Anomalie-Labels erzeugen
    ```bash
    # optional: CSV-Pfad angeben
    python isolation_forest_training.py --csv dataset/zboss.csv
    ```

- Ergebnis: features/zigbee_features.csv und labels/labels.csv

2a. Türsensor-Adresse ermitteln
    ```bash
    python identify_door_sensors.py --csv dataset/zboss.csv
    ```
    - Gibt Kandidaten-Adressen sortiert nach durchschnittlicher Sendehäufigkeit aus

2b. "Haustür geöffnet"-Labels erzeugen
    ```bash
    python generate_door_labels.py --door-src 0x1234
    ```
    - Erwartet die Zigbee-Adresse des Türsensors als Hex-Wert
    - Schreibt Labels in `labels/door_labels.csv`

3. Random Forest trainieren
    ```bash
    python train_rf.py
    ```
- Liest features/zigbee_features.csv und labels/labels.csv
- Gibt Klassifikations-Report aus
- Speichert Modell in rf_model.pkl

4. Random Forest testen
    ```bash
    python test_rf.py
    ```
- Liest rf_model.pkl und die ersten 400 Zeilen aus features/zigbee_features.csv
- Gibt Sample-Vorhersagen aus
