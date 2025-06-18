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
│   └── zboss.csv             # Roh-CSV mit Spalten: frame.time_epoch, wpan.src16, wpan.dst16, frame.len
├── features/
│   └── zigbee_features.csv   # (wird erzeugt) aggregierte Features pro 10-Sekunden-Bin
├── labels/
│   └── labels.csv            # (wird erzeugt) time_bin + label (0 = normal, 1 = Anomalie)
└── anomaly_labeling.py       # Hauptskript
```

## Usage

1. Rohdaten vorbereiten

    Aus Deiner PCAP-Datei mit tshark in dataset/zboss.csv exportieren, z. B.:
    ```bash
    tshark -r input.pcap \
    -Y "wpan" \
    -T fields \
    -e frame.time_epoch \
    -e wpan.src16 \
    -e wpan.dst16 \
    -e frame.len \
    -E header=y -E separator=, \
    > dataset/zboss.csv
    ```

2. Anomalie-Labels erzeugen
    ```bash
    python anomaly_labeling.py
    ```
   
- Ergebnis: features/zigbee_features.csv und labels/labels.csv

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