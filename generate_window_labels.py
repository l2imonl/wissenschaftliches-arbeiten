import argparse
import os
import numpy as np
import pandas as pd

# Parameters for sliding window
WINDOW_SIZE = 5  # seconds
STEP_SIZE = 1    # seconds

def _to_int(val):
    try:
        return int(str(val), 0)
    except (ValueError, TypeError):
        return np.nan

parser = argparse.ArgumentParser(
    description="Generate 'door open' labels from Zigbee CSV"
)
parser.add_argument(
    "--csv",
    default="dataset/zboss.csv",
    help="Path to CSV exported by tshark"
)
parser.add_argument(
    "--door-src",
    required=True,
    help="16-bit source address of the door sensor (e.g. 0x1234)"
)
parser.add_argument(
    "--outfile",
    default="labels/door_labels.csv",
    help="Output CSV for labels"
)

args = parser.parse_args()

# Load CSV
if not os.path.exists(args.csv):
    raise FileNotFoundError(f"Input CSV '{args.csv}' not found")

print(f"Loading {args.csv} ...")
df = pd.read_csv(args.csv)

# Convert columns
if not np.issubdtype(df["frame.time_epoch"].dtype, np.number):
    df["frame.time_epoch"] = df["frame.time_epoch"].astype(float)

df["wpan.frame_type"] = df["wpan.frame_type"].apply(_to_int)
df["wpan.src16"] = df["wpan.src16"].apply(_to_int)

door_addr = int(args.door_src, 0)

df.sort_values("frame.time_epoch", inplace=True)

start_time = df["frame.time_epoch"].min()
end_time = df["frame.time_epoch"].max()
num_bins = int(np.ceil((end_time - start_time) / STEP_SIZE))

labels = np.zeros(num_bins, dtype=int)

# Find packets from door sensor
mask = (df["wpan.src16"] == door_addr) & (df["wpan.frame_type"] == 1)
door_packets = df[mask]

for ts in door_packets["frame.time_epoch"]:
    bin_id = int((ts - start_time) // STEP_SIZE)
    if 0 <= bin_id < num_bins:
        labels[bin_id] = 1

label_df = pd.DataFrame({
    "time_bin": range(num_bins),
    "label": labels
})

os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
label_df.to_csv(args.outfile, index=False)
print(f"Labels written to {args.outfile} with {len(label_df)} rows.")
