import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description="Identify likely door sensor addresses from Zigbee traffic"
)
parser.add_argument(
    "--csv",
    default="dataset/zboss.csv",
    help="Path to CSV exported by tshark"
)
parser.add_argument(
    "--top",
    type=int,
    default=5,
    help="Number of candidate addresses to list"
)
parser.add_argument(
    "--min-gap",
    type=float,
    default=30.0,
    help="Minimum average seconds between packets"
)
parser.add_argument(
    "--max-count",
    type=int,
    default=100,
    help="Ignore devices with more packets than this"
)

args = parser.parse_args()

if not os.path.exists(args.csv):
    raise FileNotFoundError(f"Input CSV '{args.csv}' not found")

print(f"Loading {args.csv} ...")
df = pd.read_csv(args.csv)

# Helper to parse integers in decimal or hex
def _to_int(val):
    try:
        return int(str(val), 0)
    except (ValueError, TypeError):
        return np.nan

if not np.issubdtype(df["frame.time_epoch"].dtype, np.number):
    df["frame.time_epoch"] = df["frame.time_epoch"].astype(float)

df["wpan.frame_type"] = df["wpan.frame_type"].apply(_to_int)
df["wpan.src16"] = df["wpan.src16"].apply(_to_int)

df.sort_values("frame.time_epoch", inplace=True)

# consider only data frames
data_df = df[df["wpan.frame_type"] == 1]

candidates = []
for addr, grp in data_df.groupby("wpan.src16"):
    if pd.isna(addr):
        continue
    addr_int = int(addr)
    times = grp["frame.time_epoch"].values
    if len(times) < 2 or len(times) > args.max_count:
        continue
    gaps = np.diff(np.sort(times))
    avg_gap = gaps.mean()
    if avg_gap >= args.min_gap:
        candidates.append({
            "addr": addr_int,
            "count": len(times),
            "avg_gap": avg_gap,
            "med_gap": np.median(gaps),
        })

candidates.sort(key=lambda x: (-x["avg_gap"], x["count"]))

print("\nLikely door sensor addresses:")
for c in candidates[: args.top]:
    print(f"0x{c['addr']:04x}\tcount={c['count']}\tavg_gap={c['avg_gap']:.1f}s\tmed_gap={c['med_gap']:.1f}s")

