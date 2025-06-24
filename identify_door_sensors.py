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
    help="Path to CSV exported by tshark",
)
parser.add_argument(
    "--top",
    type=int,
    default=5,
    help="Number of candidate addresses to list",
)
parser.add_argument(
    "--min-gap",
    type=float,
    default=30.0,
    help="Minimum average seconds between bursts",
)
parser.add_argument(
    "--burst-max",
    type=float,
    default=1.0,
    help="Max seconds between packets of a burst",
)
parser.add_argument(
    "--min-bursts",
    type=int,
    default=2,
    help="Minimum number of bursts",
)
parser.add_argument(
    "--max-count",
    type=int,
    default=100,
    help="Ignore devices with more packets than this",
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
    times = np.sort(grp["frame.time_epoch"].values)
    if len(times) < 2 or len(times) > args.max_count:
        continue
    gaps = np.diff(times)
    avg_gap = gaps.mean()

    burst_indices = np.where(gaps <= args.burst_max)[0]
    burst_count = len(burst_indices)
    if burst_count < args.min_bursts or avg_gap < args.min_gap:
        continue

    candidates.append({
        "addr": addr_int,
        "count": len(times),
        "avg_gap": avg_gap,
        "bursts": burst_count,
        "med_gap": np.median(gaps),
    })

candidates.sort(key=lambda x: (-x["bursts"], -x["avg_gap"]))

print("\nLikely door sensor addresses:")
for c in candidates[: args.top]:
    print(
        f"0x{c['addr']:04x}\tcount={c['count']}\tbursts={c['bursts']}\tavg_gap={c['avg_gap']:.1f}s\tmed_gap={c['med_gap']:.1f}s"
    )
