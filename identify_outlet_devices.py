import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description="Identify likely smart outlet addresses from Zigbee traffic"
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
    "--min-count",
    type=int,
    default=50,
    help="Minimum number of packets required",
)
parser.add_argument(
    "--max-gap",
    type=float,
    default=30.0,
    help="Maximum average seconds between packets",
)
parser.add_argument(
    "--max-dst",
    type=int,
    default=2,
    help="Ignore devices sending to more than this many destinations",
)
parser.add_argument(
    "--min-frame-len",
    type=int,
    default=40,
    help="Ignore devices with mean frame length below this",
)
parser.add_argument(
    "--max-frame-len",
    type=int,
    default=100,
    help="Ignore devices with mean frame length above this",
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

for col in ["wpan.frame_type", "wpan.src16", "wpan.dst16"]:
    df[col] = df[col].apply(_to_int)

df["frame.len"] = pd.to_numeric(df.get("frame.len"), errors="coerce")

df.sort_values("frame.time_epoch", inplace=True)

# consider only data frames
sub_df = df[df["wpan.frame_type"] == 1]

candidates = []
for addr, grp in sub_df.groupby("wpan.src16"):
    if pd.isna(addr):
        continue
    count = len(grp)
    if count < args.min_count:
        continue
    dst_count = grp["wpan.dst16"].nunique()
    if dst_count > args.max_dst:
        continue
    mean_len = grp["frame.len"].dropna().mean()
    if not np.isnan(mean_len) and (mean_len < args.min_frame_len or mean_len > args.max_frame_len):
        continue
    times = np.sort(grp["frame.time_epoch"].values)
    if len(times) < 2:
        continue
    gaps = np.diff(times)
    avg_gap = gaps.mean()
    if avg_gap > args.max_gap:
        continue
    candidates.append({
        "addr": int(addr),
        "count": count,
        "avg_gap": avg_gap,
        "dst_count": int(dst_count),
        "mean_len": mean_len,
    })

candidates.sort(key=lambda x: (-x["count"], x["avg_gap"]))

print("\nLikely smart outlet addresses:")
for c in candidates[: args.top]:
    print(
        f"0x{c['addr']:04x}\tcount={c['count']}\tavg_gap={c['avg_gap']:.1f}s\tdst={c['dst_count']}\tmean_len={c['mean_len']:.1f}"
    )
