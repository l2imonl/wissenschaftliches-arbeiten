import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description="Identify likely window sensor addresses from Zigbee traffic"
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
parser.add_argument(
    "--max-dst",
    type=int,
    default=1,
    help="Ignore devices sending to more than this many destinations",
)
parser.add_argument(
    "--max-frame-len",
    type=int,
    default=60,
    help="Ignore devices with mean frame length above this",
)
parser.add_argument(
    "--min-cv",
    type=float,
    default=0.5,
    help="Minimum coefficient of variation of packet gaps",
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
df["wpan.dst16"] = df["wpan.dst16"].apply(_to_int)
df["frame.len"] = pd.to_numeric(df.get("frame.len"), errors="coerce")

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
    dst_count = grp["wpan.dst16"].nunique()
    if dst_count > args.max_dst:
        continue
    mean_len = grp["frame.len"].dropna().mean()
    if not np.isnan(mean_len) and mean_len > args.max_frame_len:
        continue
    gaps = np.diff(np.sort(times))
    if len(gaps) == 0:
        continue
    avg_gap = gaps.mean()
    gap_cv = gaps.std() / avg_gap if avg_gap > 0 else 0
    if avg_gap >= args.min_gap and gap_cv >= args.min_cv:
        candidates.append({
            "addr": addr_int,
            "count": len(times),
            "avg_gap": avg_gap,
            "med_gap": np.median(gaps),
        })

candidates.sort(key=lambda x: (-x["avg_gap"], x["count"]))

print("\nLikely window sensor addresses:")
for c in candidates[: args.top]:
    print(f"0x{c['addr']:04x}\tcount={c['count']}\tavg_gap={c['avg_gap']:.1f}s\tmed_gap={c['med_gap']:.1f}s")

