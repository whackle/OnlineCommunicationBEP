#!/usr/bin/env python
"""
quick_look.py — Run the NLP filter on sample data and display results.

Usage (from repo root):
    .venv/bin/python scripts/quick_look.py

No arguments needed. Uses config.yaml defaults (sample mode).
Produces outputs/nlp_filter/conflict_posts.parquet on first run,
then reads it back and prints a rich summary + 20 example posts.
"""

import json
import sys
from pathlib import Path

# ── Add repo root to path so blueccgl package is importable ──────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blueccgl.config import load_config
from blueccgl.seed import seed_everything

seed_everything(42)

cfg = load_config(ROOT / "config.yaml")
output_dir = ROOT / "outputs"


# ── Step 1: Run the filter if output doesn't exist ──────────────────────────
parquet_path = output_dir / "nlp_filter" / "conflict_posts.parquet"
stats_path   = output_dir / "nlp_filter" / "filter_stats.json"

if not parquet_path.exists():
    print("Running nlp_filter stage on sample data...\n")
    from blueccgl.stages.nlp_filter import run as nlp_run
    nlp_run(cfg, output_dir)
else:
    print(f"Found existing output: {parquet_path}")
    print("(Delete outputs/nlp_filter/ to re-run the filter)\n")


# ── Step 2: Load and display results ─────────────────────────────────────────
import pyarrow.parquet as pq
import pandas as pd

table = pq.read_table(parquet_path)
df = table.to_pandas()

stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

print("=" * 65)
print("  SAMPLE DATA — NLP FILTER RESULTS")
print("=" * 65)

total_scanned = stats.get("total_posts_scanned", "?")
conflict_n    = len(df)
pct           = conflict_n / total_scanned * 100 if isinstance(total_scanned, int) else 0.0

print(f"\n  Posts scanned          : {total_scanned:,}" if isinstance(total_scanned, int) else f"\n  Posts scanned: ?")
print(f"  Conflict posts found   : {conflict_n:,}  ({pct:.2f}%)")
print(f"  Unique users           : {df['user_id'].nunique():,}")
print(f"  Reposts                : {df['is_repost'].sum():,}  ({df['is_repost'].mean()*100:.1f}%)")
print(f"  Original posts         : {(~df['is_repost']).sum():,}")

# Event window distribution
print(f"\n{'─'*65}")
print("  BY EVENT WINDOW")
print(f"{'─'*65}")
event_counts = df["event_window"].value_counts()
event_order  = [
    "E1_Oct7_Attack", "E2_AlAhli_Hospital", "E3_Ceasefire",
    "E4_ICJ_Ruling", "E5_Bushnell", "E6_Flour_Massacre", "baseline",
]
max_count = event_counts.max() if len(event_counts) > 0 else 1
for event in event_order:
    count = event_counts.get(event, 0)
    bar   = "█" * int(count / max_count * 30)
    print(f"  {event:<28} {count:>4}  {bar}")

# Sentiment distribution
print(f"\n{'─'*65}")
print("  SENTIMENT DISTRIBUTION")
print(f"{'─'*65}")
sent_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
for label, name in sent_map.items():
    count = (df["sent_label"] == label).sum()
    pct_s = count / conflict_n * 100 if conflict_n > 0 else 0
    bar   = "█" * int(pct_s / 2)
    print(f"  {name:<10} {count:>4}  ({pct_s:5.1f}%)  {bar}")

# Keyword tier breakdown
if stats.get("by_tier"):
    print(f"\n{'─'*65}")
    print("  KEYWORD TIER HITS")
    print(f"{'─'*65}")
    for tier, count in sorted(stats["by_tier"].items(), key=lambda x: -x[1]):
        print(f"  {tier:<20} {count:>4}")

# Language breakdown (top 5)
print(f"\n{'─'*65}")
print("  LANGUAGE BREAKDOWN (top 5)")
print(f"{'─'*65}")
lang_counts = {}
for langs_list in df["langs"]:
    for lang in (langs_list if langs_list is not None and len(langs_list) > 0 else []):
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
for lang, cnt in sorted(lang_counts.items(), key=lambda x: -x[1])[:5]:
    print(f"  {lang:<10} {cnt:>4}")

# Example posts
print(f"\n{'─'*65}")
print("  20 EXAMPLE CONFLICT POSTS")
print(f"{'─'*65}")
sample = df.sample(min(20, len(df)), random_state=42).sort_values("timestamp")
for _, row in sample.iterrows():
    kw_dict = json.loads(row["conflict_keywords_matched"]) if row["conflict_keywords_matched"] else {}
    kw_flat = [kw for hits in kw_dict.values() for kw in hits][:3]
    sent_label = {0: "NEG", 1: "NEU", 2: "POS"}.get(int(row["sent_label"]), "?")
    repost_flag = "[R]" if row["is_repost"] else "   "
    ts = str(row["timestamp"])[:16]
    text = str(row["text"])[:100].replace("\n", " ")
    print(f"\n  {repost_flag} {ts}  [{row['event_window']}]  sent={sent_label}")
    print(f"  keywords: {kw_flat}")
    print(f"  \"{text}\"")

print(f"\n{'='*65}")
print(f"  Full data: {parquet_path}")
print(f"  Stats:     {stats_path}")
print(f"{'='*65}\n")
