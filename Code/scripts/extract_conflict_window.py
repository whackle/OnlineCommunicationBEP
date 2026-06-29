#!/usr/bin/env python
"""
Stream user_posts.tar.gz and extract a lean Parquet of ALL posts
in the conflict window (Oct 1 2023 – Mar 18 2024).

Only stores cascade-relevant fields: post_id, user_id, date, repost_from,
reposted_author, sent_label, sent_score. Text is NOT stored (saves ~90% space).

Usage:
    .venv/bin/python scripts/extract_conflict_window.py

Output:
    Data/conflict_window_posts.parquet  (~1-2 GB estimated)

This is a one-time preprocessing step. The cascade_construction stage
reads this Parquet for reposter discovery instead of re-scanning the tar.
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blueccgl.data.events import CONFLICT_START, CONFLICT_END, parse_date

TAR_PATH    = ROOT / "Data" / "user_posts.tar.gz"
OUTPUT_PATH = ROOT / "Data" / "conflict_window_posts.parquet"

# Schema: lean fields only — no text
SCHEMA = pa.schema([
    pa.field("post_id",         pa.int64()),
    pa.field("user_id",         pa.int64()),
    pa.field("timestamp",       pa.timestamp("s")),
    pa.field("repost_from",     pa.int64()),
    pa.field("reposted_author", pa.int64()),
    pa.field("sent_label",      pa.int8()),
    pa.field("sent_score",      pa.float32()),
])

FLUSH_EVERY = 500_000  # rows between Parquet writes


def main():
    import tarfile
    from tqdm import tqdm

    if not TAR_PATH.exists():
        print(f"ERROR: {TAR_PATH} not found")
        sys.exit(1)

    if OUTPUT_PATH.exists():
        print(f"Output already exists: {OUTPUT_PATH}")
        print("Delete it to re-extract.")
        sys.exit(0)

    print(f"Streaming: {TAR_PATH} ({TAR_PATH.stat().st_size / 1e9:.1f} GB)")
    print(f"Output:    {OUTPUT_PATH}")
    print(f"Window:    {CONFLICT_START.date()} → {CONFLICT_END.date()}")
    print()

    writer = pq.ParquetWriter(OUTPUT_PATH, SCHEMA, compression="snappy")
    batch: list[dict] = []

    total_posts    = 0
    window_posts   = 0
    files_done     = 0
    malformed      = 0
    t0 = time.time()

    with tarfile.open(TAR_PATH, "r:gz") as tf:
        for member in tf:
            if not member.name.endswith(".jsonl"):
                continue

            f = tf.extractfile(member)
            if f is None:
                continue

            files_done += 1
            raw = f.read().decode("utf-8", errors="replace")

            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue

                total_posts += 1
                post = _safe_parse(line)
                if post is None:
                    malformed += 1
                    continue

                dt = parse_date(post.get("date"))
                if dt is None:
                    continue
                if not (CONFLICT_START <= dt <= CONFLICT_END):
                    continue

                window_posts += 1
                batch.append({
                    "post_id":         post.get("post_id"),
                    "user_id":         post.get("user_id"),
                    "timestamp":       dt,
                    "repost_from":     post.get("repost_from"),
                    "reposted_author": post.get("reposted_author"),
                    "sent_label":      post.get("sent_label"),
                    "sent_score":      post.get("sent_score"),
                })

                if len(batch) >= FLUSH_EVERY:
                    _flush(writer, batch)
                    elapsed = time.time() - t0
                    rate = total_posts / elapsed
                    print(
                        f"  files={files_done}  "
                        f"total={total_posts:,}  "
                        f"window={window_posts:,}  "
                        f"rate={rate:,.0f} posts/s  "
                        f"elapsed={elapsed:.0f}s",
                        flush=True,
                    )

            # File-level progress (prints every 100 files)
            if files_done % 100 == 0:
                elapsed = time.time() - t0
                print(
                    f"  files={files_done:,}  "
                    f"posts={total_posts:,}  "
                    f"in_window={window_posts:,}  "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

    _flush(writer, batch)
    writer.close()

    elapsed = time.time() - t0
    size_mb = OUTPUT_PATH.stat().st_size / 1e6

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Files processed   : {files_done:,}")
    print(f"  Total posts       : {total_posts:,}")
    print(f"  Conflict window   : {window_posts:,}  ({window_posts/total_posts*100:.1f}%)")
    print(f"  Malformed lines   : {malformed:,}")
    print(f"  Output size       : {size_mb:.0f} MB")
    print(f"  Time              : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output            : {OUTPUT_PATH}")
    print(f"{'='*60}")


def _flush(writer: pq.ParquetWriter, batch: list[dict]) -> None:
    if not batch:
        return
    arrays = {
        "post_id":         pa.array([r["post_id"] for r in batch],         type=pa.int64()),
        "user_id":         pa.array([r["user_id"] for r in batch],         type=pa.int64()),
        "timestamp":       pa.array([r["timestamp"] for r in batch],       type=pa.timestamp("s")),
        "repost_from":     pa.array([r["repost_from"] for r in batch],     type=pa.int64()),
        "reposted_author": pa.array([r["reposted_author"] for r in batch], type=pa.int64()),
        "sent_label":      pa.array(
            [int(x) if x is not None else 1 for x in (r["sent_label"] for r in batch)],
            type=pa.int8()
        ),
        "sent_score":      pa.array(
            [float(x) if x is not None else 0.5 for x in (r["sent_score"] for r in batch)],
            type=pa.float32()
        ),
    }
    writer.write_table(pa.table(arrays, schema=SCHEMA))
    batch.clear()


def _safe_parse(line: str) -> dict | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        parts = re.split(r"\}\s*\{", line)
        if len(parts) > 1:
            try:
                return json.loads(parts[0] + "}")
            except json.JSONDecodeError:
                pass
        return None


if __name__ == "__main__":
    main()
