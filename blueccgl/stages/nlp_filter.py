"""
Stage: nlp_filter
Scans user_posts JSONL files, matches conflict keywords, assigns event windows,
and writes a filtered Parquet index of conflict-related posts.

Reads:
  config.data.sample_dir  (when config.nlp_filter.use_sample_dir == true)
  config.data.raw_dir/user_posts.tar.gz  (full data mode)

Writes:
  {output_dir}/nlp_filter/conflict_posts.parquet
  {output_dir}/nlp_filter/filter_stats.json
"""

from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path
from typing import Generator

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from blueccgl.data.events import assign_event, in_conflict_window, parse_date
from blueccgl.data.keywords import is_conflict, match_keywords


# Output schema
_SCHEMA = pa.schema([
    pa.field("post_id",                   pa.int64()),
    pa.field("user_id",                   pa.int64()),
    pa.field("timestamp",                 pa.timestamp("s")),
    pa.field("text",                      pa.string()),
    pa.field("langs",                     pa.list_(pa.string())),
    pa.field("sent_label",                pa.int8()),
    pa.field("sent_score",                pa.float32()),
    pa.field("is_repost",                 pa.bool_()),
    pa.field("repost_from",               pa.int64()),
    pa.field("reposted_author",           pa.int64()),
    pa.field("conflict_keywords_matched", pa.string()),  # JSON-encoded dict
    pa.field("event_window",              pa.string()),
])

_BATCH_SIZE = 10_000  # rows buffered before writing to parquet


def run(cfg: dict, output_dir: Path) -> None:
    out_dir = output_dir / "nlp_filter"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "conflict_posts.parquet"
    stats_path   = out_dir / "filter_stats.json"

    use_sample  = cfg["nlp_filter"].get("use_sample_dir", True)
    batch_every = cfg["nlp_filter"].get("batch_write_every", 10)

    if use_sample:
        sample_dir = Path(cfg["data"]["sample_dir"])
        if not sample_dir.exists():
            raise FileNotFoundError(f"sample_dir not found: {sample_dir}")
        file_iter   = _iter_sample_dir(sample_dir)
        total_files = len(list(sample_dir.glob("*.jsonl")))
        mode_label  = f"sample ({sample_dir})"
    else:
        # Try raw_dir first, then Data/ directly (user may not have a raw/ subdir)
        tar_path = Path(cfg["data"]["raw_dir"]) / "user_posts.tar.gz"
        if not tar_path.exists():
            tar_path = Path("Data") / "user_posts.tar.gz"
        if not tar_path.exists():
            raise FileNotFoundError(
                f"user_posts.tar.gz not found. Looked in:\n"
                f"  {Path(cfg['data']['raw_dir']) / 'user_posts.tar.gz'}\n"
                f"  Data/user_posts.tar.gz"
            )
        file_iter   = _iter_tar(tar_path)
        total_files = None  # unknown until we scan
        mode_label  = f"tar ({tar_path})"

    print(f"[nlp_filter] Mode: {mode_label}")
    print(f"[nlp_filter] Output: {parquet_path}")

    writer = None
    stats = {
        "total_posts_scanned": 0,
        "posts_in_conflict_window": 0,
        "conflict_posts": 0,
        "malformed_lines": 0,
        "files_processed": 0,
        "by_event": {},
        "by_tier": {},
    }

    batch: list[dict] = []

    def flush_batch() -> None:
        nonlocal writer
        if not batch:
            return
        table = pa.table(_batch_to_arrays(batch), schema=_SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, _SCHEMA)
        writer.write_table(table)
        batch.clear()

    with tqdm(total=total_files, unit="file", desc="nlp_filter") as pbar:
        for fname, lines in file_iter:
            stats["files_processed"] += 1
            pbar.set_postfix({"conflict": stats["conflict_posts"]}, refresh=False)
            pbar.update(1)

            for line in lines:
                stats["total_posts_scanned"] += 1
                post = _safe_parse(line)
                if post is None:
                    stats["malformed_lines"] += 1
                    continue

                dt = parse_date(post.get("date"))
                if dt is None or not in_conflict_window(dt):
                    continue
                stats["posts_in_conflict_window"] += 1

                text = post.get("text") or ""
                if not text or not is_conflict(text):
                    continue

                event = assign_event(dt)
                if event is None:
                    continue

                kw = match_keywords(text)
                stats["conflict_posts"] += 1
                stats["by_event"][event] = stats["by_event"].get(event, 0) + 1
                for tier in kw:
                    stats["by_tier"][tier] = stats["by_tier"].get(tier, 0) + 1

                batch.append({
                    "post_id":                   post.get("post_id"),
                    "user_id":                   post.get("user_id"),
                    "timestamp":                 dt,
                    "text":                      text,
                    "langs":                     post.get("langs") or [],
                    "sent_label":                post.get("sent_label"),
                    "sent_score":                post.get("sent_score"),
                    "is_repost":                 post.get("repost_from") is not None,
                    "repost_from":               post.get("repost_from"),
                    "reposted_author":           post.get("reposted_author"),
                    "conflict_keywords_matched": json.dumps(kw),
                    "event_window":              event,
                })

                if len(batch) >= _BATCH_SIZE:
                    flush_batch()

            if stats["files_processed"] % batch_every == 0:
                flush_batch()

    flush_batch()
    if writer:
        writer.close()

    stats_sorted = {
        **stats,
        "by_event": dict(sorted(stats["by_event"].items())),
        "by_tier":  dict(sorted(stats["by_tier"].items())),
    }
    stats_path.write_text(json.dumps(stats_sorted, indent=2))
    _print_summary(stats_sorted, parquet_path)


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _iter_sample_dir(sample_dir: Path) -> Generator[tuple[str, list[str]], None, None]:
    for fpath in sorted(sample_dir.glob("*.jsonl")):
        with open(fpath, encoding="utf-8") as f:
            yield fpath.name, f.readlines()


def _iter_tar(tar_path: Path) -> Generator[tuple[str, list[str]], None, None]:
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.name.endswith(".jsonl"):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            yield member.name, f.read().decode("utf-8", errors="replace").splitlines(keepends=True)


def _safe_parse(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
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


def _batch_to_arrays(batch: list[dict]) -> dict[str, pa.Array]:
    def col(key):
        return [row.get(key) for row in batch]

    return {
        "post_id":                    pa.array(col("post_id"),           type=pa.int64()),
        "user_id":                    pa.array(col("user_id"),            type=pa.int64()),
        "timestamp":                  pa.array(col("timestamp"),          type=pa.timestamp("s")),
        "text":                       pa.array(col("text"),               type=pa.string()),
        "langs":                      pa.array(col("langs"),              type=pa.list_(pa.string())),
        "sent_label":                 pa.array(
            [int(x) if x is not None else 1 for x in col("sent_label")], type=pa.int8()
        ),
        "sent_score":                 pa.array(
            [float(x) if x is not None else 0.5 for x in col("sent_score")], type=pa.float32()
        ),
        "is_repost":                  pa.array(col("is_repost"),          type=pa.bool_()),
        "repost_from":                pa.array(col("repost_from"),        type=pa.int64()),
        "reposted_author":            pa.array(col("reposted_author"),    type=pa.int64()),
        "conflict_keywords_matched":  pa.array(col("conflict_keywords_matched"), type=pa.string()),
        "event_window":               pa.array(col("event_window"),       type=pa.string()),
    }


def _print_summary(stats: dict, parquet_path: Path) -> None:
    total  = stats["total_posts_scanned"]
    conf   = stats["conflict_posts"]
    in_win = stats["posts_in_conflict_window"]
    pct    = conf / total * 100 if total > 0 else 0.0

    print(f"\n[nlp_filter] ── Summary ─────────────────────────────────────")
    print(f"  Files processed       : {stats['files_processed']:,}")
    print(f"  Total posts scanned   : {total:,}")
    print(f"  In conflict window    : {in_win:,}  ({in_win/total*100:.1f}%)")
    print(f"  Conflict posts found  : {conf:,}  ({pct:.2f}%)")
    print(f"  Malformed lines       : {stats['malformed_lines']:,}")
    print(f"\n  By event window:")
    for event, count in stats["by_event"].items():
        bar = "█" * min(40, max(1, count * 40 // max(stats["by_event"].values(), default=1)))
        print(f"    {event:<28} {count:>5,}  {bar}")
    print(f"\n  By keyword tier:")
    for tier, count in stats["by_tier"].items():
        print(f"    {tier:<20} {count:>5,}")
    print(f"\n  Output: {parquet_path}")
    print(f"[nlp_filter] ────────────────────────────────────────────────\n")
