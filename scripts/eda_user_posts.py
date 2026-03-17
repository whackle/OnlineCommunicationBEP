"""
EDA on user_posts sample (100 JSONL files)
Covers: schema, temporal range, volume, languages, sentiment, reposts, conflict keywords
"""

import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("/home/rayanmahmoud/Documents/repos/OnlineCommunicationBEP/Data/user_posts")

# ── Conflict keyword tiers ──────────────────────────────────────────────────
TIER1 = [
    r"\bgaza\b", r"\bhamas\b", r"\bisrael\b", r"\bpalestine\b", r"\bpalestinian\b",
    r"\bidf\b", r"\bnetanyahu\b", r"\bceasefire\b", r"\bhostage\b", r"\bhostages\b",
    r"\bwest\s+bank\b", r"\brafah\b", r"\bgenocide\b", r"\boccupation\b",
    r"\bzionist\b", r"\bzionism\b", r"\bapartheid\b", r"\bnakba\b", r"\bintifada\b",
    r"\bal[\-\s]aqsa\b", r"\bal[\-\s]shifa\b", r"\bunrwa\b",
]

TIER2 = [
    r"#freepalestine", r"#standwithisrael", r"#gazaunderattack", r"#ceasefirenow",
    r"#bringthemhome", r"#fromtherivertothesea", r"#october7", r"#gazagenocide",
    r"#istandwithisrael", r"#savegaza", r"#freegaza", r"#stopthegenocide",
    r"#israelhamaswar", r"#israelpalestine", r"#gazawar",
]

TIER3_ARABIC = [
    r"فلسطين", r"غزة", r"حماس", r"طوفان.?الأقصى",
    r"المقاومة", r"شهداء", r"الأقصى", r"إبادة",
]

TIER4_HEBREW = [
    r"חרבות.?ברזל", r"עזה", r"חמאס", r"חטופים", r"ביחד.?ננצח",
]

TIER5_ENTITIES = [
    r"\bkhan\s+younis\b", r"\bjabalia\b", r"\bnuseirat\b", r"\bdeir\s+el[\-\s]balah\b",
    r"\bbe'eri\b", r"\bsderot\b", r"\bhezbollah\b", r"\bhouthi\b", r"\bicj\b",
    r"\bprcs\b", r"\bal[\-\s]ahli\b", r"\bnasser\s+hospital\b",
    r"\brefaat\s+alareer\b", r"\bhind\s+rajab\b", r"\bwael\s+dahdouh\b",
    r"\baaron\s+bushnell\b", r"\bgallant\b", r"\bmuhammad\s+deif\b",
    r"\babu\s+obeida\b", r"\bismail\s+haniyeh\b", r"\bmarwan\s+issa\b",
]

ALL_PATTERNS = TIER1 + TIER2 + TIER3_ARABIC + TIER4_HEBREW + TIER5_ENTITIES
CONFLICT_RE = re.compile("|".join(ALL_PATTERNS), re.IGNORECASE)

# ── Mega-event windows ──────────────────────────────────────────────────────
MEGA_EVENTS = {
    "E1_Oct7_Attack":       (datetime(2023, 10, 4), datetime(2023, 10, 10, 23, 59)),
    "E2_AlAhli_Hospital":   (datetime(2023, 10, 14), datetime(2023, 10, 20, 23, 59)),
    "E3_Ceasefire":          (datetime(2023, 11, 21), datetime(2023, 12, 3, 23, 59)),
    "E4_ICJ_Ruling":         (datetime(2024, 1, 23), datetime(2024, 1, 29, 23, 59)),
    "E5_Bushnell":           (datetime(2024, 2, 22), datetime(2024, 2, 28, 23, 59)),
    "E6_Flour_Massacre":     (datetime(2024, 2, 26), datetime(2024, 3, 3, 23, 59)),
}


def parse_date(d):
    """Parse integer date like 202310071230 -> datetime"""
    s = str(d)
    try:
        return datetime.strptime(s, "%Y%m%d%H%M")
    except ValueError:
        return None


def assign_event(dt):
    if dt is None:
        return None
    for name, (start, end) in MEGA_EVENTS.items():
        if start <= dt <= end:
            return name
    return "baseline"


# ── Main scan ───────────────────────────────────────────────────────────────
total_posts = 0
total_users = set()
total_files = 0

# Temporal
dates = []
year_month_counts = Counter()

# Language
lang_counts = Counter()
no_lang_count = 0

# Sentiment
sent_counts = Counter()
no_sent_count = 0

# Post types
original_count = 0
repost_count = 0
reply_count = 0
quote_count = 0

# Engagement
repost_values = []
like_values = []
reply_values_engagement = []

# Conflict
conflict_posts = 0
conflict_by_tier = Counter()
conflict_by_event = Counter()
conflict_posts_with_reposts = 0
conflict_repost_counts = []
conflict_examples = []

# Field presence
fields_seen = Counter()

# Per-file stats
file_stats = []

print("=" * 70)
print("BlueCCGL EDA — user_posts sample (100 JSONL files)")
print("=" * 70)

files = sorted(DATA_DIR.glob("*.jsonl"))
total_files = len(files)

for fpath in files:
    file_posts = 0
    file_users = set()
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Some files have multiple JSON objects per line (no newline separator)
            # Try parsing; if fails, try splitting
            try:
                post = json.loads(line)
                posts_in_line = [post]
            except json.JSONDecodeError:
                # Try to split concatenated JSON objects
                # Simple heuristic: split on }{ 
                parts = re.split(r'\}\s*\{', line)
                posts_in_line = []
                for i, part in enumerate(parts):
                    if i > 0:
                        part = '{' + part
                    if i < len(parts) - 1:
                        part = part + '}'
                    try:
                        posts_in_line.append(json.loads(part))
                    except json.JSONDecodeError:
                        continue

            for post in posts_in_line:
                total_posts += 1
                file_posts += 1

                uid = post.get("user_id")
                if uid is not None:
                    total_users.add(uid)
                    file_users.add(uid)

                # Field presence
                for k in post.keys():
                    fields_seen[k] += 1

                # Date
                dt = parse_date(post.get("date"))
                if dt:
                    dates.append(dt)
                    ym = dt.strftime("%Y-%m")
                    year_month_counts[ym] += 1

                # Language
                langs = post.get("langs")
                if langs and len(langs) > 0:
                    for lang in langs:
                        lang_counts[lang] += 1
                else:
                    no_lang_count += 1

                # Sentiment
                sl = post.get("sent_label")
                if sl is not None:
                    sent_counts[sl] += 1
                else:
                    no_sent_count += 1

                # Post type
                if post.get("repost_from") is not None:
                    repost_count += 1
                elif post.get("reply_to") is not None or post.get("replied_author") is not None:
                    reply_count += 1
                elif post.get("quotes") is not None or post.get("quoted_author") is not None:
                    quote_count += 1
                else:
                    original_count += 1

                # Engagement
                rc = post.get("repost_count")
                lc = post.get("like_count")
                rpc = post.get("reply_count")
                if rc is not None:
                    repost_values.append(rc)
                if lc is not None:
                    like_values.append(lc)
                if rpc is not None:
                    reply_values_engagement.append(rpc)

                # Conflict keyword matching
                text = post.get("text", "")
                if text and CONFLICT_RE.search(text):
                    conflict_posts += 1
                    # Which tiers matched
                    for i, pat in enumerate(TIER1):
                        if re.search(pat, text, re.IGNORECASE):
                            conflict_by_tier["Tier1_Core"] += 1
                            break
                    for i, pat in enumerate(TIER2):
                        if re.search(pat, text, re.IGNORECASE):
                            conflict_by_tier["Tier2_Hashtags"] += 1
                            break
                    for pat in TIER3_ARABIC:
                        if re.search(pat, text, re.IGNORECASE):
                            conflict_by_tier["Tier3_Arabic"] += 1
                            break
                    for pat in TIER4_HEBREW:
                        if re.search(pat, text, re.IGNORECASE):
                            conflict_by_tier["Tier4_Hebrew"] += 1
                            break
                    for pat in TIER5_ENTITIES:
                        if re.search(pat, text, re.IGNORECASE):
                            conflict_by_tier["Tier5_Entities"] += 1
                            break

                    # Event window
                    event = assign_event(dt)
                    if event:
                        conflict_by_event[event] += 1

                    # Cascade potential (has reposts?)
                    if rc and rc >= 1:
                        conflict_posts_with_reposts += 1
                        conflict_repost_counts.append(rc)

                    # Collect some examples (first 20)
                    if len(conflict_examples) < 20:
                        conflict_examples.append({
                            "date": str(dt) if dt else "?",
                            "text": text[:200],
                            "reposts": rc,
                            "likes": lc,
                            "langs": langs,
                            "sent_label": sl,
                        })

    file_stats.append({"file": fpath.name, "posts": file_posts, "users": len(file_users)})

# ── Report ──────────────────────────────────────────────────────────────────

print(f"\n📁 FILES: {total_files}")
print(f"📝 TOTAL POSTS: {total_posts:,}")
print(f"👤 UNIQUE USERS: {len(total_users):,}")

# Per-file distribution
post_counts = [fs["posts"] for fs in file_stats]
user_counts = [fs["users"] for fs in file_stats]
print(f"\n── Per-File Distribution ──")
print(f"   Posts/file: min={min(post_counts):,}  max={max(post_counts):,}  "
      f"mean={sum(post_counts)/len(post_counts):,.0f}  median={sorted(post_counts)[len(post_counts)//2]:,}")
print(f"   Users/file: min={min(user_counts):,}  max={max(user_counts):,}  "
      f"mean={sum(user_counts)/len(user_counts):,.0f}")

# Schema
print(f"\n── Field Presence (out of {total_posts:,} posts) ──")
for field, count in fields_seen.most_common():
    pct = count / total_posts * 100
    print(f"   {field:25s}: {count:>10,}  ({pct:5.1f}%)")

# Temporal
print(f"\n── Temporal Range ──")
if dates:
    dates_sorted = sorted(dates)
    print(f"   Earliest post: {dates_sorted[0]}")
    print(f"   Latest post:   {dates_sorted[-1]}")
    print(f"   Span:          {(dates_sorted[-1] - dates_sorted[0]).days} days")
    
    # Posts in conflict window (Oct 1 2023 - Mar 18 2024)
    conflict_window_start = datetime(2023, 10, 1)
    conflict_window_end = datetime(2024, 3, 18, 23, 59)
    in_window = sum(1 for d in dates if conflict_window_start <= d <= conflict_window_end)
    print(f"   Posts in conflict window (Oct 2023 - Mar 2024): {in_window:,} ({in_window/total_posts*100:.1f}%)")

    print(f"\n── Monthly Post Volume ──")
    for ym in sorted(year_month_counts.keys()):
        bar = "█" * (year_month_counts[ym] // 500)
        print(f"   {ym}: {year_month_counts[ym]:>8,}  {bar}")

# Language
print(f"\n── Language Distribution (top 20) ──")
print(f"   Posts with no lang tag: {no_lang_count:,} ({no_lang_count/total_posts*100:.1f}%)")
for lang, count in lang_counts.most_common(20):
    pct = count / total_posts * 100
    print(f"   {lang:10s}: {count:>8,}  ({pct:5.1f}%)")

# Sentiment
print(f"\n── Sentiment Distribution ──")
sent_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
for sl, label in sent_labels.items():
    count = sent_counts.get(sl, 0)
    pct = count / total_posts * 100
    print(f"   {label:10s}: {count:>8,}  ({pct:5.1f}%)")
print(f"   No label : {no_sent_count:>8,}  ({no_sent_count/total_posts*100:5.1f}%)")

# Post types
print(f"\n── Post Types ──")
for label, count in [("Original", original_count), ("Repost", repost_count),
                      ("Reply", reply_count), ("Quote", quote_count)]:
    pct = count / total_posts * 100
    print(f"   {label:10s}: {count:>8,}  ({pct:5.1f}%)")

# Engagement
print(f"\n── Engagement Stats ──")
if repost_values:
    rv = sorted(repost_values)
    print(f"   Repost count: min={rv[0]}  median={rv[len(rv)//2]}  "
          f"mean={sum(rv)/len(rv):.1f}  p95={rv[int(len(rv)*0.95)]}  "
          f"p99={rv[int(len(rv)*0.99)]}  max={rv[-1]}")
if like_values:
    lv = sorted(like_values)
    print(f"   Like count:   min={lv[0]}  median={lv[len(lv)//2]}  "
          f"mean={sum(lv)/len(lv):.1f}  p95={lv[int(len(lv)*0.95)]}  "
          f"p99={lv[int(len(lv)*0.99)]}  max={lv[-1]}")
if reply_values_engagement:
    rpv = sorted(reply_values_engagement)
    print(f"   Reply count:  min={rpv[0]}  median={rpv[len(rpv)//2]}  "
          f"mean={sum(rpv)/len(rpv):.1f}  p95={rpv[int(len(rpv)*0.95)]}  "
          f"p99={rpv[int(len(rpv)*0.99)]}  max={rpv[-1]}")

# ── CONFLICT ANALYSIS ───────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"🔴 CONFLICT KEYWORD ANALYSIS")
print(f"{'=' * 70}")
print(f"\n   Conflict-related posts: {conflict_posts:,} / {total_posts:,} "
      f"({conflict_posts/total_posts*100:.2f}%)")

print(f"\n── Matches by Tier ──")
for tier, count in conflict_by_tier.most_common():
    print(f"   {tier:20s}: {count:>6,}")

print(f"\n── Conflict Posts by Event Window ──")
for event in ["E1_Oct7_Attack", "E2_AlAhli_Hospital", "E3_Ceasefire",
              "E4_ICJ_Ruling", "E5_Bushnell", "E6_Flour_Massacre", "baseline"]:
    count = conflict_by_event.get(event, 0)
    print(f"   {event:25s}: {count:>6,}")

print(f"\n── Cascade Potential ──")
print(f"   Conflict posts with ≥1 repost:  {conflict_posts_with_reposts:,}")
if conflict_repost_counts:
    crc = sorted(conflict_repost_counts)
    gte5 = sum(1 for x in crc if x >= 5)
    gte10 = sum(1 for x in crc if x >= 10)
    gte50 = sum(1 for x in crc if x >= 50)
    gte100 = sum(1 for x in crc if x >= 100)
    print(f"   Conflict posts with ≥5 reposts:  {gte5:,}")
    print(f"   Conflict posts with ≥10 reposts: {gte10:,}")
    print(f"   Conflict posts with ≥50 reposts: {gte50:,}")
    print(f"   Conflict posts with ≥100 reposts: {gte100:,}")
    print(f"   Repost distribution: min={crc[0]}  median={crc[len(crc)//2]}  "
          f"mean={sum(crc)/len(crc):.1f}  max={crc[-1]}")

# Extrapolation
print(f"\n── Extrapolation to Full Dataset ──")
# Full dataset: 235M posts, 4.1M users. Our sample: {total_posts} posts, {len(total_users)} users.
scale_factor = 235_567_116 / total_posts if total_posts > 0 else 0
print(f"   Sample: {total_posts:,} posts from {len(total_users):,} users")
print(f"   Full dataset: 235,567,116 posts from 4,099,699 users")
print(f"   Scale factor: ~{scale_factor:,.0f}x")
print(f"   Estimated conflict posts (full): ~{int(conflict_posts * scale_factor):,}")
if conflict_repost_counts:
    print(f"   Estimated conflict posts with ≥5 reposts (full): ~{int(gte5 * scale_factor):,}")
    print(f"   Estimated conflict posts with ≥10 reposts (full): ~{int(gte10 * scale_factor):,}")

print(f"\n── Sample Conflict Posts ──")
for i, ex in enumerate(conflict_examples[:15]):
    print(f"\n   [{i+1}] {ex['date']}  | reposts={ex['reposts']}  likes={ex['likes']}  "
          f"sent={ex['sent_label']}  langs={ex['langs']}")
    print(f"       \"{ex['text']}\"")

print(f"\n{'=' * 70}")
print("EDA COMPLETE")
print(f"{'=' * 70}")
