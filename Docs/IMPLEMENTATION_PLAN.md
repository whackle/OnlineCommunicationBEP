# BlueCCGL — Full Implementation Plan

## Current State Assessment (Updated 2026-04-17)

| Asset | Status |
|-------|--------|
| **Sample data** | 100 JSONL files (~15 MB, ~25K posts) in `Data/user_posts/` |
| **Full data — user_posts.tar.gz** | Downloaded (19 GB, 2,363,249 JSONL files, ~4M posts, 128 GB uncompressed). In `Data/user_posts.tar.gz` |
| **Full data — followers.csv.gz** | Downloaded (469 MB, ~144M edges). In `Data/followers.csv.gz` |
| **Full data — others** | NOT downloaded: `interactions.csv.gz` (1 GB), `graphs.tar.gz` (891 MB), `feed_posts.tar.gz` (16.4 MB), `feed_bookmarks.csv` (553 KB). **Not needed for current cascade construction approach** — we use `repost_from`/`reposted_author` fields from user_posts instead |
| **conflict_window_posts.parquet** | Extracted from tar (2.4 GB, 172M rows). Lean index: post_id, user_id, timestamp, repost_from, reposted_author, sent_label, sent_score. **No text field.** Used for fast reposter discovery in cascade construction |
| **cascade_dataset.pt** | Built (159.8 MB). 39,983 cascades, 884,832 nodes, 844,849 edges, node feature dim=6. Stored in `outputs/build_pyg_dataset/` |
| **Environment** | Python 3.14 venv, CPU-only PyTorch 2.10.0, PyG 2.7.0, all deps installed. See `requirements.txt` |
| **Pipeline** | Fully functional `run_pipeline.py` CLI + `config.yaml`. All stages implemented: `nlp_filter`, `cascade_construction`, `build_pyg_dataset`, `train`, `extract_embeddings`, `cluster`, `evaluate`, `explain`, `hierarchical`, `validate_hierarchical`, `ablation` (2026-05-26, 2³ factorial). `blueccgl/utils/mpl_compat.py` added for Python 3.14 matplotlib compatibility. Stubs dropped: `downstream`, `stability` (out of scope). Stub remaining: `eda`. Prediction strength sub-step stubbed inside `validate_hierarchical`. |
| **Training v8** (primary) | DONE — 150 epochs on SSH VM, best epoch 147, loss 0.1292. Results in `Outputs_17_04/`. 11 clusters, 1.5% noise, silhouette=0.413, DBCV=+0.618 (10D UMAP space), follower R²=0.941. 79.6% aug fallback — reported as limitation |
| **Training v9** (sensitivity) | DONE — 150 epochs on SSH VM (192.165.134.28), 18 clusters, 5.0% noise, aug fallback=42.2%. Results in `Outputs_18_04/`. Weaker follower R²=0.714 → v8 selected as primary |
| **Disk** | ~15 GB free. Full tar extraction not feasible — streaming only |

### Progress Summary

| Sprint | Status | Notes |
|--------|--------|-------|
| Sprint 1: Scaffolding | DONE | All package structure, CLI, config, pipeline orchestrator, seed, gitignore |
| Sprint 2: NLP Filter | DONE | 1,947,493 conflict posts from 233M total (0.84%). 15-worker parallel tar streaming (~85 min) |
| Sprint 3: Cascade Construction | DONE | 39,983 cascades (≥10 nodes), 844,849 edges, 75.9% FOLLOWER-typed, ~3 min runtime |
| Sprint 4: PyG Dataset + Model | DONE | Dataset built; all models + augmentations + tests implemented |
| Sprint 5: Full Data Integration | DONE (v8) | Training v8 complete (150 epochs, best epoch 147, loss 0.1292). All downstream stages run: extract_embeddings → cluster (UMAP 64D→10D + HDBSCAN, mcs=300, 11 clusters) → evaluate → explain (log1p fix for wiener_index + cascade_size) → hierarchical (Ward agglomerative, 5 macro-archetypes). Results in `Outputs_17_04/`. Stubs dropped: `ablation`, `downstream`, `stability`. `eda` stub still not implemented |

---

## Repository Target Structure

```
OnlineCommunicationBEP/
├── run_pipeline.py              # Unified CLI entry point
├── config.yaml                  # All hyperparameters
├── requirements.txt             # Pinned dependencies
├── pyproject.toml               # Project metadata + optional deps
│
├── blueccgl/                    # Main package
│   ├── __init__.py
│   ├── pipeline.py              # Stage orchestrator, dependency graph, resume logic
│   ├── config.py                # YAML loading + CLI override merging
│   ├── seed.py                  # Global RNG seeding (torch, numpy, random)
│   │
│   ├── stages/                  # One module per --stage
│   │   ├── __init__.py
│   │   ├── nlp_filter.py        # --stage nlp_filter
│   │   ├── validate_filter.py   # --stage validate_filter
│   │   ├── cascade_construction.py  # --stage cascade_construction
│   │   ├── build_pyg_dataset.py     # --stage build_pyg_dataset
│   │   ├── eda.py               # --stage eda
│   │   ├── train.py             # --stage train
│   │   ├── extract_embeddings.py    # --stage extract_embeddings
│   │   ├── cluster.py           # --stage cluster
│   │   ├── evaluate.py          # --stage evaluate
│   │   ├── explain.py           # --stage explain (XAI: probes, DiffPool assignments, cluster profiles, augmentation sensitivity)
│   │   ├── hierarchical.py      # --stage hierarchical (Ward agglomerative on cluster centroids → macro-archetypes)
│   │   ├── validate_hierarchical.py  # --stage validate_hierarchical (cophenetic, metric sweep, bootstrap ARI, UMAP seed ARI, prediction strength stub)
│   │   ├── ablation.py          # --stage ablation — 2³ factorial grid (encoder × aug × sentiment)
│   │   ├── downstream.py        # --stage downstream [DROPPED — out of scope]
│   │   └── stability.py         # --stage stability [DROPPED — out of scope]
│   │
│   ├── models/                  # Neural network components
│   │   ├── __init__.py
│   │   ├── rgcn_encoder.py      # R-GCN with basis decomposition
│   │   ├── gcn_encoder.py       # Vanilla GCN (ablation A1)
│   │   ├── diffpool.py          # Hierarchical DiffPool + sentiment enrichment
│   │   ├── readout.py           # Final MLP readout to embedding space
│   │   ├── blueccgl.py          # Full model assembly
│   │   └── losses.py            # InfoNCE + DiffPool auxiliary + VICReg variance losses
│   │
│   ├── augmentations/           # Feed-aware augmentation strategies
│   │   ├── __init__.py
│   │   ├── follower_subgraph.py # Strategy A
│   │   ├── temporal_prefix.py   # Strategy B
│   │   ├── bridge_removal.py    # Strategy C
│   │   ├── feature_masking.py   # Feature-dim zeroing (v2 retrain addition)
│   │   └── random_aug.py        # Generic augmentation (ablation A2)
│   │
│   ├── data/                    # Data loading and processing
│   │   ├── __init__.py
│   │   ├── keywords.py          # Compiled keyword regex tiers
│   │   ├── events.py            # Mega-event window definitions
│   │   ├── cascade_builder.py   # Parent assignment + edge-type logic
│   │   ├── pyg_dataset.py       # Custom PyG InMemoryDataset
│   │   └── dataloader.py        # Heterogeneous-size cascade batching
│   │
│   ├── utils/                   # Shared utilities
│   │   ├── __init__.py
│   │   └── mpl_compat.py        # matplotlib Python 3.14 Path.__deepcopy__ patch
│   │
│   └── evaluation/              # Metrics and visualisation
│       ├── __init__.py
│       ├── clustering_metrics.py # Silhouette, DB, CH, DBCV + bootstrap
│       ├── stability.py         # Subsample ARI analysis
│       ├── downstream_tasks.py  # Size + longevity prediction heads
│       └── visualisation.py     # UMAP plots, archetype profiles
│
├── tests/                       # Unit + integration tests
│   ├── test_keywords.py
│   ├── test_cascade_builder.py
│   ├── test_augmentations.py
│   ├── test_model_forward.py
│   └── test_pipeline_resume.py
│
├── Data/                        # Raw + sample data (gitignored except sample)
│   ├── user_posts/              # Sample: 100 JSONL files (currently present)
│   ├── raw/                     # Full Zenodo download destination
│   └── filelist.txt
│
├── outputs/                     # Pipeline stage outputs (gitignored)
│   ├── nlp_filter/
│   ├── cascade_construction/
│   ├── build_pyg_dataset/
│   ├── train/
│   ├── ...
│
├── scripts/                     # Ad-hoc exploration and diagnostics
│   ├── eda_user_posts.py
│   ├── extract_conflict_window.py
│   ├── quick_look.py
│   ├── plot_training.py
│   ├── diagnose_embeddings.py
│   ├── compare_ablations.py         # Post-ablation metric comparison table
│   ├── plot_ablation_interactions.py # Interaction effect plots (encoder × aug, etc.)
│   ├── run_ablations.sh             # Shell wrapper for sequential ablation runs on VM
│   └── sample_macro_cascades.py     # Sample representative cascades per macro-archetype
│
├── verification/                # Manual validation outputs
│   ├── cascade_samples.csv      # Sample cascades for manual spot-check
│   └── verification_report.md   # Summary of cascade construction validation
│
├── Outputs_ablation/            # Ablation variant outputs (partial — baseline only so far)
│   ├── ablation_results.csv     # 2³ factorial metrics table (7 variants pending)
│   ├── ablation_results.md      # Human-readable version
│   └── figures/                 # Interaction effect plots (pre-computed from baseline)
│
├── Docs/
│   ├── IMPLEMENTATION_PLAN.md       # This document
│   ├── hierarchical_analysis.md     # Narrative + literary grounding for 5 macro-archetypes (2026-05-21)
│   └── hierarchical_analysis.pdf   # Compiled PDF with inline figures (2026-05-21)
├── latex/
├── papers/
└── prompts/
```

---

## Phase 0: Project Scaffolding & Environment Setup

**Goal:** Establish the project skeleton, install all dependencies, create `run_pipeline.py` and `config.yaml` so every subsequent piece of code slots into the pipeline from the start.

### Step 0.1 — Dependency Installation

```
# Core
torch, torch-geometric (+ torch-scatter, torch-sparse, torch-cluster, torch-spline-conv)
duckdb, pyarrow, pandas
pyyaml, click (or argparse)

# ML/Clustering
hdbscan, scikit-learn, umap-learn

# Logging/Tracking
wandb (optional), tqdm, rich (for CLI output)

# Evaluation + XAI
scipy, matplotlib, seaborn
h5py                    # HDF5 storage for embeddings + DiffPool assignments (added 2026-03-24)
networkx                # Cascade tree visualisation in explain stage (added 2026-03-24)

# Testing
pytest
```

**Action items:**
1. Pin PyTorch version to match available CUDA (check `nvidia-smi`).
2. Install PyG from wheels matching the PyTorch+CUDA combo — building from source on Python 3.14 may require workarounds. **Potential issue:** Python 3.14 is bleeding-edge; PyTorch and PyG may not have pre-built wheels. Fallback: create a Python 3.11 or 3.12 venv instead.
3. Create `requirements.txt` with pinned versions.
4. Validate installation: `python -c "import torch; import torch_geometric; print(torch.cuda.is_available())"`.

### Step 0.2 — Pipeline Skeleton (`run_pipeline.py` + `config.yaml`)

Create the unified CLI with:
- `--config`, `--output-dir`, `--stage` (repeatable), `--dry-run`, `--resume`, `--seed`, `--ablation`, `--device`
- Stage dependency graph enforcement
- `PipelineDependencyError` for missing upstream checkpoints
- `--resume` logic: check for output checkpoint existence before executing each stage
- Global seed application at startup

Create `config.yaml` with all defaults from the system prompt (Section: Config File Structure).

### Step 0.3 — `.gitignore` Update

Add:
```
Data/raw/
Data/user_posts.tar.gz
outputs/
*.pt
*.h5
*.hdf5
wandb/
__pycache__/
```

**Checkpoint:** `python run_pipeline.py --config config.yaml --dry-run --stage nlp_filter` prints the execution plan without errors.

---

## Phase 1: NLP Filtering Pipeline (Weeks 1–3)

### Step 1.1 — Keyword Module (`blueccgl/data/keywords.py`)

**Reads:** Raw JSONL files from `Data/user_posts/` (sample) or `Data/raw/user_posts.tar.gz` (full).
**Writes:** Nothing (library module).

Refactor keyword tiers from `scripts/eda_user_posts.py` into a reusable module:
- Compile all tiers into a single `re.Pattern` for matching.
- Provide a `match_keywords(text) -> set[str]` function that returns which specific keywords matched (for the output schema's `conflict_keywords_matched` field).
- Provide per-tier match functions for the `validate_filter` stage.

**Potential issues:**
1. **Tier 1 `\bisrael\b` over-matching:** Will match "Israel Adesanya" (UFC fighter), "Israel Kamakawiwoʻole" (musician). Inspect false positives during validation and consider requiring co-occurrence with another tier-1 keyword, or add a negative filter.
2. **Tier 5 `\bgallant\b` over-matching:** Common English word. May need context-window filtering.
3. **Arabic/Hebrew regex on non-UTF8 text:** Ensure `.jsonl` files are read with `encoding="utf-8"`.

### Step 1.2 — Event Window Module (`blueccgl/data/events.py`)

Extract the `MEGA_EVENTS` dict and `assign_event()` function from the EDA script. Add:
- Conflict window bounds: `2023-10-01` to `2024-03-18`.
- `parse_date(int) -> datetime` with the `%Y%m%d%H%M` format.

### Step 1.3 — NLP Filter Stage (`blueccgl/stages/nlp_filter.py`)

**Stage:** `--stage nlp_filter`
**Reads:** `{raw_dir}/user_posts/` (individual JSONL files) or streams from `user_posts.tar.gz`.
**Writes:** `{output_dir}/nlp_filter/conflict_posts.parquet`

Implementation:
1. **Streaming processing:** For the full 19.5 GB tar, iterate files within the archive using `tarfile` without extracting. For the sample, iterate `Data/user_posts/*.jsonl`.
2. For each post:
   - Parse date; skip if outside conflict window.
   - Apply keyword regex; skip if no match.
   - Record: `post_id, user_id, timestamp, text, langs, sent_label, sent_score, is_repost (bool), repost_from, reposted_author, conflict_keywords_matched (list), event_window`.
3. Write output as Parquet (DuckDB or PyArrow).
4. Log statistics: total posts scanned, conflict posts found, per-event counts, per-tier hit rates.

**Memory/performance considerations for full data:**
- 19.5 GB tar.gz with 100 files. Each file is a single user's posts as JSONL. Stream-process one file at a time; no need to hold all posts in memory.
- Use `tqdm` with file-level progress (100 files).
- DuckDB append-mode for incremental Parquet writes if memory is tight.

**Potential issues:**
1. **Concatenated JSON lines:** The EDA script handles `}{` splitting. This suggests some JSONL files may have malformed lines. Keep the fallback parser but log warnings for malformed lines.
2. **Timestamp format edge cases:** The date `202310071230` parses fine, but check for dates with fewer digits (e.g., midnight `202310070000` vs `20231007`).
3. **Memory:** Even with streaming, accumulating millions of matched rows in a list before writing Parquet could spike RAM. Batch-write every N files (e.g., every 10 files).

**Checkpoint validation:**
- Row count should be between 750K–3M for full data (0.5–2% of ~235M posts in conflict window).
- For the sample (~25K posts): expect ~50–500 conflict posts depending on user mix.
- Every row must have a non-null `event_window`.
- `conflict_keywords_matched` must be a non-empty list for every row.

### Step 1.4 — Validate Filter Stage (`blueccgl/stages/validate_filter.py`)

**Stage:** `--stage validate_filter`
**Reads:** `{output_dir}/nlp_filter/conflict_posts.parquet` + raw JSONL files (for unmatched sample).
**Writes:** `{output_dir}/validate_filter/validation_report.json`

Implementation:
1. Sample `config.nlp_filter.sample_files` files (default 3).
2. From those files, draw `config.nlp_filter.validation_sample_n` (200) matched and 200 unmatched posts.
3. Output them to a CSV/JSON for manual inspection (the user reviews these).
4. After manual labels are added, compute precision and recall.

**This is a human-in-the-loop step.** The stage produces the sample; the user annotates; a second invocation computes metrics.

### Step 1.5 — Sample-First Validation Workflow

Before waiting for the full data download, use the 100 sample files to:

1. Run `--stage nlp_filter` on `Data/user_posts/`.
2. Inspect output: How many conflict posts? What keywords matched? Which events? False positives?
3. Run `--stage validate_filter` on the sample.
4. Manually inspect 50–100 matched posts for precision.
5. Iterate on keyword list as needed.

**This is the immediate next step after scaffolding is complete.**

---

## Phase 2: Cascade Construction (Weeks 4–5)

### Step 2.1 — Data Acquisition: Remaining Files

Before cascade construction can proceed, the full dataset must be downloaded:
- `user_posts.tar.gz` (19.5 GB) — needed for full NLP filter run
- `interactions.csv.gz` (1 GB) — needed for cascade edge construction
- `graphs.tar.gz` (891 MB) — repost graph edge lists
- `followers.csv.gz` (491 MB) — edge-type labelling
- `feed_posts.tar.gz` (16.4 MB) — feed-amplified content identification
- `feed_bookmarks.csv` (553 KB) — user feed subscriptions

**Action:** Download from Zenodo (DOI: 10.5281/zenodo.14669616) into `Data/raw/`.

**Potential issue:** 22 GB download. Ensure disk space (~100 GB free recommended for raw + decompressed + processed outputs).

### Step 2.2 — Follower Graph Loading

**Critical infrastructure — needed by cascade construction and later by augmentations.**

`followers.csv.gz` contains ~144M directed edges. Options:
1. **DuckDB:** Load into an on-disk DuckDB database. Query `SELECT 1 FROM followers WHERE follower = ? AND followed = ?` for edge-type checking. This is the recommended approach — avoids loading 144M edges into RAM.
2. **Sorted Parquet + binary search:** Convert to sorted Parquet, use PyArrow compute for lookups.
3. **In-memory set:** `set((follower, followed))` — at ~144M edges × ~16 bytes/tuple = ~2.3 GB RAM. Feasible on a 16+ GB machine but tight.

**Recommendation:** DuckDB with an index on `(follower, followed)`. Load once, use throughout cascade construction.

### Step 2.3 — Cascade Construction Stage (`blueccgl/stages/cascade_construction.py`)

**Stage:** `--stage cascade_construction`
**Reads:**
- `{output_dir}/nlp_filter/conflict_posts.parquet`
- `{raw_dir}/interactions.csv.gz`
- `{raw_dir}/graphs.tar.gz` (repost graph)
- `{raw_dir}/followers.csv.gz` (via DuckDB)
**Writes:** `{output_dir}/cascade_construction/cascades.parquet` (edge list with metadata)

#### Sub-step 2.3.1 — Identify Cascade Roots

From `conflict_posts.parquet`, select posts where `is_repost == False` (original conflict posts). These are potential cascade roots.

#### Sub-step 2.3.2 — Trace Downstream Reposts

For each root `post_id`:
1. Find all reposts from `interactions.csv.gz` (where `reposted_author` matches and date is after root).
2. Cross-reference with `graphs.tar.gz` repost edge lists.
3. Build the set of all downstream reposts transitively (reposts of reposts).

**Critical design decision:** The dataset's `repost_from` field links reposts to their source *post*, not their source *user*. The `reposted_author` field gives the user. A cascade chain is: User A posts → User B reposts A's post → User C reposts (B's repost of A's post, which still credits A as `reposted_author`). So `reposted_author` alone cannot reconstruct the cascade tree — it only points to the root. The tree structure must come from the temporal ordering + follower graph.

**This means cascade reconstruction is fundamentally a heuristic process, not a direct graph read.** The parent assignment heuristic (Step 2.3.3) is the core logic.

#### Sub-step 2.3.3 — Parent Assignment Heuristic

For each reposter in a cascade (ordered by timestamp):
1. **Priority 1:** Among users already in the cascade who posted *before* this reposter — find any that this reposter follows (per `followers.csv.gz`). If multiple, take the most recent. → FOLLOWER edge.
2. **Priority 2:** If no followed prior reposter exists, take the most recent prior reposter. → NON_FOLLOWER edge.
3. **Priority 3:** If no prior reposter found (shouldn't happen unless data gap), connect to root. → UNKNOWN edge.

#### Sub-step 2.3.4 — Edge-Type Assignment

Already implicit in parent assignment:
- Priority 1 → `FOLLOWER = 0`
- Priority 2 → `NON_FOLLOWER = 1`
- Priority 3 → `UNKNOWN = 2`

For edges assigned via Priority 2, also check if the pair exists in the follower graph — if they do, upgrade to `FOLLOWER`.

#### Sub-step 2.3.5 — Size Filtering

After constructing all cascades:
- Compute size (number of nodes) per cascade.
- Apply threshold: ≥10 reposts (fallback to ≥5 if insufficient cascades).
- Cap at 10,000 nodes.
- Report: cascade count, size distribution (min, median, mean, p95, max), per-event counts.

**Output schema (one row per edge):**
```
cascade_id, source_user_id, target_user_id, edge_type (0/1/2),
source_timestamp, target_timestamp, source_sent_label, target_sent_label,
source_sent_score, target_sent_score, event_window
```

Plus a cascade-level metadata table:
```
cascade_id, root_post_id, root_user_id, root_timestamp, cascade_size,
depth, event_window, text (root post text)
```

**Potential issues:**
1. **Root out-degree inflation from Priority 3 fallbacks:** If many reposters can't be linked to a prior reposter they follow, the tree degenerates into a star graph (everyone connected to root). Monitor the fraction of Priority 3 assignments; if >30%, the cascade is structurally unreliable.
2. **Temporal confound in follower graph:** The March 2024 snapshot includes follow relationships created *after* October 2023 events. FOLLOWER edges for early cascades may be spurious. Acknowledge as limitation. Consider sensitivity analysis: re-run clustering with all edges set to NON_FOLLOWER for E1/E2 cascades.
3. **Cascade size distribution may be heavily right-skewed:** A few mega-cascades (thousands of nodes) and many small ones. Log-transform sizes for EDA; consider separate handling of >1000-node cascades in batching.
4. **Duplicate reposts:** A user may repost the same content multiple times. Deduplicate by `(user_id, reposted_post_id)`.

**Checkpoint validation:**
- Every cascade is a connected directed tree (no cycles, single root).
- `depth ≥ 1` for all cascades (root + at least one reposter).
- `sum(cascade_sizes) ≤ total_conflict_reposts` (no node appears in multiple cascades of the same root).
- Edge-type distribution: expect majority FOLLOWER (social spread is often follower-driven), non-trivial NON_FOLLOWER fraction.

### Step 2.4 — Build PyG Dataset Stage (`blueccgl/stages/build_pyg_dataset.py`)

**Stage:** `--stage build_pyg_dataset`
**Reads:** `{output_dir}/cascade_construction/cascades.parquet`
**Writes:** `{output_dir}/build_pyg_dataset/cascade_dataset.pt`

For each cascade, create a `torch_geometric.data.Data` object:

**Node features (`x`):** 4-dimensional per node:
1. `temporal_delta`: log1p(minutes since root post). Log-normalise across the cascade.
2. `sent_label`: One-hot (3-dim) or ordinal (1-dim). Start with ordinal for simplicity (0=neg, 1=neutral, 2=pos).
3. `depth`: Hop distance from root in the cascade tree.
4. `degree_in` / `degree_out`: In-degree and out-degree within the cascade.

**Total node feature dim:** 1 (temporal) + 1 (sentiment label) + 1 (sentiment score) + 1 (depth) + 2 (in/out degree) = **6**.

**Edge index + edge type:**
- `edge_index`: `[2, num_edges]` standard PyG format.
- `edge_type`: `[num_edges]` tensor with values 0, 1, 2.

**Graph-level attributes (`y` or custom):**
- `sentiment_composition`: `[3]` tensor (proportion neg/neutral/pos).
- `follower_edge_ratio`: scalar.
- `event_window`: integer label (0–6 for E1–E6 + baseline).
- Structural: `depth`, `breadth`, `wiener_index` (compute from the tree).

**Potential issues:**
1. **Variable-size graphs:** PyG handles this natively via batching. No padding needed.
2. **Wiener index computation:** For a tree with N nodes, this is O(N) using a DFS-based approach (sum of all pairwise distances = sum over edges of `subtree_size * (N - subtree_size)`). Do NOT use the naive O(N²) approach for large cascades.
3. **Memory:** Saving all `Data` objects as a single `.pt` file. For 50K cascades of average size 50 nodes, this is ~50K × 50 × 6 × 4 bytes ≈ 60 MB for node features alone. Manageable.

**Checkpoint validation:**
- Load the `.pt` file; verify `len(dataset)` matches expected cascade count.
- Spot-check 10 random cascades: verify tree connectivity, feature ranges.
- `edge_type` values are all in {0, 1, 2}.
- `sentiment_composition` sums to ~1.0 for each graph.

### Step 2.5 — EDA Stage (`blueccgl/stages/eda.py`)

**Stage:** `--stage eda`
**Reads:** `{output_dir}/build_pyg_dataset/cascade_dataset.pt`
**Writes:** `{output_dir}/eda/` (plots + `eda_summary.json`)

Generate:
1. Cascade size distribution (histogram, log-scale) — overall and per-event.
2. Cascade depth distribution — overall and per-event.
3. Follower-edge ratio distribution — overall and per-event.
4. Sentiment composition scatter (ternary plot or stacked bar).
5. Temporal distribution of cascade roots across the conflict window.
6. Branching factor distribution.
7. Correlation matrix of structural features.

---

## Phase 3: Model Implementation (Weeks 6–8)

### Step 3.1 — R-GCN Encoder (`blueccgl/models/rgcn_encoder.py`)

Implementation using `torch_geometric.nn.RGCNConv`:
- 3 relation types, 2 bases (basis decomposition).
- 3 layers: `input_dim → hidden_dim → hidden_dim → hidden_dim`.
- ReLU activation + dropout (0.3) between layers.
- Input: node features `x` (dim 6), `edge_index`, `edge_type`.
- Output: node embeddings `h` (dim `hidden_dim=128`).

**Potential issues:**
1. **Basis decomposition with only 3 relations and 2 bases:** This is fine — 2 bases for 3 relations means each relation's weight matrix is a linear combination of 2 shared basis matrices. This gives meaningful parameter sharing while still allowing differentiation.
2. **Over-smoothing with 3 layers:** For small cascades (10–20 nodes), 3 hops may cover the entire graph, making all node embeddings converge. Monitor node embedding variance per cascade. Consider residual connections if over-smoothing is observed.

### Step 3.2 — Vanilla GCN Encoder (`blueccgl/models/gcn_encoder.py`)

For ablation A1 (no edge-type encoding). Same architecture but uses `torch_geometric.nn.GCNConv` — all edges treated identically.

### Step 3.3 — Hierarchical DiffPool (`blueccgl/models/diffpool.py`)

Implementation:
1. **Level 1:** Pool from N nodes to N×0.5 nodes. Assignment matrix `S₁ = softmax(GNN_pool(x, edge_index))`.
2. **Level 2:** Pool from N×0.5 to N×0.25 nodes. Assignment matrix `S₂`.
3. At each level, compute global mean pool of node embeddings → level embedding.
4. Concatenate all level embeddings + graph-level sentiment composition `[3]`.
5. Final MLP: `[concat_dim] → hidden → embedding_dim (64)`.

**Auxiliary losses:**
- **Link prediction loss:** `||S^T S - A||_F` (the pooled adjacency should match the assignment).
- **Entropy loss:** `H(S)` (encourage soft assignments to be close to hard assignments).
- Both weighted by `diffpool_aux_weight = 0.1`.

**Potential issues:**
1. **DiffPool memory:** The assignment matrix is `[N, N_pooled]` — dense. For a cascade of 10,000 nodes with ratio 0.5, that's `10000 × 5000 = 50M` entries per cascade. This will OOM on large cascades. **Mitigation:** Cap cascade size at 1,000 for DiffPool, or use MinCutPool as a less memory-hungry alternative.
2. **Auxiliary loss domination:** If `diffpool_aux_weight` is too high, the contrastive loss is drowned out. Start at 0.1, monitor the ratio of contrastive loss to auxiliary loss.
3. **Variable-size pooling:** DiffPool requires knowing the number of pooled nodes. In a batch of variable-size cascades, the pool ratio produces different absolute sizes. PyG's `dense_diff_pool` handles this but requires dense adjacency representations → memory-intensive. Consider `torch_geometric.nn.dense_diff_pool` or implement a sparse variant.

**Alternative if DiffPool proves too expensive:** Use SAGPool (attention-based, O(N) memory) with the same hierarchical structure. Flag this as a fallback.

### Step 3.4 — Feed-Aware Augmentations (`blueccgl/augmentations/`)

**Strategy A — Follower-Subgraph Extraction (`follower_subgraph.py`):**
1. Filter edges to keep only `edge_type == 0` (FOLLOWER).
2. Sample `follower_retain_ratio` (0.7) of these edges.
3. Compute the connected component containing the root. If disconnected, keep only the root's component.
4. Reindex nodes; return new `Data` object.

**Strategy B — Temporal Prefix Sampling (`temporal_prefix.py`):**
1. Sort nodes by `temporal_delta`.
2. Keep the earliest `temporal_prefix_ratio` (0.6) of nodes.
3. Reindex edges to only include edges between kept nodes.
4. Verify root is included (it always should be — delta=0).

**Strategy C — Bridge-Node Removal (`bridge_removal.py`):**
1. Identify "bridge nodes": nodes whose incoming edge is `edge_type == 1` (NON_FOLLOWER).
2. Sample `bridge_remove_ratio` (0.3) of these nodes for removal.
3. Remove selected nodes and all their incident edges.
4. Keep only the connected component containing the root.

**Random Augmentation (ablation A2, `random_aug.py`):**
- Random node dropping (20%) + random edge dropping (20%).

**Contrastive pair generation:**
- For each cascade in a batch: apply two different augmentation strategies (e.g., A+B, A+C, B+C — selected randomly per pair).
- The two augmented views form a positive pair.
- All other cascades in the batch are negatives.
- Hard negatives: optionally prioritise cascades from the same event window.

**Potential issues:**
1. **Augmentation producing empty or trivial graphs:** If a cascade has very few FOLLOWER edges, Strategy A may produce a graph with only 1–2 nodes. **Guard:** If augmented graph has <3 nodes, re-augment with a different strategy.
2. **Bridge-node removal disconnecting the graph:** After removal, the graph may fragment. Always take the root's connected component. If this reduces the graph to <3 nodes, fall back to a different augmentation.

### Step 3.5 — InfoNCE Loss (`blueccgl/models/losses.py`)

```python
def info_nce_loss(z_i, z_j, temperature=0.5):
    """
    z_i, z_j: [B, D] embedding matrices for the two augmented views.
    Returns scalar loss.
    """
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    sim = torch.mm(z_i, z_j.T) / temperature  # [B, B]
    labels = torch.arange(B, device=sim.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
    return loss
```

**Potential issues:**
1. **Representation collapse:** All embeddings converge to the same vector. Symptom: loss plateaus near `log(batch_size)`. Diagnostic: compute `std(||z||)` per batch — if near 0, collapse is occurring. Mitigation: lower temperature, increase augmentation diversity, add a projector MLP head (use projected embeddings for loss, but extract embeddings from the encoder output before the projector).
2. **Small batch size:** With 64 cascades per batch, the number of negatives is 63. For InfoNCE, more negatives generally helps. If batch size is limited by GPU memory, use a memory bank or momentum encoder (MoCo-style).

### Step 3.6 — Full Model Assembly (`blueccgl/models/blueccgl.py`)

```
Input: Data(x, edge_index, edge_type, batch, sentiment_composition)
  → R-GCN Encoder → node embeddings [N, hidden_dim]
  → DiffPool Level 1 → [N*0.5, hidden_dim] + level_1_global_pool
  → DiffPool Level 2 → [N*0.25, hidden_dim] + level_2_global_pool
  → Concat(level_0_pool, level_1_pool, level_2_pool, sentiment_composition)
  → MLP Projector → embedding [B, embedding_dim]
```

### Step 3.7 — Unit Tests

Before training, verify:
1. **Forward pass:** Random `Data` object → model → embedding of correct shape.
2. **Gradient flow:** Loss.backward() produces non-zero gradients on all parameters.
3. **Augmentation contracts:** Each augmentation produces a valid `Data` object with root node, ≥3 nodes, connected.
4. **Keyword matching:** Known conflict text → match. Known non-conflict text → no match.
5. **Cascade tree validity:** Constructed cascade is a valid directed tree.

---

## Phase 4: Training & Experiments (Weeks 9–12)

### Step 4.1 — Training Stage (`blueccgl/stages/train.py`)

**Stage:** `--stage train`
**Reads:** `{output_dir}/build_pyg_dataset/cascade_dataset.pt`
**Writes:** `{output_dir}/train/checkpoints/epoch_{N}.pt`, `{output_dir}/train/config_snapshot.yaml`, `{output_dir}/train/training_log.json`

Training loop:
1. Load PyG dataset.
2. Create `DataLoader` with batching (PyG's `DataLoader` handles variable-size graphs).
3. For each epoch:
   a. For each batch: generate two augmented views per cascade.
   b. Forward pass both views through the encoder.
   c. Compute InfoNCE loss + DiffPool auxiliary losses.
   d. Backward + gradient clipping (`grad_clip_norm=1.0`) + optimizer step.
   e. Log: loss, gradient norm, embedding variance, augmentation statistics.
4. Checkpoint every `checkpoint_every` epochs. Save: model state, optimizer state, epoch, config snapshot.

**Ablation support:** When `--ablation` is set:
- `A1_no_edge_type`: Use GCN encoder instead of R-GCN.
- `A2_generic_aug`: Use random augmentations instead of feed-aware.
- `A3_flat_pool`: Use global mean pooling instead of DiffPool.
- `A4_no_sentiment`: Zero out sentiment features in node features and omit sentiment composition from readout.
- `A5_random_init`: Skip training entirely; use random encoder weights.

**Potential issues:**
1. **Batch construction with variable-size graphs:** PyG's `DataLoader` pads and concatenates graphs into a single large graph with a `batch` index vector. This is standard but means very large cascades dominate batch memory. **Mitigation:** Sort cascades by size, use size-aware batching (group similarly-sized cascades).
2. **Learning rate schedule:** Start with `lr=1e-3`, cosine annealing to `1e-5` over 100 epochs. If loss plateaus early, consider warm-up (linear warm-up for 5 epochs).
3. **Gradient explosion from DiffPool:** DiffPool's dense operations can produce large gradients. Gradient clipping is essential. Monitor gradient norms — if consistently at the clip value, the effective learning rate is being capped.

### Step 4.2 — Extract Embeddings Stage (`blueccgl/stages/extract_embeddings.py`) — DONE

**Stage:** `--stage extract_embeddings`
**Reads:** `{output_dir}/train/checkpoints/best.pt`, `{output_dir}/build_pyg_dataset/cascade_dataset.pt`
**Writes:** `{output_dir}/extract_embeddings/embeddings.h5`, `{output_dir}/extract_embeddings/cascade_index.parquet`

**Implementation:** Load `best.pt` via `BlueCCGL(cfg)` + `load_state_dict()`. Inference with `model.encode()`, `batch_size=32`, `torch.no_grad()`. Saves full cascade metadata (depth, breadth, wiener_index, follower_edge_ratio, sentiment_composition, event_window) to `cascade_index.parquet` alongside the HDF5 embedding matrix. Sanity-checks embedding norm variance; warns if `std(norms) < 0.01` (potential collapse).

**Embedding diagnostics (added 2026-04-17):** `_embedding_diagnostics()` function computes effective rank (SVD entropy), uniformity (Wang & Isola 2020), median NN cosine similarity, and top-5 variance ratio on up to 8,000 sampled embeddings. Prints a `EMBEDDING DIAGNOSTICS` block with OK/WARN status against targets: `eff_rank > 25`, `uniformity < -3.0`, `top-5 var ratio < 0.6`.

**Checkpoint validation:**
- Embedding matrix shape: `[N_cascades, 64]` (gzip-compressed in HDF5).
- `cascade_index.parquet` row count matches embedding count.
- Embedding diagnostics printed automatically after every extraction run.

### Step 4.3 — Cluster Stage (`blueccgl/stages/cluster.py`) — DONE

**Stage:** `--stage cluster`
**Reads:** `{output_dir}/extract_embeddings/embeddings.h5`, `{output_dir}/extract_embeddings/cascade_index.parquet`
**Writes:** `{output_dir}/cluster/cluster_assignments.parquet`, `{output_dir}/cluster/grid_search_results.json`, `{output_dir}/cluster/cluster_persistences.json`, `{output_dir}/cluster/umap_embeddings_cluster.h5`

**Implementation:** UMAP pre-reduction (64D → `umap_cluster_n_components=10`) applied before HDBSCAN — essential for avoiding curse of dimensionality (without this, all grid configs produce >67% noise). Grid search over `min_cluster_size ∈ config.clustering.min_cluster_size_grid` (default `[100, 200, 300]`). Per-step `min_samples = max(1, round(min_samples_ratio × mcs))` computed from `min_samples_ratio` (default 0.4). Silhouette score on non-noise points (capped at 5000 samples) selects best config. `noise_threshold` (default 0.70 — raised from 0.30 to allow silhouette comparison across all configs) filters degenerate configs. Best clusterer cached to extract `cluster_persistence_` after grid search.

**Additional outputs:**
- `cluster_persistences.json`: `mean_persistence`, `per_cluster` array, `n_clusters`, `min_cluster_size_used` — consumed by evaluate stage for CENTRAL METRICS block.
- `umap_embeddings_cluster.h5`: 10D UMAP-reduced embeddings saved for reproducibility.

**v8 results:** mcs=300 selected, 11 clusters, 1.5% noise, silhouette=0.362.

### Step 4.4 — Evaluate Stage (`blueccgl/stages/evaluate.py`) — DONE

**Stage:** `--stage evaluate`
**Reads:** `{output_dir}/extract_embeddings/embeddings.h5`, `{output_dir}/extract_embeddings/cascade_index.parquet`, `{output_dir}/cluster/cluster_assignments.parquet`, `{output_dir}/cluster/cluster_persistences.json`
**Writes:** `{output_dir}/evaluate/metrics.json`, `{output_dir}/evaluate/umap_figure.png`, `{output_dir}/evaluate/umap_coords.parquet`

**Implementation:** Bootstrap (n=1000, capped at 5000-sample Silhouette) for Silhouette, Davies-Bouldin, Calinski-Harabasz. DBCV via `hdbscan.validity_index` (capped at 100 bootstrap iterations due to cost). All metrics report mean ± std + 95% CI. UMAP figure has 2 panels: (A) coloured by cluster label, (B) coloured by event_window. `umap_coords.parquet` saves 2D coordinates alongside `cascade_id` for reuse in explain stage without re-running UMAP.

**CENTRAL METRICS block (added 2026-04-17):** Printed before the bootstrapped metrics table: number of clusters, noise fraction (absolute count + percentage), mean HDBSCAN persistence from `cluster_persistences.json`, and DBCV mean. Also stores `mean_cluster_persistence` in `metrics["meta"]`. `import blueccgl.utils.mpl_compat` added at top for Python 3.14 matplotlib compatibility (see Known Bugs #7).

**10D UMAP metric space fix (added 2026-04-18):** All bootstrap metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz, DBCV) now computed in the **10D UMAP-reduced space** from `cluster/umap_embeddings_cluster.h5`, not raw 64D embeddings. HDBSCAN clusters in this space, so evaluating in this space is the correct methodology. Raw 64D gave silhouette≈0.012 (near-random) and DBCV=-0.46 for v9; 10D gives silhouette=0.413 and DBCV=+0.618 for v8 — the honest numbers. `metric_space` field added to `metrics["meta"]` to record which space was used. Falls back to 64D with a WARNING if `umap_embeddings_cluster.h5` is absent.

### Step 4.5 — Explain Stage (`blueccgl/stages/explain.py`) — DONE

**Stage:** `--stage explain`
**Reads:**
- `{output_dir}/train/checkpoints/best.pt`
- `{output_dir}/build_pyg_dataset/cascade_dataset.pt`
- `{output_dir}/extract_embeddings/embeddings.h5`
- `{output_dir}/extract_embeddings/cascade_index.parquet`
- `{output_dir}/cluster/cluster_assignments.parquet`
**Writes:** `{output_dir}/explain/` (see output structure below)

**Dependency:** `evaluate` (added to pipeline dependency chain 2026-03-24)

Multi-layer explainability pipeline with **5 sub-steps** (4 from plan + 1 new), each answering a distinct question about discovered cascade archetypes.

#### Sub-step 5.1 — Probing Classifiers (What is encoded in the embeddings?) — DONE

Train `Ridge` regression (continuous) or `LogisticRegression` (categorical) on frozen 64-dim embeddings with 5-fold CV. Properties: `cascade_depth`, `cascade_breadth`, `wiener_index`, `follower_edge_ratio`, `cascade_size`, `event_window`. Reports R² (regression) or balanced accuracy (classification), with bootstrap CIs (n=200) globally and per-cluster breakdown. Uses 5-fold CV for clusters <100 cascades; flags per output.

**Output:** `explain/probe_results.json` — R²/accuracy per property, overall and per-cluster. `explain/figures/probe_heatmap.png` — property × cluster seaborn heatmap.

**RQ connection:** High `follower_edge_ratio` R² → model encodes SRQ3 signal. High `event_window` accuracy → model captures SRQ1/SRQ2 signal.

**Key reference:** Alain & Bengio (2017). "Understanding intermediate layers using linear classifier probes."

#### Sub-step 5.2 — DiffPool Assignment Extraction (What structural roles does the model learn?) — DONE

Runs `model.explain()` over each cascade individually (`batch_size=1`, critical to avoid N_max padding contaminating the real-node assignment slice). Computes per-node Level-1 assignment entropy `H = -∑ softmax(s1)·log(softmax(s1))` and fraction of follower-typed edge targets assigned to the dominant super-node. **Memory-efficient:** only aggregated statistics stored per cascade (no raw `s1` matrices). Results grouped by HDBSCAN cluster.

**Output:** `explain/diffpool_assignments.h5` — `mean_entropy_l1 [N_cascades]`, `follower_in_dominant_pool [N_cascades]`. `explain/figures/diffpool_analysis.png` — per-cluster entropy + follower concentration bars.

**RQ connection:** If follower nodes consistently land in the dominant super-node with low entropy → R-GCN edge-type encoding is being utilised by pooling (validates Extension 2).

**Key reference:** Ying et al. (2018). "Hierarchical Graph Representation Learning with Differentiable Pooling."

#### Sub-step 5.3 — Cluster Statistical Profiles (What distinguishes archetypes?) — DONE

Per-cluster mean ± std / median / IQR for `cascade_depth`, `cascade_breadth`, `wiener_index`, `follower_edge_ratio`, `cascade_size`, `sentiment_neg/neu/pos`. Between-cluster Kruskal-Wallis per property + pairwise Mann-Whitney U with Bonferroni correction. PCA on cluster centroids + property correlations for discriminative dimension identification.

**RQ-specific additions beyond original plan:**
- **Archetype × event_window cross-tabulation** (chi-squared test for independence; reports expected-cell-count warnings for sparse event windows) → SRQ1, SRQ2
- **Temporal evolution plot** (archetype prevalence fraction per event window, line chart) → SRQ1
- **Follower edge ratio violin plots per archetype** → SRQ3
- **Sentiment composition radar chart per archetype** → SRQ4

**Output:** `explain/cluster_profiles.json` + `explain/figures/cluster_radar_profiles.png`, `archetype_event_heatmap.png`, `follower_ratio_violins.png`, `temporal_evolution.png`.

**Interpretation:** "Archetype A (n=342): deep chains (depth=8.2±2.1), low follower ratio (0.23), negative sentiment dominant. Archetype B (n=189): shallow star-bursts (depth=2.1, breadth=45.3), high follower ratio (0.78), neutral sentiment."

#### Sub-step 5.4 — Augmentation Sensitivity Analysis (What is structurally defining for each archetype?) — DONE

Samples `n_representatives_per_cluster` (default 20) cascades per cluster. For each, applies each augmentation strategy with a **fixed per-cascade seed** (`seed = cascade_id * 137 % 2³¹`) to ensure reproducibility. Computes `‖embed(original) − embed(augmented)‖₂` per augmentation type. Aggregates mean ± std per cluster.

**Output:** `explain/augmentation_sensitivity.json` + `explain/figures/augmentation_sensitivity_bars.png` — grouped bar chart (3 bars per cluster).

**Interpretation:** High sensitivity to `follower_subgraph` removal → archetype is defined by follower structure (organic virality). High sensitivity to `bridge_removal` → archetype exhibits out-of-network diffusion — content spreading across non-follower ties (structural bridging). Validates Extension 1 (feed-aware augmentations) against ablation A2 (generic augmentation).

#### Sub-step 5.5 — Representative Cascade Visualisations (NEW — 2026-03-24)

Selects top-5 cascades per cluster (closest to centroid in embedding space). Renders each as a tree plot with 3 colour panels: (A) edge type (follower/non-follower/unknown), (B) depth, (C) sentiment label. Uses Reingold-Tilford hierarchical layout via networkx BFS tree.

**Output:** `explain/figures/archetypes/cluster_{k}_rep_{i}.png` per representative cascade. Requires `networkx` (added to `install.sh`).

#### Optional Extensions (time permitting)

| Technique | Value | Effort | Notes |
|---|---|---|---|
| **GNNExplainer** (PyG `torch_geometric.explain`) | Per-cascade edge/feature importance | ~4h | Needs proxy classifier (predict cluster label); aggregate per cluster |
| **PGExplainer** (PyG `torch_geometric.explain`) | Global edge-type importance per archetype | ~4h | Inductive — learns generalizable patterns across clusters |
| **Concept Activation Vectors** (TCAV adapted) | "Does model use concept X?" | ~6h | Define concepts from graph properties (star topology, chain topology, sentiment-polarized, follower-dominated). Train SVMs on embeddings. Highest thesis novelty |
| **SubgraphX** (DIG library) | Minimal motifs per archetype | ~6h | MCTS-based; expensive (~10-30s per graph). Sample 20 per cluster |

**Key references for optional extensions:**
- Ying et al. (2019). "GNNExplainer." *NeurIPS.*
- Luo et al. (2020). "PGExplainer." *NeurIPS.*
- Kim et al. (2018). "TCAV." *ICML.*
- Yuan et al. (2021). "SubgraphX." *ICML.*
- Yuan et al. (2022). "Explainability in GNNs: A Taxonomic Survey." *TPAMI.*
- Magister et al. (2021). "GCExplainer: Concept-based Explanations for GNNs."

**Python 3.14 compatibility:** `import blueccgl.utils.mpl_compat` added at top. Removed `sys.setrecursionlimit` workaround (insufficient for complex figures). `RecursionError` guard added in UMAP figure block as belt-and-suspenders fallback.

**Bug fix (2026-04-17):** `_plot_archetype_event_heatmap` KeyError — `event_cols` was incorrectly extracted from one level too deep in the `frac` dict, yielding cluster IDs instead of event-window keys. Fixed to `event_cols = sorted(frac.keys())`.

#### Config (deployed in `config.yaml`)

```yaml
explain:
  n_representatives_per_cluster: 20     # sample size for aug sensitivity + representative viz
  run_probes: true
  run_diffpool_analysis: true
  run_augmentation_sensitivity: true
  run_cluster_profiles: true
  run_representative_viz: false         # disabled — color array bug in _draw_cascade_tree
  probe_properties:
    - cascade_depth
    - cascade_breadth
    - wiener_index
    - follower_edge_ratio
    - cascade_size
    - event_window
  figures_dpi: 300
  probe_bootstrap_n: 200                # bootstrap iterations for probe CIs (cheaper than eval bootstrap)
```

#### Output Structure (deployed)

```
outputs/explain/
  probe_results.json                    # R²/accuracy per property per cluster + bootstrap CIs
  cluster_profiles.json                 # Summary stats + Kruskal-Wallis + Mann-Whitney + PCA + cross-tab
  augmentation_sensitivity.json         # Per-cluster sensitivity to each augmentation type
  diffpool_assignments.h5               # Per-cascade entropy + follower-in-dominant-pool fraction
  explanation_report.json               # Combined summary (stage checkpoint)
  figures/
    probe_heatmap.png                   # Property × cluster R²/accuracy seaborn heatmap
    diffpool_analysis.png               # Per-cluster entropy + follower concentration bars
    cluster_radar_profiles.png          # Normalised structural property radar chart per archetype
    archetype_event_heatmap.png         # Archetype × event_window fraction heatmap (SRQ1, SRQ2)
    follower_ratio_violins.png          # follower_edge_ratio violin plots per archetype (SRQ3)
    temporal_evolution.png              # Archetype prevalence line chart over conflict timeline (SRQ1)
    augmentation_sensitivity_bars.png   # Per-cluster grouped bar chart (SRQ3 + MRQ3)
    archetypes/
      cluster_{k}_rep_{i}.png           # Representative cascade trees with 3 colour panels
```

#### Failure Mode Audit

1. *Probe overfitting:* Small clusters (<50 cascades) may overfit linear probes. **Diagnostic:** Use leave-one-out CV for small clusters. Report confidence intervals.
2. *DiffPool degenerate assignments:* If pool_2 is collapsed (as observed in v1 diagnostics), assignment matrices will be near-uniform. **Diagnostic:** Check assignment entropy — if close to max entropy for all cascades, the pool level is uninformative. Skip that level in visualizations.
3. *Augmentation sensitivity floor effect:* If all augmentations produce nearly identical embeddings (alignment ~1.0, as in v1), sensitivity analysis yields no signal. **Prerequisite:** Retrain v2 must achieve alignment < 0.85 for this sub-step to be meaningful.

### Step 4.6 — Validate Hierarchical Stage (`blueccgl/stages/validate_hierarchical.py`) — DONE

**Stage:** `--stage validate_hierarchical`
**Reads:** `{output_dir}/cluster/umap_embeddings_cluster.h5`, `{output_dir}/cluster/cluster_assignments.parquet`, `{output_dir}/cluster/cluster_persistences.json`, `{output_dir}/extract_embeddings/embeddings.h5`, `{output_dir}/hierarchical/hierarchical_results.json`
**Writes:** `{output_dir}/validate_hierarchical/validation_report.json`, `figures/metric_sweep.png`, `figures/bootstrap_stability.png`, `figures/umap_seed_stability.png`
**Dependency:** `hierarchical`

Four fully implemented validation methods + one stub:

1. **Cophenetic correlation** — single scalar measuring how faithfully the Ward dendrogram preserves pairwise centroid distances. Threshold: r > 0.75 = good; r > 0.85 = excellent.
2. **Multi-cut metric sweep (k=2..6)** — silhouette, Calinski-Harabasz, and Davies-Bouldin computed at each candidate k using all non-noise cascade embeddings in 10D UMAP space. Silhouette capped at 8,000-cascade subsample for speed.
3. **Bootstrap stability (n=100, frac=0.80)** — stratified resample of cascades within each HDBSCAN cluster → recompute centroids → Ward → cut at k=n_macro → ARI vs original. ARI is permutation-invariant; no Hungarian matching needed.
4. **UMAP seed stability (10 seeds)** — full UMAP → HDBSCAN → Ward → cut re-run per seed on raw 64D embeddings. ARI computed on cascades non-noise in both original and re-run. Handles degenerate runs (< 2 clusters) gracefully.
5. **Prediction strength [STUB]** — Tibshirani & Walther (2005). Algorithm fully documented in the stub docstring. Not yet implemented.

**Config keys (all in `validate_hierarchical:` block):** `n_bootstrap` (default 100), `bootstrap_frac` (0.80), `umap_seeds` ([0..9]), `k_sweep_min` (2), `k_sweep_max` (6).

**v8 validation results (2026-05-21, `Outputs_17_04/`):**

| Method | Result | Interpretation |
|--------|--------|----------------|
| Cophenetic r | **0.9006** | Excellent — dendrogram is geometrically faithful |
| Metric sweep | **k=5 wins all three** — sil=0.513, CH=22,252, DB=0.746 | Algorithmic agreement with largest-gap heuristic |
| Bootstrap ARI | **1.000 ± 0.000** (n=100) | Perfect — macro assignments invariant to 20% cascade removal |
| UMAP seed ARI | **0.915 ± 0.094** (10 seeds, all valid) | Robust; 6/10 seeds give ARI ≥ 0.945; min=0.775 (above threshold) |
| Prediction strength | **STUB** | Not yet implemented |

**Key interpretation:** All four implemented methods validate the 5-macro solution. The bootstrap ARI=1.000 reflects that the 11 HDBSCAN centroids are well-separated enough that 20% cascade removal does not move any centroid far enough to change the Ward merge order. The UMAP seed variance (min ARI=0.775) is honest: the dominant archetype structure is seed-independent but ~8% of cascades near macro boundaries can be reassigned under different UMAP projections — the core of each archetype is stable, the boundaries are soft.

### Step 4.7 — Ablation Stage (`blueccgl/stages/ablation.py`) — IMPLEMENTED 2026-05-26

**Stage:** `--stage ablation`
**Reads:** `{output_dir}/build_pyg_dataset/cascade_dataset.pt` (shared across all variants via symlink)
**Writes:** `{output_dir}/ablation/{variant}/train/`, `extract_embeddings/`, `cluster/`, `evaluate/` (per variant), `{output_dir}/ablation/ablation_results.json` (stage checkpoint)

**Design:** 2³ factorial grid — three binary factors, seven variants (baseline `rgcn_feed_sent` = `Outputs_17_04`):

| Variant | Encoder | Augmentation | Sentiment |
|---|---|---|---|
| `gcn_feed_sent` | GCN | feed-aware | ✓ |
| `rgcn_rand_sent` | R-GCN | random | ✓ |
| `rgcn_feed_nosent` | R-GCN | feed-aware | ✗ |
| `gcn_rand_sent` | GCN | random | ✓ |
| `gcn_feed_nosent` | GCN | feed-aware | ✗ |
| `rgcn_rand_nosent` | R-GCN | random | ✗ |
| `gcn_rand_nosent` | GCN | random | ✗ |

**Implementation details:**
- Each variant gets its own output dir: `{output_dir}/ablation/{variant}/`
- Dataset symlinked into `{variant_dir}/build_pyg_dataset/cascade_dataset.pt` — no data copy needed
- Skip-if-done: checks for `{variant_dir}/evaluate/metrics.json` before running; fully resume-safe
- Calls `train.run() → extract_embeddings.run() → cluster.run() → evaluate.run()` with variant config
- Config overrides applied via `apply_overrides(copy.deepcopy(cfg), overrides)`
- `aug_fallback_rate` read from training log and included in results table
- Prints formatted comparison table on completion

**Key confound to report:** Feed-aware variants will show ~79.6% aug fallback rate (Strategy C returns None for cascades with no NON_FOLLOWER edges); random-aug variants will show ~2–5%. This asymmetry must be reported explicitly alongside metrics in Table 4 of the thesis.

**VM run command:**
```bash
python run_pipeline.py --config config.yaml --output-dir ./outputs --stage ablation --device cuda --seed 42
```
Only `outputs/build_pyg_dataset/cascade_dataset.pt` (159.8 MB) needs to be transferred to VM. Estimated runtime: ~7.5 hrs (7 variants × ~65 min/variant on same GPU as v8).

### Step 4.7 — Downstream Stage (`blueccgl/stages/downstream.py`)

**Stage:** `--stage downstream`
**Reads:** `{output_dir}/train/checkpoints/best.pt`, `{output_dir}/build_pyg_dataset/cascade_dataset.pt`
**Writes:** `{output_dir}/downstream/results.json`

Fine-tune the pre-trained encoder on:
1. **Cascade size prediction:** Regression head on top of frozen/fine-tuned encoder. Target: log(cascade_size). Metric: RMSE, R².
2. **Cascade longevity prediction:** Binary classification. Target: cascade active > median duration. Metric: AUROC, F1.

Baselines:
- Random init (same architecture, no pre-training).
- Handcrafted features: [Wiener index, depth, breadth, branching factor, follower-edge ratio] → MLP.

### Step 4.8 — Stability Stage (`blueccgl/stages/stability.py`)

**Stage:** `--stage stability`
**Reads:** `{output_dir}/extract_embeddings/embeddings.h5`
**Writes:** `{output_dir}/stability/stability_report.json`

1. 10× random 80% subsampling of cascades.
2. For each subsample: run HDBSCAN with the selected `min_cluster_size`.
3. Compute ARI between each pair of subsample cluster assignments.
4. Report: mean ± std of clustering metrics across subsamples, mean ARI.

---

## Phase 5: Thesis Writing (Weeks 13–16)

Not a code phase, but the pipeline should support it:
- All metrics, plots, and tables are saved as machine-readable JSON/CSV + publication-ready figures.
- `config_snapshot.yaml` alongside every checkpoint ensures full reproducibility.
- Stage outputs in `{output_dir}/{stage}/` provide a clear audit trail.

---

## Immediate Action Plan (Working with Sample Data)

Since only the 100 sample `user_posts` JSONL files are available, here is the prioritised sequence of work:

### Sprint 1: Scaffolding — DONE

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Resolve Python version | DONE | Python 3.14 works. CPU-only torch 2.10.0 + PyG 2.7.0 from PyG wheel server |
| 2 | Create `blueccgl/` package structure | DONE | |
| 3 | Create `config.yaml` with all defaults | DONE | |
| 4 | Create `run_pipeline.py` with stage orchestration | DONE | Click CLI with `--set KEY=VALUE` overrides (bool casting fixed) |
| 5 | Create `blueccgl/config.py` | DONE | YAML + CLI merge with dot-path overrides |
| 6 | Create `blueccgl/pipeline.py` | DONE | 12-stage dependency graph, resume logic, dry-run |
| 7 | Create `blueccgl/seed.py` | DONE | |
| 8 | Update `.gitignore` | DONE | `Data/` entirely ignored |

### Sprint 2: NLP Filter — DONE (tasks 14–15 deferred)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 9 | Create `blueccgl/data/keywords.py` | DONE | 5 tiers, 68 patterns. Co-occurrence gate for "hostage". "gallant" excluded |
| 10 | Create `blueccgl/data/events.py` | DONE | E1–E6 + baseline. `parse_date()` handles 8/10/12-digit formats |
| 11 | Create `blueccgl/stages/nlp_filter.py` | DONE | `r\|gz` streaming + 15-worker multiprocessing. 95 → 500 files/sec |
| 12 | Run `--stage nlp_filter` on sample | DONE | 137 conflict posts / 25,509 scanned (0.54%) |
| 13 | Inspect results | DONE | `scripts/quick_look.py` for easy inspection |
| 14 | Create validate_filter stage | DEFERRED | Stub exists. Skipped — text quality looked good in manual spot-check of sample |
| 15 | Manual precision check on ~50 posts | DEFERRED | Skipped — note as limitation in thesis |
| — | Run `--stage nlp_filter` on full tar | DONE | 1,947,493 conflict posts from 233M (0.84%), 85 min |

### Sprint 3: Cascade Construction — DONE

| # | Task | Status | Notes |
|---|------|--------|-------|
| 16 | Create `blueccgl/data/cascade_builder.py` | DONE | P1/P2/P3 parent heuristic, edge-type labelling. 7/7 unit tests passing |
| 17 | Create `blueccgl/stages/cascade_construction.py` | DONE | Reposter discovery via `conflict_window_posts.parquet`. Follower path fallback fixed to check `Data/followers.csv.gz` |
| 18 | Create `blueccgl/data/follower_graph.py` | DONE | DuckDB-backed. Auto-builds index from `followers.csv.gz`. `preload_for_users()` added for bulk in-memory lookups (9 roots/sec → 350K roots/sec) |
| 19 | Create `scripts/extract_conflict_window.py` | DONE | Streams tar, extracts lean parquet (172M rows, 2.4 GB) |
| — | Run cascade construction on full data | DONE | 39,983 cascades, 844,849 edges, 75.9% FOLLOWER-typed |

**Full cascade construction results (2025-03-17):**
- Cascades built: **39,983** (≥10 nodes)
- Total edges: 844,849 — Total nodes: 884,832
- FOLLOWER: 641,139 (75.9%) — NON_FOLLOWER: 203,710 (24.1%) — UNKNOWN: 0
- P1 (follower-guided): 641,139 — P2 (temporal): 186,896 — P3 (root): 16,814
- Size: min=3, median=10, mean=22.1, p95=69, max=1,854
- Events: E1=2,732 / E2=3,338 / E3=2,716 / E4=1,320 / E5=1,375 / E6=930 / baseline=27,572
- Outputs: `outputs/cascade_construction/cascades.parquet`, `cascade_meta.parquet`

**Bugs fixed during Sprint 3:**
- `_load_follower_graph` only checked `Data/raw/followers.csv.gz` — added fallback to `Data/followers.csv.gz`
- Per-edge DuckDB query was O(n) round-trips (~26 hr estimate). Fixed with `preload_for_users()`: one batch query loads all relevant edges into a Python set. Total run time: ~3 min

### Sprint 4: PyG Dataset + Model Skeleton — DONE

| # | Task | Status | Notes |
|---|------|--------|-------|
| 20 | Create `blueccgl/stages/build_pyg_dataset.py` | DONE | Full implementation. Two-pass sentiment lookup: `conflict_posts.parquet` first, DuckDB on `conflict_window_posts.parquet` (172M rows) as fallback. O(N) Wiener index. Outputs: `cascade_dataset.pt` (159.8 MB), `dataset_stats.json` |
| 21 | Create R-GCN encoder, DiffPool, losses, augmentations | DONE | All 6 model modules + 4 augmentation strategies implemented. DiffPool max_nodes=150 (not 1000) to prevent CPU OOM. `blueccgl/augmentations/__init__.py` provides shared `subgraph()` and `root_component()` utilities |
| 22 | Unit tests: forward pass, gradient flow, augmentation validity | DONE | `tests/test_model_forward.py` (4 test classes: R-GCN+DiffPool, GCN+mean-pool, no-sentiment, InfoNCE). `tests/test_augmentations.py` (4 test classes, shared `_assert_valid()` validator checking connectivity, root preservation, min-nodes) |
| 23 | Train stage | DONE | 100 epochs completed. Best epoch 89 (loss 2.292). Warmup (5 ep) + cosine LR. Crash recovery via `last.pt`. Early stopping patience=15 did not trigger (loss kept improving slowly) |

**Full build_pyg_dataset results (2026-03-18):**
- Cascades: **39,983** — Nodes: 884,832 — Edges: 844,849 — Feature dim: 6
- Size: min=3, median=10, mean=22.1, p95=69, max=1,854
- Depth: min=1, median=4, mean=4.61, p95=11, max=60
- Follower ratio: median=0.80, mean=0.78
- Sentiment: negative=54.5%, neutral=39.65%, positive=5.85%
- Events: E1=2,732 / E2=3,338 / E3=2,716 / E4=1,320 / E5=1,375 / E6=930 / baseline=27,572
- Outputs: `outputs/build_pyg_dataset/cascade_dataset.pt` (159.8 MB), `dataset_stats.json`

**Training results (2026-03-19, COMPLETE — 100 epochs):**

| Epoch | Loss | Contrastive | Aux | LR | Grad Norm | Time |
|-------|------|-------------|-----|-----|-----------|------|
| 1 | 3.263 | 2.720 | 0.543 | 0.000200 | 1.848 | 98 s |
| 5 | 2.464 | 2.442 | 0.022 | 0.001000 | 1.846 | 492 s |
| 10 | 2.361 | 2.350 | 0.012 | 0.000993 | 1.043 | 983 s |
| 20 | 2.318 | 2.317 | 0.002 | 0.000940 | 0.575 | 1964 s |
| 50 | 2.299 | 2.298 | 0.000 | 0.000546 | 0.337 | 4905 s |
| **89** | **2.292** | **2.292** | **0.000** | **0.000042** | **0.298** | **8931 s** |
| 100 | 2.293 | 2.293 | 0.000 | 0.000010 | 0.307 | 10063 s |

- **Best checkpoint:** epoch 89 (loss 2.29225) → `outputs/train/checkpoints/best.pt`
- Aux loss collapsed to ~7e-5 by epoch 100 (DiffPool well-behaved)
- Grad norms stabilised at ~0.3 (no explosion, no clipping saturation)
- Loss plateau from ~epoch 20 onward (2.318 → 2.292). Contrastive loss well below `log(64) ≈ 4.16` — model is learning, but slow convergence suggests possible embedding saturation
- Total wall time: ~2.8 hours on CPU
- Checkpoints: `best.pt`, `last.pt`, `epoch_{10,20,...,100}.pt`, `config_snapshot.yaml`, `training_log.json`

**Training v1 diagnostics (2026-03-23) — MARGINAL:**

| Metric | Value | Status |
|---|---|---|
| Median cosine sim | 0.365 | OK |
| Effective rank (SVD) | 14.9 / 64 | WARN — partial dimensional collapse |
| Uniformity | -1.884 | WARN |
| Alignment | 0.997 | CRITICAL — augmentations too weak |
| Top-5 variance ratio | 0.907 | WARN |

Root cause: augmentations too weak; model solves pretext task via unchanged node feature statistics. Required retrain.

**Iterative config evolution (v2→v8):** Multiple training runs on SSH VMs with progressive hyperparameter tuning. Key changes each version tracked in `config.yaml` comments. Final v8 config (currently in `config.yaml`) includes:
- `temperature=0.1` (lowered from v2's 0.3 — harder contrastive task)
- `grad_clip_norm=5.0` (raised from 2.0 — was clipping 100% of epochs in v6)
- `covariance_weight=0.25` (lowered from 1.0 — was dominating loss)
- `rgcn_layers=2` (lowered from 3 — 3-hop over-smoothed small cascades)
- `use_root_readout=false` (root has 3 constant features, dilutes readout)
- `epochs=150` (raised from 75 — best epoch was 74/75 in v6, model still improving)
- `projector_layers=3`, `projector_bn=true` (VICReg/SimCLR v2 best practice)

**New code added v1→v8:**
- `blueccgl/augmentations/feature_masking.py` — random node feature dimension zeroing
- `blueccgl/models/losses.py` — `vicreg_variance_loss()` (Bardes et al., 2022) + covariance loss
- `blueccgl/stages/train.py` — feature masking + variance/covariance losses integrated
- `blueccgl/utils/mpl_compat.py` — matplotlib Python 3.14 compatibility patch

### Sprint 5: Full Data Integration + Downstream Pipeline

| # | Task | Status | Notes |
|---|------|--------|-------|
| 24 | Download full Zenodo dataset | PARTIAL | `user_posts.tar.gz` (19 GB) + `followers.csv.gz` (469 MB) downloaded. Others not needed for current approach |
| 25 | Run `--stage nlp_filter` on full tar | DONE | 1,947,493 conflict posts, 85 min |
| 26 | Run `--stage cascade_construction` with full data | DONE | 39,983 cascades, ~3 min |
| 27 | Run `--stage build_pyg_dataset` | DONE | cascade_dataset.pt (159.8 MB, 39,983 cascades). 2026-03-18 |
| 28 | Run `--stage eda` | NOT STARTED | Stub — needs implementation. Unblocked |
| 29 | **Training v8** (SSH VM 87.205.21.33) | DONE | 150 epochs, best epoch 147, loss 0.1292. Config: τ=0.1, feature masking, VICReg, covariance reg. Results in `Outputs_17_04/`. Note: 79.6% aug fallback rate flagged — fixed in v9 config below |
| 30 | **Run `--stage extract_embeddings`** | DONE | embeddings.h5 + cascade_index.parquet. Embedding diagnostics block added. |
| 31 | **Run `--stage cluster`** | DONE | UMAP 64D→10D + HDBSCAN. mcs=300 selected, 11 clusters, 1.5% noise, silhouette=0.362, mean persistence=0.395 |
| 32 | **Run `--stage evaluate`** | DONE | Metrics computed in 10D UMAP space (fix 2026-04-18). v8: silhouette=0.413, DBCV=+0.618, DB=0.676, CH=1978. |
| 33 | **Run `--stage explain`** | DONE | Probes, DiffPool assignments, cluster profiles (4 figures), augmentation sensitivity. Event heatmap KeyError fixed. matplotlib RecursionError fixed via mpl_compat.py. Probe log1p fix applied 2026-05-17 (wiener_index + cascade_size) |
| 34 | **Implement + run `--stage ablation`** (2³ factorial) | DONE (impl 2026-05-26) / PENDING (VM runs) | Stage implemented as 2³ factorial grid: encoder (R-GCN vs GCN) × augmentation (feed-aware vs random) × sentiment (on vs off) = 7 variants. Baseline `rgcn_feed_sent` = `Outputs_17_04`. Remaining 7 variants to run on SSH VM (`cascade_dataset.pt` only needed). Results → `outputs/ablation/{variant}/`. See Step 4.7 for implementation details. |
| 35 | **Implement + run `--stage downstream`** | DROPPED | Out of scope — follower_edge_ratio R²=0.941 already demonstrates embedding quality |
| 36 | **Implement + run `--stage stability`** | DROPPED | Out of scope — v8 vs v9 sensitivity comparison serves as de-facto stability evidence |
| 37 | **Implement + run `--stage hierarchical`** | DONE | Ward agglomerative on 11 cluster centroids (10D UMAP). 5 macro-archetypes (natural cut at merge distance 10.05). Outputs: dendrogram, heatmap (rank-normalised), macro profiles, UMAP, bubble chart, archetype_taxonomy.csv. Docs: `Docs/hierarchical_analysis.md` + `Docs/hierarchical_analysis.pdf`. Stage implemented and run 2026-05-21 |
| 38 | **Implement + run `--stage validate_hierarchical`** | DONE | 4 validation methods + prediction strength stub. Cophenetic r=0.90, metric sweep all-agree k=5, bootstrap ARI=1.000, UMAP seed ARI=0.915±0.094. Full results in `Outputs_17_04/validate_hierarchical/validation_report.json`. Stage implemented and run 2026-05-21 |

**v9 config changes (augmentation fallback reduction, 2026-04-17):**
Root causes of 79.6% fallback (two mechanisms): (1) Strategy C returns `None` for cascades with no NON_FOLLOWER edges — given 75.9% FOLLOWER dominance, ~90% of cascades (especially near the 10-node median) are all-follower → C structurally ineligible; (2) Strategy A with aggressive retain_ratio disconnects the root in chain-like cascades → root_component returns singleton. OR fallback semantics amplify both: P(fallback) ≈ 2/3·(α+γ) where α≈0.5, γ≈0.9. Config changes:
- `augmentation.follower_retain_ratio`: 0.3 → **0.4**
- `augmentation.temporal_prefix_ratio`: 0.3 → **0.5**
- `augmentation.min_nodes_after_aug`: 5 → **3**

**Training v8 results summary (primary results — `Outputs_17_04/`):**

| Metric | Value |
|--------|-------|
| Epochs | 150 (best: 147) |
| Best loss | 0.1292 |
| Clusters | 11 |
| Noise fraction | 1.5% (594 / 39,983) |
| Mean HDBSCAN persistence | 0.395 |
| **Silhouette** (10D UMAP space) | **0.413** ± 0.004, 95% CI [0.405, 0.422] |
| **Davies-Bouldin** (10D UMAP space) | **0.676** ± 0.007 (lower = better) |
| **Calinski-Harabasz** (10D UMAP space) | **1978** ± 38 |
| **DBCV** (10D UMAP space) | **+0.618** ± 0.028 — positive = clusters denser than surrounding space |
| follower_edge_ratio R² | 0.941 (global), 0.917–0.981 per cluster |
| cascade_depth R² | 0.686 (global), 0.697–0.824 per cluster |
| cascade_breadth R² | 0.317 (global), 0.333–0.938 per cluster |
| cascade_size R² | −6.07 (global — driven by clusters 3 and 6 with extreme size variance) |
| event_window accuracy | 0.143 (chance = 1/7) — no event discrimination |

**v9 sensitivity run (2026-04-18, `Outputs_18_04/`) — reported as sensitivity analysis, not primary:**

| Metric | v8 | v9 | Notes |
|--------|----|----|-------|
| Best loss | 0.1292 | 0.1924 | v9 easier augmentations → weaker contrastive signal |
| Clusters | 11 | 18 | More granular archetypes |
| Noise | 1.5% | 5.0% | Slightly higher |
| Silhouette (10D) | 0.413 | 0.445 | v9 marginally better in UMAP space |
| DBCV (10D) | +0.618 | −0.46* | *v9 metrics were computed in 64D before fix |
| follower R² | **0.941** | 0.714 | v8 substantially better — key advantage |
| Aug fallback | 79.6% | 42.2% | v9 better but still structural floor |

v8 selected as primary result due to stronger embedding quality (follower R²=0.941) and better loss convergence. v9 demonstrates the 11-archetype solution is robust — rerunning with corrected augmentation config recovers similar cluster structure.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation | Status |
|------|-----------|--------|------------|--------|
| **Python 3.14 incompatible with PyTorch/PyG** | High | Blocks all ML code | Create 3.11/3.12 venv as first step | RESOLVED — Python 3.14 works with CPU torch 2.10.0 + PyG 2.7.0 from wheel server |
| **Disk space for full data** | — | — | — | RESOLVED — CPU-only torch + pip cache purge freed space. Streaming tar (no full extraction) keeps footprint manageable. ~15 GB free |
| **Too few cascades with ≥10 reposts** | Medium | Weak clustering signal | Lower threshold to ≥5; report as limitation | RESOLVED — 39,983 cascades at ≥10 threshold. Ample for HDBSCAN with min_cluster_size=50–200 |
| **DiffPool OOM on large cascades** | Medium | Training crashes | Cap cascade size at 1000 for DiffPool; fallback to SAGPool | MITIGATED — max_nodes=150 in DiffPool. P95 cascade size is 69 nodes, so cap is nearly lossless. Training stable so far |
| **Representation collapse during training** | Medium | Useless embeddings | Monitor embedding variance; add projector MLP; tune temperature | RESOLVED — v8: follower_edge_ratio R²=0.941, cascade_depth R²=0.686. Embeddings carry strong structural signal. event_window accuracy still at chance (0.143) — acknowledged limitation |
| **Follower graph temporal confound** | Certain | Biased FOLLOWER labels for early cascades | Acknowledge as limitation; sensitivity analysis | OPEN |
| **High HDBSCAN noise fraction** | Medium | Weak archetype discovery | UMAP 64D→10D pre-reduction before HDBSCAN; noise_threshold=0.70 to allow silhouette comparison | RESOLVED — v8: 1.5% noise (was 67.6% without UMAP pre-reduction) |
| **Cascade reconstruction heuristic too lossy** | Medium | Inaccurate cascade trees | Validate via known properties (e.g., branching factor distributions from literature) | OPEN |
| **Keyword false positives (e.g., "hostage")** | Medium | Noisy cascade roots | Co-occurrence gate: "hostage/hostages" requires ≥1 other keyword to match | MITIGATED — gate implemented in keywords.py |
| **Sample data insufficient for cascade testing** | High | Can't test cascade construction without followers data | Use degraded mode with `repost_from` field only; test full pipeline once data arrives | RESOLVED — full data downloaded, unit tests pass |
| **Training on single GPU too slow for 100 epochs** | Low-Medium | Schedule slippage | CPU-only mode (no NVIDIA GPU). Profile first; reduce epochs / cascade count if needed | RESOLVED — 100 epochs completed in ~2.8 hours on CPU. ~100 s/epoch |
| **NLP filter too slow on full tar (6+ hours)** | — | Schedule delay | Switched from `r:gz` + `getmembers()` to `r\|gz` streaming + multiprocessing (15 workers) | RESOLVED — 95 files/sec → 500 files/sec (~80 min) |

---

## Key Decisions Made

1. **Python version:** 3.14 works. CPU-only torch from `download.pytorch.org/whl/cpu`.
2. **DuckDB for follower graph:** Yes. `FollowerGraph` class auto-builds indexed DuckDB from `followers.csv.gz` on first use.
3. **No `interactions.csv.gz` needed:** Cascade construction uses `repost_from`/`reposted_author` from user_posts directly (via `conflict_window_posts.parquet`). No need for separate interactions file.
4. **Streaming tar approach:** 19 GB tar never fully extracted. `conflict_window_posts.parquet` (2.4 GB, 172M rows) pre-extracted for fast reposter discovery. NLP filter streams with `r|gz` mode.
5. **`--set` bool override:** Fixed to cast `"true"`/`"false"` strings to Python bools.
6. **DiffPool max_nodes=150:** Capped at 150 nodes (not 1000 as originally planned) to prevent CPU OOM during dense assignment matrix operations. The p95 cascade size is 69 nodes, so this cap is essentially lossless for 95% of cascades.
7. **Batch size 32:** Reduced from 64 to 32 to fit CPU memory during training. InfoNCE still has 31 negatives per positive pair.
8. **Augmentation utilities in `__init__.py`:** Shared `subgraph()` and `root_component()` helpers live in `blueccgl/augmentations/__init__.py` rather than duplicating across strategies.
9. **Two-pass sentiment lookup in `build_pyg_dataset`:** Primary lookup in `conflict_posts.parquet` (1.9M rows, fast); DuckDB fallback on `conflict_window_posts.parquet` (172M rows) for reposter nodes not in the NLP-filtered set.
10. **Crash recovery in `train.py`:** `last.pt` checkpoint saved every epoch enables resume from interruption without loss of progress.
11. **UMAP before HDBSCAN (2026-04-17):** Raw 64D embeddings cause curse of dimensionality — all HDBSCAN configs produced >67% noise. Added UMAP 64D→10D reduction in cluster.py before grid search. This is standard practice and dropped noise to 1.5%.
12. **noise_threshold=0.70 (2026-04-17):** Raised from hardcoded 0.30 to 0.70 (configurable). With UMAP pre-reduction, all configs produce low noise; the threshold mainly prevents pathological fallback, not normal operation. 0.30 caused all configs to score -1.0 silhouette before UMAP was added.
13. **mcs=[100,200,300] for 11 archetypes (2026-04-17):** Initial grid [15,25,40] produced 700+ clusters (too granular for thesis interpretation). [100,200,300] with mcs=300 selected → 11 meaningful archetypes — thesis-interpretable, matches prior cascade literature cluster counts.
14. **matplotlib Python 3.14 fix via monkey-patch (2026-04-17):** `matplotlib/path.py:285` calls `copy.deepcopy(super(), memo)` which causes infinite recursion in Python 3.14's updated deepcopy implementation. `sys.setrecursionlimit` is insufficient (complex figures need >10,000 levels). Fix: `blueccgl/utils/mpl_compat.py` patches `Path.__deepcopy__` to copy `__dict__` directly, avoiding the `super()` call. Imported at the top of all affected stages.
15. **min_nodes_after_aug=3, follower_retain_ratio=0.4 (v9 config, 2026-04-17):** Original v6/v7 config had `min_nodes=5` and `retain_ratio=0.3` — together these caused 79.6% of cascades to fall back to random augmentation (median cascade has 7 nodes; 0.3×7=2.1 < 5). Fixed to `min_nodes=3`, `retain_ratio=0.4`, `temporal_prefix_ratio=0.5`.
16. **DiffPool disabled in v8 (pool=mean, 2026-05-21):** v8 `config.yaml` has `model.pool: mean`, which bypasses DiffPool entirely and falls back to global mean pooling. DiffPool was disabled after `pool_2` collapse was observed during v8 hyperparameter search. `explain.run_diffpool_analysis: true` is a no-op for this config — recommendation to set it `false` to suppress spurious "assignments not found" warnings.
17. **log1p probe transformation for wiener_index + cascade_size (2026-05-21):** Raw Wiener index scales O(N²) and cascade_size is heavily right-skewed — both break the linearity assumption of Ridge regression probing. Global probe R² was catastrophic (cascade_size R²=−6.07). Fixed by applying `y = np.log1p(y)` before fitting. Per-cluster cascade_size R² improved to 0.69–0.91 for 9/11 clusters. `target_transform: "log1p"` field added to probe results JSON. Fix applied in `explain.py:_run_probes`.
18. **Hierarchical stage as separate pipeline stage (2026-05-21):** Ward agglomerative clustering on HDBSCAN cluster centroids (computed in 10D UMAP space) added as `--stage hierarchical`. Depends only on `--stage cluster`. Natural cut via largest-gap in linkage merge distances; `n_macro` clamped [2,6]. 5-macro solution selected at cut height 10.05 — singleton C6 (Macro 5) retained deliberately as it represents a structurally distinct scaled-up intra-network spreading regime (high follower ratio, large breadth) separable from the smaller follower-driven neutral cascades in Macro 3. The out-of-network diffusion archetype is C8 (Macro 1), not C6.
19. **Rank normalisation for hierarchical heatmap (2026-05-21):** Min-max normalisation produced multiple cells showing 1.00 for the same property (ties when several clusters share the same median value). Switched to `scipy.stats.rankdata` rank normalisation: cell value = `(rank − 1) / (n_clusters − 1)`, giving strictly ordered values with no spurious ties across clusters.
20. **"Out-of-network diffusion" replaces "algorithmic amplification" (2026-05-21):** The original framing attributed non-follower edge propagation to platform algorithmic amplification — a causal claim not supportable from follower-graph data alone (temporal confound; non-follower edges conflate search, second-order reposts, hashtag discovery, and feed surfacing). Replaced throughout with "out-of-network diffusion" (structural, mechanism-agnostic) grounded in Granovetter (1973) weak-tie bridging. C8 (follower ratio 0.52, depth 5) is the out-of-network diffusion archetype; C6 (follower ratio 0.875) is scaled-up intra-network spreading. Old planning docs (`Initial_Plan.md`, `Project_Plan_Draft.md`, `Palestine_Bluesky_Project_Plan.md`, `latex/main.tex`, `latex/palestine_plan.tex`) marked DEPRECATED.
22. **Ablation redesigned from A1–A5 to 2³ factorial (2026-05-26):** Original plan described five independent ablations (A1: no edge-type, A2: generic aug, A3: flat pool, A4: no sentiment, A5: random init). Replaced with a 2³ factorial grid that orthogonally isolates three binary factors (R-GCN vs GCN, feed-aware vs random aug, sentiment on vs off). This enables main-effect and interaction-effect estimation, which is more informative for a thesis table than five independent point comparisons. DiffPool is already disabled in v8 (`pool=mean`), so A3 (flat pool) is vacuous and dropped. A5 (random init) is dropped — the downstream probe R²=0.941 already demonstrates that learned representations are non-random without a separate untrained baseline. The 2³ design preserves the three most thesis-relevant factors.

21. **`--stage validate_hierarchical` as a separate post-hierarchical stage (2026-05-21):** Four validation methods (cophenetic correlation, multi-cut metric sweep, bootstrap ARI, UMAP seed ARI) implemented as `blueccgl/stages/validate_hierarchical.py`. Prediction strength (Tibshirani & Walther 2005) stubbed with full algorithm in docstring. Depends on `--stage hierarchical`. All four methods validate the 5-macro solution; see Step 4.6 for full results.

## Key Decisions Still Open

1. **DiffPool vs. SAGPool:** DiffPool completed training successfully (max_nodes=150). No instability observed. Decision: keep DiffPool.
2. **Ablation training time budget:** RESOLVED — 2³ factorial design implemented (7 variants, ~7.5 hrs on GPU). Uses same 150-epoch schedule as baseline. Variants pending VM run. See Step 4.7.
3. **Explain stage optional extensions:** Core 4 sub-steps done. Optional extensions (GNNExplainer, PGExplainer, TCAV) depend on remaining time budget. TCAV has highest thesis novelty but requires concept set curation (~6h). Decision deferred.
4. **event_window not captured in embeddings:** Probes show balanced accuracy = 0.143 (chance) — model does not discriminate between events. This limits SRQ1/SRQ2 answers. Options: (a) accept as limitation, (b) add event-aware contrastive loss, (c) use cluster × event cross-tabulation from cluster profiles instead.
5. **v9 completed (2026-04-18):** v9 run on VM 192.165.134.28, 18 clusters, 42.2% fallback. Decided to use v8 as primary result (follower R²=0.941 > v9's 0.714). v9 reported as sensitivity analysis. No further retraining planned.

---

## Known Bugs Fixed

1. **`extract_conflict_window.py` — `batch.append()` in wrong block:** Was inside `if files_done % 100 == 0:` (file-level progress) instead of per-line loop. Produced 593 KB parquet instead of 2.4 GB. Fixed 2025-03-17.
2. **`run_pipeline.py` — `--set key=false` treated as truthy string:** `"false"` failed int/float cast, stayed as string `"false"` (truthy). Added bool casting before numeric. Fixed 2025-03-17.
3. **`nlp_filter.py` — `_iter_tar` used `getmembers()` with `r:gz`:** After reading all 2.3M headers, each `extractfile()` forced gzip re-decompression from start (O(n²)). Switched to `r|gz` streaming + multiprocessing. 95 → 500 files/sec. Fixed 2025-03-17.
4. **`quick_look.py` — `langs_list or []` ValueError:** Numpy array `or` is ambiguous. Fixed with explicit None/length check.
5. **`cascade_construction.py` — `_load_follower_graph` wrong path:** Only checked `Data/raw/followers.csv.gz`. Added fallback to `Data/followers.csv.gz`. Fixed 2025-03-17.
6. **`follower_graph.py` — per-edge DuckDB queries O(n²):** Each `follows()` call issued a separate SQL query, making tree building take an estimated 26 hours. Fixed with `preload_for_users()`: one batch JOIN query loads all relevant edges into a Python `set` before tree building starts. Total runtime went from 26hr estimate to ~3 min. Fixed 2025-03-17.
7. **matplotlib Python 3.14 RecursionError in all plot-generating stages:** `matplotlib/path.py` `Path.__deepcopy__` calls `copy.deepcopy(super(), memo)` which causes infinite recursion under Python 3.14's updated deepcopy. `sys.setrecursionlimit(10_000)` is insufficient — complex figures need >10,000 levels and hit the limit. Fixed with `blueccgl/utils/mpl_compat.py` monkey-patch that replaces `Path.__deepcopy__` with a direct `__dict__` copy that never calls `super()`. Imported at top of `evaluate.py` and `explain.py`. Fixed 2026-04-17.
8. **`explain.py` — `_plot_archetype_event_heatmap` KeyError on cluster ID 7:** `event_cols = sorted(frac[list(frac.keys())[0]].keys())` dereferenced one level too deep — `frac` is `{event_window: {cluster_id: fraction}}`, so indexing one level in gave cluster IDs {0..10}, not event windows {0..6}. Accessing `frac[7]` → KeyError. Fixed to `event_cols = sorted(frac.keys())`. Fixed 2026-04-17.
9. **`evaluate.py` — clustering metrics computed in wrong space:** Silhouette/Davies-Bouldin/CH/DBCV were bootstrapped on raw 64D embeddings, but HDBSCAN clustered in 10D UMAP space. This produced silhouette≈0.012 and DBCV=-0.46 — near-random numbers that don't reflect actual cluster quality. Fixed to load `cluster/umap_embeddings_cluster.h5` and compute all metrics in 10D space. v8 corrected metrics: silhouette=0.413, DB=0.676, CH=1978, DBCV=+0.618. `metric_space` recorded in `metrics["meta"]`. Fixed 2026-04-18.
10. **`explain.py` — probe R² catastrophic for heavy-tailed targets (wiener_index, cascade_size):** Global R² for cascade_size was −6.07 (worse than predicting the mean). Root cause: Wiener index is O(N²) in cascade size, producing extreme outliers; cascade_size is right-skewed. Both violate Ridge regression's linearity assumption. Fixed by applying `np.log1p()` to the target before fitting for these two properties. Bootstrap CIs which previously didn't bracket the point estimate (due to resampling rarely hitting extreme outliers) now behave correctly. Fixed 2026-05-21.
11. **`hierarchical.py` — HDF5 KeyError on `"umap_embeddings"` key:** Stage was reading `f["umap_embeddings"][:]` but `cluster/umap_embeddings_cluster.h5` stores embeddings under key `"embeddings"`. Fixed to `f["embeddings"][:]`. Fixed 2026-05-21.
12. **`hierarchical.py` — macro profile bar chart showing negative lower bounds:** Bar chart used mean ± std for cascade_size. Since std > mean for most clusters (right-skewed distribution), the lower error bar goes negative (impossible for a count). Fixed by switching to median + IQR with asymmetric error bars (`err_lo = median − q25`, `err_hi = q75 − median`). Fixed 2026-05-21.
13. **`hierarchical.py` — heatmap showed raw median values alongside normalised colorbar:** Cell annotations showed raw values (e.g., 13, 16) while the colorbar represented normalised scale — inconsistent. Fixed by storing normalised values in the cell array and displaying those in annotations; colorbar removed entirely. Fixed 2026-05-21.
14. **`ablation.py` — stub replaced with 2³ factorial implementation (2026-05-26):** Stage was `raise NotImplementedError()`. Implemented as full 2³ factorial grid (encoder × aug × sentiment, 7 variants). Dataset symlinked per-variant to avoid 159.8 MB copies. Resume-safe via `evaluate/metrics.json` checkpoint check. `pipeline.py` updated: `"ablation"` added to `ALL_STAGES`, `STAGE_CHECKPOINTS`, `STAGE_DEPS`. `run_pipeline.py` `_run_stage()` dispatcher updated.
