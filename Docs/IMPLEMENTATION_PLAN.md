# BlueCCGL — Full Implementation Plan

## Current State Assessment (Updated 2026-03-23)

| Asset | Status |
|-------|--------|
| **Sample data** | 100 JSONL files (~15 MB, ~25K posts) in `Data/user_posts/` |
| **Full data — user_posts.tar.gz** | Downloaded (19 GB, 2,363,249 JSONL files, ~4M posts, 128 GB uncompressed). In `Data/user_posts.tar.gz` |
| **Full data — followers.csv.gz** | Downloaded (469 MB, ~144M edges). In `Data/followers.csv.gz` |
| **Full data — others** | NOT downloaded: `interactions.csv.gz` (1 GB), `graphs.tar.gz` (891 MB), `feed_posts.tar.gz` (16.4 MB), `feed_bookmarks.csv` (553 KB). **Not needed for current cascade construction approach** — we use `repost_from`/`reposted_author` fields from user_posts instead |
| **conflict_window_posts.parquet** | Extracted from tar (2.4 GB, 172M rows). Lean index: post_id, user_id, timestamp, repost_from, reposted_author, sent_label, sent_score. **No text field.** Used for fast reposter discovery in cascade construction |
| **cascade_dataset.pt** | Built (159.8 MB). 39,983 cascades, 884,832 nodes, 844,849 edges, node feature dim=6. Stored in `outputs/build_pyg_dataset/` |
| **Environment** | Python 3.14 venv, CPU-only PyTorch 2.10.0, PyG 2.7.0, all deps installed. See `requirements.txt` |
| **Pipeline** | Fully functional `run_pipeline.py` CLI + `config.yaml`. Stages `nlp_filter`, `cascade_construction`, `build_pyg_dataset`, and `train` fully implemented and executed. Downstream stages (`extract_embeddings`, `cluster`, `evaluate`, `ablation`, `downstream`, `stability`, `eda`) are stubs requiring implementation |
| **Training v1** | DONE — 50 epochs, τ=0.1, cosine LR. Loss 1.589→0.035. Diagnostics: MARGINAL — alignment=0.997 (augmentations too weak), eff_rank=14.9/64 (dimensional collapse), uniformity=-1.88. Archived as `outputs/train/run_v1_tau01/` |
| **Training v2** | PENDING RETRAIN — stronger augmentations (retain 0.5/prefix 0.4/bridge 0.5), feature masking 25%, τ=0.3, VICReg variance reg (0.1), DiffPool aux weight 0.5. Target: alignment 0.6–0.85, eff_rank >25, uniformity <-3.0 |
| **Disk** | ~15 GB free. Full tar extraction not feasible — streaming only |

### Progress Summary

| Sprint | Status | Notes |
|--------|--------|-------|
| Sprint 1: Scaffolding | DONE | All package structure, CLI, config, pipeline orchestrator, seed, gitignore |
| Sprint 2: NLP Filter | DONE | 1,947,493 conflict posts from 233M total (0.84%). 15-worker parallel tar streaming (~85 min) |
| Sprint 3: Cascade Construction | DONE | 39,983 cascades (≥10 nodes), 844,849 edges, 75.9% FOLLOWER-typed, ~3 min runtime |
| Sprint 4: PyG Dataset + Model | DONE | Dataset built; all models + augmentations + tests implemented; training v1 complete (50 epochs) |
| Sprint 5: Full Data Integration | IN PROGRESS | Training v1 diagnostics: MARGINAL (alignment=0.997, eff_rank=14.9). Retrain v2 ready with stronger augmentations, feature masking, τ=0.3, VICReg variance reg. Stages 30–35 (extract → evaluate → ablation → downstream → stability) need implementation |

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
│   │   ├── ablation.py          # --stage ablation
│   │   ├── downstream.py        # --stage downstream
│   │   └── stability.py         # --stage stability
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
├── scripts/                     # Ad-hoc exploration (legacy)
│   └── eda_user_posts.py
│
├── Docs/
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

# Evaluation
scipy, matplotlib, seaborn

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

### Step 4.2 — Extract Embeddings Stage (`blueccgl/stages/extract_embeddings.py`)

**Stage:** `--stage extract_embeddings`
**Reads:** `{output_dir}/train/checkpoints/best.pt`, `{output_dir}/build_pyg_dataset/cascade_dataset.pt`
**Writes:** `{output_dir}/extract_embeddings/embeddings.h5`

1. Load trained model (eval mode, no augmentation).
2. Forward pass all cascades.
3. Save: `embeddings` array `[N_cascades, embedding_dim]`, `cascade_ids`, `event_windows`.
4. Use HDF5 for storage (efficient for large matrices + metadata).

**Checkpoint validation:**
- Embedding matrix shape: `[N_cascades, 64]`.
- No NaN or Inf values.
- Variance of embeddings > 0 (no collapse).
- Cosine similarity histogram: should be roughly centred, not peaked at 1.0.

### Step 4.3 — Cluster Stage (`blueccgl/stages/cluster.py`)

**Stage:** `--stage cluster`
**Reads:** `{output_dir}/extract_embeddings/embeddings.h5`
**Writes:** `{output_dir}/cluster/cluster_assignments.parquet`, `{output_dir}/cluster/hdbscan_results.json`

1. Load embeddings.
2. For each `min_cluster_size` in `[50, 100, 200]`:
   a. Fit HDBSCAN.
   b. Record: cluster labels, noise fraction, number of clusters, DBCV score.
3. Select best `min_cluster_size` by DBCV (or manual inspection).
4. Save cluster assignments for the best configuration.

**Potential issues:**
1. **High noise fraction (>30%):** Indicates embeddings don't form well-separated clusters. Investigate: (a) embedding quality (collapse?), (b) augmentation quality, (c) cascade homogeneity (all cascades look similar).
2. **Too few clusters (<3):** HDBSCAN may find only 1–2 clusters + noise. Lower `min_cluster_size`. If still few, try reducing `min_samples`.
3. **Too many clusters (>20):** May indicate fragmentation. Increase `min_cluster_size`.

### Step 4.4 — Evaluate Stage (`blueccgl/stages/evaluate.py`)

**Stage:** `--stage evaluate`
**Reads:** `{output_dir}/cluster/cluster_assignments.parquet`, `{output_dir}/extract_embeddings/embeddings.h5`
**Writes:** `{output_dir}/evaluate/metrics.json`, `{output_dir}/evaluate/umap_*.png`, `{output_dir}/evaluate/archetype_profiles.json`

1. Compute metrics (excluding noise points):
   - Silhouette Score
   - Davies-Bouldin Index
   - Calinski-Harabasz Index
   - DBCV
2. Bootstrap confidence intervals (1000 resamples) for each metric.
3. UMAP projection (2D) of embeddings, coloured by:
   - Cluster assignment
   - Event window
   - Sentiment composition (dominant sentiment)
   - Follower-edge ratio (continuous colour scale)
4. Archetype characterisation: for each cluster, compute mean ± std of all structural and sentiment features.

### Step 4.5 — Explain Stage (`blueccgl/stages/explain.py`)

**Stage:** `--stage explain`
**Reads:**
- `{output_dir}/train/checkpoints/best.pt`
- `{output_dir}/build_pyg_dataset/cascade_dataset.pt`
- `{output_dir}/extract_embeddings/embeddings.h5`
- `{output_dir}/cluster/cluster_assignments.parquet`
**Writes:** `{output_dir}/explain/` (see output structure below)

**Dependency:** `cluster` (runs after clustering assigns archetypes)

Multi-layer explainability pipeline with 4 sub-steps, each answering a distinct question about discovered cascade archetypes.

#### Sub-step 5.1 — Probing Classifiers (What is encoded in the embeddings?)

Train lightweight linear classifiers (logistic regression / shallow MLP) on frozen 64-dim embeddings to predict interpretable graph properties already stored on each `Data` object: `cascade_depth`, `cascade_breadth`, `wiener_index`, `follower_edge_ratio`, `sentiment_composition`, `cascade_size`, `event_window`.

**Output:** `explain/probe_results.json` — R²/accuracy per property, overall and per-cluster.

**Interpretation:** If wiener_index is predictable with R²=0.87 but sentiment gives R²=0.12, the contrastive objective has encoded structural virality but not sentiment dynamics. Per-cluster breakdown characterises each archetype in terms of what the model sees in it.

**Key reference:** Alain & Bengio (2017). "Understanding intermediate layers using linear classifier probes."

#### Sub-step 5.2 — DiffPool Assignment Extraction (What structural roles does the model learn?)

The `HierarchicalDiffPool` already computes soft assignment matrices `s1` (Level 1) and `s2` (Level 2) during forward pass — this is free explainability currently being discarded.

**Implementation:**
1. Add `return_assignments=False` parameter to `HierarchicalDiffPool.forward()` in [diffpool.py](blueccgl/models/diffpool.py). When True, also return `s1`, `s2`.
2. Add `explain(batch)` method to `BlueCCGL` in [blueccgl.py](blueccgl/models/blueccgl.py) that returns `(embedding, s1, s2, node_embeddings_h)`.
3. Run inference over all cascades, extracting assignment matrices.
4. For each cascade: compute assignment entropy per node (high entropy = ambiguous role, low = clear role), hard partition via argmax, super-node feature profiles (mean depth, sentiment, degree of member nodes).
5. Aggregate super-node characterisations per HDBSCAN cluster.

**Output:** `explain/diffpool_assignments.h5` — assignment matrices, entropy maps, super-node profiles.

**Interpretation:** "In Archetype A, DiffPool consistently groups high-degree early adopters into super-node 0 and sentiment-polarised leaf nodes into super-node 1." Visualise as node-coloured cascade tree layouts.

**Key reference:** Ying et al. (2018). "Hierarchical Graph Representation Learning with Differentiable Pooling."

#### Sub-step 5.3 — Cluster Statistical Profiles (What distinguishes archetypes?)

For each HDBSCAN cluster, compute summary statistics of all graph-level properties and run statistical tests.

**Implementation:**
1. Per-cluster: mean ± std of `cascade_depth`, `cascade_breadth`, `wiener_index`, `follower_edge_ratio`, `sentiment_composition` (3 dims), `cascade_size`.
2. Between-cluster: Kruskal-Wallis test per property, then pairwise Mann-Whitney U with Bonferroni correction.
3. Discriminative embedding dimensions: PCA on cluster centroids → identify which of the 64 dims have highest between-cluster variance → correlate with probing classifier weights from Sub-step 5.1.

**Output:** `explain/cluster_profiles.json` — per-archetype profiles, pairwise significance tables, discriminative dimensions.

**Interpretation:** "Archetype A (n=342): deep chains (depth=8.2±2.1), low follower ratio (0.23), negative sentiment dominant. Archetype B (n=189): shallow star-bursts (depth=2.1, breadth=45.3), high follower ratio (0.78), neutral sentiment."

#### Sub-step 5.4 — Augmentation Sensitivity Analysis (What is structurally defining for each archetype?)

The three feed-aware augmentations each destroy specific information. By measuring which augmentation changes the embedding most, we reveal what the model considers structurally essential.

**Implementation:**
1. For each cascade in a sample (e.g., 500 per cluster): apply each augmentation separately, compute `||embed(original) - embed(augmented)||₂`.
2. Aggregate per cluster: mean sensitivity to follower_subgraph vs. temporal_prefix vs. bridge_removal.

**Output:** `explain/augmentation_sensitivity.json` — per-cluster sensitivity scores per augmentation type.

**Interpretation:** "Cluster 3 is highly sensitive to bridge_removal (non-follower entry nodes are critical) but invariant to follower_subgraph → this archetype is defined by cross-community bridges, not follower structure."

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

#### Config Addition

```yaml
explain:
  n_representatives_per_cluster: 20     # sample size for per-instance analyses
  run_probes: true
  run_diffpool_analysis: true
  run_augmentation_sensitivity: true
  run_cluster_profiles: true
  probe_properties: [cascade_depth, cascade_breadth, wiener_index,
                     follower_edge_ratio, cascade_size, event_window]
  figures_dpi: 300
```

#### Output Structure

```
outputs/explain/
  probe_results.json                    # R²/accuracy per property per cluster
  cluster_profiles.json                 # Summary stats + statistical tests + discriminative dims
  augmentation_sensitivity.json         # Per-cluster sensitivity to each augmentation
  diffpool_assignments.h5               # Assignment matrices, entropy, super-node profiles
  explanation_report.json               # Combined summary (stage checkpoint)
  figures/
    cluster_radar_profiles.png          # Radar charts per archetype
    probe_heatmap.png                   # Property × cluster R² heatmap
    diffpool_prototype_graphs.png       # Representative cascade per cluster with DiffPool coloring
    augmentation_sensitivity_bars.png   # Per-cluster bar chart of sensitivity per augmentation
```

#### Failure Mode Audit

1. *Probe overfitting:* Small clusters (<50 cascades) may overfit linear probes. **Diagnostic:** Use leave-one-out CV for small clusters. Report confidence intervals.
2. *DiffPool degenerate assignments:* If pool_2 is collapsed (as observed in v1 diagnostics), assignment matrices will be near-uniform. **Diagnostic:** Check assignment entropy — if close to max entropy for all cascades, the pool level is uninformative. Skip that level in visualizations.
3. *Augmentation sensitivity floor effect:* If all augmentations produce nearly identical embeddings (alignment ~1.0, as in v1), sensitivity analysis yields no signal. **Prerequisite:** Retrain v2 must achieve alignment < 0.85 for this sub-step to be meaningful.

### Step 4.6 — Ablation Stage (`blueccgl/stages/ablation.py`)

**Stage:** `--stage ablation`
**Reads:** `{output_dir}/build_pyg_dataset/cascade_dataset.pt`
**Writes:** `{output_dir}/ablation/{variant}/` (mirrors train → extract → cluster → evaluate outputs), `{output_dir}/ablation/comparison_table.csv`

For each ablation variant (A1–A5):
1. Train with the variant's config override.
2. Extract embeddings.
3. Cluster.
4. Evaluate.
5. Collect metrics into a comparison table.

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

**Training v2 (2026-03-23, τ=0.1 → τ=0.3, retrained with stronger augmentations):**

After diagnostics revealed the v1 model was MARGINAL (see below), the following config changes were applied:

| Parameter | v1 | v2 | Rationale |
|---|---|---|---|
| `augmentation.follower_retain_ratio` | 0.7 | 0.5 | Stronger structural perturbation |
| `augmentation.temporal_prefix_ratio` | 0.6 | 0.4 | Forces learning beyond temporal shortcuts |
| `augmentation.bridge_remove_ratio` | 0.3 | 0.5 | Removes more non-follower entry nodes |
| `augmentation.feature_mask_ratio` | 0.0 | 0.25 | **NEW** — randomly zeros 25% of node feature dims per view. Blocks feature-statistics shortcut |
| `training.temperature` | 0.1 | 0.3 | Harder contrastive task — prevents pretext saturation |
| `training.variance_weight` | 0.0 | 0.1 | **NEW** — VICReg variance regularisation. Penalises dims with std < 1.0 to prevent dimensional collapse |
| `model.diffpool_aux_weight` | 0.1 | 0.5 | Keeps DiffPool assignment matrices non-degenerate |

**New code added for v2:**
- `blueccgl/augmentations/feature_masking.py` — random node feature dimension zeroing
- `blueccgl/models/losses.py` — added `vicreg_variance_loss()` (Bardes et al., 2022)
- `blueccgl/stages/train.py` — integrated feature masking into augmentor + variance loss into training loop + `variance` field in training log

**Target metrics for v2:** alignment 0.6–0.85, effective rank >25, uniformity < -3.0

**Embedding diagnostics (v1, 2026-03-23):**

| Metric | Value | Status |
|---|---|---|
| Median cosine sim | 0.365 | OK (<0.5) |
| Effective rank (SVD) | 14.9 / 64 | WARN — partial dimensional collapse |
| Uniformity | -1.884 | WARN — moderate (should be <-3.0) |
| Alignment | 0.997 | **CRITICAL** — augmentations too weak, model matches pairs trivially |
| Top-5 variance ratio | 0.907 | 91% of variance in 5 dims |
| Pool level 2 eff_dim | 3.65 / 128 | WARN — final DiffPool layer degenerate |
| Intra vs inter-event gap | <0.06 | No event discrimination |

**Verdict: MARGINAL → retrain required.** Root cause: augmentations don't destroy enough information; model solves pretext task via unchanged node feature statistics rather than learning structural cascade patterns.

### Sprint 5: Full Data Integration + Downstream Pipeline

| # | Task | Status | Notes |
|---|------|--------|-------|
| 24 | Download full Zenodo dataset | PARTIAL | `user_posts.tar.gz` (19 GB) + `followers.csv.gz` (469 MB) downloaded. Others not needed for current approach |
| 25 | Run `--stage nlp_filter` on full tar | DONE | 1,947,493 conflict posts, 85 min |
| 26 | Run `--stage cascade_construction` with full data | DONE | 39,983 cascades, ~3 min |
| 27 | Run `--stage build_pyg_dataset` | DONE | cascade_dataset.pt (159.8 MB, 39,983 cascades). 2026-03-18 |
| 28 | Run `--stage eda` | NOT STARTED | Stub — needs implementation. Unblocked |
| 29 | Run `--stage train` v1 | DONE | 50 epochs, τ=0.1. Diagnostics: MARGINAL (alignment=0.997, eff_rank=14.9). 2026-03-23 |
| 29b | **Retrain v2** | READY | Stronger augmentations, feature masking, τ=0.3, VICReg variance reg, aux weight 0.5. Config committed. Archive old run first: `mv outputs/train/train outputs/train/run_v1_tau01` |
| 30 | **Implement + run `--stage extract_embeddings`** | NOT STARTED | Stub — blocked on #29b. Forward pass all cascades through best.pt, save [39983, 64] embeddings to HDF5 |
| 31 | **Implement + run `--stage cluster`** | NOT STARTED | Stub — blocked on #30. HDBSCAN grid search over min_cluster_size={50, 100, 200} |
| 32 | **Implement + run `--stage evaluate`** | NOT STARTED | Stub — blocked on #31. Silhouette, DB, CH, DBCV + bootstrap CIs + UMAP plots |
| 33 | **Implement + run `--stage explain`** | NOT STARTED | New stage — blocked on #31. 4 sub-steps: probing classifiers, DiffPool assignment extraction, cluster statistical profiles, augmentation sensitivity analysis. Requires model changes: `explain()` method on BlueCCGL, `return_assignments` on DiffPool |
| 34 | **Implement + run `--stage ablation`** (A1–A5) | NOT STARTED | Stub — blocked on #32. Each variant: train → extract → cluster → evaluate |
| 35 | **Implement + run `--stage downstream`** | NOT STARTED | Stub — blocked on #30. Size + longevity prediction heads |
| 36 | **Implement + run `--stage stability`** | NOT STARTED | Stub — blocked on #31. 10× 80% subsample ARI |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation | Status |
|------|-----------|--------|------------|--------|
| **Python 3.14 incompatible with PyTorch/PyG** | High | Blocks all ML code | Create 3.11/3.12 venv as first step | RESOLVED — Python 3.14 works with CPU torch 2.10.0 + PyG 2.7.0 from wheel server |
| **Disk space for full data** | — | — | — | RESOLVED — CPU-only torch + pip cache purge freed space. Streaming tar (no full extraction) keeps footprint manageable. ~15 GB free |
| **Too few cascades with ≥10 reposts** | Medium | Weak clustering signal | Lower threshold to ≥5; report as limitation | RESOLVED — 39,983 cascades at ≥10 threshold. Ample for HDBSCAN with min_cluster_size=50–200 |
| **DiffPool OOM on large cascades** | Medium | Training crashes | Cap cascade size at 1000 for DiffPool; fallback to SAGPool | MITIGATED — max_nodes=150 in DiffPool. P95 cascade size is 69 nodes, so cap is nearly lossless. Training stable so far |
| **Representation collapse during training** | Medium | Useless embeddings | Monitor embedding variance; add projector MLP; tune temperature | DIAGNOSED — v1 shows partial dimensional collapse (eff_rank=14.9/64) and near-perfect alignment (0.997). Not full collapse but MARGINAL. v2 retrain addresses with feature masking, VICReg variance reg, τ=0.3, stronger augmentations |
| **Follower graph temporal confound** | Certain | Biased FOLLOWER labels for early cascades | Acknowledge as limitation; sensitivity analysis | OPEN |
| **High HDBSCAN noise fraction** | Medium | Weak archetype discovery | Try multiple `min_cluster_size` values; consider UMAP pre-reduction | OPEN |
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

## Key Decisions Still Open

1. **DiffPool vs. SAGPool:** DiffPool completed training successfully (max_nodes=150). No instability observed. Decision: keep DiffPool.
2. **Embedding quality validation:** v1 diagnosed as MARGINAL (alignment=0.997, eff_rank=14.9). v2 retrain config committed. **Must re-run diagnostics after v2 retrain** and verify alignment < 0.85, eff_rank > 25, uniformity < -3.0 before proceeding to clustering.
3. **HDBSCAN `min_cluster_size`:** Grid search over {50, 100, 200} planned. With 39,983 cascades and 69% baseline (27,572), event-specific cascades are smaller per event (930–3,338). May need min_cluster_size=50 to avoid overcollapsing.
4. **Ablation training time budget:** Each A1–A5 variant requires a full training run (~2.8 hrs on CPU). All 5 = ~14 hrs. Consider reducing epochs for ablations (e.g., 50 epochs) if wall-clock is too tight.
5. **Explain stage scope:** Core 4 sub-steps (probes, DiffPool assignments, cluster profiles, augmentation sensitivity) are planned. Optional extensions (GNNExplainer, PGExplainer, TCAV) depend on remaining time budget. TCAV has highest thesis novelty but requires concept set curation (~6h). Decision deferred until after clustering results are available.

---

## Known Bugs Fixed

1. **`extract_conflict_window.py` — `batch.append()` in wrong block:** Was inside `if files_done % 100 == 0:` (file-level progress) instead of per-line loop. Produced 593 KB parquet instead of 2.4 GB. Fixed 2025-03-17.
2. **`run_pipeline.py` — `--set key=false` treated as truthy string:** `"false"` failed int/float cast, stayed as string `"false"` (truthy). Added bool casting before numeric. Fixed 2025-03-17.
3. **`nlp_filter.py` — `_iter_tar` used `getmembers()` with `r:gz`:** After reading all 2.3M headers, each `extractfile()` forced gzip re-decompression from start (O(n²)). Switched to `r|gz` streaming + multiprocessing. 95 → 500 files/sec. Fixed 2025-03-17.
4. **`quick_look.py` — `langs_list or []` ValueError:** Numpy array `or` is ambiguous. Fixed with explicit None/length check.
5. **`cascade_construction.py` — `_load_follower_graph` wrong path:** Only checked `Data/raw/followers.csv.gz`. Added fallback to `Data/followers.csv.gz`. Fixed 2025-03-17.
6. **`follower_graph.py` — per-edge DuckDB queries O(n²):** Each `follows()` call issued a separate SQL query, making tree building take an estimated 26 hours. Fixed with `preload_for_users()`: one batch JOIN query loads all relevant edges into a Python `set` before tree building starts. Total runtime went from 26hr estimate to ~3 min. Fixed 2025-03-17.
