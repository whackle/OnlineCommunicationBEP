# BlueCCGL — Full Implementation Plan

## Current State Assessment (Starting Point)

| Asset | Status |
|-------|--------|
| **Sample data** | 100 JSONL files from `user_posts.tar.gz` (~15 MB, ~25K posts across 100 users) in `Data/user_posts/` |
| **Full data** | Not yet downloaded. Need: `user_posts.tar.gz` (19.5 GB), `interactions.csv.gz` (1 GB), `graphs.tar.gz` (891 MB), `followers.csv.gz` (491 MB), `feed_posts.tar.gz` (16.4 MB), `feed_bookmarks.csv` (553 KB) |
| **EDA script** | `scripts/eda_user_posts.py` — standalone EDA with keyword tiers, event windows, conflict matching. **Not yet integrated into pipeline.** |
| **Environment** | Python 3.14 venv (empty — no packages installed yet) |
| **Docs** | Project plan (`Docs/Palestine_Bluesky_Project_Plan.md`), system prompt (`prompts/system_prompt.md`), LaTeX proposal draft |
| **Pipeline** | Does not exist yet. `run_pipeline.py` and `config.yaml` need to be created from scratch |

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
│   │   └── losses.py            # InfoNCE + DiffPool auxiliary losses
│   │
│   ├── augmentations/           # Feed-aware augmentation strategies
│   │   ├── __init__.py
│   │   ├── follower_subgraph.py # Strategy A
│   │   ├── temporal_prefix.py   # Strategy B
│   │   ├── bridge_removal.py    # Strategy C
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

### Step 4.5 — Ablation Stage (`blueccgl/stages/ablation.py`)

**Stage:** `--stage ablation`
**Reads:** `{output_dir}/build_pyg_dataset/cascade_dataset.pt`
**Writes:** `{output_dir}/ablation/{variant}/` (mirrors train → extract → cluster → evaluate outputs), `{output_dir}/ablation/comparison_table.csv`

For each ablation variant (A1–A5):
1. Train with the variant's config override.
2. Extract embeddings.
3. Cluster.
4. Evaluate.
5. Collect metrics into a comparison table.

### Step 4.6 — Downstream Stage (`blueccgl/stages/downstream.py`)

**Stage:** `--stage downstream`
**Reads:** `{output_dir}/train/checkpoints/best.pt`, `{output_dir}/build_pyg_dataset/cascade_dataset.pt`
**Writes:** `{output_dir}/downstream/results.json`

Fine-tune the pre-trained encoder on:
1. **Cascade size prediction:** Regression head on top of frozen/fine-tuned encoder. Target: log(cascade_size). Metric: RMSE, R².
2. **Cascade longevity prediction:** Binary classification. Target: cascade active > median duration. Metric: AUROC, F1.

Baselines:
- Random init (same architecture, no pre-training).
- Handcrafted features: [Wiener index, depth, breadth, branching factor, follower-edge ratio] → MLP.

### Step 4.7 — Stability Stage (`blueccgl/stages/stability.py`)

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

### Sprint 1: Scaffolding (do first)

| # | Task | Deliverable |
|---|------|-------------|
| 1 | Resolve Python version (3.14 may lack PyTorch wheels) | Working venv with PyTorch + PyG |
| 2 | Create `blueccgl/` package structure | Empty `__init__.py` files, directory tree |
| 3 | Create `config.yaml` with all defaults | Config file |
| 4 | Create `run_pipeline.py` with stage orchestration | CLI entry point, `--dry-run` works |
| 5 | Create `blueccgl/config.py` (YAML + CLI merge) | Config loader |
| 6 | Create `blueccgl/pipeline.py` (dependency graph, resume) | Pipeline orchestrator |
| 7 | Create `blueccgl/seed.py` | RNG seeding |
| 8 | Update `.gitignore` | Ignore large files |

### Sprint 2: NLP Filter on Sample (validates pipeline + keywords)

| # | Task | Deliverable |
|---|------|-------------|
| 9 | Create `blueccgl/data/keywords.py` | Keyword module |
| 10 | Create `blueccgl/data/events.py` | Event window module |
| 11 | Create `blueccgl/stages/nlp_filter.py` | NLP filter stage |
| 12 | Run `--stage nlp_filter` on sample data | `outputs/nlp_filter/conflict_posts.parquet` |
| 13 | Inspect results: conflict post count, event distribution, keyword hits | Sanity check |
| 14 | Create `blueccgl/stages/validate_filter.py` | Validation sample generator |
| 15 | Manual precision check on ~50 matched posts | Keyword list refinement |

### Sprint 3: Cascade Construction on Sample (validates tree-building logic)

| # | Task | Deliverable |
|---|------|-------------|
| 16 | Create `blueccgl/data/cascade_builder.py` | Parent assignment + edge-type logic |
| 17 | Create `blueccgl/stages/cascade_construction.py` | Cascade construction stage |
| 18 | **Note:** Without `followers.csv.gz` and `interactions.csv.gz`, cascade construction can only run in a *degraded mode* using `repost_from` / `reposted_author` fields within `user_posts` + all edges marked UNKNOWN | Partial cascade test |
| 19 | Verify: cascade tree validity, size distribution from sample | Sanity check |

### Sprint 4: PyG Dataset + Model Skeleton

| # | Task | Deliverable |
|---|------|-------------|
| 20 | Create `blueccgl/stages/build_pyg_dataset.py` | PyG dataset builder |
| 21 | Create R-GCN encoder, DiffPool, losses, augmentations | Model code |
| 22 | Unit tests: forward pass, gradient flow, augmentation validity | Green tests |
| 23 | Test training loop on sample cascades (expect poor results — too little data) | Training code verified |

### Sprint 5: Full Data Integration (once downloaded)

| # | Task | Deliverable |
|---|------|-------------|
| 24 | Download full Zenodo dataset | `Data/raw/` populated |
| 25 | Run `--stage nlp_filter` on full `user_posts.tar.gz` | Full conflict post index |
| 26 | Run `--stage cascade_construction` with all data sources | Full cascade dataset |
| 27 | Run `--stage build_pyg_dataset` | Full PyG dataset |
| 28 | Run `--stage eda` | EDA plots + statistics |
| 29 | Run `--stage train` | Trained BlueCCGL model |
| 30 | Run `--stage extract_embeddings → cluster → evaluate` | Archetype discovery |
| 31 | Run `--stage ablation` | Ablation comparison table |
| 32 | Run `--stage downstream` | Transfer evaluation |
| 33 | Run `--stage stability` | Stability report |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Python 3.14 incompatible with PyTorch/PyG** | High | Blocks all ML code | Create 3.11/3.12 venv as first step |
| **Too few cascades with ≥10 reposts** | Medium | Weak clustering signal | Lower threshold to ≥5; report as limitation |
| **DiffPool OOM on large cascades** | Medium | Training crashes | Cap cascade size at 1000 for DiffPool; fallback to SAGPool |
| **Representation collapse during training** | Medium | Useless embeddings | Monitor embedding variance; add projector MLP; tune temperature |
| **Follower graph temporal confound** | Certain | Biased FOLLOWER labels for early cascades | Acknowledge as limitation; sensitivity analysis |
| **High HDBSCAN noise fraction** | Medium | Weak archetype discovery | Try multiple `min_cluster_size` values; consider UMAP pre-reduction |
| **Cascade reconstruction heuristic too lossy** | Medium | Inaccurate cascade trees | Validate via known properties (e.g., branching factor distributions from literature) |
| **Keyword false positives (e.g., "Israel" matching non-conflict content)** | Medium | Noisy cascade roots | Manual validation; context-aware filtering for ambiguous keywords |
| **Sample data insufficient for cascade testing** | High | Can't test cascade construction without `interactions.csv.gz` / `followers.csv.gz` | Use degraded mode with `repost_from` field only; test full pipeline once data arrives |
| **Training on single GPU too slow for 100 epochs** | Low-Medium | Schedule slippage | Profile first; reduce epochs / cascade count if needed; use mixed precision (fp16) |

---

## Key Decisions to Make Early

1. **Python version:** 3.14 or downgrade? → Check PyTorch wheel availability immediately.
2. **DuckDB vs. in-memory for follower graph:** Depends on available RAM. Default to DuckDB.
3. **DiffPool vs. SAGPool:** Try DiffPool first; switch if OOM. Wire the pooling module as a config toggle.
4. **Wiener index computation:** Implement the O(N) tree-based algorithm, not the naive O(N²).
5. **Cascade size threshold:** Can't decide until full data is processed. Build the code to accept a config parameter.
6. **Hard negatives:** Implement as optional (config flag `augmentation.hard_negatives: true`). Can be disabled for initial experiments.
