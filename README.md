# BlueCCGL: Affective-Structural Archetypes of Crisis Information Cascades on Bluesky

Unsupervised contrastive graph learning framework for discovering crisis information cascade archetypes on Bluesky. Applied to the Israel-Palestine conflict (October 2023 – March 2024) using the Failla & Rossetti (2024) Zenodo dataset.

---

## Repository Structure

```
OnlineCommunicationBEP/
├── Data/                        # Raw and sample data (see below for download)
│   ├── user_posts/              # 100 sample JSONL files for dev/testing
│   ├── user_posts.tar.gz        # Full dataset (19.5 GB) — download separately
│   ├── followers.csv.gz         # Directed follower graph (469 MB) — download separately
│   ├── followers.duckdb         # Auto-built index on first cascade construction run
│   └── conflict_window_posts.parquet  # Pre-extracted lean index (2.4 GB, 172M rows)
└── Code/                        # All pipeline code
    ├── blueccgl/                # Main Python package
    │   ├── stages/              # One module per pipeline stage
    │   ├── models/              # R-GCN encoder, DiffPool, losses
    │   ├── augmentations/       # Feed-aware augmentation strategies
    │   ├── data/                # Keyword filter, cascade builder, PyG dataset
    │   ├── evaluation/          # Clustering metrics, visualisation
    │   └── utils/               # Shared utilities
    ├── scripts/                 # Ad-hoc analysis and diagnostic scripts
    ├── tests/                   # Unit and integration tests
    ├── run_pipeline.py          # Unified CLI entry point
    ├── config.yaml              # All hyperparameters and paths
    ├── requirements.txt         # Pinned dependencies
    └── install.sh               # Environment setup script
```

---

## Data Download

The dataset is from Failla & Rossetti (2024), hosted on Zenodo:

**DOI:** [10.5281/zenodo.14669616](https://doi.org/10.5281/zenodo.14669616)

Only two files are required:

| File | Size | Purpose |
|------|------|---------|
| `user_posts.tar.gz` | 19.5 GB | Post text, timestamps, RoBERTa sentiment labels, repost metadata |
| `followers.csv.gz` | 469 MB | Directed follower graph (~144M edges) for cascade edge-type labelling |

Place both files in the `Data/` directory at the repo root.

The remaining dataset files (`interactions.csv.gz`, `graphs.tar.gz`, `feed_posts.tar.gz`, `feed_bookmarks.csv`) are not used — cascade construction uses `repost_from`/`reposted_author` fields directly from `user_posts.tar.gz`.

**For development without the full dataset:** the 100 sample JSONL files in `Data/user_posts/` are sufficient to run and test all pipeline stages. Set `nlp_filter.use_sample_dir: true` in `config.yaml` (the default).

---

## Installation

```bash
cd Code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.14, CPU-only PyTorch 2.10.0 + PyG 2.7.0. GPU training is supported — set `--device cuda` when running the pipeline.

---

## Running the Pipeline

All commands are run from the `Code/` directory.

### Full pipeline (in order)

```bash
python run_pipeline.py --config config.yaml --stage nlp_filter
python run_pipeline.py --config config.yaml --stage cascade_construction
python run_pipeline.py --config config.yaml --stage build_pyg_dataset
python run_pipeline.py --config config.yaml --stage train --device cuda
python run_pipeline.py --config config.yaml --stage extract_embeddings
python run_pipeline.py --config config.yaml --stage cluster
python run_pipeline.py --config config.yaml --stage evaluate
python run_pipeline.py --config config.yaml --stage explain
python run_pipeline.py --config config.yaml --stage hierarchical
python run_pipeline.py --config config.yaml --stage validate_hierarchical
```

### Run multiple stages in one command

```bash
python run_pipeline.py --config config.yaml \
  --stage extract_embeddings \
  --stage cluster \
  --stage evaluate
```

### Resume interrupted runs

```bash
python run_pipeline.py --config config.yaml --stage train --resume
```

Checks for existing stage checkpoints and skips completed stages.

### Override config values at the command line

```bash
# Train without sentiment features (no-sentiment ablation)
python run_pipeline.py --config config.yaml --stage train \
  --set model.use_sentiment=false \
  --output-dir ./outputs_no_sentiment

# Use a different output directory
python run_pipeline.py --config config.yaml --stage train \
  --output-dir ./my_run
```

### Dry run (print execution plan without running)

```bash
python run_pipeline.py --config config.yaml --dry-run --stage train
```

---

## Pipeline Stages

| Stage | Input | Output | Description |
|-------|-------|--------|-------------|
| `nlp_filter` | `user_posts.tar.gz` or `Data/user_posts/` | `conflict_posts.parquet` | Keyword filter across 5 tiers (English, hashtags, Arabic, Hebrew, named entities). Extracts conflict-related posts within the Oct 2023–Mar 2024 window. |
| `cascade_construction` | `conflict_posts.parquet` + `followers.csv.gz` | `cascades.parquet`, `cascade_meta.parquet` | Builds directed repost trees. Labels each edge as FOLLOWER (reposter follows inferred parent) or NON_FOLLOWER. Applies size filter (≥10 nodes). |
| `build_pyg_dataset` | `cascades.parquet` | `cascade_dataset.pt` | Converts each cascade to a `torch_geometric.data.Data` object with 6-dimensional node features: temporal delta, sentiment label, sentiment score, depth, in-degree, out-degree. |
| `train` | `cascade_dataset.pt` | `checkpoints/`, `training_log.json` | Contrastive self-supervised training. R-GCN encoder + feed-aware augmentations + InfoNCE + VICReg loss. Produces 64-dimensional cascade embeddings. |
| `extract_embeddings` | `best.pt` + `cascade_dataset.pt` | `embeddings.h5`, `cascade_index.parquet` | Passes all cascades through the frozen encoder. Runs embedding diagnostics (effective rank, uniformity, alignment). |
| `cluster` | `embeddings.h5` | `cluster_assignments.parquet`, `umap_embeddings_cluster.h5` | UMAP 64D→10D reduction then HDBSCAN grid search over `min_cluster_size`. Selects best config by Silhouette score. |
| `evaluate` | `embeddings.h5` + `cluster_assignments.parquet` | `metrics.json`, `umap_figure.png` | Bootstrap clustering metrics (Silhouette, DBCV, Davies-Bouldin, Calinski-Harabasz) computed in 10D UMAP space. 2-panel UMAP figure. |
| `explain` | Embeddings + cluster assignments + model | `probe_results.json`, `cluster_profiles.json`, figures | Linear probing, cluster statistical profiles, augmentation sensitivity analysis. |
| `hierarchical` | `umap_embeddings_cluster.h5` | `hierarchical_results.json`, dendrogram figures | Ward agglomerative clustering on HDBSCAN cluster centroids. Largest-gap cut selects number of macro-archetypes. |
| `validate_hierarchical` | Hierarchical results + embeddings | `validation_report.json`, stability figures | Cophenetic correlation, metric sweep (k=2..6), bootstrap ARI (n=100), UMAP seed ARI (10 seeds). |
| `ablation` | `cascade_dataset.pt` | `ablation_results.json` | 2³ factorial design: encoder (R-GCN vs GCN) × augmentation (feed-aware vs random) × sentiment (on vs off). Runs 7 variants. |

---

## Key Configuration Options

All options live in `Code/config.yaml`. Key ones to know:

```yaml
nlp_filter:
  use_sample_dir: true      # true = use Data/user_posts/ sample; false = stream full tar

model:
  encoder: rgcn             # rgcn | gcn
  use_sentiment: true       # false for no-sentiment ablation

augmentation:
  strategy: feed_aware      # feed_aware | random

clustering:
  min_cluster_size_grid: [100, 200, 300]  # HDBSCAN grid search values
  umap_cluster_n_components: 10           # dimensionality before HDBSCAN
```

---

## Pre-extraction Step (full data only)

Before running `cascade_construction` on the full dataset, extract the lean post index from the tar archive. This is a one-time step that takes ~85 minutes and produces `Data/conflict_window_posts.parquet` (2.4 GB), used for fast reposter lookup during cascade building:

```bash
cd Code
python scripts/extract_conflict_window.py
```

The follower DuckDB index (`Data/followers.duckdb`) is built automatically on the first `cascade_construction` run from `followers.csv.gz`.

---

## Outputs

Each stage writes to `Code/outputs/{stage}/` by default. Override with `--output-dir`. Key outputs:

| File | Stage | Description |
|------|-------|-------------|
| `outputs/nlp_filter/conflict_posts.parquet` | nlp_filter | 1.95M conflict-related posts |
| `outputs/cascade_construction/cascades.parquet` | cascade_construction | Edge list for all 39,983 cascades |
| `outputs/build_pyg_dataset/cascade_dataset.pt` | build_pyg_dataset | PyG dataset (159.8 MB) |
| `outputs/train/checkpoints/best.pt` | train | Best model checkpoint |
| `outputs/extract_embeddings/embeddings.h5` | extract_embeddings | 64D embeddings [N_cascades, 64] |
| `outputs/cluster/cluster_assignments.parquet` | cluster | HDBSCAN cluster labels |
| `outputs/evaluate/metrics.json` | evaluate | Bootstrapped clustering metrics |
| `outputs/hierarchical/hierarchical_results.json` | hierarchical | Macro-archetype assignments |

---

## Tests

```bash
cd Code
pytest tests/
```

---

## Citation

Failla, A. & Rossetti, G. (2024). "I'm in the Bluesky Tonight: Insights from a Year's Worth of Social Data". *PLOS ONE* 19.11. DOI: [10.1371/journal.pone.0310330](https://doi.org/10.1371/journal.pone.0310330)
