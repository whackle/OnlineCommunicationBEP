#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# run_ablations.sh — 2³ factorial ablation grid for BlueCCGL
#
# Axes:
#   encoder       : R-GCN (default) vs GCN (A1)
#   augmentation  : feed-aware (default) vs random (A2)
#   sentiment     : enabled (default) vs disabled (A4)
#
# The baseline (R-GCN + feed-aware + sentiment) is already complete in
# Outputs_17_04, so only the remaining 7 variants are trained here.
#
# Each variant re-uses the pre-built PyG dataset via a symlink so that
# only train → extract_embeddings → cluster → evaluate need to run.
#
# Usage:
#   bash scripts/run_ablations.sh [--dry-run]
#
# Options:
#   --dry-run   Print commands without executing them.
#
# Outputs are written to Outputs_ablation/<variant_name>/
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

BASELINE="Outputs_17_04"
OUT_ROOT="Outputs_ablation"
CONFIG="config.yaml"
DRY_RUN=0

for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done

# ── helpers ──────────────────────────────────────────────────────────────────

log() { echo "[run_ablations] $*"; }

maybe_run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  DRY-RUN: $*"
    else
        "$@"
    fi
}

# Prepare an output directory: create it and symlink the shared PyG dataset.
# The pipeline's --resume flag will skip build_pyg_dataset if the checkpoint
# (cascade_dataset.pt) already exists.
prepare_dir() {
    local dir="$1"
    local dataset_src
    dataset_src="$(pwd)/$BASELINE/build_pyg_dataset/cascade_dataset.pt"

    if [[ ! -f "$dataset_src" ]]; then
        log "ERROR: baseline dataset not found at $dataset_src"
        exit 1
    fi

    if [[ $DRY_RUN -eq 0 ]]; then
        mkdir -p "$dir/build_pyg_dataset"
        # Only create the symlink if it doesn't exist yet
        if [[ ! -e "$dir/build_pyg_dataset/cascade_dataset.pt" ]]; then
            ln -s "$dataset_src" "$dir/build_pyg_dataset/cascade_dataset.pt"
        fi
    else
        echo "  DRY-RUN: mkdir -p $dir/build_pyg_dataset"
        echo "  DRY-RUN: ln -s $dataset_src $dir/build_pyg_dataset/cascade_dataset.pt"
    fi
}

# Run one variant.
#   $1 = short name (used for output dir suffix)
#   $2 = human-readable label (for logging)
#   rest = extra args forwarded to run_pipeline.py
run_variant() {
    local name="$1"
    local label="$2"
    shift 2
    local dir="$OUT_ROOT/$name"

    log "──────────────────────────────────────────────"
    log "Variant : $name ($label)"
    log "Out dir : $dir"
    log "Extra   : $*"
    log "──────────────────────────────────────────────"

    prepare_dir "$dir"

    maybe_run python run_pipeline.py \
        --config "$CONFIG" \
        --output-dir "$dir" \
        --stage build_pyg_dataset \
        --stage train \
        --stage extract_embeddings \
        --stage cluster \
        --stage evaluate \
        --resume \
        "$@"

    log "Done: $name"
    echo ""
}

# ── guard: baseline must exist ────────────────────────────────────────────────

if [[ ! -f "$BASELINE/evaluate/metrics.json" ]]; then
    log "ERROR: baseline ($BASELINE) has no evaluate/metrics.json."
    log "       Run the full pipeline on the baseline first."
    exit 1
fi

if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY-RUN MODE — no commands will execute"
    echo ""
fi

# ── 2³ grid (7 variants; baseline = Outputs_17_04) ───────────────────────────
#
#  name                encoder    augmentation   sentiment
#  ──────────────────  ─────────  ─────────────  ─────────
#  rgcn_feed_sent      R-GCN      feed-aware     ON      ← BASELINE (Outputs_17_04)
#  gcn_feed_sent       GCN        feed-aware     ON
#  rgcn_rand_sent      R-GCN      random         ON
#  rgcn_feed_nosent    R-GCN      feed-aware     OFF
#  gcn_rand_sent       GCN        random         ON
#  gcn_feed_nosent     GCN        feed-aware     OFF
#  rgcn_rand_nosent    R-GCN      random         OFF
#  gcn_rand_nosent     GCN        random         OFF    ← weakest baseline

run_variant "gcn_feed_sent" \
    "GCN encoder, feed-aware aug, sentiment ON" \
    --ablation A1_no_edge_type

run_variant "rgcn_rand_sent" \
    "R-GCN encoder, random aug, sentiment ON" \
    --ablation A2_generic_aug

run_variant "rgcn_feed_nosent" \
    "R-GCN encoder, feed-aware aug, sentiment OFF" \
    --ablation A4_no_sentiment

run_variant "gcn_rand_sent" \
    "GCN encoder, random aug, sentiment ON" \
    --set model.encoder=gcn \
    --set augmentation.strategy=random

run_variant "gcn_feed_nosent" \
    "GCN encoder, feed-aware aug, sentiment OFF" \
    --set model.encoder=gcn \
    --set model.use_sentiment=false

run_variant "rgcn_rand_nosent" \
    "R-GCN encoder, random aug, sentiment OFF" \
    --set augmentation.strategy=random \
    --set model.use_sentiment=false

run_variant "gcn_rand_nosent" \
    "GCN encoder, random aug, sentiment OFF (weakest)" \
    --set model.encoder=gcn \
    --set augmentation.strategy=random \
    --set model.use_sentiment=false

# ── done ──────────────────────────────────────────────────────────────────────

log "All 7 ablation variants complete."
log ""
log "Next steps:"
log "  python scripts/compare_ablations.py"
log "  python scripts/plot_ablation_interactions.py"
