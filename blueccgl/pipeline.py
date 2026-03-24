"""Stage orchestration: dependency graph, resume logic, dry-run."""

from __future__ import annotations

from pathlib import Path


class PipelineDependencyError(RuntimeError):
    pass


# Ordered stage list (topological order).
ALL_STAGES: list[str] = [
    "nlp_filter",
    "validate_filter",
    "cascade_construction",
    "build_pyg_dataset",
    "eda",
    "train",
    "extract_embeddings",
    "cluster",
    "evaluate",
    "explain",
    "ablation",
    "downstream",
    "stability",
]

# Maps each stage to the output checkpoint file it produces.
STAGE_CHECKPOINTS: dict[str, str] = {
    "nlp_filter":           "nlp_filter/conflict_posts.parquet",
    "validate_filter":      "validate_filter/validation_report.json",
    "cascade_construction": "cascade_construction/cascades.parquet",
    "build_pyg_dataset":    "build_pyg_dataset/cascade_dataset.pt",
    "eda":                  "eda/eda_summary.json",
    "train":                "train/checkpoints/best.pt",
    "extract_embeddings":   "extract_embeddings/embeddings.h5",
    "cluster":              "cluster/cluster_assignments.parquet",
    "evaluate":             "evaluate/metrics.json",
    "explain":              "explain/explanation_report.json",
    "ablation":             "ablation/comparison_table.csv",
    "downstream":           "downstream/results.json",
    "stability":            "stability/stability_report.json",
}

# Upstream dependency: each stage requires its upstream checkpoint to exist.
STAGE_DEPS: dict[str, str | None] = {
    "nlp_filter":           None,
    "validate_filter":      "nlp_filter",
    "cascade_construction": "nlp_filter",
    "build_pyg_dataset":    "cascade_construction",
    "eda":                  "build_pyg_dataset",
    "train":                "build_pyg_dataset",
    "extract_embeddings":   "train",
    "cluster":              "extract_embeddings",
    "evaluate":             "cluster",
    "explain":              "evaluate",
    "ablation":             "build_pyg_dataset",
    "downstream":           "train",
    "stability":            "extract_embeddings",
}


def checkpoint_path(stage: str, output_dir: Path) -> Path:
    return output_dir / STAGE_CHECKPOINTS[stage]


def checkpoint_exists(stage: str, output_dir: Path) -> bool:
    return checkpoint_path(stage, output_dir).exists()


def resolve_stages(requested: list[str], output_dir: Path, resume: bool) -> list[str]:
    """
    Returns the ordered list of stages that will actually run.
    With --resume, skips stages whose checkpoint already exists.
    Raises PipelineDependencyError if an upstream dep is missing.
    """
    for stage in requested:
        if stage not in ALL_STAGES:
            raise ValueError(f"Unknown stage '{stage}'. Valid stages: {ALL_STAGES}")

    # Validate dependencies
    for stage in requested:
        dep = STAGE_DEPS[stage]
        if dep is not None and dep not in requested and not checkpoint_exists(dep, output_dir):
            raise PipelineDependencyError(
                f"Stage '{stage}' requires '{dep}' to have run first. "
                f"Expected checkpoint: {checkpoint_path(dep, output_dir)}"
            )

    # Sort requested stages into pipeline order
    ordered = [s for s in ALL_STAGES if s in requested]

    if resume:
        ordered = [s for s in ordered if not checkpoint_exists(s, output_dir)]

    return ordered


def print_plan(requested: list[str], output_dir: Path, resume: bool) -> None:
    """Print the execution plan for --dry-run."""
    print("\n[dry-run] Pipeline execution plan")
    print(f"  Output dir : {output_dir}")
    print(f"  Resume     : {resume}")
    print()

    try:
        to_run = resolve_stages(requested, output_dir, resume)
    except (PipelineDependencyError, ValueError) as e:
        print(f"  ERROR: {e}")
        return

    for stage in ALL_STAGES:
        if stage not in requested:
            continue
        cp = checkpoint_path(stage, output_dir)
        exists = cp.exists()
        if stage in to_run:
            status = "RUN"
        else:
            status = "SKIP (checkpoint exists)" if exists else "SKIP"
        dep = STAGE_DEPS[stage]
        dep_str = f"  [dep: {dep}]" if dep else ""
        print(f"  {'→' if stage in to_run else '·'} {stage:<25} {status}{dep_str}")
    print()
