#!/usr/bin/env python
"""
BlueCCGL unified pipeline entry point.

Example usage:
    python run_pipeline.py --config config.yaml --output-dir ./outputs \\
        --stage nlp_filter --stage cascade_construction --dry-run
"""

import sys
from pathlib import Path

import click

from blueccgl.config import load_config, apply_overrides
from blueccgl.pipeline import (
    ALL_STAGES,
    resolve_stages,
    print_plan,
    PipelineDependencyError,
)
from blueccgl.seed import seed_everything


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", "config_path", default="config.yaml", show_default=True,
              type=click.Path(exists=True), help="Path to YAML config file.")
@click.option("--output-dir", default="./outputs", show_default=True,
              type=click.Path(), help="Root directory for all stage outputs.")
@click.option("--stage", "stages", multiple=True,
              type=click.Choice(ALL_STAGES, case_sensitive=False),
              help="Stage(s) to execute (repeatable). Order is enforced by dependency graph.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print execution plan without running anything.")
@click.option("--resume", is_flag=True, default=False,
              help="Skip stages whose output checkpoint already exists.")
@click.option("--seed", default=42, show_default=True, type=int,
              help="Global random seed.")
@click.option("--ablation", default=None,
              type=click.Choice(
                  ["A1_no_edge_type", "A2_generic_aug", "A3_flat_pool",
                   "A4_no_sentiment", "A5_random_init"],
                  case_sensitive=False,
              ),
              help="Ablation variant for the train stage.")
@click.option("--device", default=None, type=click.Choice(["cuda", "cpu"]),
              help="Compute device. Defaults to cuda if available, else cpu.")
@click.option("--set", "overrides", multiple=True, metavar="KEY=VALUE",
              help="Override config values, e.g. --set training.lr=5e-4")
def main(config_path, output_dir, stages, dry_run, resume, seed, ablation, device, overrides):
    output_dir = Path(output_dir)

    # Load + patch config
    cfg = load_config(config_path)

    if overrides:
        parsed = {}
        for kv in overrides:
            if "=" not in kv:
                raise click.BadParameter(f"Override must be KEY=VALUE, got: {kv}")
            k, v = kv.split("=", 1)
            # Attempt type cast: bool → int → float → str
            if v.lower() == "true":
                v = True
            elif v.lower() == "false":
                v = False
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            parsed[k] = v
        cfg = apply_overrides(cfg, parsed)

    # Ablation config patches
    if ablation:
        ablation_overrides = {
            "A1_no_edge_type": {"model.encoder": "gcn"},
            "A2_generic_aug":  {"augmentation.strategy": "random"},
            "A3_flat_pool":    {"model.pool": "mean"},
            "A4_no_sentiment": {"model.use_sentiment": False},
            "A5_random_init":  {},  # handled inside train stage
        }
        cfg = apply_overrides(cfg, ablation_overrides[ablation])

    # Device resolution
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if not stages:
        click.echo("No --stage specified. Use --help to see available stages.")
        sys.exit(1)

    stages = list(stages)

    if dry_run:
        print_plan(stages, output_dir, resume)
        return

    # Seed before anything stochastic
    seed_everything(seed)
    click.echo(f"[pipeline] seed={seed}  device={device}  output_dir={output_dir}")

    # Resolve execution order
    try:
        to_run = resolve_stages(stages, output_dir, resume)
    except PipelineDependencyError as e:
        click.echo(f"[pipeline] ERROR: {e}", err=True)
        sys.exit(1)

    if not to_run:
        click.echo("[pipeline] Nothing to run (all checkpoints exist and --resume is set).")
        return

    # Create output subdirectories
    output_dir.mkdir(parents=True, exist_ok=True)

    # Execute stages in order
    for stage in to_run:
        click.echo(f"\n[pipeline] ── Starting stage: {stage} ──")
        _run_stage(stage, cfg, output_dir, seed, device, ablation)
        click.echo(f"[pipeline] ── Completed stage: {stage} ──")


def _run_stage(stage: str, cfg: dict, output_dir: Path, seed: int, device: str, ablation: str | None) -> None:
    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    if stage == "nlp_filter":
        from blueccgl.stages.nlp_filter import run
        run(cfg, output_dir)

    elif stage == "validate_filter":
        from blueccgl.stages.validate_filter import run
        run(cfg, output_dir)

    elif stage == "cascade_construction":
        from blueccgl.stages.cascade_construction import run
        run(cfg, output_dir)

    elif stage == "build_pyg_dataset":
        from blueccgl.stages.build_pyg_dataset import run
        run(cfg, output_dir, device)

    elif stage == "eda":
        from blueccgl.stages.eda import run
        run(cfg, output_dir)

    elif stage == "train":
        from blueccgl.stages.train import run
        run(cfg, output_dir, device, seed, ablation)

    elif stage == "extract_embeddings":
        from blueccgl.stages.extract_embeddings import run
        run(cfg, output_dir, device)

    elif stage == "cluster":
        from blueccgl.stages.cluster import run
        run(cfg, output_dir)

    elif stage == "evaluate":
        from blueccgl.stages.evaluate import run
        run(cfg, output_dir)

    elif stage == "ablation":
        from blueccgl.stages.ablation import run
        run(cfg, output_dir, device, seed)

    elif stage == "downstream":
        from blueccgl.stages.downstream import run
        run(cfg, output_dir, device, seed)

    elif stage == "stability":
        from blueccgl.stages.stability import run
        run(cfg, output_dir, seed)

    else:
        raise ValueError(f"No implementation for stage: {stage}")


if __name__ == "__main__":
    main()
