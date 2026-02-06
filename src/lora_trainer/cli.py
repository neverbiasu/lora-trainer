"""CLI entry point - flat argparse, no subcommands"""
import argparse
import sys
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create command line parser (flat, no subcommands)"""
    parser = argparse.ArgumentParser(
        prog="lora-trainer",
        description="LoRA Trainer - Minimalist but not simplistic",
    )

    # -- Data --
    data = parser.add_argument_group("Data")
    data.add_argument("--dataset", type=Path, help="Dataset directory")
    data.add_argument("--resolution", type=int, default=512, help="Training resolution")
    data.add_argument("--cache-latents", action="store_true", help="Enable latent caching")
    data.add_argument("--no-bucketing", action="store_true", help="Disable aspect-ratio bucketing")

    # -- Model --
    model = parser.add_argument_group("Model")
    model.add_argument("--base-model", type=str, help="Base model identifier (e.g. sd15)")
    model.add_argument("--model-path", type=Path, help="Custom model path (overrides --base-model)")
    model.add_argument("--rank", type=int, default=32, help="LoRA rank")
    model.add_argument("--alpha", type=float, default=32.0, help="LoRA alpha")

    # -- Training --
    training = parser.add_argument_group("Training")
    training.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    training.add_argument("--lr-scheduler", type=str, default="cosine", help="LR scheduler: cosine / constant")
    training.add_argument("--batch-size", type=int, default=4, help="Batch size")
    training.add_argument("--gradient-accumulation", type=int, default=1, help="Gradient accumulation steps")
    training.add_argument("--max-steps", type=int, help="Max training steps (default: auto)")
    training.add_argument("--preset", choices=["quick", "balanced", "quality"], help="Training preset")
    training.add_argument("--seed", type=int, default=42, help="Random seed")

    # -- Optimization --
    optim = parser.add_argument_group("Optimization")
    optim.add_argument("--mixed-precision", choices=["fp16", "bf16", "fp32"], default="fp16", help="Mixed precision mode")
    optim.add_argument("--enable-xformers", action="store_true", help="Enable xformers memory optimization")
    optim.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")

    # -- Output --
    output = parser.add_argument_group("Output")
    output.add_argument("--output-dir", type=Path, default=Path("./output"), help="Output directory")
    output.add_argument("--save-every-n-steps", type=int, default=500, help="Checkpoint save frequency")
    output.add_argument("--sample-every-n-steps", type=int, default=250, help="Sample generation frequency")
    output.add_argument("--sample-prompts", type=Path, help="Sample prompt file")

    # -- Mode --
    mode = parser.add_argument_group("Mode")
    mode.add_argument("--config", type=Path, help="YAML configuration file")
    mode.add_argument("--resume", type=Path, help="Resume from run directory or checkpoint")
    mode.add_argument("--validate-only", action="store_true", help="Validate dataset and exit")
    mode.add_argument("--export-only", action="store_true", help="Export model from run and exit (requires --resume)")
    mode.add_argument("--dry-run", action="store_true", help="Preview resolved config and exit")
    mode.add_argument("--verbose", action="store_true", help="Verbose logging")

    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Validate argument combinations"""
    has_source = args.config or args.dataset or args.resume
    if not has_source:
        print("❌ [E041] Must provide --dataset, --config, or --resume", file=sys.stderr)
        sys.exit(1)

    if args.export_only and not args.resume:
        print("❌ [E041] --export-only requires --resume", file=sys.stderr)
        sys.exit(1)


def main():
    """CLI main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    _validate_args(args)

    # Dispatch based on mode flags
    if args.validate_only:
        # TODO: run validation pipeline
        raise NotImplementedError("validate-only mode")
    elif args.export_only:
        # TODO: load run, export model
        raise NotImplementedError("export-only mode")
    elif args.dry_run:
        # TODO: resolve config, print preview
        raise NotImplementedError("dry-run mode")
    elif args.resume:
        # TODO: load checkpoint, continue training
        raise NotImplementedError("resume mode")
    else:
        # Default: validate → train → export
        raise NotImplementedError("training pipeline")


if __name__ == "__main__":
    main()
