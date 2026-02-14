"""CLI entry point - flat argparse, no subcommands."""
import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.lora_trainer.config_manager import ConfigManager, deep_merge
from src.lora_trainer.errors import LoRATrainerError
from src.lora_trainer.presets import get_preset
from src.lora_trainer.run_manager import RunManager
from src.lora_trainer.trainer import Trainer


logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create command line parser (flat, no subcommands)"""
    parser = argparse.ArgumentParser(
        prog="lora-trainer",
        description="LoRA Trainer - Minimalist but not simplistic",
    )

    data = parser.add_argument_group("Data")
    data.add_argument("--dataset", type=Path, help="Dataset directory")
    data.add_argument("--resolution", type=int, default=512, help="Training resolution")
    data.add_argument("--cache-latents", action="store_true", help="Enable latent caching")
    data.add_argument("--no-bucketing", action="store_true", help="Disable aspect-ratio bucketing")

    model = parser.add_argument_group("Model")
    model.add_argument("--base-model", type=str, help="Base model identifier (e.g. sd15)")
    model.add_argument("--model-path", type=Path, help="Custom model path (overrides --base-model)")
    model.add_argument("--rank", type=int, default=32, help="LoRA rank")
    model.add_argument("--alpha", type=float, default=32.0, help="LoRA alpha")

    training = parser.add_argument_group("Training")
    training.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    training.add_argument("--lr-scheduler", type=str, default="cosine", help="LR scheduler: cosine / constant")
    training.add_argument("--batch-size", type=int, default=4, help="Batch size")
    training.add_argument("--gradient-accumulation", type=int, default=1, help="Gradient accumulation steps")
    training.add_argument("--max-steps", type=int, help="Max training steps (default: auto)")
    training.add_argument("--preset", choices=["quick", "balanced", "quality"], help="Training preset")
    training.add_argument("--seed", type=int, default=42, help="Random seed")

    optim = parser.add_argument_group("Optimization")
    optim.add_argument("--mixed-precision", choices=["fp16", "bf16", "fp32"], default="fp16", help="Mixed precision mode")
    optim.add_argument("--enable-xformers", action="store_true", help="Enable xformers memory optimization")
    optim.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")

    output = parser.add_argument_group("Output")
    output.add_argument("--output-dir", type=Path, default=Path("./output"), help="Output directory")
    output.add_argument("--run-dir", type=Path, help="Base directory for run artifacts")
    output.add_argument("--save-every-n-steps", type=int, default=500, help="Checkpoint save frequency")
    output.add_argument("--sample-every-n-steps", type=int, default=250, help="Sample generation frequency")
    output.add_argument("--sample-prompts", type=Path, help="Sample prompt file")

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


def _build_explicit_namespace(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Return a namespace with only explicitly provided CLI values."""
    defaults: dict[str, object] = {
        action.dest: action.default
        for action in parser._actions
        if action.dest != "help"
    }
    explicit: dict[str, object] = {}

    for key, value in vars(args).items():
        default = defaults.get(key)
        if value != default:
            explicit[key] = value

    return argparse.Namespace(**explicit)


def _build_resolved_config(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> dict:
    """Resolve final config: defaults < YAML < preset < explicit CLI."""
    manager = ConfigManager()

    config = manager.resolve(config_path=args.config, args=None)

    if args.preset:
        config = deep_merge(config, get_preset(args.preset))

    explicit_args = _build_explicit_namespace(parser, args)
    cli_overrides = manager.extract_cli_overrides(explicit_args)
    config = deep_merge(config, cli_overrides)

    if args.run_dir is not None:
        config.setdefault("export", {})["output_dir"] = str(args.run_dir)
    elif "output" in config and "output_dir" in config["output"]:
        config.setdefault("export", {})["output_dir"] = config["output"]["output_dir"]

    config["_config_path"] = str(args.config) if args.config else None
    config["_config_version"] = str(config.get("config_version", "0.1.0"))
    config = manager.normalize_config(config)

    return config


def main():
    """CLI main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    _validate_args(args)

    resolved_config = _build_resolved_config(parser, args)
    config_manager = ConfigManager()
    try:
        config_manager.validate_or_raise(resolved_config)
    except LoRATrainerError as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(yaml.safe_dump(resolved_config, sort_keys=False))
        return

    if args.validate_only:
        print("✅ config validation passed")
        return
    elif args.export_only:
        print("❌ [E042] export-only mode is not implemented yet", file=sys.stderr)
        sys.exit(2)
    elif args.resume:
        print("❌ [E042] resume mode is not implemented yet", file=sys.stderr)
        sys.exit(2)
    else:
        logging.basicConfig(
            level=logging.INFO if not args.verbose else logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        trainer = None
        try:
            trainer = Trainer(resolved_config)
            run_dir = trainer.start()
            print(f"✅ run initialized: {run_dir}")

            trainer.train()
            print(f"✅ training completed")

            trainer.end()
            print(f"✅ exported final model")

        except KeyboardInterrupt:
            print("\n⚠️  training interrupted", file=sys.stderr)
            if trainer is not None:
                trainer.end()
            sys.exit(130)
        except Exception as error:
            print(f"❌ training failed: {error}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
