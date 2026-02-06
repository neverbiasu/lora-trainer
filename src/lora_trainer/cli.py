"""CLI entry point"""
import argparse
import sys
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create command line parser"""
    parser = argparse.ArgumentParser(
        prog="lora-trainer",
        description="LoRA Trainer - Minimalist but not simplistic",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # train command
    train_parser = subparsers.add_parser("train", help="Train LoRA model")
    # TODO: add arguments
    
    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    # TODO: add arguments
    
    # export command
    export_parser = subparsers.add_parser("export", help="Export model")
    # TODO: add arguments
    
    # resume command
    resume_parser = subparsers.add_parser("resume", help="Resume training")
    # TODO: add arguments
    
    # info command
    info_parser = subparsers.add_parser("info", help="Show run information")
    # TODO: add arguments
    
    return parser


def main():
    """CLI main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # TODO: implement command dispatch
    print(f"Command: {args.command}")
    print("Not implemented yet.")


if __name__ == "__main__":
    main()
