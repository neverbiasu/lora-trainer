# LoRA Trainer

> Minimal yet practical LoRA trainer — CLI-first, reproducibility-first.

## Features

- ✅ **Minimal design**: single CLI entrypoint, YAML config, sensible defaults
- ✅ **Reproducible**: records config, data, code version, and training curves
- ✅ **VRAM-friendly**: SD1.5 on 8GB GPUs (rank=32, batch=4)
- ✅ **Presets**: quick/balanced/quality out of the box
- ✅ **Actionable errors**: clear errors with fix suggestions

## Quick Start

### Install

```bash
pip install -e .

# Optional: xformers acceleration (recommended)
pip install -e ".[xformers]"
```

### M1 Quick Run

```bash
# Preview resolved config (no side effects)
lora-trainer --config examples/config_basic.yaml --dataset ./my_data --dry-run

# Initialize a run directory and save config snapshot
lora-trainer --config examples/config_basic.yaml --dataset ./my_data --run-dir ./outputs
```

### Dataset

Dataset structure:
```
my_data/
├── image_001.png
├── image_001.txt    # "a photo of sks person"
├── image_002.png
├── image_002.txt
└── ...
```

Each image must have a matching `.txt` caption file.

M1 currently includes config resolution/validation and run initialization. Training/export flows are scaffolded for the next milestone.

## Config Example

```yaml
# config.yaml
model:
  base_model: sd15

data:
  dataset_path: ./my_data
  resolution: 512
  cache_latents: auto

lora:
  rank: 32
  alpha: 32

training:
  learning_rate: 1e-4
  batch_size: 4
  max_train_steps: 1500
  preset: balanced
```

## Command Reference (M1)

Use a flat CLI (no subcommands):

- `--config`: load YAML config
- `--dry-run`: print resolved config and exit
- `--run-dir`: set run output base directory
- `--validate-only`: validate config and exit

## MVP Scope

**Supported**:
- SD1.5 (512x512)
- Standard LoRA (rank 8-128)
- Aspect-ratio bucketing
- Latent caching
- Mixed precision (fp16)
- xformers optimization

**Not supported (Phase 2)**:
- SDXL
- LoHA/LoKr
- Data augmentation
- Multi-GPU

## Project Structure

```
lora-trainer/
├── src/lora_trainer/          # Source code
├── docs/                      # Design documents
│   ├── design/                # Detailed design
│   └── references/            # Reference implementations (13 repos)
├── tests/                     # Tests
└── examples/                  # Example configs
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format and lint
black src/ tests/
ruff check src/ tests/
```

## License

MIT
