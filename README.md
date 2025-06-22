# LLM Training Stack

A complete LLM training pipeline built entirely with agentic tools (Claude Code, Cursor) with minimal manual editing.

## Overview

This project demonstrates a full machine learning lifecycle for large language models, spanning from initial pretraining through midtraining to final posttraining phases. The unique constraint is that all development must be done using AI coding assistants with little to no manual code editing.

## Project Structure

```
llmstack/
├── models/        # Model architectures and configurations
├── pretraining/   # Initial model training from scratch
├── midtraining/   # Intermediate training and fine-tuning
├── posttraining/  # Final training phase and optimization
└── README.md
```

## Training Phases

### Pretraining
- Foundation model training from scratch
- Large-scale unsupervised learning
- Base model architecture implementation

### Midtraining
- Domain-specific fine-tuning
- Intermediate model refinement
- Performance optimization

### Posttraining
- Final model polishing
- Alignment and safety training
- Production-ready model preparation

## Development Methodology

This project serves as an experiment in agentic development, where AI coding assistants handle the majority of implementation work with minimal human intervention in the actual coding process.

## Quick Start with Modal

### Setup
```bash
# Install dependencies
pip install -e .

# Set up Modal (requires Modal account)
modal setup
```

### Test Modal Setup
```bash
# Test imports and GPU access
modal run test_modal.py --test imports

# Test data download
modal run test_modal.py --test download
```

### Run Pretraining Pipeline
```bash
# Quick validation (recommended) - complete pipeline in 2-3 minutes
uv run modal run modal_pretraining.py::run_pretraining --step full --gpu-tier t4 --quick true

# Medium validation - more documents for thorough testing
uv run modal run modal_pretraining.py::run_pretraining --step full --gpu-tier t4 --quick false

# Run individual steps for debugging
uv run modal run modal_pretraining.py::run_pretraining --step download --quick true
uv run modal run modal_pretraining.py::run_pretraining --step process --quick true
uv run modal run modal_pretraining.py::run_pretraining --step train --quick true

# Production training on A100 (for larger experiments)
uv run modal run modal_pretraining.py::run_pretraining --step full --gpu-tier a100 --quick false

# Artifact management
uv run modal run modal_pretraining.py::run_pretraining --list-files
uv run modal run modal_pretraining.py::run_pretraining --download "results/full_result.json"
uv run modal run modal_pretraining.py::run_pretraining --download "models/checkpoints/final_model.pt"
uv run modal run modal_pretraining.py::run_pretraining --cleanup
```

#### Expected Results
The quick validation mode will show:
- **Training loss**: 9.59 → 7.30 (good learning progression)
- **Evaluation loss**: 7.67 → 7.22 (model improving)
- **Perplexity**: 2150 → 1370 (significant improvement)
- **Complete in**: 2-3 minutes end-to-end

### Local Development
```bash
# Run pipeline locally for development
cd pretraining
python pipeline.py --step full
```