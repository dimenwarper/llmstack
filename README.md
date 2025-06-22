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
# Run full pipeline on T4 GPU
modal run modal_pretraining.py --step full --gpu-tier t4

# Run individual steps
modal run modal_pretraining.py --step download
modal run modal_pretraining.py --step process
modal run modal_pretraining.py --step train

# Run on A100 for serious training
modal run modal_pretraining.py --step full --gpu-tier a100

# List artifacts
modal run modal_pretraining.py --list-files

# Download specific results
modal run modal_pretraining.py --download "results/full_result.json"

# Clean up artifacts
modal run modal_pretraining.py --cleanup
```

### Local Development
```bash
# Run pipeline locally for development
cd pretraining
python pipeline.py --step full
```