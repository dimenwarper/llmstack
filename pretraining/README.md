# Pretraining Pipeline

A research-backed pretraining pipeline implementing best practices from recent papers.

## Overview

This pretraining pipeline implements the systematic 4-step approach validated by research:

1. **Data Sources** → 2. **Data Curation** → 3. **Data Selection** → 4. **Data Sampling** → 5. **Model Training**

## Research Foundation

Based on findings from:
- **Longpre et al. (2023)**: "A Pretrainer's Guide to Training Data"
- **Eldan et al. (2024)**: "Data, Data Everywhere: A Guide for Pretraining Dataset Construction"

Key implemented findings:
- Data age causes 0.4 performance points degradation per year
- Quality filtering improves performance 1-6% despite removing 10%+ of data
- UniMax sampling optimal for English, Alpha sampling optimal for code
- DSIR selection provides 0.3-0.4 point improvements

## Pipeline Architecture

### 1. Data Sources (`toy_dataset_generator.py`)
Downloads real datasets from HuggingFace:
- **Web**: C4 (Common Crawl)
- **Books**: BookCorpus  
- **Code**: The Stack v1.2
- **News**: CC-News
- **Academic**: ArXiv abstracts

### 2. Data Processing (`data_processing.py`)
- **Exact Deduplication**: Hash-based duplicate removal
- **Fuzzy Deduplication**: Similarity-based duplicate detection
- **Quality Filtering**: KenLM perplexity + heuristic filters
- **Multi-Attribute Classification**: Quality, toxicity, domain classification

### 3. Data Selection (`data_selection.py`)
- **DSIR**: Domain Selection via Importance Resampling
- **Attribute-Based Selection**: Quality and domain-weighted selection
- **Hybrid Selection**: Combines DSIR + attribute-based approaches

### 4. Data Sampling (`data_sampling.py`)
- **UniMax Sampling**: Uniform domain sampling (optimal for English)
- **Alpha Sampling**: Power-law domain sampling (optimal for code)
- **Attribute-Based Sampling**: Fine-grained quality-based sampling

### 5. Model Training (`training.py`, `model.py`)
- **Small Transformer**: RoPE, SwiGLU, untied embeddings
- **Training Loop**: AdamW optimizer, cosine LR schedule, gradient clipping
- **Monitoring**: Perplexity tracking, checkpointing, metrics logging

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
python pipeline.py --step full
```

### Run Individual Steps
```bash
# Download data
python pipeline.py --step download

# Process data
python pipeline.py --step process

# Select data
python pipeline.py --step select

# Sample data
python pipeline.py --step sample

# Train model
python pipeline.py --step train
```

### Test Individual Components
```bash
# Test data processing
python data_processing.py

# Test data selection
python data_selection.py

# Test data sampling
python data_sampling.py

# Test model
python model.py

# Test training
python training.py
```

## Configuration

Edit `config.py` to customize:
- Dataset composition and sizes
- Quality filtering thresholds
- Sampling strategies
- Model architecture
- Training hyperparameters

## Output Structure

```
pretraining/
├── data/
│   ├── raw/                 # Downloaded raw datasets
│   └── processed/           # Processed and selected data
├── models/
│   └── checkpoints/         # Model checkpoints
├── logs/                    # Training logs and metrics
└── *.py                     # Pipeline components
```

## Key Features

### Research-Backed Filtering
- Perplexity threshold: 5000 (KenLM-based)
- Character filters: <25% non-alphanumeric
- Content filters: <15% numbers, <20% URLs
- Document length: 50-100k words

### Domain-Specific Optimization
- **English datasets**: UniMax sampling (1 epoch optimal)
- **Code datasets**: Alpha sampling (α=1.3)
- **Quality-based**: Fine-grained attribute sampling

### Expected Improvements
- Data curation: ~2.3 point improvement
- Optimal deduplication: ~0.5 point improvement
- DSIR selection: ~0.3-0.4 point improvement
- Attribute-based sampling: ~1 point improvement

## Monitoring

The pipeline tracks:
- Processing statistics (deduplication rates, quality distribution)
- Selection scores and rankings
- Training metrics (loss, perplexity, learning rate)
- Domain and quality distributions throughout pipeline

## Extending the Pipeline

To add new data sources:
1. Add dataset config in `toy_dataset_generator.py`
2. Update domain weights in `pipeline.py`
3. Add domain-specific filters in `data_processing.py`

To add new sampling strategies:
1. Implement new sampler class in `data_sampling.py`
2. Update `create_sampler()` factory function
3. Add config options in `config.py`