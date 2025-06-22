# LLM Training Stack - Context

## Project Overview
This project creates a complete LLM training pipeline using only agentic tools (Claude Code, Cursor) with minimal manual editing. The goal is to demonstrate end-to-end model development through AI-assisted coding.

## Project Structure
```
llmstack/
├── models/        # Model architectures and configurations
├── pretraining/   # Initial model training from scratch
├── midtraining/   # Intermediate training and fine-tuning
├── posttraining/  # Final training phase and optimization
├── references/    # Research papers and helpful references
├── CONTEXT.md     # Project context and requirements
└── README.md      # Project overview
```

## Reference Materials
The `references/` directory contains research papers, technical documentation, and other materials that inform the implementation of each training phase. These resources guide architectural decisions and best practices throughout the development process.

## Training Pipeline Stages

### 1. Pretraining
- **Purpose**: Train foundation model from scratch
- **Data**: Large-scale text corpora
- **Techniques**: Unsupervised learning, next-token prediction
- **Output**: Base language model

### 2. Midtraining  
- **Purpose**: Domain-specific adaptation and fine-tuning
- **Data**: Curated domain-specific datasets
- **Techniques**: Supervised fine-tuning, task-specific training
- **Output**: Specialized model for target domain

### 3. Posttraining
- **Purpose**: Final optimization and alignment
- **Data**: Human feedback, safety datasets
- **Techniques**: RLHF, constitutional AI, safety training
- **Output**: Production-ready aligned model

## Key Research Findings

### Data Curation Principles (Longpre et al., 2023)
Based on experiments with 28 1.5B parameter models, the following empirically-validated principles guide our implementation:

### Data Age & Temporal Alignment
- **Temporal misalignment causes 0.4 performance points degradation per year**
- Finetuning cannot overcome temporally misaligned pretraining data
- Prioritize recent data collection over older archives
- Plan for regular model retraining cycles

### Quality & Toxicity Filtering Trade-offs
- **Quality filtering improves performance by 1-6% despite removing 10%+ of data**
- Toxicity filtering reduces toxic generation but hurts general performance
- Use targeted filtering strategies based on intended use case:
  - General performance: Apply moderate quality filtering (T=0.975)
  - Toxicity identification: Use inverse toxicity filters
  - Safety-critical: Accept performance trade-offs for toxicity filtering

### Domain Composition Priority
1. **Common Crawl** (27% of data, highest impact when removed: -4.8%)
2. **Books** (7% of data, -2.7% performance impact)
3. **OpenWeb Text** (7% of data, -1.4% impact)
4. **Academic/PubMed/Code** (moderate specialized impact)
- **Key insight**: Heterogeneous data trumps domain-specific volume
- Include all available high-quality sources rather than being selective

### Dataset Construction Methodology (Eldan et al., 2024)
Systematic 4-step pipeline approach validated on 2B-8B parameter models:

#### Data Sourcing Strategy (2T+ tokens total)
- **English (1.2T)**: Web crawl (889B), News (94B), Books (35B), Scientific (33B)
- **Multilingual (596B)**: Web crawl (540B), Parallel corpora (56B) across 52 languages  
- **Code (212B)**: The Stack v1.2 across 43 programming languages
- **Web crawl composition**: Multiple Common Crawl snapshots + C4 URL re-crawling

#### Data Processing Pipeline
1. **Exact Deduplication**: 128-bit hashes, select one document per group
2. **Fuzzy Deduplication**: Prioritize older documents (significant performance improvement)
3. **Quality Filtering**: KenLM perplexity + heuristic filters
   - N-gram LM perplexity threshold: 5000
   - Content filters: <25% non-alphanumeric, <15% numbers, <20% URLs
   - Document length: 50-100k words, mean word length 3-10 characters
4. **Multi-Attribute Classification**: Quality (3-class), Toxicity, Domain (27 categories)

#### Data Selection & Sampling
- **DSIR Selection**: Domain Selection via Importance Resampling at source level (95% rate)
- **Sampling by Domain**:
  - English: UniMax sampling (1 epoch optimal)
  - Multilingual: UniMax slightly better than alpha sampling (α=1.3)
  - Code: Alpha sampling (α=1.3) superior due to limited cross-language transfer
- **Attribute-Based Sampling**: Fine-grained approach better for quality, grouped for domain

**Expected Pipeline Improvements**:
- Data curation: ~2.3 point improvement
- Optimal deduplication: ~0.5 point improvement  
- DSIR selection: ~0.3-0.4 point improvement
- Attribute-based sampling: ~1 point improvement

## Implementation Status

### ✅ Completed Pretraining Pipeline
A comprehensive research-backed pretraining pipeline has been implemented in `/pretraining/`:

#### Core Components
1. **Configuration System** (`config.py`) - Research-backed parameter settings
2. **Data Download** (`toy_dataset_generator.py`) - Real HuggingFace datasets
3. **Data Processing** (`data_processing.py`) - Deduplication + quality filtering
4. **Data Selection** (`data_selection.py`) - DSIR implementation + hybrid selection
5. **Data Sampling** (`data_sampling.py`) - UniMax, Alpha, attribute-based sampling
6. **Model Architecture** (`model.py`) - Small transformer with RoPE, SwiGLU
7. **Training Loop** (`training.py`) - Complete training with monitoring
8. **Pipeline Orchestrator** (`pipeline.py`) - Coordinates all steps

#### Dataset Implementation
- **Web**: C4 (Common Crawl) - 400 documents
- **Books**: BookCorpus - 100 documents  
- **Code**: The Stack v1.2 - 100 documents
- **News**: CC-News - 100 documents
- **Academic**: ArXiv abstracts - 50 documents

#### Research Implementation
- ✅ **Temporal alignment** - Prioritizes recent data collection
- ✅ **Quality filtering** - KenLM perplexity (5000 threshold) + heuristics
- ✅ **Domain composition** - Research-recommended weights and sampling
- ✅ **DSIR selection** - 95% selection rate with importance resampling
- ✅ **Sampling strategies** - UniMax for English, Alpha (α=1.3) for code
- ✅ **Expected improvements** - ~4+ point gains from systematic pipeline

#### Pipeline Architecture
```
Data Sources → Curation → Selection → Sampling → Training
     ↓             ↓          ↓         ↓         ↓
  Download    Deduplication  DSIR    UniMax/   Model
  (HF APIs)   Quality Filter        Alpha     Training
```

#### Usage
```bash
# Full pipeline
python pretraining/pipeline.py --step full

# Individual steps
python pretraining/pipeline.py --step [download|process|select|sample|train]
```

## Technical Requirements
- Framework: PyTorch/Transformers ✅
- Hardware: GPU cluster support ✅
- Data handling: Efficient data loading and preprocessing with configurable filtering ✅
- Monitoring: Training metrics, toxicity tracking, and domain composition logging ✅
- Checkpointing: Model saving and resumption with temporal metadata ✅

## Development Constraints
- All code written by AI assistants
- Minimal manual editing allowed
- Focus on clean, maintainable architecture
- Comprehensive documentation generated by AI

## Success Metrics
- Functional training pipeline for each phase
- Clear separation of concerns between stages
- Scalable and configurable architecture
- Complete documentation and examples