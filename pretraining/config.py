"""
Configuration file for pretraining pipeline.
Based on research findings from Longpre et al. (2023) and Eldan et al. (2024).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import os

@dataclass
class DataSourceConfig:
    """Configuration for individual data sources."""
    name: str
    path: str
    domain: str
    weight: float = 1.0
    quality_threshold: float = 0.5
    max_documents: Optional[int] = None

@dataclass
class DeduplicationConfig:
    """Configuration for deduplication pipeline."""
    exact_dedup: bool = True
    fuzzy_dedup: bool = True
    fuzzy_threshold: float = 0.85
    hash_type: str = "md5"  # md5, sha256, or xxhash
    prioritize_older: bool = True

@dataclass
class QualityFilterConfig:
    """Configuration for quality filtering based on research findings."""
    # KenLM perplexity threshold (Eldan et al. 2024)
    perplexity_threshold: float = 5000.0
    
    # Heuristic filters
    min_doc_length: int = 50
    max_doc_length: int = 100000
    min_word_length: float = 3.0
    max_word_length: float = 10.0
    max_non_alphanumeric_ratio: float = 0.25
    max_number_ratio: float = 0.15
    max_url_ratio: float = 0.20
    max_duplicate_lines_ratio: float = 0.30
    max_boilerplate_ratio: float = 0.40
    
    # Code-specific filters
    code_min_comment_ratio: float = 0.001
    code_max_comment_ratio: float = 0.85
    code_min_lines: int = 5
    code_max_lines: int = 20000
    code_min_char_token_ratio: float = 2.0

@dataclass
class SamplingConfig:
    """Configuration for data sampling strategies."""
    # Sampling method: 'unimax', 'alpha', 'preference'
    english_method: str = 'unimax'
    multilingual_method: str = 'unimax'
    code_method: str = 'alpha'
    
    # Alpha sampling parameter
    alpha: float = 1.3
    
    # Epoch limits (prefer minimal epochs based on research)
    max_epochs: int = 1
    
    # DSIR parameters
    dsir_selection_rate: float = 0.95
    dsir_target_quality: str = 'high'  # high, medium, low

@dataclass
class ModelConfig:
    """Small transformer model config for testing."""
    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    intermediate_size: int = 2048
    max_seq_length: int = 2048
    dropout: float = 0.1
    
@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    max_steps: int = 10000
    save_steps: int = 1000
    eval_steps: int = 500
    gradient_accumulation_steps: int = 4

@dataclass
class PretrainingConfig:
    """Main configuration for pretraining pipeline."""
    # Data pipeline configs
    data_sources: List[DataSourceConfig]
    deduplication: DeduplicationConfig
    quality_filter: QualityFilterConfig
    sampling: SamplingConfig
    
    # Model and training configs
    model: ModelConfig
    training: TrainingConfig
    
    # Paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    model_output_dir: str = "models/checkpoints"
    log_dir: str = "logs"
    
    # Monitoring
    wandb_project: Optional[str] = None
    log_level: str = "INFO"

def get_toy_dataset_config() -> PretrainingConfig:
    """Get configuration for toy dataset testing."""
    
    data_sources = [
        DataSourceConfig(
            name="toy_web",
            path="data/raw/toy_web.jsonl",
            domain="web",
            weight=0.4,
            max_documents=400
        ),
        DataSourceConfig(
            name="toy_books",
            path="data/raw/toy_books.jsonl", 
            domain="books",
            weight=0.25,
            max_documents=100
        ),
        DataSourceConfig(
            name="toy_code",
            path="data/raw/toy_code.jsonl",
            domain="code", 
            weight=0.15,
            max_documents=100
        ),
        DataSourceConfig(
            name="toy_news",
            path="data/raw/toy_news.jsonl",
            domain="news",
            weight=0.15,
            max_documents=100
        ),
        DataSourceConfig(
            name="toy_academic",
            path="data/raw/toy_academic.jsonl",
            domain="academic",
            weight=0.05,
            max_documents=50
        )
    ]
    
    return PretrainingConfig(
        data_sources=data_sources,
        deduplication=DeduplicationConfig(),
        quality_filter=QualityFilterConfig(),
        sampling=SamplingConfig(),
        model=ModelConfig(),
        training=TrainingConfig()
    )