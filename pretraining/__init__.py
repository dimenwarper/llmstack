"""
Pretraining pipeline for LLM training stack.
Research-backed implementation of data processing, selection, and training.
"""

from .config import get_toy_dataset_config, PretrainingConfig
from .toy_dataset_generator import ToyDatasetDownloader
from .data_processing import DataProcessor, ProcessedDocument
from .data_selection import DSIRSelector, HybridSelector, create_target_distribution
from .data_sampling import create_sampler, SamplingConfig
from .model import create_model, SmallTransformer
from .training import PretrainingTrainer, PretrainingDataset
from .pipeline import PretrainingPipeline

__all__ = [
    "get_toy_dataset_config",
    "PretrainingConfig", 
    "ToyDatasetDownloader",
    "DataProcessor",
    "ProcessedDocument",
    "DSIRSelector",
    "HybridSelector", 
    "create_target_distribution",
    "create_sampler",
    "SamplingConfig",
    "create_model",
    "SmallTransformer",
    "PretrainingTrainer",
    "PretrainingDataset",
    "PretrainingPipeline",
]