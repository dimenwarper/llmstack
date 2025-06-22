"""
Main pretraining pipeline orchestrator.
Coordinates data processing, selection, sampling, and training.
"""

import os
import json
import logging
from typing import List, Dict, Any
import argparse

from .config import get_toy_dataset_config, PretrainingConfig
from .toy_dataset_generator import ToyDatasetDownloader
from .data_processing import DataProcessor
from .data_selection import HybridSelector, create_target_distribution
from .data_sampling import create_sampler, SamplingConfig
from .model import create_model
from .training import PretrainingDataset, PretrainingTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PretrainingPipeline:
    """Main pretraining pipeline."""
    
    def __init__(self, config: PretrainingConfig):
        self.config = config
        self.raw_documents = []
        self.processed_documents = []
        self.selected_documents = []
        self.sampled_documents = []
        
        # Initialize components
        self.data_processor = DataProcessor(config)
        self.data_selector = None
        self.data_sampler = None
        
        # Create output directories
        os.makedirs(config.raw_data_dir, exist_ok=True)
        os.makedirs(config.processed_data_dir, exist_ok=True)
        os.makedirs(config.model_output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        logger.info("Pretraining pipeline initialized")
    
    def step1_download_data(self):
        """Step 1: Download raw data."""
        logger.info("Step 1: Downloading raw data...")
        
        # Update downloader with our modified config
        downloader = ToyDatasetDownloader()
        
        # Override the default sample sizes with our config
        logger.info("Updating dataset sample sizes for fast validation:")
        for data_source in self.config.data_sources:
            domain = data_source.domain
            if domain in downloader.dataset_configs:
                old_size = downloader.dataset_configs[domain]['sample_size']
                downloader.dataset_configs[domain]['sample_size'] = data_source.max_documents
                logger.info(f"  {domain}: {old_size} -> {data_source.max_documents} documents")
            else:
                logger.warning(f"  Domain {domain} not found in downloader configs")
        
        dataset = downloader.download_all_data()
        downloader.save_dataset(dataset, self.config.raw_data_dir)
        
        # Load all documents into memory for processing
        self.raw_documents = []
        for domain, docs in dataset.items():
            for doc in docs:
                doc_dict = {
                    'text': doc.text,
                    'domain': doc.domain,
                    'quality': doc.quality,
                    'source': doc.source,
                    'metadata': doc.metadata
                }
                self.raw_documents.append(doc_dict)
        
        logger.info(f"Downloaded {len(self.raw_documents)} raw documents")
        
        # Save statistics
        stats = {
            'total_documents': len(self.raw_documents),
            'domains': list(dataset.keys()),
            'domain_counts': {domain: len(docs) for domain, docs in dataset.items()}
        }
        
        stats_path = os.path.join(self.config.log_dir, "raw_data_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def step2_process_data(self):
        """Step 2: Process data (deduplication, quality filtering)."""
        logger.info("Step 2: Processing data...")
        
        if not self.raw_documents:
            raise ValueError("No raw documents found. Run step1_download_data first.")
        
        # Apply data processing pipeline
        self.processed_documents = self.data_processor.process_dataset(self.raw_documents)
        
        # Save processed documents
        processed_path = os.path.join(self.config.processed_data_dir, "processed_documents.jsonl")
        with open(processed_path, 'w') as f:
            for doc in self.processed_documents:
                doc_dict = {
                    'text': doc.text,
                    'domain': doc.domain,
                    'quality_score': doc.quality_score,
                    'quality_class': doc.quality_class,
                    'toxicity_score': doc.toxicity_score,
                    'hash_id': doc.hash_id,
                    'is_duplicate': doc.is_duplicate,
                    'metadata': doc.metadata
                }
                f.write(json.dumps(doc_dict, ensure_ascii=False) + '\n')
        
        logger.info(f"Processed {len(self.processed_documents)} documents")
        
        # Save processing statistics
        quality_counts = {'high': 0, 'medium': 0, 'low': 0}
        domain_counts = {}
        
        for doc in self.processed_documents:
            quality_counts[doc.quality_class] += 1
            domain_counts[doc.domain] = domain_counts.get(doc.domain, 0) + 1
        
        stats = {
            'total_processed': len(self.processed_documents),
            'quality_distribution': quality_counts,
            'domain_distribution': domain_counts,
            'avg_quality_score': sum(doc.quality_score for doc in self.processed_documents) / len(self.processed_documents),
            'avg_toxicity_score': sum(doc.toxicity_score for doc in self.processed_documents) / len(self.processed_documents)
        }
        
        stats_path = os.path.join(self.config.log_dir, "processing_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def step3_select_data(self):
        """Step 3: Select data using DSIR."""
        logger.info("Step 3: Selecting data...")
        
        if not self.processed_documents:
            raise ValueError("No processed documents found. Run step2_process_data first.")
        
        # Create target distribution from high-quality documents
        target_docs = create_target_distribution(
            self.processed_documents, 
            self.config.sampling.dsir_target_quality
        )
        
        if len(target_docs) < 5:
            logger.warning(f"Only {len(target_docs)} target documents found, using all processed docs")
            # Convert ProcessedDocument objects to dicts
            self.selected_documents = []
            for doc in self.processed_documents:
                doc_dict = {
                    'text': doc.text,
                    'domain': doc.domain,
                    'quality': doc.quality_class,
                    'quality_score': doc.quality_score,
                    'toxicity_score': doc.toxicity_score,
                    'metadata': doc.metadata
                }
                self.selected_documents.append(doc_dict)
        else:
            # Initialize hybrid selector
            self.data_selector = HybridSelector()
            
            # Domain weights based on research findings
            domain_weights = {
                'web': 1.0,      # High priority (Common Crawl equivalent)
                'books': 0.8,    # High quality source
                'code': 0.6,     # Important for technical capability
                'news': 0.5,     # Good quality, moderate priority
                'academic': 0.4  # Specialized content
            }
            
            quality_weights = {
                'high': 1.0,
                'medium': 0.7,
                'low': 0.3
            }
            
            self.data_selector.fit(target_docs, quality_weights, domain_weights)
            
            # Convert processed documents to dict format for selection
            doc_dicts = []
            for doc in self.processed_documents:
                doc_dict = {
                    'text': doc.text,
                    'domain': doc.domain,
                    'quality': doc.quality_class,
                    'quality_score': doc.quality_score,
                    'toxicity_score': doc.toxicity_score,
                    'metadata': doc.metadata
                }
                doc_dicts.append(doc_dict)
            
            # Select documents
            selected_dicts = self.data_selector.select_documents(
                doc_dicts, 
                self.config.sampling.dsir_selection_rate
            )
            
            self.selected_documents = selected_dicts
        
        logger.info(f"Selected {len(self.selected_documents)} documents")
        
        # Save selected documents
        selected_path = os.path.join(self.config.processed_data_dir, "selected_documents.jsonl")
        with open(selected_path, 'w') as f:
            for doc in self.selected_documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    def step4_sample_data(self):
        """Step 4: Sample data for training."""
        logger.info("Step 4: Sampling data...")
        
        if not self.selected_documents:
            raise ValueError("No selected documents found. Run step3_select_data first.")
        
        # Create sampler based on domain composition
        domain_counts = {}
        for doc in self.selected_documents:
            domain_counts[doc['domain']] = domain_counts.get(doc['domain'], 0) + 1
        
        # Determine sampling method based on domain distribution
        if domain_counts.get('code', 0) > len(self.selected_documents) * 0.3:
            sampling_method = 'alpha'  # Good for code-heavy datasets
        else:
            sampling_method = 'unimax'  # Good for general English content
        
        sampling_config = SamplingConfig(
            method=sampling_method,
            alpha=self.config.sampling.alpha,
            max_epochs=self.config.sampling.max_epochs
        )
        
        self.data_sampler = create_sampler(sampling_config)
        self.data_sampler.fit(self.selected_documents)
        
        # For now, just use all selected documents as sampled
        # In practice, you would sample batches during training
        self.sampled_documents = self.selected_documents
        
        logger.info(f"Prepared sampling for {len(self.sampled_documents)} documents")
    
    def step5_train_model(self):
        """Step 5: Train the model."""
        logger.info("Step 5: Training model...")
        
        if not self.sampled_documents:
            raise ValueError("No sampled documents found. Run step4_sample_data first.")
        
        # Create model
        model = create_model(self.config.model)
        
        # Create datasets
        # Split into train/eval (80/20)
        split_idx = int(len(self.sampled_documents) * 0.8)
        train_docs = self.sampled_documents[:split_idx]
        eval_docs = self.sampled_documents[split_idx:]
        
        train_dataset = PretrainingDataset(train_docs, None, self.config.model.max_seq_length)
        eval_dataset = PretrainingDataset(eval_docs, None, self.config.model.max_seq_length) if eval_docs else None
        
        # Create trainer
        trainer = PretrainingTrainer(self.config, model, train_dataset, eval_dataset)
        
        # Train model
        metrics = trainer.train()
        
        logger.info("Training completed!")
        
        return metrics
    
    def run_full_pipeline(self):
        """Run the complete pretraining pipeline."""
        logger.info("Starting full pretraining pipeline...")
        
        try:
            self.step1_download_data()
            self.step2_process_data()
            self.step3_select_data()
            self.step4_sample_data()
            metrics = self.step5_train_model()
            
            logger.info("Pipeline completed successfully!")
            return metrics
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def save_pipeline_state(self, path: str):
        """Save pipeline state for debugging/resumption."""
        state = {
            'config': self.config.__dict__,
            'num_raw_documents': len(self.raw_documents),
            'num_processed_documents': len(self.processed_documents),
            'num_selected_documents': len(self.selected_documents),
            'num_sampled_documents': len(self.sampled_documents)
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Pipeline state saved to {path}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Pretraining Pipeline')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--step', type=str, choices=['download', 'process', 'select', 'sample', 'train', 'full'],
                       default='full', help='Pipeline step to run')
    
    args = parser.parse_args()
    
    # Get configuration
    if args.config:
        # Would load custom config from file
        config = get_toy_dataset_config()
    else:
        config = get_toy_dataset_config()
    
    # Override for testing - reduce training steps
    config.training.max_steps = 50
    config.training.eval_steps = 25
    config.training.save_steps = 50
    
    # Create pipeline
    pipeline = PretrainingPipeline(config)
    
    # Run specified step(s)
    if args.step == 'download':
        pipeline.step1_download_data()
    elif args.step == 'process':
        pipeline.step2_process_data()
    elif args.step == 'select':
        pipeline.step3_select_data()
    elif args.step == 'sample':
        pipeline.step4_sample_data()
    elif args.step == 'train':
        pipeline.step5_train_model()
    elif args.step == 'full':
        pipeline.run_full_pipeline()
    
    # Save pipeline state
    state_path = os.path.join(config.log_dir, "pipeline_state.json")
    pipeline.save_pipeline_state(state_path)

if __name__ == "__main__":
    main()