"""
Training loop for pretraining with monitoring and logging.
Based on research-backed training practices.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from typing import List, Dict, Any, Optional
import logging
from dataclasses import asdict
import time
import math
from tqdm import tqdm

from .model import create_model
from .data_processing import DataProcessor
from .data_selection import create_target_distribution, HybridSelector
from .data_sampling import create_sampler, SamplingConfig

logger = logging.getLogger(__name__)

class PretrainingDataset(Dataset):
    """Dataset for pretraining."""
    
    def __init__(self, documents: List[Dict], tokenizer, max_length: int = 2048):
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        doc = self.documents[idx]
        text = doc['text'] if isinstance(doc, dict) else doc.text
        
        # Simple tokenization (would use proper tokenizer in practice)
        tokens = self._simple_tokenize(text)
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))  # Pad with 0
        
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(tokens[1:], dtype=torch.long),
            'attention_mask': torch.ones(len(tokens) - 1, dtype=torch.bool)
        }
    
    def _simple_tokenize(self, text: str) -> List[int]:
        """Simple tokenization for testing (replace with real tokenizer)."""
        # Very basic tokenization - just map characters to integers
        # In practice, use a proper tokenizer like SentencePiece
        vocab_size = 32000
        
        # Convert to bytes and map to token range
        tokens = []
        for char in text.lower():
            if char.isalnum():
                token = ord(char) % (vocab_size - 100) + 100
            elif char.isspace():
                token = 50  # Space token
            else:
                token = 51  # Other punctuation
            tokens.append(token)
        
        # Add BOS and EOS tokens
        tokens = [1] + tokens + [2]  # 1=BOS, 2=EOS
        
        return tokens

class TrainingMetrics:
    """Track training metrics."""
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'eval_loss': [],
            'perplexity': [],
            'learning_rate': [],
            'step': [],
            'epoch': [],
            'tokens_seen': 0,
            'best_eval_loss': float('inf')
        }
    
    def update(self, step: int, epoch: int, train_loss: float, 
               eval_loss: Optional[float] = None, lr: float = 0.0):
        """Update metrics."""
        self.metrics['step'].append(step)
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['learning_rate'].append(lr)
        
        if eval_loss is not None:
            self.metrics['eval_loss'].append(eval_loss)
            perplexity = math.exp(min(eval_loss, 10))  # Clip to prevent overflow
            self.metrics['perplexity'].append(perplexity)
            
            if eval_loss < self.metrics['best_eval_loss']:
                self.metrics['best_eval_loss'] = eval_loss
    
    def save(self, path: str):
        """Save metrics to file."""
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self, path: str):
        """Load metrics from file."""
        with open(path, 'r') as f:
            self.metrics = json.load(f)

class PretrainingTrainer:
    """Main training class."""
    
    def __init__(self, config, model, train_dataset, eval_dataset=None):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Setup optimizer with cosine schedule
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Calculate total steps for learning rate schedule
        self.total_steps = config.training.max_steps
        self.warmup_steps = int(self.total_steps * config.training.warmup_ratio)
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for debugging
        )
        
        if eval_dataset:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Metrics tracking
        self.metrics = TrainingMetrics()
        self.global_step = 0
        self.current_epoch = 0
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Model has {self.model.get_num_params():,} parameters")
    
    def get_lr(self, step: int) -> float:
        """Get learning rate with cosine schedule and warmup."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.config.training.learning_rate * step / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return self.config.training.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Compute loss
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update weights
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def evaluate(self) -> float:
        """Evaluate model on validation set."""
        if not self.eval_dataset:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids)
                
                loss_fn = nn.CrossEntropyLoss(ignore_index=0)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, step: int, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'step': step,
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'metrics': self.metrics.metrics
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint at step {step} to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['step']
        self.current_epoch = checkpoint['epoch']
        self.metrics.metrics = checkpoint['metrics']
        
        logger.info(f"Loaded checkpoint from step {self.global_step}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Total steps: {self.total_steps}")
        logger.info(f"Warmup steps: {self.warmup_steps}")
        
        # Create output directories
        os.makedirs(self.config.model_output_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        step = 0
        accumulation_step = 0
        
        while step < self.total_steps:
            self.current_epoch += 1
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
            
            for batch in pbar:
                # Update learning rate
                lr = self.get_lr(step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Training step
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                accumulation_step += 1
                
                # Update metrics every few steps
                if accumulation_step % self.config.training.gradient_accumulation_steps == 0:
                    step += 1
                    self.global_step = step
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{loss:.4f}",
                        'lr': f"{lr:.2e}",
                        'step': step
                    })
                    
                    # Evaluation
                    if step % self.config.training.eval_steps == 0:
                        eval_loss = self.evaluate()
                        self.metrics.update(step, self.current_epoch, loss, eval_loss, lr)
                        
                        logger.info(f"Step {step}: train_loss={loss:.4f}, eval_loss={eval_loss:.4f}, "
                                  f"perplexity={math.exp(min(eval_loss, 10)):.2f}")
                    else:
                        self.metrics.update(step, self.current_epoch, loss, None, lr)
                    
                    # Save checkpoint
                    if step % self.config.training.save_steps == 0:
                        checkpoint_path = os.path.join(
                            self.config.model_output_dir, 
                            f"checkpoint-{step}.pt"
                        )
                        self.save_checkpoint(step, checkpoint_path)
                        
                        # Save metrics
                        metrics_path = os.path.join(self.config.log_dir, "metrics.json")
                        self.metrics.save(metrics_path)
                    
                    # Check if we've reached max steps
                    if step >= self.total_steps:
                        break
            
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {self.current_epoch} complete. Average loss: {avg_epoch_loss:.4f}")
            
            if step >= self.total_steps:
                break
        
        # Final checkpoint and evaluation
        final_checkpoint_path = os.path.join(self.config.model_output_dir, "final_model.pt")
        self.save_checkpoint(step, final_checkpoint_path)
        
        final_eval_loss = self.evaluate()
        logger.info(f"Training complete! Final eval loss: {final_eval_loss:.4f}")
        
        return self.metrics

def main():
    """Test training loop."""
    from config import get_toy_dataset_config
    
    # Get configuration
    config = get_toy_dataset_config()
    
    # Create model
    model = create_model(config.model)
    
    # Create dummy datasets for testing
    dummy_docs = [
        {'text': f'This is training document {i} with some content for testing the model.'}
        for i in range(100)
    ]
    
    train_dataset = PretrainingDataset(dummy_docs[:80], None, config.model.max_seq_length)
    eval_dataset = PretrainingDataset(dummy_docs[80:], None, config.model.max_seq_length)
    
    # Create trainer
    trainer = PretrainingTrainer(config, model, train_dataset, eval_dataset)
    
    # Train for a few steps
    config.training.max_steps = 10
    config.training.eval_steps = 5
    config.training.save_steps = 10
    
    metrics = trainer.train()
    
    print("Training completed!")
    print(f"Final metrics: {metrics.metrics}")

if __name__ == "__main__":
    main()