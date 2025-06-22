"""
Data sampling strategies for pretraining.
Implements UniMax, Alpha sampling, and domain-specific sampling
based on research findings from Eldan et al. (2024).
"""

import numpy as np
import random
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)

@dataclass
class SamplingConfig:
    """Configuration for sampling strategies."""
    method: str  # 'unimax', 'alpha', 'preference'
    alpha: float = 1.3
    max_epochs: int = 1
    temperature: float = 1.0
    
class DataSampler:
    """Base class for data sampling strategies."""
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.domain_stats = {}
        self.sampling_weights = {}
    
    def _compute_domain_statistics(self, documents: List[Dict]) -> Dict[str, Any]:
        """Compute statistics for each domain."""
        domain_counts = defaultdict(int)
        domain_docs = defaultdict(list)
        
        for doc in documents:
            domain = doc.get('domain', 'unknown')
            domain_counts[domain] += 1
            domain_docs[domain].append(doc)
        
        total_docs = len(documents)
        domain_stats = {}
        
        for domain, count in domain_counts.items():
            domain_stats[domain] = {
                'count': count,
                'proportion': count / total_docs,
                'documents': domain_docs[domain]
            }
        
        return domain_stats
    
    def fit(self, documents: List[Dict]):
        """Fit sampler on document collection."""
        self.domain_stats = self._compute_domain_statistics(documents)
        self.sampling_weights = self._compute_sampling_weights()
        
        logger.info(f"Fitted sampler on {len(documents)} documents across {len(self.domain_stats)} domains")
        for domain, stats in self.domain_stats.items():
            logger.info(f"  {domain}: {stats['count']} docs ({stats['proportion']:.3f})")
    
    def _compute_sampling_weights(self) -> Dict[str, float]:
        """Compute sampling weights for each domain."""
        raise NotImplementedError("Subclasses must implement _compute_sampling_weights")
    
    def sample_batch(self, batch_size: int) -> List[Dict]:
        """Sample a batch of documents."""
        raise NotImplementedError("Subclasses must implement sample_batch")

class UniMaxSampler(DataSampler):
    """
    UniMax sampling - samples each domain uniformly until one is exhausted.
    Optimal for English data according to research findings.
    """
    
    def __init__(self, config: SamplingConfig):
        super().__init__(config)
        self.domain_iterators = {}
        self.epoch_count = defaultdict(int)
    
    def _compute_sampling_weights(self) -> Dict[str, float]:
        """UniMax uses uniform sampling across domains."""
        num_domains = len(self.domain_stats)
        return {domain: 1.0 / num_domains for domain in self.domain_stats.keys()}
    
    def _reset_domain_iterator(self, domain: str):
        """Reset iterator for a domain."""
        docs = self.domain_stats[domain]['documents']
        random.shuffle(docs)
        self.domain_iterators[domain] = iter(docs)
        self.epoch_count[domain] += 1
        
        if self.epoch_count[domain] > self.config.max_epochs:
            logger.warning(f"Domain {domain} exceeded max_epochs ({self.config.max_epochs})")
    
    def fit(self, documents: List[Dict]):
        """Fit UniMax sampler."""
        super().fit(documents)
        
        # Initialize iterators for each domain
        for domain in self.domain_stats.keys():
            self._reset_domain_iterator(domain)
    
    def sample_batch(self, batch_size: int) -> List[Dict]:
        """Sample batch using UniMax strategy."""
        batch = []
        domains = list(self.domain_stats.keys())
        
        for _ in range(batch_size):
            # Round-robin through domains
            domain = domains[len(batch) % len(domains)]
            
            try:
                doc = next(self.domain_iterators[domain])
                batch.append(doc)
            except StopIteration:
                # Domain exhausted, reset if under epoch limit
                if self.epoch_count[domain] < self.config.max_epochs:
                    self._reset_domain_iterator(domain)
                    doc = next(self.domain_iterators[domain])
                    batch.append(doc)
                else:
                    # Skip this domain
                    continue
        
        return batch

class AlphaSampler(DataSampler):
    """
    Alpha sampling - samples according to p_i^alpha / sum(p_j^alpha).
    Optimal for code data according to research findings.
    """
    
    def __init__(self, config: SamplingConfig):
        super().__init__(config)
        self.document_pool = []
        self.domain_probabilities = {}
    
    def _compute_sampling_weights(self) -> Dict[str, float]:
        """Compute alpha-weighted sampling probabilities."""
        domain_props = {domain: stats['proportion'] 
                       for domain, stats in self.domain_stats.items()}
        
        # Apply alpha weighting
        alpha_weights = {domain: prop ** self.config.alpha 
                        for domain, prop in domain_props.items()}
        
        # Normalize
        total_weight = sum(alpha_weights.values())
        normalized_weights = {domain: weight / total_weight 
                            for domain, weight in alpha_weights.items()}
        
        return normalized_weights
    
    def fit(self, documents: List[Dict]):
        """Fit Alpha sampler."""
        super().fit(documents)
        
        # Create weighted document pool
        self.document_pool = []
        for domain, stats in self.domain_stats.items():
            weight = self.sampling_weights[domain]
            domain_docs = stats['documents']
            
            # Add documents with appropriate weights
            for doc in domain_docs:
                doc['sampling_weight'] = weight
                self.document_pool.append(doc)
        
        # Shuffle pool
        random.shuffle(self.document_pool)
        
        logger.info(f"Alpha sampler prepared pool of {len(self.document_pool)} documents")
        for domain, weight in self.sampling_weights.items():
            logger.info(f"  {domain}: weight = {weight:.3f}")
    
    def sample_batch(self, batch_size: int) -> List[Dict]:
        """Sample batch using Alpha sampling."""
        # Sample documents according to their weights
        weights = [doc['sampling_weight'] for doc in self.document_pool]
        
        # Convert to numpy for efficient sampling
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        # Sample indices
        indices = np.random.choice(
            len(self.document_pool), 
            size=min(batch_size, len(self.document_pool)),
            replace=True,
            p=weights
        )
        
        batch = [self.document_pool[i] for i in indices]
        return batch

class PreferenceSampler(DataSampler):
    """
    Preference-based sampling with explicit domain preferences.
    Generally underperforms according to research.
    """
    
    def __init__(self, config: SamplingConfig, domain_preferences: Dict[str, float]):
        super().__init__(config)
        self.domain_preferences = domain_preferences
        self.document_pool = []
    
    def _compute_sampling_weights(self) -> Dict[str, float]:
        """Use explicit domain preferences."""
        # Normalize preferences
        total_pref = sum(self.domain_preferences.values())
        normalized_prefs = {domain: pref / total_pref 
                           for domain, pref in self.domain_preferences.items()}
        
        # Only include domains that exist in data
        weights = {}
        for domain in self.domain_stats.keys():
            weights[domain] = normalized_prefs.get(domain, 0.1)  # Default small weight
        
        return weights
    
    def fit(self, documents: List[Dict]):
        """Fit Preference sampler."""
        super().fit(documents)
        
        # Create weighted document pool similar to Alpha sampler
        self.document_pool = []
        for domain, stats in self.domain_stats.items():
            weight = self.sampling_weights[domain]
            domain_docs = stats['documents']
            
            for doc in domain_docs:
                doc['sampling_weight'] = weight
                self.document_pool.append(doc)
        
        random.shuffle(self.document_pool)
    
    def sample_batch(self, batch_size: int) -> List[Dict]:
        """Sample batch using preference weights."""
        weights = [doc['sampling_weight'] for doc in self.document_pool]
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        indices = np.random.choice(
            len(self.document_pool),
            size=min(batch_size, len(self.document_pool)),
            replace=True,
            p=weights
        )
        
        batch = [self.document_pool[i] for i in indices]
        return batch

class AttributeBasedSampler(DataSampler):
    """
    Attribute-based sampling using quality and other attributes.
    Creates fine-grained buckets based on document attributes.
    """
    
    def __init__(self, config: SamplingConfig, 
                 quality_weights: Dict[str, float] = None,
                 attribute_sampling: str = 'fine_grained'):
        super().__init__(config)
        self.quality_weights = quality_weights or {'high': 2.0, 'medium': 1.0, 'low': 0.5}
        self.attribute_sampling = attribute_sampling  # 'fine_grained' or 'grouped'
        self.attribute_buckets = {}
    
    def _create_attribute_buckets(self, documents: List[Dict]):
        """Create buckets based on document attributes."""
        buckets = defaultdict(list)
        
        if self.attribute_sampling == 'fine_grained':
            # Create buckets for each (domain, quality) combination
            for doc in documents:
                domain = doc.get('domain', 'unknown')
                
                if hasattr(doc, 'quality_class'):
                    quality = doc.quality_class
                else:
                    quality = doc.get('quality', 'medium')
                
                bucket_key = f"{domain}_{quality}"
                buckets[bucket_key].append(doc)
        
        else:  # grouped
            # Create buckets across entire corpus by quality
            for doc in documents:
                if hasattr(doc, 'quality_class'):
                    quality = doc.quality_class
                else:
                    quality = doc.get('quality', 'medium')
                
                buckets[quality].append(doc)
        
        return dict(buckets)
    
    def _compute_sampling_weights(self) -> Dict[str, float]:
        """Compute weights for attribute buckets."""
        weights = {}
        
        for bucket_key, docs in self.attribute_buckets.items():
            if self.attribute_sampling == 'fine_grained':
                # Extract quality from bucket key
                quality = bucket_key.split('_')[-1]
            else:
                # Bucket key is the quality
                quality = bucket_key
            
            # Base weight from quality
            base_weight = self.quality_weights.get(quality, 1.0)
            
            # Scale by bucket size (optional)
            size_factor = len(docs) / sum(len(b) for b in self.attribute_buckets.values())
            
            weights[bucket_key] = base_weight * size_factor
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def fit(self, documents: List[Dict]):
        """Fit attribute-based sampler."""
        # Create attribute buckets
        self.attribute_buckets = self._create_attribute_buckets(documents)
        
        # Compute domain stats from buckets
        self.domain_stats = {bucket: {'documents': docs, 'count': len(docs)} 
                           for bucket, docs in self.attribute_buckets.items()}
        
        # Compute sampling weights
        self.sampling_weights = self._compute_sampling_weights()
        
        logger.info(f"Attribute-based sampler created {len(self.attribute_buckets)} buckets")
        for bucket, weight in self.sampling_weights.items():
            count = len(self.attribute_buckets[bucket])
            logger.info(f"  {bucket}: {count} docs, weight = {weight:.3f}")
    
    def sample_batch(self, batch_size: int) -> List[Dict]:
        """Sample batch using attribute-based weights."""
        batch = []
        bucket_keys = list(self.attribute_buckets.keys())
        bucket_weights = [self.sampling_weights[key] for key in bucket_keys]
        
        for _ in range(batch_size):
            # Sample bucket according to weights
            bucket_idx = np.random.choice(len(bucket_keys), p=bucket_weights)
            bucket_key = bucket_keys[bucket_idx]
            
            # Sample document from selected bucket
            bucket_docs = self.attribute_buckets[bucket_key]
            doc = random.choice(bucket_docs)
            
            batch.append(doc)
        
        return batch

def create_sampler(config: SamplingConfig, domain: str = None, **kwargs) -> DataSampler:
    """Factory function to create appropriate sampler based on domain and config."""
    
    # Domain-specific recommendations from research
    if domain == 'english' or config.method == 'unimax':
        return UniMaxSampler(config)
    
    elif domain == 'code' or config.method == 'alpha':
        return AlphaSampler(config)
    
    elif config.method == 'preference':
        domain_prefs = kwargs.get('domain_preferences', {})
        return PreferenceSampler(config, domain_prefs)
    
    elif config.method == 'attribute':
        quality_weights = kwargs.get('quality_weights')
        attribute_sampling = kwargs.get('attribute_sampling', 'fine_grained')
        return AttributeBasedSampler(config, quality_weights, attribute_sampling)
    
    else:
        # Default to UniMax
        return UniMaxSampler(config)

def main():
    """Test sampling strategies."""
    # Create test documents
    test_docs = [
        {'text': 'Web document 1', 'domain': 'web', 'quality': 'high'},
        {'text': 'Web document 2', 'domain': 'web', 'quality': 'medium'},
        {'text': 'Code document 1', 'domain': 'code', 'quality': 'high'},
        {'text': 'Code document 2', 'domain': 'code', 'quality': 'low'},
        {'text': 'Book document 1', 'domain': 'books', 'quality': 'high'},
    ]
    
    # Test UniMax sampling
    config = SamplingConfig(method='unimax', max_epochs=1)
    sampler = create_sampler(config, domain='english')
    sampler.fit(test_docs)
    
    batch = sampler.sample_batch(10)
    print(f"UniMax sampled {len(batch)} documents:")
    domain_counts = Counter(doc['domain'] for doc in batch)
    print(f"  Domain distribution: {dict(domain_counts)}")
    
    # Test Alpha sampling
    config = SamplingConfig(method='alpha', alpha=1.3)
    sampler = create_sampler(config, domain='code')
    sampler.fit(test_docs)
    
    batch = sampler.sample_batch(10)
    print(f"Alpha sampled {len(batch)} documents:")
    domain_counts = Counter(doc['domain'] for doc in batch)
    print(f"  Domain distribution: {dict(domain_counts)}")

if __name__ == "__main__":
    main()