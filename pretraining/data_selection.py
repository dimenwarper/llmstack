"""
Data selection using Domain Selection via Importance Resampling (DSIR).
Based on findings from Eldan et al. (2024).
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class SelectionTarget:
    """Target distribution for DSIR selection."""
    name: str
    documents: List[str]  # Sample documents representing target distribution
    weights: Dict[str, float]  # Domain weights

class DSIRSelector:
    """Domain Selection via Importance Resampling implementation."""
    
    def __init__(self, selection_rate: float = 0.95, max_features: int = 10000):
        self.selection_rate = selection_rate
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.target_distribution = None
        self.source_distributions = {}
    
    def _create_target_distribution(self, target_docs: List[str]) -> np.ndarray:
        """Create target distribution from high-quality documents."""
        # Fit vectorizer on target documents
        target_vectors = self.vectorizer.fit_transform(target_docs)
        
        # Compute mean target distribution
        target_distribution = np.mean(target_vectors.toarray(), axis=0)
        
        # Normalize to probability distribution
        target_distribution = target_distribution / np.sum(target_distribution)
        
        return target_distribution
    
    def _compute_source_distribution(self, source_docs: List[str]) -> np.ndarray:
        """Compute source distribution for a set of documents."""
        if not source_docs:
            return np.zeros(self.max_features)
        
        # Transform documents using fitted vectorizer
        source_vectors = self.vectorizer.transform(source_docs)
        
        # Compute mean source distribution
        source_distribution = np.mean(source_vectors.toarray(), axis=0)
        
        # Normalize to probability distribution
        source_distribution = source_distribution / (np.sum(source_distribution) + 1e-10)
        
        return source_distribution
    
    def _compute_importance_weights(self, source_distribution: np.ndarray) -> np.ndarray:
        """Compute importance weights for resampling."""
        if self.target_distribution is None:
            raise ValueError("Target distribution not set. Call fit() first.")
        
        # Compute importance weights: p_target / p_source
        weights = self.target_distribution / (source_distribution + 1e-10)
        
        # Clip extreme weights to prevent instability
        weights = np.clip(weights, 0.1, 10.0)
        
        return weights
    
    def _compute_document_scores(self, documents: List[str], importance_weights: np.ndarray) -> List[float]:
        """Compute selection scores for individual documents."""
        if not documents:
            return []
        
        # Transform documents
        doc_vectors = self.vectorizer.transform(documents)
        
        # Compute weighted scores for each document
        scores = []
        for i in range(doc_vectors.shape[0]):
            doc_vector = doc_vectors[i].toarray().flatten()
            # Weighted sum of features
            score = np.sum(doc_vector * importance_weights)
            scores.append(score)
        
        return scores
    
    def fit(self, target_docs: List[str]):
        """Fit DSIR selector on target distribution."""
        logger.info(f"Fitting DSIR selector on {len(target_docs)} target documents")
        
        # Create target distribution
        self.target_distribution = self._create_target_distribution(target_docs)
        
        logger.info("DSIR selector fitted successfully")
    
    def select_documents(self, source_docs: List[Dict], source_name: str) -> List[Dict]:
        """Select documents from source using DSIR."""
        if self.target_distribution is None:
            raise ValueError("Must fit selector first")
        
        logger.info(f"Selecting documents from {source_name} ({len(source_docs)} documents)")
        
        # Extract text from documents
        texts = [doc['text'] for doc in source_docs]
        
        # Compute source distribution
        source_distribution = self._compute_source_distribution(texts)
        
        # Compute importance weights
        importance_weights = self._compute_importance_weights(source_distribution)
        
        # Compute document scores
        scores = self._compute_document_scores(texts, importance_weights)
        
        # Select top documents based on scores
        num_select = int(len(source_docs) * self.selection_rate)
        
        # Sort by score and select top documents
        doc_scores = list(zip(source_docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_docs = [doc for doc, score in doc_scores[:num_select]]
        
        # Add selection metadata
        for i, (doc, score) in enumerate(doc_scores[:num_select]):
            doc['selection_score'] = score
            doc['selection_rank'] = i
            doc['selected_by'] = 'dsir'
        
        logger.info(f"Selected {len(selected_docs)} documents from {source_name}")
        
        return selected_docs

class AttributeBasedSelector:
    """Attribute-based selection using quality and domain information."""
    
    def __init__(self, quality_weights: Dict[str, float] = None, 
                 domain_weights: Dict[str, float] = None):
        self.quality_weights = quality_weights or {'high': 1.0, 'medium': 0.7, 'low': 0.3}
        self.domain_weights = domain_weights or {}
    
    def select_documents(self, documents: List[Dict], 
                        selection_rate: float = 0.95) -> List[Dict]:
        """Select documents based on quality and domain attributes."""
        logger.info(f"Attribute-based selection from {len(documents)} documents")
        
        # Compute selection scores
        scored_docs = []
        for doc in documents:
            score = self._compute_attribute_score(doc)
            scored_docs.append((doc, score))
        
        # Sort by score and select top documents
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        num_select = int(len(documents) * selection_rate)
        
        selected_docs = []
        for i, (doc, score) in enumerate(scored_docs[:num_select]):
            doc['selection_score'] = score
            doc['selection_rank'] = i
            doc['selected_by'] = 'attribute'
            selected_docs.append(doc)
        
        logger.info(f"Selected {len(selected_docs)} documents via attribute-based selection")
        
        return selected_docs
    
    def _compute_attribute_score(self, doc: Dict) -> float:
        """Compute selection score based on document attributes."""
        score = 0.0
        
        # Quality component
        if hasattr(doc, 'quality_class'):
            quality = doc.quality_class
        else:
            quality = doc.get('quality', 'medium')
        
        quality_score = self.quality_weights.get(quality, 0.5)
        score += quality_score
        
        # Domain component
        domain = doc.get('domain', 'unknown')
        domain_score = self.domain_weights.get(domain, 0.5)
        score += domain_score
        
        # Additional scoring factors
        if hasattr(doc, 'quality_score'):
            score += doc.quality_score
        
        if hasattr(doc, 'toxicity_score'):
            # Lower toxicity is better
            score += (1.0 - doc.toxicity_score)
        
        return score

class HybridSelector:
    """Hybrid selector combining DSIR and attribute-based selection."""
    
    def __init__(self, dsir_weight: float = 0.6, attribute_weight: float = 0.4):
        self.dsir_weight = dsir_weight
        self.attribute_weight = attribute_weight
        self.dsir_selector = None
        self.attribute_selector = None
    
    def fit(self, target_docs: List[str], 
            quality_weights: Dict[str, float] = None,
            domain_weights: Dict[str, float] = None):
        """Fit hybrid selector."""
        # Initialize DSIR selector
        self.dsir_selector = DSIRSelector()
        self.dsir_selector.fit(target_docs)
        
        # Initialize attribute selector
        self.attribute_selector = AttributeBasedSelector(quality_weights, domain_weights)
        
        logger.info("Hybrid selector fitted successfully")
    
    def select_documents(self, documents: List[Dict], 
                        selection_rate: float = 0.95) -> List[Dict]:
        """Select documents using hybrid approach."""
        if self.dsir_selector is None or self.attribute_selector is None:
            raise ValueError("Must fit selector first")
        
        logger.info(f"Hybrid selection from {len(documents)} documents")
        
        # Get DSIR scores
        texts = [doc['text'] for doc in documents]
        source_distribution = self.dsir_selector._compute_source_distribution(texts)
        importance_weights = self.dsir_selector._compute_importance_weights(source_distribution)
        dsir_scores = self.dsir_selector._compute_document_scores(texts, importance_weights)
        
        # Get attribute scores
        attribute_scores = [self.attribute_selector._compute_attribute_score(doc) for doc in documents]
        
        # Normalize scores
        dsir_scores = np.array(dsir_scores)
        attribute_scores = np.array(attribute_scores)
        
        dsir_scores = (dsir_scores - np.min(dsir_scores)) / (np.max(dsir_scores) - np.min(dsir_scores) + 1e-10)
        attribute_scores = (attribute_scores - np.min(attribute_scores)) / (np.max(attribute_scores) - np.min(attribute_scores) + 1e-10)
        
        # Combine scores
        combined_scores = (self.dsir_weight * dsir_scores + 
                          self.attribute_weight * attribute_scores)
        
        # Select top documents
        num_select = int(len(documents) * selection_rate)
        top_indices = np.argsort(combined_scores)[::-1][:num_select]
        
        selected_docs = []
        for i, idx in enumerate(top_indices):
            doc = documents[idx]
            doc['selection_score'] = combined_scores[idx]
            doc['dsir_score'] = dsir_scores[idx]
            doc['attribute_score'] = attribute_scores[idx]
            doc['selection_rank'] = i
            doc['selected_by'] = 'hybrid'
            selected_docs.append(doc)
        
        logger.info(f"Selected {len(selected_docs)} documents via hybrid selection")
        
        return selected_docs

def create_target_distribution(processed_docs: List[Dict], 
                             target_quality: str = 'high') -> List[str]:
    """Create target distribution from high-quality documents."""
    target_docs = []
    
    for doc in processed_docs:
        # Get quality from processed document or original
        if hasattr(doc, 'quality_class'):
            quality = doc.quality_class
        else:
            quality = doc.get('quality', 'medium')
        
        if quality == target_quality:
            if hasattr(doc, 'text'):
                target_docs.append(doc.text)
            else:
                target_docs.append(doc['text'])
    
    # If not enough high-quality docs, include medium quality
    if len(target_docs) < 10:
        logger.warning(f"Only {len(target_docs)} {target_quality} quality docs found, including medium quality")
        for doc in processed_docs:
            if hasattr(doc, 'quality_class'):
                quality = doc.quality_class
            else:
                quality = doc.get('quality', 'medium')
            
            if quality == 'medium':
                if hasattr(doc, 'text'):
                    target_docs.append(doc.text)
                else:
                    target_docs.append(doc['text'])
    
    return target_docs

def main():
    """Test data selection."""
    # Create test documents
    test_docs = [
        {'text': 'High-quality document with excellent content and proper structure.', 
         'quality': 'high', 'domain': 'web'},
        {'text': 'Medium quality document with decent content.', 
         'quality': 'medium', 'domain': 'web'},
        {'text': 'Low quality bad grammar txt.', 
         'quality': 'low', 'domain': 'web'},
        {'text': 'Another high-quality document with comprehensive information.', 
         'quality': 'high', 'domain': 'books'},
    ]
    
    # Create target distribution
    target_docs = create_target_distribution(test_docs, 'high')
    
    # Test DSIR selector
    dsir_selector = DSIRSelector()
    dsir_selector.fit(target_docs)
    selected = dsir_selector.select_documents(test_docs, 'test_source')
    
    print(f"DSIR selected {len(selected)} documents")
    for doc in selected:
        print(f"  Score: {doc['selection_score']:.3f}, Quality: {doc['quality']}")

if __name__ == "__main__":
    main()