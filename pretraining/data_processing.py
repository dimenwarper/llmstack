"""
Data processing pipeline for pretraining.
Implements deduplication, quality filtering, and multi-attribute classification
based on research findings from Eldan et al. (2024) and Longpre et al. (2023).
"""

import hashlib
import json
import re
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from difflib import SequenceMatcher
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessedDocument:
    """Document after processing pipeline."""
    text: str
    domain: str
    quality_score: float
    quality_class: str  # high, medium, low
    toxicity_score: float
    hash_id: str
    is_duplicate: bool
    metadata: Dict[str, Any]

class ExactDeduplicator:
    """Exact deduplication using hash-based matching."""
    
    def __init__(self, hash_type: str = "md5"):
        self.hash_type = hash_type
        self.seen_hashes: Set[str] = set()
        self.hash_to_doc: Dict[str, Dict] = {}
    
    def _compute_hash(self, text: str) -> str:
        """Compute hash for text."""
        if self.hash_type == "md5":
            return hashlib.md5(text.encode('utf-8')).hexdigest()
        elif self.hash_type == "sha256":
            return hashlib.sha256(text.encode('utf-8')).hexdigest()
        else:
            # Simple hash for testing
            return str(hash(text))
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """Remove exact duplicates, keeping older documents."""
        processed = []
        
        for doc in documents:
            text_hash = self._compute_hash(doc['text'])
            doc['hash_id'] = text_hash
            
            if text_hash in self.seen_hashes:
                # Duplicate found
                existing_doc = self.hash_to_doc[text_hash]
                # Keep older document (lower index assumed to be older)
                doc['is_duplicate'] = True
                logger.debug(f"Duplicate found: {text_hash[:8]}...")
            else:
                # New document
                self.seen_hashes.add(text_hash)
                self.hash_to_doc[text_hash] = doc
                doc['is_duplicate'] = False
                processed.append(doc)
        
        logger.info(f"Exact deduplication: {len(documents)} -> {len(processed)} documents")
        return processed

class FuzzyDeduplicator:
    """Fuzzy deduplication using similarity matching."""
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.processed_docs: List[Dict] = []
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        # Simple character-level similarity
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _compute_ngram_signature(self, text: str, n: int = 3) -> Set[str]:
        """Compute n-gram signature for faster similarity computation."""
        words = text.lower().split()
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.add(ngram)
        return ngrams
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """Remove fuzzy duplicates."""
        processed = []
        
        for doc in documents:
            text = doc['text']
            is_duplicate = False
            
            # Check against already processed documents
            for existing_doc in processed:
                similarity = self._compute_similarity(text, existing_doc['text'])
                if similarity >= self.threshold:
                    is_duplicate = True
                    doc['is_duplicate'] = True
                    doc['similar_to'] = existing_doc['hash_id']
                    logger.debug(f"Fuzzy duplicate found (similarity: {similarity:.3f})")
                    break
            
            if not is_duplicate:
                doc['is_duplicate'] = False
                processed.append(doc)
        
        logger.info(f"Fuzzy deduplication: {len(documents)} -> {len(processed)} documents")
        return processed

class QualityFilter:
    """Quality filtering based on research findings."""
    
    def __init__(self, config):
        self.config = config
        
    def _compute_perplexity_score(self, text: str) -> float:
        """Simplified perplexity estimation (placeholder for KenLM)."""
        # Simple heuristic based on word frequency patterns
        words = text.split()
        if not words:
            return float('inf')
        
        # Basic indicators of text quality
        avg_word_length = sum(len(word) for word in words) / len(words)
        unique_ratio = len(set(words)) / len(words)
        
        # Lower score = better quality (inverse of perplexity)
        score = 1000 * (1 / (avg_word_length * unique_ratio + 0.1))
        return min(score, self.config.perplexity_threshold * 2)
    
    def _apply_heuristic_filters(self, text: str, domain: str) -> Dict[str, Any]:
        """Apply heuristic filters based on research findings."""
        filters = {}
        
        # Basic text statistics
        char_count = len(text)
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return {"passes": False, "reason": "empty_text"}
        
        # Document length filter
        filters['length_ok'] = self.config.min_doc_length <= word_count <= self.config.max_doc_length
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / word_count
        filters['word_length_ok'] = self.config.min_word_length <= avg_word_length <= self.config.max_word_length
        
        # Character-level filters
        non_alphanumeric = sum(1 for c in text if not c.isalnum() and not c.isspace())
        non_alpha_ratio = non_alphanumeric / char_count if char_count > 0 else 1
        filters['non_alpha_ok'] = non_alpha_ratio <= self.config.max_non_alphanumeric_ratio
        
        # Number ratio
        number_chars = sum(1 for c in text if c.isdigit())
        number_ratio = number_chars / char_count if char_count > 0 else 0
        filters['number_ratio_ok'] = number_ratio <= self.config.max_number_ratio
        
        # URL ratio (simple heuristic)
        url_count = len(re.findall(r'http[s]?://\S+', text))
        url_ratio = url_count / word_count if word_count > 0 else 0
        filters['url_ratio_ok'] = url_ratio <= self.config.max_url_ratio
        
        # Duplicate lines
        lines = text.split('\n')
        unique_lines = len(set(lines))
        duplicate_ratio = 1 - (unique_lines / len(lines)) if lines else 0
        filters['duplicate_lines_ok'] = duplicate_ratio <= self.config.max_duplicate_lines_ratio
        
        # Domain-specific filters
        if domain == "code":
            filters.update(self._apply_code_filters(text))
        
        # Overall pass/fail
        filters['passes'] = all(filters.values())
        
        return filters
    
    def _apply_code_filters(self, text: str) -> Dict[str, bool]:
        """Apply code-specific filters."""
        lines = text.split('\n')
        code_lines = [line for line in lines if line.strip()]
        
        if not code_lines:
            return {'code_length_ok': False, 'code_structure_ok': False}
        
        # Comment ratio
        comment_lines = sum(1 for line in code_lines 
                          if line.strip().startswith('#') or 
                             line.strip().startswith('//') or 
                             '/*' in line)
        comment_ratio = comment_lines / len(code_lines) if code_lines else 0
        
        # Lines of code filter
        loc_ok = self.config.code_min_lines <= len(code_lines) <= self.config.code_max_lines
        
        # Comment ratio filter
        comment_ok = self.config.code_min_comment_ratio <= comment_ratio <= self.config.code_max_comment_ratio
        
        return {
            'code_length_ok': loc_ok,
            'code_comment_ok': comment_ok
        }
    
    def process_documents(self, documents: List[Dict]) -> List[ProcessedDocument]:
        """Apply quality filtering to documents."""
        processed = []
        
        for doc in documents:
            text = doc['text']
            domain = doc['domain']
            
            # Compute perplexity score
            perplexity = self._compute_perplexity_score(text)
            
            # Apply heuristic filters
            filter_results = self._apply_heuristic_filters(text, domain)
            
            # Determine quality class
            if perplexity <= self.config.perplexity_threshold and filter_results['passes']:
                if perplexity <= self.config.perplexity_threshold * 0.3:
                    quality_class = "high"
                elif perplexity <= self.config.perplexity_threshold * 0.7:
                    quality_class = "medium"
                else:
                    quality_class = "low"
            else:
                quality_class = "low"
            
            # Normalize quality score (0-1, higher is better)
            quality_score = max(0, 1 - (perplexity / (self.config.perplexity_threshold * 2)))
            
            # Simple toxicity placeholder (would use real classifier)
            toxicity_score = self._estimate_toxicity(text)
            
            processed_doc = ProcessedDocument(
                text=text,
                domain=domain,
                quality_score=quality_score,
                quality_class=quality_class,
                toxicity_score=toxicity_score,
                hash_id=doc.get('hash_id', ''),
                is_duplicate=doc.get('is_duplicate', False),
                metadata={
                    **doc.get('metadata', {}),
                    'perplexity': perplexity,
                    'filter_results': filter_results,
                    'original_quality': doc.get('quality', 'unknown')
                }
            )
            
            processed.append(processed_doc)
        
        logger.info(f"Quality filtering processed {len(processed)} documents")
        return processed
    
    def _estimate_toxicity(self, text: str) -> float:
        """Simple toxicity estimation (placeholder for real classifier)."""
        # Simple heuristic based on text patterns
        toxic_indicators = [
            'hate', 'violence', 'threat', 'attack', 'kill', 'die', 'stupid', 'idiot',
            'racist', 'sexist', 'discrimination', 'harassment'
        ]
        
        text_lower = text.lower()
        toxic_count = sum(1 for indicator in toxic_indicators if indicator in text_lower)
        
        # Normalize to 0-1 score
        return min(toxic_count / 10.0, 1.0)

class DataProcessor:
    """Main data processing pipeline."""
    
    def __init__(self, config):
        self.config = config
        self.exact_dedup = ExactDeduplicator(config.deduplication.hash_type)
        self.fuzzy_dedup = FuzzyDeduplicator(config.deduplication.fuzzy_threshold)
        self.quality_filter = QualityFilter(config.quality_filter)
    
    def process_dataset(self, raw_documents: List[Dict]) -> List[ProcessedDocument]:
        """Run complete processing pipeline."""
        logger.info(f"Starting data processing pipeline with {len(raw_documents)} documents")
        
        # Step 1: Exact deduplication
        if self.config.deduplication.exact_dedup:
            documents = self.exact_dedup.process_documents(raw_documents)
        else:
            documents = raw_documents
        
        # Step 2: Fuzzy deduplication  
        if self.config.deduplication.fuzzy_dedup:
            documents = self.fuzzy_dedup.process_documents(documents)
        
        # Step 3: Quality filtering
        processed_documents = self.quality_filter.process_documents(documents)
        
        # Statistics
        self._print_statistics(raw_documents, processed_documents)
        
        return processed_documents
    
    def _print_statistics(self, raw_docs: List[Dict], processed_docs: List[ProcessedDocument]):
        """Print processing statistics."""
        logger.info(f"Processing complete: {len(raw_docs)} -> {len(processed_docs)} documents")
        
        # Quality distribution
        quality_counts = defaultdict(int)
        domain_counts = defaultdict(int)
        
        for doc in processed_docs:
            quality_counts[doc.quality_class] += 1
            domain_counts[doc.domain] += 1
        
        logger.info(f"Quality distribution: {dict(quality_counts)}")
        logger.info(f"Domain distribution: {dict(domain_counts)}")
        
        # Average scores
        avg_quality = np.mean([doc.quality_score for doc in processed_docs])
        avg_toxicity = np.mean([doc.toxicity_score for doc in processed_docs])
        
        logger.info(f"Average quality score: {avg_quality:.3f}")
        logger.info(f"Average toxicity score: {avg_toxicity:.3f}")

def main():
    """Test data processing pipeline."""
    from .config import get_toy_dataset_config
    
    config = get_toy_dataset_config()
    processor = DataProcessor(config)
    
    # Load test documents
    test_docs = [
        {
            'text': 'This is a high-quality document with proper grammar and structure. It contains meaningful content that would be useful for training.',
            'domain': 'web',
            'quality': 'high',
            'metadata': {}
        },
        {
            'text': 'bad txt no grammar much wrong',
            'domain': 'web', 
            'quality': 'low',
            'metadata': {}
        },
        {
            'text': 'This is a high-quality document with proper grammar and structure. It contains meaningful content that would be useful for training.',  # Duplicate
            'domain': 'web',
            'quality': 'high', 
            'metadata': {}
        }
    ]
    
    processed = processor.process_dataset(test_docs)
    
    for doc in processed:
        print(f"Text: {doc.text[:50]}...")
        print(f"Quality: {doc.quality_class} ({doc.quality_score:.3f})")
        print(f"Toxicity: {doc.toxicity_score:.3f}")
        print(f"Duplicate: {doc.is_duplicate}")
        print("---")

if __name__ == "__main__":
    main()