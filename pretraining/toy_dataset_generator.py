"""
Download and prepare toy dataset for testing pretraining pipeline.
Downloads subsets from HuggingFace datasets based on research paper recommendations.

Datasets based on CONTEXT.md research findings:
- Web: C4 (Common Crawl)
- Books: BookCorpus, Project Gutenberg
- Code: The Stack v1.2 
- News: CC-News
- Academic: ArXiv, PubMed abstracts
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datasets import load_dataset
import random

@dataclass
class Document:
    """Represents a document in the dataset."""
    text: str
    domain: str
    quality: str  # high, medium, low
    source: str
    metadata: Dict[str, Any]

class ToyDatasetDownloader:
    """Downloads real datasets from HuggingFace for testing pipeline."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.dataset_configs = self._get_dataset_configs()
    
    def _get_dataset_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration for datasets to download based on research papers."""
        return {
            "web": {
                "dataset": "c4",
                "config": "en",
                "split": "train",
                "text_column": "text",
                "sample_size": 400,
                "domain": "web"
            },
            "books": {
                "dataset": "bookcorpus",
                "config": None,
                "split": "train", 
                "text_column": "text",
                "sample_size": 100,
                "domain": "books"
            },
            "code": {
                "dataset": "bigcode/the-stack-dedup",
                "config": "data",
                "split": "train",
                "text_column": "content",
                "sample_size": 100,
                "domain": "code"
            },
            "news": {
                "dataset": "cc_news",
                "config": None,
                "split": "train",
                "text_column": "text", 
                "sample_size": 100,
                "domain": "news"
            },
            "academic": {
                "dataset": "scientific_papers",
                "config": "arxiv",
                "split": "train",
                "text_column": "abstract",
                "sample_size": 50,
                "domain": "academic"
            }
        }
    
    def _estimate_quality(self, text: str, domain: str) -> str:
        """Simple heuristic to estimate document quality."""
        # Basic quality indicators
        word_count = len(text.split())
        avg_word_length = sum(len(word) for word in text.split()) / max(word_count, 1)
        
        # Count non-alphanumeric characters
        non_alpha_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        
        # Count uppercase ratio
        upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Domain-specific quality assessment
        if domain == "code":
            # Code quality indicators
            has_comments = "#" in text or "//" in text or "/*" in text
            has_structure = any(keyword in text.lower() for keyword in ["def", "function", "class", "import", "return"])
            if has_comments and has_structure and word_count > 20:
                return "high"
            elif has_structure and word_count > 10:
                return "medium"
            else:
                return "low"
        
        elif domain == "academic":
            # Academic quality indicators
            has_citations = any(marker in text for marker in ["et al.", "Figure", "Table", "Abstract"])
            formal_language = avg_word_length > 5 and non_alpha_ratio < 0.1
            if has_citations and formal_language and word_count > 100:
                return "high"
            elif formal_language and word_count > 50:
                return "medium"
            else:
                return "low"
        
        else:
            # General quality indicators
            if (word_count > 100 and 
                avg_word_length > 4 and 
                non_alpha_ratio < 0.15 and 
                upper_ratio < 0.3):
                return "high"
            elif (word_count > 50 and 
                  avg_word_length > 3 and 
                  non_alpha_ratio < 0.25):
                return "medium"
            else:
                return "low"
    
    def download_domain_data(self, domain: str) -> List[Document]:
        """Download data for a specific domain."""
        config = self.dataset_configs[domain]
        
        print(f"Downloading {domain} data from {config['dataset']}...")
        
        try:
            # Load dataset
            if config['config']:
                dataset = load_dataset(config['dataset'], config['config'], split=config['split'], streaming=True, trust_remote_code=True)
            else:
                dataset = load_dataset(config['dataset'], split=config['split'], streaming=True, trust_remote_code=True)
            
            # Sample documents
            documents = []
            text_column = config['text_column']
            
            # Take first N samples and shuffle
            samples = []
            for i, item in enumerate(dataset):
                if i >= config['sample_size'] * 3:  # Get 3x to allow for filtering
                    break
                if text_column in item and item[text_column]:
                    samples.append(item)
            
            # Shuffle and take subset
            random.shuffle(samples)
            samples = samples[:config['sample_size']]
            
            for item in samples:
                text = item[text_column]
                
                # Skip very short or very long texts
                if len(text.split()) < 10 or len(text.split()) > 10000:
                    continue
                
                quality = self._estimate_quality(text, domain)
                
                metadata = {
                    "length": len(text),
                    "word_count": len(text.split()),
                    "original_dataset": config['dataset']
                }
                
                # Add domain-specific metadata
                if domain == "code" and "language" in item:
                    metadata["language"] = item["language"]
                elif domain == "academic" and "title" in item:
                    metadata["title"] = item["title"]
                
                doc = Document(
                    text=text,
                    domain=domain,
                    quality=quality,
                    source=config['dataset'],
                    metadata=metadata
                )
                documents.append(doc)
            
            print(f"Downloaded {len(documents)} {domain} documents")
            return documents
            
        except Exception as e:
            print(f"Error downloading {domain} data: {e}")
            print(f"Falling back to placeholder data for {domain}")
            return self._generate_placeholder_data(domain, config['sample_size'])
    
    def _generate_placeholder_data(self, domain: str, count: int) -> List[Document]:
        """Generate placeholder data if download fails."""
        documents = []
        
        placeholder_texts = {
            "web": "This is a sample web document about various topics including technology, science, and current events. It contains multiple paragraphs with different types of information.",
            "books": "Chapter 1: The Beginning. In a small town nestled between rolling hills and ancient forests, our story begins with a young protagonist who would soon discover that their ordinary life was about to become extraordinary.",
            "code": "def example_function(x, y):\n    \"\"\"\n    This is an example function that demonstrates basic Python syntax.\n    \"\"\"\n    result = x + y\n    return result\n\nif __name__ == '__main__':\n    print(example_function(1, 2))",
            "news": "Breaking News: Local community comes together to address important issues affecting residents. City officials announced new initiatives aimed at improving quality of life for all citizens.",
            "academic": "Abstract: This paper presents a comprehensive analysis of recent developments in the field. Our methodology involves systematic review of existing literature and empirical evaluation of proposed approaches. Results indicate significant improvements over baseline methods."
        }
        
        base_text = placeholder_texts.get(domain, "Sample text content for testing purposes.")
        
        for i in range(count):
            # Add variation
            text = f"{base_text} Document {i+1} with additional content for variation."
            quality = ["high", "medium", "low"][i % 3]
            
            doc = Document(
                text=text,
                domain=domain,
                quality=quality,
                source=f"placeholder_{domain}",
                metadata={"length": len(text), "word_count": len(text.split()), "placeholder": True}
            )
            documents.append(doc)
        
        return documents
    
    def download_all_data(self) -> Dict[str, List[Document]]:
        """Download data for all domains."""
        dataset = {}
        
        for domain in self.dataset_configs.keys():
            dataset[domain] = self.download_domain_data(domain)
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, List[Document]], output_dir: str):
        """Save dataset to JSONL files."""
        os.makedirs(output_dir, exist_ok=True)
        
        for domain, documents in dataset.items():
            output_path = os.path.join(output_dir, f"toy_{domain}.jsonl")
            with open(output_path, 'w', encoding='utf-8') as f:
                for doc in documents:
                    json_doc = {
                        "text": doc.text,
                        "domain": doc.domain,
                        "quality": doc.quality,
                        "source": doc.source,
                        "metadata": doc.metadata
                    }
                    f.write(json.dumps(json_doc, ensure_ascii=False) + '\n')
            
            print(f"Saved {len(documents)} {domain} documents to {output_path}")

def main():
    """Download toy dataset."""
    downloader = ToyDatasetDownloader()
    dataset = downloader.download_all_data()
    
    # Create output directory
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    downloader.save_dataset(dataset, output_dir)
    
    # Print statistics
    total_docs = sum(len(docs) for docs in dataset.values())
    print(f"\nDownloaded toy dataset with {total_docs} documents:")
    for domain, docs in dataset.items():
        quality_counts = {}
        for doc in docs:
            quality_counts[doc.quality] = quality_counts.get(doc.quality, 0) + 1
        print(f"  {domain}: {len(docs)} docs ({quality_counts})")

if __name__ == "__main__":
    main()