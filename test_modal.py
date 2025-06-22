"""
Test script for Modal pretraining pipeline.
Quick validation of the Modal setup.
"""

import modal
from modal_utils import image, app

@app.function(image=image, gpu="T4", memory=8192, timeout=300)
def test_imports():
    """Test that all imports work correctly in Modal environment."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.get_device_name()}")
        
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        
        import datasets
        print(f"Datasets version: {datasets.__version__}")
        
        from pretraining import get_toy_dataset_config
        config = get_toy_dataset_config()
        print(f"Config loaded successfully: {config.model.vocab_size} vocab size")
        
        return {"status": "success", "message": "All imports successful"}
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.function(image=image, timeout=300)
def test_data_download():
    """Test data download functionality."""
    try:
        from pretraining import ToyDatasetDownloader
        
        downloader = ToyDatasetDownloader()
        print("Testing data download for web domain...")
        
        # Try to download just a few documents for testing
        web_docs = downloader.download_domain_data("web")
        print(f"Downloaded {len(web_docs)} web documents")
        
        if web_docs:
            doc = web_docs[0]
            print(f"Sample document: {doc.text[:100]}...")
            
        return {"status": "success", "documents": len(web_docs)}
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.local_entrypoint()
def main(test: str = "imports"):
    """
    Run tests for Modal pretraining setup.
    
    Args:
        test: Type of test to run (imports, download)
    """
    
    if test == "imports":
        print("=== Testing Imports ===")
        result = test_imports.remote()
        
    elif test == "download":
        print("=== Testing Data Download ===")
        result = test_data_download.remote()
        
    else:
        print(f"Unknown test: {test}")
        return
    
    print(f"Test result: {result}")
    
    if result["status"] == "success":
        print("✅ Test passed!")
    else:
        print("❌ Test failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")