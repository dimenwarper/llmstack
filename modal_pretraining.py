"""
Modal-compatible pretraining pipeline.
Adapts the pretraining pipeline to run on Modal infrastructure.
"""

import modal
from modal_utils import image, app, VOLUME_CONFIG
from pretraining import PretrainingPipeline, get_toy_dataset_config

# Update config for Modal environment
@app.function(
    image=image,
    volumes=VOLUME_CONFIG,
    gpu="T4",  # Start with T4 for testing, can upgrade to A100
    memory=16384,  # 16GB RAM
    timeout=3600,  # 1 hour timeout
)
def run_pretraining_step(step: str = "full", config_overrides: dict = None, quick_validation: bool = True):
    """Run a pretraining pipeline step on Modal."""
    import os
    import json
    from pretraining import PretrainingPipeline, get_toy_dataset_config
    
    # Get base config
    config = get_toy_dataset_config()
    
    # Apply any overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                if hasattr(getattr(config, key), '__dict__'):
                    # Handle nested config objects
                    for subkey, subvalue in value.items():
                        setattr(getattr(config, key), subkey, subvalue)
                else:
                    setattr(config, key, value)
    
    # Update paths for Modal environment
    config.raw_data_dir = "/artifacts/data/raw"
    config.processed_data_dir = "/artifacts/data/processed"
    config.model_output_dir = "/artifacts/models/checkpoints"
    config.log_dir = "/artifacts/logs"
    
    # Reduce everything for fast validation
    config.training.max_steps = 20  # Very few steps for validation
    config.training.eval_steps = 10
    config.training.save_steps = 20
    config.training.batch_size = 2  # Smaller batch for T4
    
    # Make model even smaller for debugging
    config.model.hidden_size = 256
    config.model.num_layers = 4
    config.model.num_heads = 4
    config.model.intermediate_size = 1024
    config.model.max_seq_length = 512  # Much shorter sequences
    
    # Reduce dataset sizes for fast processing
    if quick_validation:
        for data_source in config.data_sources:
            if data_source.name == "toy_web":
                data_source.max_documents = 5   # Tiny for validation
            elif data_source.name == "toy_books":
                data_source.max_documents = 3   # Tiny for validation  
            elif data_source.name == "toy_code":
                data_source.max_documents = 3   # Tiny for validation
            elif data_source.name == "toy_news":
                data_source.max_documents = 3   # Tiny for validation
            elif data_source.name == "toy_academic":
                data_source.max_documents = 2   # Tiny for validation
        print("🚀 Quick validation mode: Using tiny dataset (~16 documents total)")
    else:
        for data_source in config.data_sources:
            if data_source.name == "toy_web":
                data_source.max_documents = 20  # Down from 400
            elif data_source.name == "toy_books":
                data_source.max_documents = 10  # Down from 100
            elif data_source.name == "toy_code":
                data_source.max_documents = 10  # Down from 100
            elif data_source.name == "toy_news":
                data_source.max_documents = 10  # Down from 100
            elif data_source.name == "toy_academic":
                data_source.max_documents = 5   # Down from 50
    
    print(f"Running pretraining step: {step}")
    print(f"Config: {config}")
    
    # Create and run pipeline
    pipeline = PretrainingPipeline(config)
    
    try:
        if step == "download":
            pipeline.step1_download_data()
            result = {"status": "success", "step": "download", "documents": len(pipeline.raw_documents)}
            
        elif step == "process":
            # Load raw documents if not already loaded
            if not pipeline.raw_documents:
                print("Raw documents not found, running download step first...")
                pipeline.step1_download_data()
            pipeline.step2_process_data()
            result = {"status": "success", "step": "process", "documents": len(pipeline.processed_documents)}
            
        elif step == "select":
            # Load processed documents if not already loaded
            if not pipeline.processed_documents:
                print("Processed documents not found, running previous steps...")
                pipeline.step1_download_data()
                pipeline.step2_process_data()
            pipeline.step3_select_data()
            result = {"status": "success", "step": "select", "documents": len(pipeline.selected_documents)}
            
        elif step == "sample":
            # Load selected documents if not already loaded  
            if not pipeline.selected_documents:
                print("Selected documents not found, running previous steps...")
                pipeline.step1_download_data()
                pipeline.step2_process_data()
                pipeline.step3_select_data()
            pipeline.step4_sample_data()
            result = {"status": "success", "step": "sample", "documents": len(pipeline.sampled_documents)}
            
        elif step == "train":
            # Load sampled documents if not already loaded
            if not pipeline.sampled_documents:
                print("Sampled documents not found, running previous steps...")
                pipeline.step1_download_data()
                pipeline.step2_process_data()
                pipeline.step3_select_data()
                pipeline.step4_sample_data()
            metrics = pipeline.step5_train_model()
            result = {"status": "success", "step": "train", "metrics": metrics.metrics}
            
        elif step == "full":
            metrics = pipeline.run_full_pipeline()
            result = {"status": "success", "step": "full", "metrics": metrics.metrics}
            
        else:
            result = {"status": "error", "message": f"Unknown step: {step}"}
        
        # Save result to artifacts
        result_path = f"/artifacts/results/{step}_result.json"
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Step {step} completed successfully!")
        print(f"Result: {result}")
        
        return result
        
    except Exception as e:
        import traceback
        full_traceback = traceback.format_exc()
        error_result = {"status": "error", "step": step, "error": str(e), "traceback": full_traceback}
        print(f"Error in step {step}: {e}")
        print(f"Full traceback:\n{full_traceback}")
        return error_result

@app.function(
    image=image,
    volumes=VOLUME_CONFIG,
    gpu="A100",  # Bigger GPU for full training
    memory=32768,  # 32GB RAM
    timeout=7200,  # 2 hour timeout
)
def run_full_pretraining(config_overrides: dict = None):
    """Run the complete pretraining pipeline with high-end resources."""
    
    # Override config for full training
    full_config_overrides = {
        "training": {
            "max_steps": 1000,
            "eval_steps": 100,
            "save_steps": 200,
            "batch_size": 16,
            "learning_rate": 2e-4
        }
    }
    
    if config_overrides:
        # Merge user overrides with full training overrides
        for key, value in config_overrides.items():
            if key in full_config_overrides:
                full_config_overrides[key].update(value)
            else:
                full_config_overrides[key] = value
    
    return run_pretraining_step.remote("full", full_config_overrides)

@app.function(image=image, volumes=VOLUME_CONFIG)
def list_artifacts():
    """List all artifacts in the volume."""
    import os
    
    artifacts = {}
    base_path = "/artifacts"
    
    if os.path.exists(base_path):
        for root, dirs, files in os.walk(base_path):
            rel_path = os.path.relpath(root, base_path)
            if rel_path == ".":
                rel_path = ""
            
            artifacts[rel_path] = {
                "directories": dirs,
                "files": files,
                "file_count": len(files)
            }
    
    return artifacts

@app.function(image=image, volumes=VOLUME_CONFIG)
def download_artifact(artifact_path: str):
    """Download an artifact file from the volume."""
    import os
    
    full_path = f"/artifacts/{artifact_path}"
    
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            return f.read()
    else:
        return f"Artifact not found: {artifact_path}"

@app.function(image=image, volumes=VOLUME_CONFIG)
def cleanup_artifacts():
    """Clean up old artifacts."""
    import shutil
    import os
    
    artifacts_path = "/artifacts"
    if os.path.exists(artifacts_path):
        shutil.rmtree(artifacts_path)
        os.makedirs(artifacts_path)
        return "Artifacts cleaned up successfully"
    else:
        return "No artifacts to clean up"

@app.local_entrypoint()
def run_pretraining(
    step: str = "full",
    gpu_tier: str = "t4",  # t4, a100
    config_file: str = None,
    list_files: bool = False,
    download: str = None,
    cleanup: bool = False,
    quick: bool = True  # Default to quick validation mode
):
    """
    Main entrypoint for Modal pretraining pipeline.
    
    Args:
        step: Pipeline step to run (download, process, select, sample, train, full)
        gpu_tier: GPU tier to use (t4, a100)  
        config_file: Path to custom config JSON file
        list_files: List artifacts in volume
        download: Download specific artifact file
        cleanup: Clean up artifacts
    """
    
    if cleanup:
        result = cleanup_artifacts.remote()
        print(result)
        return
    
    if list_files:
        artifacts = list_artifacts.remote()
        print("=== ARTIFACTS ===")
        for path, info in artifacts.items():
            print(f"{path or 'root'}: {info['file_count']} files")
            for file in info['files'][:5]:  # Show first 5 files
                print(f"  - {file}")
            if len(info['files']) > 5:
                print(f"  ... and {len(info['files']) - 5} more")
        return
    
    if download:
        content = download_artifact.remote(download)
        print(f"=== ARTIFACT: {download} ===")
        print(content)
        return
    
    # Load config overrides if provided
    config_overrides = None
    if config_file:
        import json
        with open(config_file, 'r') as f:
            config_overrides = json.load(f)
    
    # Run the appropriate function based on GPU tier
    if gpu_tier == "a100" and step == "full":
        result = run_full_pretraining.remote(config_overrides)
    else:
        result = run_pretraining_step.remote(step, config_overrides, quick)
    
    print(f"=== PRETRAINING RESULT ===")
    print(result)