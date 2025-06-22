import tomli
import modal
from typing import Union
from pathlib import PurePosixPath
import subprocess

def get_python_modules():
    """Get all top-level Python module directories in current directory"""
    result = subprocess.run(
        ["find", ".", "-maxdepth", "1", "-type", "d"],
        capture_output=True,
        text=True
    )
    dirs = result.stdout.strip().split("\n")
    python_modules = [
        d[2:] for d in dirs  # Remove "./" prefix
        if d != "." and 
        (d.startswith("./") and not d.startswith("./.")) and  # Exclude hidden dirs
        (subprocess.run(
            ["test", "-f", f"{d}/__init__.py"],
            capture_output=True
        ).returncode == 0)  # Check for __init__.py
    ]
    return python_modules

python_module_dirs = get_python_modules()


def create_modal_image_from_pyproject(pyproject_path="pyproject.toml"):
    """
    Parse a pyproject.toml file and create a Modal image with the dependencies.
    
    Args:
        pyproject_path (str): Path to the pyproject.toml file
        
    Returns:
        tuple: (modal.Image, modal.App) The configured Modal image and app
    """
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomli.load(f)
    
    project_info = pyproject_data.get("project", {})
    project_name = project_info.get("name", "default_app")
    dependencies = project_info.get("dependencies", [])
    
    app = modal.App(project_name)
    
    image = (modal.Image.debian_slim()
             .pip_install("uv")
    )
    
    nl = "\n"
    print(f"===Deps to install===\n*{(nl+'*').join(dependencies)}")
    if dependencies:
        image = image.pip_install(*dependencies)

    image = (
        image
        .env(
            dict(
                HUGGINGFACE_HUB_CACHE="/pretrained",
                HF_HUB_ENABLE_HF_TRANSFER="1",
                TQDM_DISABLE="true",
            )
        )
        .entrypoint([])
        .add_local_file(pyproject_path, remote_path="/root/" + pyproject_path)
    )

    print(f"\n===Adding local packages===\n*{(nl+'*').join(python_module_dirs)}")
    
    for pm in python_module_dirs:
        image = image.add_local_python_source(pm)
    
    print(f"Created Modal image for '{project_name}' with {len(dependencies)} dependencies")
    
    return image, app

image, app = create_modal_image_from_pyproject()

# Volumes for models/data 
artifact_volume = modal.Volume.from_name(
    "artifact-vol", create_if_missing=True
)
VOLUME_CONFIG: dict[Union[str, PurePosixPath], modal.Volume] = {
    "/artifacts/": artifact_volume,
}

@app.function(image=image)
def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr

@app.local_entrypoint()
def main(command: str, local: str):
    if local == "true":
        _stdout, _stderr = run_command.local(command)
    else:
        _stdout, _stderr = run_command.remote(command)
    print("\n=== STDOUT ===")
    print(_stdout)
    print("\n=== STDERR ===") 
    print(_stderr)