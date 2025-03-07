#!/usr/bin/env python
"""Generate requirements.txt from pyproject.toml"""
import tomli
import sys
from pathlib import Path

def generate_requirements():
    # Load pyproject.toml
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"
    
    with open(pyproject_path, "rb") as f:
        pyproject = tomli.load(f)
    
    # Extract dependencies
    dependencies = pyproject["project"]["dependencies"]
    dev_dependencies = pyproject["project"]["optional-dependencies"]["dev"]
    
    # Write requirements.txt
    requirements_path = project_root / "requirements.txt"
    with open(requirements_path, "w") as f:
        f.write("# Main dependencies\n")
        for dep in dependencies:
            f.write(f"{dep}\n")
        
        f.write("\n# Development dependencies\n")
        for dep in dev_dependencies:
            f.write(f"{dep}\n")
    
    print(f"Generated requirements.txt with {len(dependencies)} main dependencies and {len(dev_dependencies)} dev dependencies")

if __name__ == "__main__":
    generate_requirements()
