#!/usr/bin/env python
import subprocess
import sys
from pathlib import Path

def run_command(command):
    """Run a shell command and print output"""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return result.returncode == 0

def main():
    """Install minimal dependencies for KenPom data collection"""
    print("\n=== Installing minimal dependencies for KenPom data collection ===\n")
    
    # Essential packages for KenPom collection
    essential_packages = [
        "pandas",
        "requests",
        "beautifulsoup4",
        "lxml",
        "pyyaml"
    ]
    
    # Use either UV or pip based on what's available
    if run_command(["which", "uv"]):
        # Create a Python 3.9 environment with UV
        print("\nCreating Python 3.9 environment with UV...")
        if not run_command(["uv", "venv", "--python", "3.9"]):
            print("Failed to create UV environment with Python 3.9")
            print("Trying with default Python version...")
            if not run_command(["uv", "venv"]):
                print("Failed to create UV environment")
                return False
                
        # Install essential packages with UV
        print("\nInstalling essential packages with UV...")
        install_cmd = ["uv", "pip", "install"] + essential_packages
        if not run_command(install_cmd):
            print("Failed to install packages with UV")
            return False
    else:
        # Fall back to standard venv + pip
        import venv
        
        print("\nCreating virtual environment with standard venv...")
        venv_path = Path(".venv")
        venv.create(venv_path, with_pip=True)
        
        # Determine pip path
        if sys.platform == 'win32':
            pip_path = venv_path / "Scripts" / "pip"
        else:
            pip_path = venv_path / "bin" / "pip"
        
        # Install with pip
        print("\nInstalling essential packages with pip...")
        install_cmd = [str(pip_path), "install"] + essential_packages
        if not run_command(install_cmd):
            print("Failed to install packages with pip")
            return False
    
    print("\n=== Installation Complete ===")
    print("\nTo activate the environment:")
    
    if sys.platform == 'win32':
        print(".venv\\Scripts\\activate")
    else:
        print("source .venv/bin/activate")
    
    print("\nTo run the KenPom collector:")
    print("python scripts/download_kenpom_data.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
