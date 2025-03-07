#!/usr/bin/env python
import os
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.credentials import CredentialsManager

def setup_credentials():
    """Interactive script to set up credentials"""
    print("\n=== Basketball Prediction System Credentials Setup ===\n")
    
    credentials_path = Path("credentials.yaml")
    
    # Load existing credentials or create template
    creds_manager = CredentialsManager(credentials_path)
    credentials = creds_manager.credentials
    
    # Helper function to get user input with default
    def get_input(prompt, default=""):
        if default:
            user_input = input(f"{prompt} [{default}]: ")
            return user_input if user_input else default
        else:
            return input(f"{prompt}: ")
    
    # KenPom credentials
    print("\n-- KenPom Credentials --")
    print("These are needed to scrape data from the KenPom website.")
    print("You need an active KenPom subscription.")
    
    kenpom = credentials.get('kenpom', {})
    kenpom_username = get_input("KenPom Username", kenpom.get('username', ''))
    kenpom_password = get_input("KenPom Password", kenpom.get('password', ''))
    
    # Kaggle credentials
    print("\n-- Kaggle Credentials --")
    print("These are needed to download competition data via the Kaggle API.")
    print("You can find these in your Kaggle account settings.")
    
    kaggle = credentials.get('kaggle', {})
    kaggle_username = get_input("Kaggle Username", kaggle.get('username', ''))
    kaggle_key = get_input("Kaggle API Key", kaggle.get('key', ''))
    
    # Update credentials dictionary
    credentials = {
        'kenpom': {
            'username': kenpom_username,
            'password': kenpom_password
        },
        'kaggle': {
            'username': kaggle_username,
            'key': kaggle_key
        }
    }
    
    # Save credentials
    try:
        with open(credentials_path, 'w') as f:
            yaml.dump(credentials, f, default_flow_style=False)
        print(f"\nCredentials saved to {credentials_path}")
        print("Note: This file contains sensitive information and should never be committed to version control.")
    except Exception as e:
        print(f"Error saving credentials: {str(e)}")
        return False
    
    # Set up .gitignore if it doesn't exist
    gitignore_path = Path(".gitignore")
    gitignore_content = ""
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()
    
    # Make sure credentials file is in .gitignore
    if "credentials.yaml" not in gitignore_content:
        with open(gitignore_path, 'a') as f:
            f.write("\n# Credentials file - contains sensitive information\n")
            f.write("credentials.yaml\n")
            f.write("credentials.json\n")
        print("Added credentials files to .gitignore")
    
    print("\nCredentials setup complete! You can now run the data collection and prediction pipeline.")
    return True

if __name__ == "__main__":
    setup_credentials()
