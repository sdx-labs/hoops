import yaml
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CredentialsManager:
    """Manages loading and providing access to sensitive credentials"""
    
    def __init__(self, credentials_file="credentials.yaml"):
        """
        Initialize the credentials manager
        
        Args:
            credentials_file: Path to credentials file
        """
        self.credentials_path = Path(credentials_file)
        self.credentials = {}
        self.load_credentials()
        
    def load_credentials(self):
        """Load credentials from file or environment variables"""
        if self.credentials_path.exists():
            try:
                with open(self.credentials_path, 'r') as f:
                    self.credentials = yaml.safe_load(f) or {}
                logger.info(f"Loaded credentials from {self.credentials_path}")
            except Exception as e:
                logger.error(f"Error loading credentials from file: {str(e)}")
        else:
            logger.warning(f"Credentials file not found: {self.credentials_path}")
            # Create empty structure
            self.credentials = {
                "kenpom": {"username": "", "password": ""},
                "kaggle": {"username": "", "key": ""}
            }
            
        # Allow environment variables to override file settings
        # Example: HOOPS_KENPOM_USERNAME will override credentials['kenpom']['username']
        for service in self.credentials:
            for key in self.credentials.get(service, {}):
                env_var = f"HOOPS_{service.upper()}_{key.upper()}"
                if env_var in os.environ:
                    self.credentials[service][key] = os.environ[env_var]
                    logger.info(f"Using environment variable {env_var} for {service}.{key}")
    
    def get_kenpom_credentials(self):
        """
        Get KenPom credentials
        
        Returns:
            tuple: (username, password)
        """
        kenpom = self.credentials.get('kenpom', {})
        username = kenpom.get('username', '')
        password = kenpom.get('password', '')
        
        if not username or not password:
            logger.warning("KenPom credentials not configured")
            
        return username, password
    
    def get_kaggle_credentials(self):
        """
        Get Kaggle API credentials
        
        Returns:
            tuple: (username, key)
        """
        kaggle = self.credentials.get('kaggle', {})
        username = kaggle.get('username', '')
        key = kaggle.get('key', '')
        
        if not username or not key:
            logger.warning("Kaggle credentials not configured")
            
        return username, key
    
    def save_template(self):
        """
        Save a template credentials file if one doesn't exist
        """
        if not self.credentials_path.exists():
            template = {
                "kenpom": {
                    "username": "your_kenpom_username",
                    "password": "your_kenpom_password"
                },
                "kaggle": {
                    "username": "your_kaggle_username",
                    "key": "your_kaggle_key"
                }
            }
            
            try:
                with open(self.credentials_path, 'w') as f:
                    yaml.dump(template, f, default_flow_style=False)
                logger.info(f"Created credentials template at {self.credentials_path}")
                print(f"Created credentials template at {self.credentials_path}")
                print("Please edit this file to add your actual credentials")
            except Exception as e:
                logger.error(f"Error creating credentials template: {str(e)}")
