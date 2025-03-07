import os
import kaggle
import zipfile
from pathlib import Path
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KaggleDataDownloader:
    def __init__(self, competition_name="march-machine-learning-mania-2025", data_dir="data/raw"):
        """
        Initialize the downloader with competition name and target directory
        """
        self.competition_name = competition_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_competition_data(self, force=False):
        """
        Download competition data files using Kaggle API
        
        Args:
            force: If True, download data even if it already exists
        """
        # Check if data already exists
        if not force and self.verify_required_files():
            logger.info("All required data files already exist. Skipping download.")
            return True
            
        try:
            logger.info(f"Downloading data from competition: {self.competition_name}")
            kaggle.api.authenticate()
            kaggle.api.competition_download_files(
                self.competition_name, 
                path=str(self.data_dir)
            )
            
            # Unzip the downloaded data
            zip_path = self.data_dir / f"{self.competition_name}.zip"
            if zip_path.exists():
                logger.info(f"Extracting data from {zip_path}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(str(self.data_dir))
                logger.info("Data extraction completed")
                
                # Optional: remove zip file after extraction
                os.remove(zip_path)
                logger.info(f"Removed zip file: {zip_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading competition data: {str(e)}")
            return False
            
    def verify_required_files(self, required_files=None):
        """
        Verify that all required files are present
        
        Returns:
            bool: True if all required files are present, False otherwise
        """
        if required_files is None:
            required_files = ["MDataFiles", "WDataFiles", "2025SubmissionFormat"]
            
        missing_files = []
        for file_pattern in required_files:
            matches = list(self.data_dir.glob(f"*{file_pattern}*"))
            if not matches:
                missing_files.append(file_pattern)
                
        if missing_files:
            logger.warning(f"Missing required files: {missing_files}")
            return False
        else:
            logger.info("All required files are present")
            return True
    
    def check_data_folder_exists(self):
        """
        Check if data folder exists and contains any CSV files
        
        Returns:
            bool: True if folder exists and contains CSV files, False otherwise
        """
        if not self.data_dir.exists():
            return False
            
        csv_files = list(self.data_dir.glob("*.csv"))
        return len(csv_files) > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Kaggle competition data')
    parser.add_argument('--force', action='store_true', help='Force download even if data exists')
    args = parser.parse_args()
    
    downloader = KaggleDataDownloader()
    downloader.download_competition_data(force=args.force)
    downloader.verify_required_files()
