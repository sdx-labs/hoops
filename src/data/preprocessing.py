import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from src.data.download import KaggleDataDownloader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasketballDataPreprocessor:
    def __init__(self, raw_data_dir="data/raw", processed_data_dir="data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = KaggleDataDownloader(data_dir=raw_data_dir)
        
    def ensure_data_available(self, download=True, force_download=False):
        """
        Ensure that raw data is available, downloading it if necessary and allowed
        
        Args:
            download: Whether to download data if missing
            force_download: Whether to force download even if data exists
            
        Returns:
            bool: True if data is available, False otherwise
        """
        # Check if data exists
        if self.downloader.verify_required_files():
            return True
            
        # If data doesn't exist and download is allowed
        if download:
            logger.info("Required data files missing, attempting to download...")
            return self.downloader.download_competition_data(force=force_download)
        else:
            logger.error("Required data files missing and download not allowed")
            return False
        
    def load_regular_season_results(self, gender="M", seasons=range(2015, 2025)):
        """Load and combine regular season results for multiple seasons"""
        all_results = []
        
        for season in seasons:
            try:
                file_pattern = f"{gender}RegularSeasonCompactResults_{season}.csv"
                file_paths = list(self.raw_data_dir.glob(f"*{file_pattern}*"))
                
                if not file_paths:
                    logger.warning(f"No file found matching pattern: {file_pattern}")
                    continue
                    
                file_path = file_paths[0]
                df = pd.read_csv(file_path)
                df['Season'] = season
                all_results.append(df)
                logger.info(f"Loaded {file_path.name} with {len(df)} games")
                
            except Exception as e:
                logger.error(f"Error loading {gender} regular season data for {season}: {str(e)}")
        
        if not all_results:
            logger.error(f"No regular season data found for {gender}")
            return None
            
        # Combine all seasons into one dataframe
        combined_results = pd.concat(all_results, ignore_index=True)
        logger.info(f"Combined {gender} regular season data: {len(combined_results)} games")
        return combined_results
        
    def load_tournament_results(self, gender="M", seasons=range(2015, 2025)):
        """Load and combine tournament results for multiple seasons"""
        all_results = []
        
        for season in seasons:
            try:
                file_pattern = f"{gender}NCAATourneyCompactResults_{season}.csv"
                file_paths = list(self.raw_data_dir.glob(f"*{file_pattern}*"))
                
                if not file_paths:
                    logger.warning(f"No file found matching pattern: {file_pattern}")
                    continue
                    
                file_path = file_paths[0]
                df = pd.read_csv(file_path)
                df['Season'] = season
                all_results.append(df)
                logger.info(f"Loaded {file_path.name} with {len(df)} games")
                
            except Exception as e:
                logger.error(f"Error loading {gender} tournament data for {season}: {str(e)}")
        
        if not all_results:
            logger.error(f"No tournament data found for {gender}")
            return None
            
        # Combine all seasons into one dataframe
        combined_results = pd.concat(all_results, ignore_index=True)
        logger.info(f"Combined {gender} tournament data: {len(combined_results)} games")
        return combined_results
        
    def load_team_data(self):
        """Load team information"""
        try:
            file_paths = list(self.raw_data_dir.glob("*Teams.csv"))
            if not file_paths:
                logger.error("Teams.csv file not found")
                return None
                
            teams_df = pd.read_csv(file_paths[0])
            logger.info(f"Loaded team data with {len(teams_df)} teams")
            return teams_df
            
        except Exception as e:
            logger.error(f"Error loading team data: {str(e)}")
            return None
            
    def create_matchup_results(self, gender="M", seasons=range(2015, 2025)):
        """
        Create a dataset of all game results with team1 vs team2 format
        where team1 is always the team with the lower TeamID
        """
        # Load regular season and tournament data
        regular_season = self.load_regular_season_results(gender, seasons)
        tournament = self.load_tournament_results(gender, seasons)
        
        if regular_season is None or tournament is None:
            return None
            
        # Combine regular season and tournament data
        all_games = pd.concat([regular_season, tournament], ignore_index=True)
        
        # Create normalized representation of games (team1 always has lower ID)
        games_normalized = []
        
        for _, game in all_games.iterrows():
            if game['WTeamID'] < game['LTeamID']:
                games_normalized.append({
                    'Season': game['Season'],
                    'DayNum': game.get('DayNum', 0),
                    'Team1ID': game['WTeamID'],  # Lower team ID
                    'Team2ID': game['LTeamID'],  # Higher team ID
                    'Team1Score': game['WScore'],
                    'Team2Score': game['LScore'],
                    'Result': 1,  # Team1 won
                    'ScoreDiff': game['WScore'] - game['LScore'],
                    'Gender': gender
                })
            else:
                games_normalized.append({
                    'Season': game['Season'],
                    'DayNum': game.get('DayNum', 0),
                    'Team1ID': game['LTeamID'],  # Lower team ID
                    'Team2ID': game['WTeamID'],  # Higher team ID
                    'Team1Score': game['LScore'],
                    'Team2Score': game['WScore'],
                    'Result': 0,  # Team1 lost
                    'ScoreDiff': game['LScore'] - game['WScore'],
                    'Gender': gender
                })
        
        matchups_df = pd.DataFrame(games_normalized)
        
        # Save to processed directory
        output_path = self.processed_data_dir / f"{gender}_matchup_results.csv"
        matchups_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(matchups_df)} {gender} matchups to {output_path}")
        
        return matchups_df
        
    def load_submission_format(self):
        """Load the submission format file to understand what predictions are needed"""
        try:
            file_paths = list(self.raw_data_dir.glob("*2025SubmissionFormat.csv"))
            if not file_paths:
                logger.error("Submission format file not found")
                return None
                
            submission_format = pd.read_csv(file_paths[0])
            logger.info(f"Loaded submission format with {len(submission_format)} rows")
            
            # Extract season, team1, team2 from ID column
            submission_format[['Season', 'Team1ID', 'Team2ID']] = submission_format['ID'].str.split('_', expand=True).astype(int)
            
            # Save to processed directory
            output_path = self.processed_data_dir / "submission_format.csv"
            submission_format.to_csv(output_path, index=False)
            
            return submission_format
            
        except Exception as e:
            logger.error(f"Error loading submission format: {str(e)}")
            return None
            
    def process_all_data(self, download=True, force_download=False):
        """Process all data for both genders"""
        # Ensure data is available
        if not self.ensure_data_available(download=download, force_download=force_download):
            logger.error("Cannot process data: required files not available")
            return None
            
        # Process men's data
        m_matchups = self.create_matchup_results(gender="M")
        
        # Process women's data
        w_matchups = self.create_matchup_results(gender="W")
        
        # Load submission format
        submission_format = self.load_submission_format()
        
        # Load team data
        teams = self.load_team_data()
        
        # Save team data to processed directory if available
        if teams is not None:
            teams.to_csv(self.processed_data_dir / "teams.csv", index=False)
        
        return {
            "men_matchups": m_matchups,
            "women_matchups": w_matchups,
            "submission_format": submission_format,
            "teams": teams
        }

if __name__ == "__main__":
    preprocessor = BasketballDataPreprocessor()
    processed_data = preprocessor.process_all_data()
