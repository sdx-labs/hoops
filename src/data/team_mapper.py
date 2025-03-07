import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re
import difflib
from fuzzywuzzy import fuzz, process

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TeamMapper:
    def __init__(self, raw_data_dir="data/raw", output_dir="data/external"):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapping dictionaries
        self.mens_team_map = {}
        self.womens_team_map = {}
        self.kenpom_team_map = {}
        
    def load_kaggle_team_data(self):
        """
        Load Kaggle team data (MTeams.csv, MTeamSpellings.csv, WTeams.csv, WTeamSpellings.csv)
        """
        # Load core team files
        m_teams_file = list(self.raw_data_dir.glob("*MTeams.csv"))
        w_teams_file = list(self.raw_data_dir.glob("*WTeams.csv"))
        
        if not m_teams_file:
            logger.error("MTeams.csv file not found")
            return False
            
        if not w_teams_file:
            logger.error("WTeams.csv file not found")
            return False
        
        # Load team spellings files
        m_spellings_file = list(self.raw_data_dir.glob("*MTeamSpellings.csv"))
        w_spellings_file = list(self.raw_data_dir.glob("*WTeamSpellings.csv"))
        
        if not m_spellings_file:
            logger.error("MTeamSpellings.csv file not found")
            return False
            
        if not w_spellings_file:
            logger.error("WTeamSpellings.csv file not found")
            return False
        
        # Read the files
        self.m_teams = pd.read_csv(m_teams_file[0])
        self.w_teams = pd.read_csv(w_teams_file[0])
        self.m_spellings = pd.read_csv(m_spellings_file[0])
        self.w_spellings = pd.read_csv(w_spellings_file[0])
        
        logger.info(f"Loaded {len(self.m_teams)} men's teams and {len(self.w_teams)} women's teams")
        logger.info(f"Loaded {len(self.m_spellings)} men's team spellings and {len(self.w_spellings)} women's team spellings")
        
        return True
    
    def _normalize_team_name(self, name):
        """Normalize team name for better matching"""
        if not name:
            return ""
            
        # Convert to lowercase
        name = name.lower()
        
        # Remove common prefixes/suffixes
        name = re.sub(r'^univ\.? of |^university of ', '', name)
        
        # Remove state university, etc.
        name = re.sub(r' state university$| state$| university$| college$', '', name)
        
        # Remove special characters
        name = re.sub(r'[^\w\s]', '', name)
        
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Common abbreviations
        name_map = {
            'uconn': 'connecticut',
            'ul': 'louisville',
            'lsu': 'louisiana state',
            'smu': 'southern methodist',
            'unc': 'north carolina',
            'usc': 'southern california',
            'pitt': 'pittsburgh',
            'ole miss': 'mississippi',
            'ucf': 'central florida',
            'byu': 'brigham young',
            'vcu': 'virginia commonwealth',
            'tcu': 'texas christian',
            'utep': 'texas el paso',
            'utsa': 'texas san antonio',
            'etsu': 'east tennessee state',
            'fiu': 'florida international',
            'fau': 'florida atlantic',
            'uab': 'alabama birmingham',
            'uic': 'illinois chicago',
            'uta': 'texas arlington',
            'st': 'saint',
        }
        
        # Apply name mappings
        for abbr, full in name_map.items():
            if name == abbr:
                return full
                
        return name
        
    def build_team_mapping(self):
        """
        Build comprehensive team mapping dictionaries
        """
        if not hasattr(self, 'm_teams') or not hasattr(self, 'm_spellings'):
            if not self.load_kaggle_team_data():
                return False
        
        # Create men's teams mapping
        for _, row in self.m_teams.iterrows():
            team_id = row['TeamID']
            team_name = row['TeamName']
            self.mens_team_map[team_id] = {
                'TeamName': team_name,
                'Spellings': [team_name],
                'NormalizedName': self._normalize_team_name(team_name)
            }
        
        # Add men's team spellings
        for _, row in self.m_spellings.iterrows():
            team_id = row['TeamID']
            team_name_spelling = row['TeamNameSpelling']
            
            if team_id in self.mens_team_map:
                self.mens_team_map[team_id]['Spellings'].append(team_name_spelling)
            else:
                logger.warning(f"Unknown men's team ID in spellings: {team_id}")
        
        # Create women's teams mapping
        for _, row in self.w_teams.iterrows():
            team_id = row['TeamID']
            team_name = row['TeamName']
            self.womens_team_map[team_id] = {
                'TeamName': team_name,
                'Spellings': [team_name],
                'NormalizedName': self._normalize_team_name(team_name)
            }
        
        # Add women's team spellings
        for _, row in self.w_spellings.iterrows():
            team_id = row['TeamID']
            team_name_spelling = row['TeamNameSpelling']
            
            if team_id in self.womens_team_map:
                self.womens_team_map[team_id]['Spellings'].append(team_name_spelling)
            else:
                logger.warning(f"Unknown women's team ID in spellings: {team_id}")
                
        # Create inverse lookup dictionaries for spelling to ID
        self.mens_spelling_to_id = {}
        for team_id, team_data in self.mens_team_map.items():
            for spelling in team_data['Spellings']:
                self.mens_spelling_to_id[spelling.lower()] = team_id
                # Also add normalized version
                norm_spelling = self._normalize_team_name(spelling)
                if norm_spelling:
                    self.mens_spelling_to_id[norm_spelling] = team_id
                    
        self.womens_spelling_to_id = {}
        for team_id, team_data in self.womens_team_map.items():
            for spelling in team_data['Spellings']:
                self.womens_spelling_to_id[spelling.lower()] = team_id
                # Also add normalized version
                norm_spelling = self._normalize_team_name(spelling)
                if norm_spelling:
                    self.womens_spelling_to_id[norm_spelling] = team_id
                    
        logger.info(f"Built mapping for {len(self.mens_team_map)} men's teams and {len(self.womens_team_map)} women's teams")
        return True
    
    def map_kenpom_teams(self, kenpom_data):
        """
        Map KenPom team names to Kaggle team IDs
        
        Args:
            kenpom_data: DataFrame containing KenPom data with TeamName column
        """
        if not self.mens_team_map:
            if not self.build_team_mapping():
                return None
                
        if kenpom_data is None or len(kenpom_data) == 0:
            logger.error("No KenPom data provided")
            return None
            
        # Create mapping
        kenpom_to_kaggle = {}
        unmatched_teams = set()
        
        # Get unique KenPom team names
        kenpom_teams = kenpom_data['TeamName'].unique() if 'TeamName' in kenpom_data.columns else []
        
        for kenpom_name in kenpom_teams:
            # Try direct lookup first
            team_id = self.mens_spelling_to_id.get(kenpom_name.lower())
            
            if team_id is not None:
                kenpom_to_kaggle[kenpom_name] = team_id
                continue
                
            # Try normalized name
            norm_name = self._normalize_team_name(kenpom_name)
            team_id = self.mens_spelling_to_id.get(norm_name)
            
            if team_id is not None:
                kenpom_to_kaggle[kenpom_name] = team_id
                continue
                
            # Try fuzzy matching
            best_match = None
            best_score = 0
            
            # Create list of all team names and normalized names
            all_names = []
            for tid, team_data in self.mens_team_map.items():
                for spelling in team_data['Spellings']:
                    all_names.append((tid, spelling))
                all_names.append((tid, team_data['NormalizedName']))
                
            # Find best fuzzy match
            for team_id, name in all_names:
                score = fuzz.ratio(norm_name, self._normalize_team_name(name))
                if score > best_score and score > 85:  # Threshold for a good match
                    best_score = score
                    best_match = team_id
                    
            if best_match is not None:
                kenpom_to_kaggle[kenpom_name] = best_match
            else:
                unmatched_teams.add(kenpom_name)
                
        # Create mapping DataFrame
        mapping_data = []
        for kenpom_name, kaggle_id in kenpom_to_kaggle.items():
            kaggle_name = self.mens_team_map[kaggle_id]['TeamName']
            mapping_data.append({
                'KenPomTeamName': kenpom_name,
                'KaggleTeamID': kaggle_id,
                'KaggleTeamName': kaggle_name
            })
            
        mapping_df = pd.DataFrame(mapping_data)
        
        # Report results
        logger.info(f"Mapped {len(kenpom_to_kaggle)} of {len(kenpom_teams)} KenPom teams to Kaggle IDs")
        if unmatched_teams:
            logger.warning(f"Could not match {len(unmatched_teams)} KenPom teams: {', '.join(sorted(unmatched_teams))}")
            
        # Save mapping
        mapping_file = self.output_dir / "kenpom_team_mapping.csv"
        mapping_df.to_csv(mapping_file, index=False)
        logger.info(f"Saved team mapping to {mapping_file}")
        
        return mapping_df
    
    def apply_mapping_to_kenpom_data(self, kenpom_data, mapping_df=None):
        """
        Apply team ID mapping to KenPom data
        
        Args:
            kenpom_data: DataFrame containing KenPom data
            mapping_df: DataFrame with KenPom to Kaggle mapping (optional)
        """
        if mapping_df is None:
            mapping_file = self.output_dir / "kenpom_team_mapping.csv"
            if not mapping_file.exists():
                logger.error("Team mapping file not found")
                return None
            mapping_df = pd.read_csv(mapping_file)
        
        # Convert mapping to dict for faster lookup
        kenpom_to_kaggle = dict(zip(mapping_df['KenPomTeamName'], mapping_df['KaggleTeamID']))
        
        # Add TeamID column to KenPom data
        kenpom_data['TeamID'] = kenpom_data['TeamName'].map(kenpom_to_kaggle)
        
        # Report stats
        mapped_count = kenpom_data['TeamID'].notna().sum()
        total_count = len(kenpom_data)
        logger.info(f"Applied team mapping: {mapped_count}/{total_count} rows mapped ({mapped_count/total_count:.1%})")
        
        return kenpom_data

if __name__ == "__main__":
    # Example usage
    mapper = TeamMapper()
    
    # 1. Load team data and build mapping
    mapper.build_team_mapping()
    
    # 2. If you have KenPom data, you can map it
    # kenpom_file = Path("data/external/kenpom_historical_ratings.csv")
    # if kenpom_file.exists():
    #     kenpom_data = pd.read_csv(kenpom_file)
    #     mapping_df = mapper.map_kenpom_teams(kenpom_data)
    #     
    #     # Apply mapping to add TeamID to KenPom data
    #     if mapping_df is not None:
    #         mapped_kenpom = mapper.apply_mapping_to_kenpom_data(kenpom_data, mapping_df)
    #         mapped_kenpom.to_csv("data/external/kenpom_historical_ratings_with_ids.csv", index=False)
    
    logger.info("Team mapping process completed.")
