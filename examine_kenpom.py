import os
import sys
import pandas as pd
import numpy as np
from fuzzywuzzy import process, fuzz
import matplotlib.pyplot as plt

def examine_kenpom_data(data_path='/Volumes/MINT/projects/model/data/kenpom/historical_kenpom_data.csv'):
    """
    Examine the structure of KenPom data to help with integration
    """
    if not os.path.exists(data_path):
        print(f"KenPom data file not found at: {data_path}")
        return None
    
    try:
        # Load the data
        kenpom_data = pd.read_csv(data_path)
        
        # Basic information
        print(f"KenPom data loaded with shape: {kenpom_data.shape}")
        print("\nColumns in data:")
        for col in kenpom_data.columns:
            print(f"  - {col}")
            
        # Sample data
        print("\nFirst few rows:")
        print(kenpom_data.head(3))
        
        # Check for team name column - specifically look for "Mapped ESPN Team Name" first
        team_col = None
        if 'Mapped ESPN Team Name' in kenpom_data.columns:
            team_col = 'Mapped ESPN Team Name'
            print(f"\nFound team name column: 'Mapped ESPN Team Name'")
            print(f"Sample values: {kenpom_data[team_col].head(5).tolist()}")
        elif 'Full Team Name' in kenpom_data.columns:
            team_col = 'Full Team Name'
            print(f"\nFound team name column: 'Full Team Name'")
            print(f"Sample values: {kenpom_data[team_col].head(5).tolist()}")
        else:
            # Fallback to other possible column names
            possible_team_columns = ['TeamID', 'Team_ID', 'Team ID', 'team_id', 'id', 'Team', 'team', 'NAME', 'TeamName', 'name']
            for col in possible_team_columns:
                if col in kenpom_data.columns:
                    print(f"\nPossible team name column found: '{col}'")
                    print(f"Sample values: {kenpom_data[col].head(5).tolist()}")
                    break
            
            print("\nWARNING: Neither 'Mapped ESPN Team Name' nor 'Full Team Name' column found. These columns are expected in KenPom data.")
            
        # Check for potential Season columns
        possible_season_columns = ['Season', 'Year', 'season', 'SEASON', 'year']
        found = False
        for col in possible_season_columns:
            if col in kenpom_data.columns:
                print(f"\nPossible season column found: '{col}'")
                print(f"Unique values: {sorted(kenpom_data[col].unique())}")
                found = True
                
        if not found:
            print("\nNo obvious season column found.")
            
        # Look for missing values
        print("\nMissing value counts:")
        print(kenpom_data.isnull().sum()[kenpom_data.isnull().sum() > 0])
        
        return kenpom_data
        
    except Exception as e:
        print(f"Error examining KenPom data: {str(e)}")
        return None

def create_team_mapping(kaggle_data, kenpom_data):
    """
    Create a mapping between KenPom team names and Kaggle TeamIDs
    """
    # Check if both datasets are available
    if kaggle_data is None or kenpom_data is None:
        print("Unable to create team mapping: missing data")
        return None
    
    # Identify the team name column in KenPom data - prioritize "Mapped ESPN Team Name"
    kenpom_team_col = None
    if 'Mapped ESPN Team Name' in kenpom_data.columns:
        kenpom_team_col = 'Mapped ESPN Team Name'
        print(f"Using 'Mapped ESPN Team Name' column for team mapping")
    elif 'Full Team Name' in kenpom_data.columns:
        kenpom_team_col = 'Full Team Name'
        print(f"Using 'Full Team Name' column for team mapping")
    else:
        # Fall back to other possible column names
        for col in ['Team', 'TeamName', 'team', 'NAME', 'name']:
            if col in kenpom_data.columns:
                kenpom_team_col = col
                print(f"Falling back to '{col}' column for team mapping")
                break
    
    if kenpom_team_col is None:
        print("ERROR: Could not identify team name column in KenPom data (expected 'Mapped ESPN Team Name' or 'Full Team Name')")
        return None
    
    # Get all unique KenPom team names
    kenpom_teams = kenpom_data[kenpom_team_col].unique()
    print(f"Found {len(kenpom_teams)} unique team names in KenPom data")
    print("Sample KenPom teams:", kenpom_teams[:5])
    
    # Get Kaggle team names and IDs
    if 'MTeams' in kaggle_data:
        kaggle_teams = kaggle_data['MTeams'][['TeamID', 'TeamName']]
        print(f"Found {len(kaggle_teams)} teams in Kaggle MTeams data")
        print("Sample Kaggle teams:", kaggle_teams.head(5))
    else:
        print("Could not find MTeams in Kaggle data")
        return None
    
    # Create a mapping dictionary using fuzzy matching
    print("\nBuilding team name mapping...")
    mapping = {}
    unmapped = []
    
    # Add MTeamSpellings data if available
    team_spellings = {}
    if 'MTeamSpellings' in kaggle_data:
        for _, row in kaggle_data['MTeamSpellings'].iterrows():
            team_id = row['TeamID']
            spelling = row['TeamNameSpelling'].lower()
            if team_id not in team_spellings:
                team_spellings[team_id] = []
            team_spellings[team_id].append(spelling)
    
    # Build a reverse lookup from team name variations to TeamID
    name_to_id = {}
    for _, row in kaggle_teams.iterrows():
        team_id = row['TeamID']
        team_name = row['TeamName'].lower()
        name_to_id[team_name] = team_id
        
        # Add alternate spellings
        if team_id in team_spellings:
            for spelling in team_spellings[team_id]:
                name_to_id[spelling] = team_id
    
    # Try direct matches first
    for team in kenpom_teams:
        team_lower = team.lower() if isinstance(team, str) else str(team).lower()
        
        # Try direct match
        if team_lower in name_to_id:
            mapping[team] = name_to_id[team_lower]
        else:
            unmapped.append(team)
    
    print(f"Direct matching: mapped {len(mapping)} teams, {len(unmapped)} unmapped")
    
    # Use fuzzy matching for remaining teams
    if unmapped:
        kaggle_names = list(name_to_id.keys())
        for team in unmapped[:]:  # Use a copy to avoid modifying while iterating
            team_str = str(team)
            match, score = process.extractOne(team_str, kaggle_names, scorer=fuzz.token_sort_ratio)
            
            # Only accept matches with high confidence
            if score >= 85:
                mapping[team] = name_to_id[match]
                unmapped.remove(team)
        
        print(f"After fuzzy matching: mapped {len(mapping)} teams, {len(unmapped)} unmapped")
        
        if unmapped:
            print("Sample of unmapped teams:", unmapped[:10])
            
            # Save unmapped teams for manual review
            pd.DataFrame({'UnmappedTeam': unmapped}).to_csv('/Volumes/MINT/projects/model/data/kenpom/unmapped_teams.csv', index=False)
            print("Saved unmapped teams to /Volumes/MINT/projects/model/data/kenpom/unmapped_teams.csv")
    
    # Save the mapping
    mapping_df = pd.DataFrame({
        'KenPomTeam': list(mapping.keys()),
        'KaggleTeamID': list(mapping.values())
    })
    mapping_df.to_csv('/Volumes/MINT/projects/model/data/kenpom/team_mapping.csv', index=False)
    print(f"Saved team mapping to /Volumes/MINT/projects/model/data/kenpom/team_mapping.csv")
    
    return mapping

if __name__ == "__main__":
    # Try to pip install fuzzywuzzy if not already installed
    try:
        from fuzzywuzzy import process, fuzz
    except ImportError:
        print("Installing fuzzywuzzy package for fuzzy string matching...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fuzzywuzzy"])
        from fuzzywuzzy import process, fuzz
    
    # Examine KenPom data
    kenpom_data = examine_kenpom_data()
    
    if kenpom_data is not None:
        print("\n--- Creating Team Name Mapping ---")
        from data_processing.load_data import load_kaggle_data
        kaggle_data = load_kaggle_data()
        
        mapping = create_team_mapping(kaggle_data, kenpom_data)
        
        if mapping:
            print(f"Successfully created mapping for {len(mapping)} teams")
            
            # Apply the mapping to KenPom data
            team_col = None
            for col in ['Team', 'TeamName', 'team', 'NAME', 'name']:
                if col in kenpom_data.columns:
                    team_col = col
                    break
            
            if team_col:
                kenpom_data['TeamID'] = kenpom_data[team_col].map(mapping)
                
                # Check mapping success
                mapped_count = kenpom_data['TeamID'].notna().sum()
                total_count = len(kenpom_data)
                print(f"Applied mapping to {mapped_count}/{total_count} rows ({mapped_count/total_count*100:.1f}%)")
                
                # Save the enhanced KenPom data with TeamIDs
                kenpom_data.to_csv('/Volumes/MINT/projects/model/data/kenpom/enhanced_kenpom_data.csv', index=False)
                print("Saved enhanced KenPom data to /Volumes/MINT/projects/model/data/kenpom/enhanced_kenpom_data.csv")
