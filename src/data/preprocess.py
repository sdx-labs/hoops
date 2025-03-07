import os
import pandas as pd
import yaml

def load_config():
    """Load project configuration."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_raw_data(data_type='M', season=None):
    """
    Load raw data files from the competition.
    
    Args:
        data_type: 'M' for men's data, 'W' for women's data
        season: Specific season to load, or None for all available
    
    Returns:
        Dictionary of dataframes with keys representing the file types
    """
    config = load_config()
    raw_data_path = config['paths']['data']['raw']
    
    # Define file patterns to look for
    file_prefix = 'M' if data_type == 'M' else 'W'
    
    # Common basketball data file types
    file_types = [
        'RegularSeasonCompactResults',
        'RegularSeasonDetailedResults',
        'NCAATourneyCompactResults',
        'NCAATourneyDetailedResults',
        'TeamCoaches',
        'Teams',
        'TeamSpellings',
        'SeasonResults',
        'Players',
        'TeamConferences'
    ]
    
    data_dict = {}
    
    # Look for data files in the extracted folders
    for root, dirs, files in os.walk(raw_data_path):
        for file in files:
            if file.endswith('.csv'):
                for file_type in file_types:
                    if f"{file_prefix}{file_type}" in file:
                        # Add season filter if specified
                        if season and str(season) not in file:
                            continue
                            
                        print(f"Loading {file}")
                        filepath = os.path.join(root, file)
                        df = pd.read_csv(filepath)
                        
                        key = file_type
                        if key in data_dict:
                            # Append to existing dataframe if we have multiple season files
                            data_dict[key] = pd.concat([data_dict[key], df], ignore_index=True)
                        else:
                            data_dict[key] = df
    
    if not data_dict:
        print(f"No {data_type} data files found. Make sure you've downloaded the competition data.")
    else:
        print(f"Loaded {len(data_dict)} {data_type} data files")
    
    return data_dict

def process_game_results(data_dict):
    """
    Process game results data to create features.
    
    Args:
        data_dict: Dictionary of raw dataframes
    
    Returns:
        Dataframe of processed game results
    """
    # This is a placeholder for the actual implementation
    # You'll need to adapt this to the actual data structure
    
    if 'RegularSeasonDetailedResults' in data_dict:
        games = data_dict['RegularSeasonDetailedResults'].copy()
    elif 'RegularSeasonCompactResults' in data_dict:
        games = data_dict['RegularSeasonCompactResults'].copy()
    else:
        print("No game results data found")
        return None
    
    # Example processing - create win indicator
    if 'WScore' in games.columns and 'LScore' in games.columns:
        games['ScoreDiff'] = games['WScore'] - games['LScore']
    
    # Process the games data here...
    # This is where you'd implement feature engineering based on game results
    
    return games

def save_processed_data(df, name, data_type='M'):
    """
    Save processed data to the processed data directory.
    
    Args:
        df: Dataframe to save
        name: Name of the dataframe (e.g., 'games', 'teams')
        data_type: 'M' for men's data, 'W' for women's data
    """
    config = load_config()
    processed_data_path = config['paths']['data']['processed']
    
    # Create filename with prefix for men's/women's data
    prefix = 'M' if data_type == 'M' else 'W'
    filename = f"{prefix}_{name}.csv"
    filepath = os.path.join(processed_data_path, filename)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Saved processed {data_type} {name} data to {filepath}")

if __name__ == "__main__":
    # Example usage
    men_data = load_raw_data(data_type='M')
    women_data = load_raw_data(data_type='W')
    
    if men_data:
        men_games = process_game_results(men_data)
        if men_games is not None:
            save_processed_data(men_games, 'games', 'M')
    
    if women_data:
        women_games = process_game_results(women_data)
        if women_games is not None:
            save_processed_data(women_games, 'games', 'W')
