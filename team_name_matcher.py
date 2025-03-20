import os
import sys
import pandas as pd
import numpy as np
import argparse

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def standardize_team_name(name):
    """
    Standardize team names for better matching
    """
    if pd.isna(name):
        return ""
    
    name = str(name).lower().strip()
    
    # Remove common parts that vary between datasets
    replacements = {
        " university": "",
        " college": "",
        " u.": "",
        " u": "",
        "university of ": "",
        "university ": "",
        "univ. of ": "",
        "univ ": "",
        "univ. ": "",
        " state": " st",
        "saint ": "st ",
        "st. ": "st ",
        " & ": " and ",
        "-": " ",  # Handle hyphens consistently
        ".": "",    # Remove periods
        "'": "",    # Remove apostrophes
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    return name.strip()

def match_team_names(kenpom_path, kaggle_dir, output_path=None, threshold=85):
    """
    Match KenPom team names to Kaggle TeamIDs
    """
    # Load KenPom data
    if not os.path.exists(kenpom_path):
        print(f"ERROR: KenPom file not found at {kenpom_path}")
        return None
    
    try:
        kenpom_data = pd.read_csv(kenpom_path)
        print(f"Loaded KenPom data with {len(kenpom_data)} rows")
        print(f"KenPom columns: {kenpom_data.columns.tolist()}")
    except Exception as e:
        print(f"Error loading KenPom data: {str(e)}")
        return None
    
    # Identify team name column in KenPom data - specifically look for "Mapped ESPN Team Name" first
    team_col = None
    if 'Mapped ESPN Team Name' in kenpom_data.columns:
        team_col = 'Mapped ESPN Team Name'
        print(f"Using 'Mapped ESPN Team Name' column for team mapping")
    elif 'Full Team Name' in kenpom_data.columns:
        team_col = 'Full Team Name'
        print(f"Using 'Full Team Name' column for team mapping")
    else:
        # Fall back to other possible column names
        for col in ['Team', 'TeamName', 'team', 'NAME', 'name']:
            if col in kenpom_data.columns:
                team_col = col
                print(f"Falling back to '{col}' column for team mapping")
                break
    
    if team_col is None:
        print("ERROR: Could not find team name column in KenPom data")
        print("Expected column name: 'Mapped ESPN Team Name' or 'Full Team Name'")
        return None
    
    # Load Kaggle team data
    try:
        from data_processing.load_data import load_kaggle_data
        
        # Check if we can load directly from CSV if the module isn't working
        if not os.path.exists(os.path.join(kaggle_dir, 'MTeams.csv')):
            print(f"ERROR: MTeams.csv not found in {kaggle_dir}")
            return None
        
        try:
            kaggle_data = load_kaggle_data(kaggle_dir)
        except Exception as e:
            print(f"Error using load_kaggle_data: {str(e)}")
            print("Falling back to direct CSV loading")
            
            # Direct loading as fallback
            mteams = pd.read_csv(os.path.join(kaggle_dir, 'MTeams.csv'))
            team_spellings = pd.read_csv(os.path.join(kaggle_dir, 'MTeamSpellings.csv')) if os.path.exists(os.path.join(kaggle_dir, 'MTeamSpellings.csv')) else pd.DataFrame(columns=['TeamID', 'TeamNameSpelling'])
            kaggle_data = {'MTeams': mteams, 'MTeamSpellings': team_spellings}
        
        if 'MTeams' not in kaggle_data:
            print("ERROR: MTeams not found in Kaggle data")
            return None
        
        mteams = kaggle_data['MTeams']
        print(f"Loaded {len(mteams)} teams from Kaggle data")
        
        # Load team spellings if available
        if 'MTeamSpellings' in kaggle_data:
            team_spellings = kaggle_data['MTeamSpellings']
            print(f"Loaded {len(team_spellings)} additional team spellings")
        else:
            team_spellings = pd.DataFrame(columns=['TeamID', 'TeamNameSpelling'])
    except Exception as e:
        print(f"Error loading Kaggle data: {str(e)}")
        return None
    
    # Create lookup dictionaries
    kaggle_names = {}  # TeamID -> [standardized names]
    
    # Process main team names
    print("Processing team names from MTeams.csv")
    for _, row in mteams.iterrows():
        team_id = row['TeamID']
        team_name = standardize_team_name(row['TeamName'])
        
        if team_id not in kaggle_names:
            kaggle_names[team_id] = set()
        
        kaggle_names[team_id].add(team_name)
    
    # Add alternate spellings
    print("Processing team spellings from MTeamSpellings.csv")
    for _, row in team_spellings.iterrows():
        team_id = row['TeamID']
        spelling = standardize_team_name(row['TeamNameSpelling'])
        
        if team_id not in kaggle_names:
            kaggle_names[team_id] = set()
        
        kaggle_names[team_id].add(spelling)
    
    # Reverse lookup - from name to ID
    name_to_id = {}
    for team_id, names in kaggle_names.items():
        for name in names:
            if name:  # Skip empty strings
                name_to_id[name] = team_id
    
    print(f"Created {len(name_to_id)} standardized name-to-ID mappings")
    
    # Get unique KenPom team names
    kenpom_names = kenpom_data[team_col].dropna().unique()
    print(f"Found {len(kenpom_names)} unique team names in KenPom data")
    print(f"Sample KenPom team names: {kenpom_names[:5]}")
    
    # Try direct matching first with standardization
    mapping = {}
    unmapped = []
    
    for name in kenpom_names:
        std_name = standardize_team_name(name)
        
        if std_name in name_to_id:
            mapping[name] = name_to_id[std_name]
        else:
            # Also try just the first part of names with multiple parts (e.g., "Duke Blue Devils" -> "duke")
            first_word = std_name.split()[0] if std_name.split() else ""
            if first_word and any(k.startswith(first_word) for k in name_to_id.keys()):
                for k in name_to_id.keys():
                    if k.startswith(first_word):
                        mapping[name] = name_to_id[k]
                        print(f"Matched based on first word: '{name}' -> '{k}' (ID: {name_to_id[k]})")
                        break
            else:
                unmapped.append(name)
    
    print(f"Direct matching: mapped {len(mapping)} out of {len(kenpom_names)} teams")
    
    # Try fuzzy matching for the rest
    if unmapped:
        try:
            from fuzzywuzzy import process, fuzz
            
            std_kaggle_names = list(name_to_id.keys())
            for name in unmapped[:]:
                std_name = standardize_team_name(name)
                best_match, score = process.extractOne(std_name, std_kaggle_names, scorer=fuzz.token_sort_ratio)
                
                if score >= threshold:
                    mapping[name] = name_to_id[best_match]
                    print(f"Fuzzy matched: '{name}' -> '{best_match}' with score {score}")
                    unmapped.remove(name)
            
            print(f"After fuzzy matching: mapped {len(mapping)} out of {len(kenpom_names)} teams")
        except ImportError:
            print("WARNING: fuzzywuzzy module not available for fuzzy matching")
            print("Install with: pip install fuzzywuzzy python-Levenshtein")
    
    # Report unmapped teams
    if unmapped:
        print(f"{len(unmapped)} teams could not be mapped")
        print("Sample of unmapped teams:", unmapped[:10])
        
        # Save unmapped teams for manual inspection
        unmapped_df = pd.DataFrame({'KenPomTeamName': unmapped})
        unmapped_path = os.path.join(os.path.dirname(output_path), 'unmapped_teams.csv') if output_path else 'unmapped_teams.csv'
        unmapped_df.to_csv(unmapped_path, index=False)
        print(f"Saved unmapped teams to {unmapped_path}")
    
    # Create mapping DataFrame
    mapping_df = pd.DataFrame({
        'KenPomTeam': list(mapping.keys()),
        'KaggleTeamID': list(mapping.values())
    })
    
    # Print summary statistics
    team_id_counts = mapping_df['KaggleTeamID'].value_counts()
    if team_id_counts.max() > 1:
        print(f"WARNING: Some TeamIDs matched to multiple KenPom teams")
        print("Most common:")
        for team_id, count in team_id_counts.nlargest(5).items():
            if count > 1:
                dupes = mapping_df[mapping_df['KaggleTeamID'] == team_id]['KenPomTeam'].tolist()
                team_name = mteams[mteams['TeamID'] == team_id]['TeamName'].iloc[0]
                print(f"  TeamID {team_id} ({team_name}) matched to {count} KenPom teams: {dupes}")
    
    # Save mapping
    if output_path:
        mapping_df.to_csv(output_path, index=False)
        print(f"Saved mapping to {output_path}")
        
        # Also create an enhanced KenPom dataset with TeamIDs
        kenpom_data['TeamID'] = kenpom_data[team_col].map(mapping)
        enhanced_path = os.path.join(os.path.dirname(output_path), 'enhanced_kenpom_data.csv')
        kenpom_data.to_csv(enhanced_path, index=False)
        print(f"Saved enhanced KenPom data with TeamIDs to {enhanced_path}")
    
    return mapping_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Match KenPom team names to Kaggle TeamIDs')
    parser.add_argument('--kenpom', type=str, 
                        default='/Volumes/MINT/projects/model/data/kenpom/historical_kenpom_data.csv',
                        help='Path to the KenPom data file')
    parser.add_argument('--kaggle-dir', type=str, 
                        default='/Volumes/MINT/projects/model/data/kaggle-data',
                        help='Path to the Kaggle data directory')
    parser.add_argument('--output', type=str, 
                        default='/Volumes/MINT/projects/model/data/kenpom/team_mapping.csv',
                        help='Output file for the mapping')
    parser.add_argument('--threshold', type=int, default=85,
                        help='Minimum score threshold for fuzzy matching (0-100)')
    
    try:
        # Try to import fuzzywuzzy
        from fuzzywuzzy import process
    except ImportError:
        print("Installing fuzzywuzzy package for fuzzy string matching...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fuzzywuzzy"])
        try:
            try:
                import Levenshtein as python_Levenshtein
            except ImportError:
                print("python-Levenshtein is not installed. Install it with: pip install python-Levenshtein")
        except ImportError:
            print("For faster fuzzy matching, consider installing: pip install python-Levenshtein")
    
    args = parser.parse_args()
    match_team_names(args.kenpom, args.kaggle_dir, args.output, args.threshold)
