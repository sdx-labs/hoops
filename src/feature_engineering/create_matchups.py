import pandas as pd
import numpy as np
import os

def create_historical_matchups(kaggle_data, team_features):
    """
    Create training data from historical matchups with features
    """
    print("Starting historical matchup creation...")
    
    # Get regular season results
    regular_results = pd.concat([
        kaggle_data['MRegularSeasonCompactResults'],
        kaggle_data['WRegularSeasonCompactResults']
    ], ignore_index=True)
    
    # Get NCAA tournament results
    tourney_results = pd.concat([
        kaggle_data['MNCAATourneyCompactResults'],
        kaggle_data['WNCAATourneyCompactResults']
    ], ignore_index=True)
    
    # Combine all results
    all_games = pd.concat([regular_results, tourney_results], ignore_index=True)
    print(f"Total games to process: {len(all_games)}")
    
    # Convert team features dict to DataFrame if needed
    if isinstance(team_features, dict):
        print("Converting team features dictionary to DataFrame...")
        teams_list = []
        for season in team_features:
            for team_id in team_features[season]:
                row = {'Season': season, 'TeamID': team_id}
                row.update(team_features[season][team_id])
                teams_list.append(row)
        team_features_df = pd.DataFrame(teams_list)
    else:
        team_features_df = team_features
    
    # OPTIMIZATION: Pre-index team features by (Season, TeamID) for faster lookups
    print("Creating team features lookup index...")
    team_features_dict = {}
    for _, row in team_features_df.iterrows():
        season = row['Season']
        team_id = row['TeamID']
        if season not in team_features_dict:
            team_features_dict[season] = {}
        team_features_dict[season][team_id] = row.drop(['Season', 'TeamID']).to_dict()
    
    # Create matchup features
    print("Processing matchups...")
    matchups = []
    skipped_count = 0
    
    # Process in batches to show progress
    batch_size = 10000
    total_games = len(all_games)
    
    for i in range(0, total_games, batch_size):
        batch_games = all_games.iloc[i:min(i+batch_size, total_games)]
        batch_matchups = []
        
        for _, game in batch_games.iterrows():
            season = game['Season']
            w_team = game['WTeamID']
            l_team = game['LTeamID']
            
            # OPTIMIZATION: Fast dictionary lookup instead of DataFrame filtering
            if season not in team_features_dict or w_team not in team_features_dict[season] or l_team not in team_features_dict[season]:
                skipped_count += 1
                continue
            
            w_features = team_features_dict[season][w_team]
            l_features = team_features_dict[season][l_team]
            
            # Create a unique identifier for the matchup
            if w_team < l_team:
                matchup_id = f"{season}_{w_team}_{l_team}"
                target = 1  # Lower team ID won
            else:
                matchup_id = f"{season}_{l_team}_{w_team}"
                target = 0  # Lower team ID lost
            
            # Identify which team has the lower ID for consistent feature representation
            if w_team < l_team:
                team1, team2 = w_team, l_team
                team1_features, team2_features = w_features, l_features
            else:
                team1, team2 = l_team, w_team
                team1_features, team2_features = l_features, w_features
            
            # Create matchup feature dictionary
            matchup = {'ID': matchup_id, 'Season': season, 'Team1': team1, 'Team2': team2, 'Target': target}
            
            # OPTIMIZATION: Process all features in one pass
            for col, team1_val in team1_features.items():
                team2_val = team2_features.get(col)
                
                if team2_val is None or pd.isna(team1_val) or pd.isna(team2_val):
                    continue
                
                # Raw values for each team
                matchup[f'Team1_{col}'] = team1_val
                matchup[f'Team2_{col}'] = team2_val
                
                # Difference between teams (Team1 - Team2)
                matchup[f'Diff_{col}'] = team1_val - team2_val
                
                # Ratio (if applicable and non-zero)
                if team2_val != 0:
                    matchup[f'Ratio_{col}'] = team1_val / team2_val
            
            batch_matchups.append(matchup)
        
        matchups.extend(batch_matchups)
        print(f"Processed {min(i+batch_size, total_games)}/{total_games} games ({len(matchups)} valid matchups)")
    
    print(f"Completed matchup creation. Total matchups: {len(matchups)}, Skipped: {skipped_count}")
    matchups_df = pd.DataFrame(matchups)
    return matchups_df

def create_tournament_predictions(kaggle_data, team_features, season=2025):
    """
    Create prediction dataset for all possible tournament matchups
    
    According to Kaggle requirements:
    - Need predictions for all possible matchups between teams (not just tournament teams)
    - Men's teams have IDs 1000-1999, women's teams have IDs 3000-3999
    - No matchups between men's and women's teams
    - ID format: SSSS_XXXX_YYYY where XXXX is lower TeamID, YYYY is higher TeamID
    """
    # Get all teams for the current season
    m_teams = kaggle_data['MTeams']
    w_teams = kaggle_data['WTeams']
    
    # Filter teams to include only the valid ID ranges
    m_teams = m_teams[m_teams['TeamID'].between(1000, 1999)]
    w_teams = w_teams[w_teams['TeamID'].between(3000, 3999)]
    
    all_teams = pd.concat([m_teams[['TeamID']], w_teams[['TeamID']]], ignore_index=True)
    
    # Convert team features dict to DataFrame if needed
    if isinstance(team_features, dict):
        print("Converting team features dictionary to DataFrame...")
        teams_list = []
        for s in team_features:
            if s != season:
                continue
            for team_id in team_features[s]:
                row = {'Season': s, 'TeamID': team_id}  # Ensure TeamID is explicitly included
                row.update(team_features[s][team_id])
                teams_list.append(row)
        
        if teams_list:
            team_features_df = pd.DataFrame(teams_list)
            print(f"Created features DataFrame with {len(team_features_df)} rows and columns: {team_features_df.columns.tolist()}")
        else:
            # Create an empty DataFrame with the necessary columns to avoid KeyError
            print("WARNING: No teams found in team_features dictionary for season", season)
            team_features_df = pd.DataFrame(columns=['Season', 'TeamID'])
    else:
        # Handle case where team_features is already a DataFrame
        print(f"Working with existing DataFrame of shape {team_features.shape}")
        
        # Check if TeamID column exists
        if 'TeamID' not in team_features.columns:
            # Critical error - we need to fix this
            print("ERROR: TeamID column missing from the features DataFrame")
            print(f"Available columns are: {team_features.columns.tolist()}")
            
            # Try to recover by creating a TeamID column - find any column that might have team IDs
            team_id_candidates = ['Team', 'Team1', 'Team2', 'team_id', 'team_ID']
            renamed = False
            for col in team_id_candidates:
                if col in team_features.columns:
                    team_features = team_features.rename(columns={col: 'TeamID'})
                    print(f"Renamed '{col}' column to 'TeamID'")
                    renamed = True
                    break
                    
            # If we couldn't find a suitable column, create one from the index
            if not renamed:
                print("Creating artificial TeamID column from index")
                team_features = team_features.reset_index(drop=True)
                # Create a plausible range of TeamIDs (1101-1500)
                team_features['TeamID'] = team_features.index.map(lambda i: 1101 + i % 400)
        
        # Filter for just this season
        team_features_df = team_features[team_features['Season'] == season].copy()
    
    if len(team_features_df) == 0:
        print("WARNING: No team features found for season", season)
        # Create a basic DataFrame with just TeamID for all teams in all_teams
        team_features_df = pd.DataFrame({'TeamID': all_teams['TeamID'], 'Season': season})
    
    print(f"Final team_features_df: {len(team_features_df)} rows with columns: {team_features_df.columns.tolist()}")
    
    # Create all possible matchups
    matchups = []
    
    # Get men's teams and women's teams separately
    men_teams = sorted([team for team in all_teams['TeamID'].unique() if 1000 <= team <= 1999])
    women_teams = sorted([team for team in all_teams['TeamID'].unique() if 3000 <= team <= 3999])
    
    # Process men's matchups
    for i, team1 in enumerate(men_teams):
        # Safely filter for team1 - use .copy() to avoid SettingWithCopyWarning
        team1_features = team_features_df[team_features_df['TeamID'] == team1].copy()
        
        if len(team1_features) == 0:
            # If no features, create dummy features
            print(f"Creating dummy features for team {team1}")
            team1_features = pd.DataFrame({'TeamID': [team1], 'Season': [season]})
            
            # Create dummy columns matching the structure of team_features_df
            for col in team_features_df.columns:
                if col not in ['TeamID', 'Season']:
                    team1_features[col] = 0
        
        for team2 in men_teams[i+1:]:
            team2_features = team_features_df[team_features_df['TeamID'] == team2]
            if len(team2_features) == 0:
                # If no features, create dummy features to ensure complete coverage
                team2_features = pd.DataFrame({'TeamID': [team2], 'Season': [season]})
                for col in team_features_df.columns:
                    if col not in ['TeamID', 'Season']:
                        team2_features[col] = 0
            
            # Create matchup ID in required format (SSSS_XXXX_YYYY)
            matchup_id = f"{season}_{min(team1, team2)}_{max(team1, team2)}"
            
            # Define team1 as the lower TeamID
            if team1 > team2:
                team1, team2 = team2, team1
                team1_features, team2_features = team2_features, team1_features
                
            # Create matchup dictionary
            matchup = {'ID': matchup_id, 'Season': season, 'Team1': team1, 'Team2': team2}
            
            # Add features
            for col in team1_features.columns:
                if col not in ['Season', 'TeamID']:
                    try:
                        team1_val = team1_features[col].iloc[0]
                        team2_val = team2_features[col].iloc[0]
                        
                        # Raw values for each team
                        matchup[f'Team1_{col}'] = team1_val
                        matchup[f'Team2_{col}'] = team2_val
                        
                        # Difference between teams (Team1 - Team2)
                        matchup[f'Diff_{col}'] = team1_val - team2_val
                        
                        # Ratio (if applicable and non-zero)
                        if team2_val != 0 and not pd.isna(team2_val) and not pd.isna(team1_val):
                            matchup[f'Ratio_{col}'] = team1_val / team2_val
                    except:
                        continue
            
            matchups.append(matchup)
    
    # Process women's matchups (same logic as men's)
    for i, team1 in enumerate(women_teams):
        team1_features = team_features_df[team_features_df['TeamID'] == team1]
        if len(team1_features) == 0:
            team1_features = pd.DataFrame({'TeamID': [team1], 'Season': [season]})
            for col in team_features_df.columns:
                if col not in ['TeamID', 'Season']:
                    team1_features[col] = 0
        
        for team2 in women_teams[i+1:]:
            team2_features = team_features_df[team_features_df['TeamID'] == team2]
            if len(team2_features) == 0:
                team2_features = pd.DataFrame({'TeamID': [team2], 'Season': [season]})
                for col in team_features_df.columns:
                    if col not in ['TeamID', 'Season']:
                        team2_features[col] = 0
            
            # Create matchup ID in required format
            matchup_id = f"{season}_{min(team1, team2)}_{max(team1, team2)}"
            
            # Define team1 as the lower TeamID
            if team1 > team2:
                team1, team2 = team2, team1
                team1_features, team2_features = team2_features, team1_features
                
            # Create matchup dictionary
            matchup = {'ID': matchup_id, 'Season': season, 'Team1': team1, 'Team2': team2}
            
            # Add features (same as for men)
            for col in team1_features.columns:
                if col not in ['Season', 'TeamID']:
                    try:
                        team1_val = team1_features[col].iloc[0]
                        team2_val = team2_features[col].iloc[0]
                        
                        matchup[f'Team1_{col}'] = team1_val
                        matchup[f'Team2_{col}'] = team2_val
                        matchup[f'Diff_{col}'] = team1_val - team2_val
                        
                        if team2_val != 0 and not pd.isna(team2_val) and not pd.isna(team1_val):
                            matchup[f'Ratio_{col}'] = team1_val / team2_val
                    except:
                        continue
            
            matchups.append(matchup)
    
    matchups_df = pd.DataFrame(matchups)
    print(f"Created {len(matchups_df)} prediction matchups for season {season}")
    print(f"Men's matchups: {sum((matchups_df['Team1'] < 2000) & (matchups_df['Team2'] < 2000))}")
    print(f"Women's matchups: {sum((matchups_df['Team1'] >= 3000) & (matchups_df['Team2'] >= 3000))}")
    
    return matchups_df

def save_datasets(train_df, pred_df, output_dir='/Volumes/MINT/projects/model/data/processed'):
    """
    Save the training and prediction datasets
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'training_matchups.csv')
    pred_path = os.path.join(output_dir, 'prediction_matchups.csv')
    
    train_df.to_csv(train_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    
    print(f"Saved training data ({train_df.shape[0]} matchups) to {train_path}")
    print(f"Saved prediction data ({pred_df.shape[0]} matchups) to {pred_path}")
