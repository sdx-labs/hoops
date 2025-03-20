import pandas as pd
import numpy as np
import os

def create_historical_matchups(kaggle_data, team_features):
    """
    Create training data from historical matchups with features
    """
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
    
    # Convert team features dict to DataFrame if needed
    if isinstance(team_features, dict):
        teams_list = []
        for season in team_features:
            for team_id in team_features[season]:
                row = {'Season': season, 'TeamID': team_id}
                row.update(team_features[season][team_id])
                teams_list.append(row)
        team_features_df = pd.DataFrame(teams_list)
    else:
        team_features_df = team_features
    
    # Create matchup features
    matchups = []
    
    for _, game in all_games.iterrows():
        season = game['Season']
        w_team = game['WTeamID']
        l_team = game['LTeamID']
        
        # Get team features for this season
        season_features = team_features_df[team_features_df['Season'] == season]
        
        # Get features for winning and losing teams
        w_features = season_features[season_features['TeamID'] == w_team]
        l_features = season_features[season_features['TeamID'] == l_team]
        
        if len(w_features) == 0 or len(l_features) == 0:
            continue  # Skip if features not available
        
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
        
        # Add differential features
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
        teams_list = []
        for s in team_features:
            if s != season:
                continue
            for team_id in team_features[s]:
                row = {'Season': s, 'TeamID': team_id}
                row.update(team_features[s][team_id])
                teams_list.append(row)
        team_features_df = pd.DataFrame(teams_list)
    else:
        team_features_df = team_features[team_features['Season'] == season]
    
    # Create all possible matchups
    matchups = []
    
    # Get men's teams and women's teams separately
    men_teams = sorted([team for team in all_teams['TeamID'].unique() if 1000 <= team <= 1999])
    women_teams = sorted([team for team in all_teams['TeamID'].unique() if 3000 <= team <= 3999])
    
    # Process men's matchups
    for i, team1 in enumerate(men_teams):
        team1_features = team_features_df[team_features_df['TeamID'] == team1]
        if len(team1_features) == 0:
            # If no features, create dummy features to ensure complete coverage
            team1_features = pd.DataFrame({'TeamID': [team1], 'Season': [season]})
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
