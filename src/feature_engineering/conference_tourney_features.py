import pandas as pd
import numpy as np

def build_conference_tourney_features(kaggle_data, seasons=range(2001, 2025)):
    """
    Build features from conference tournament performance
    """
    features = {}
    
    for season in seasons:
        # Get men's conference tournament data for this season
        m_conf_games = kaggle_data.get('MConferenceTourneyGames', pd.DataFrame())
        season_games = m_conf_games[m_conf_games['Season'] == season]
        
        if len(season_games) == 0:
            continue
            
        # Get unique conferences for this season
        conferences = season_games['ConfAbbrev'].unique()
        
        # Initialize season features
        season_features = {}
        
        # Process each conference tournament
        for conf in conferences:
            conf_games = season_games[season_games['ConfAbbrev'] == conf]
            
            if len(conf_games) == 0:
                continue  # Skip empty conferences
            
            # Get all teams in this conference tournament
            winning_teams = set(conf_games['WTeamID'].unique())
            losing_teams = set(conf_games['LTeamID'].unique())
            all_teams = winning_teams.union(losing_teams)
            
            # Get the champion (team that won the last game)
            try:
                last_day = conf_games['DayNum'].max()
                last_games = conf_games[conf_games['DayNum'] == last_day]
                if len(last_games) > 0:
                    champion = last_games['WTeamID'].iloc[0]
                else:
                    champion = None  # No games found on last day
            except (IndexError, KeyError):
                champion = None  # Error finding champion
            
            # Process each team
            for team_id in all_teams:
                # Initialize team features if not already present
                if team_id not in season_features:
                    season_features[team_id] = {
                        'conf_tourney_games': 0,
                        'conf_tourney_wins': 0,
                        'conf_tourney_losses': 0,
                        'is_conf_champion': 0,
                        'conf_win_rate': 0
                    }
                
                # Games won by this team
                team_wins = conf_games[conf_games['WTeamID'] == team_id]
                wins = len(team_wins)
                
                # Games lost by this team
                team_losses = conf_games[conf_games['LTeamID'] == team_id]
                losses = len(team_losses)
                
                # Update features
                season_features[team_id]['conf_tourney_games'] = wins + losses
                season_features[team_id]['conf_tourney_wins'] = wins
                season_features[team_id]['conf_tourney_losses'] = losses
                season_features[team_id]['is_conf_champion'] = 1 if team_id == champion else 0
                
                # Win rate
                if wins + losses > 0:
                    season_features[team_id]['conf_win_rate'] = wins / (wins + losses)
        
        features[season] = season_features
    
    return features

def add_conference_features_to_team_data(team_features, conf_features):
    """
    Add conference tournament features to the main team features dictionary
    """
    enhanced_features = {}
    
    # Add conference features
    for season in team_features:
        enhanced_features[season] = {}
        
        for team_id in team_features[season]:
            # Copy original features
            enhanced_features[season][team_id] = team_features[season][team_id].copy()
            
            # Add conference features if available
            if season in conf_features and team_id in conf_features[season]:
                enhanced_features[season][team_id].update(conf_features[season][team_id])
            else:
                # Add default values if not available
                enhanced_features[season][team_id].update({
                    'conf_tourney_games': 0,
                    'conf_tourney_wins': 0,
                    'conf_tourney_losses': 0,
                    'is_conf_champion': 0,
                    'conf_win_rate': 0
                })
    
    return enhanced_features
