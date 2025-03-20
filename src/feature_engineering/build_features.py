import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class TeamFeatureBuilder:
    """
    Class to build team-specific features from historical data
    """
    def __init__(self, kaggle_data, kenpom_data=None):
        self.kaggle_data = kaggle_data
        self.kenpom_data = kenpom_data
        self.team_features = {}
        self.season_features = {}
        
    def build_basic_stats(self, seasons_to_use=range(2003, 2025)):
        """
        Build basic team statistics from regular season results
        """
        features = {}
        
        for season in seasons_to_use:
            # Filter men's results for this season
            m_results = self.kaggle_data['MRegularSeasonCompactResults']
            season_results = m_results[m_results['Season'] == season]
            
            # Get all unique teams for this season
            all_teams = set(season_results['WTeamID'].unique()) | set(season_results['LTeamID'].unique())
            
            season_features = {}
            for team_id in all_teams:
                # Games won by this team
                team_wins = season_results[season_results['WTeamID'] == team_id]
                # Games lost by this team
                team_losses = season_results[season_results['LTeamID'] == team_id]
                
                # Calculate basic stats
                num_wins = len(team_wins)
                num_losses = len(team_losses)
                win_rate = num_wins / (num_wins + num_losses) if (num_wins + num_losses) > 0 else 0
                
                # Points stats
                points_scored = team_wins['WScore'].sum() + team_losses['LScore'].sum()
                points_allowed = team_wins['LScore'].sum() + team_losses['WScore'].sum()
                games_played = num_wins + num_losses
                avg_points_scored = points_scored / games_played if games_played > 0 else 0
                avg_points_allowed = points_allowed / games_played if games_played > 0 else 0
                point_diff = avg_points_scored - avg_points_allowed
                
                # Home/away performance
                home_wins = len(team_wins[team_wins['WLoc'] == 'H'])
                home_games = home_wins + len(team_losses[team_losses['WLoc'] == 'A'])
                home_win_rate = home_wins / home_games if home_games > 0 else 0
                
                away_wins = len(team_wins[team_wins['WLoc'] == 'A'])
                away_games = away_wins + len(team_losses[team_losses['WLoc'] == 'H'])
                away_win_rate = away_wins / away_games if away_games > 0 else 0
                
                # Store features for this team in this season
                season_features[team_id] = {
                    'win_rate': win_rate,
                    'num_wins': num_wins,
                    'num_losses': num_losses,
                    'avg_points_scored': avg_points_scored,
                    'avg_points_allowed': avg_points_allowed,
                    'point_diff': point_diff,
                    'home_win_rate': home_win_rate,
                    'away_win_rate': away_win_rate
                }
            
            # Store all team features for this season
            features[season] = season_features
            
        self.basic_stats = features
        return features
    
    def build_advanced_stats(self, seasons_to_use=range(2003, 2025)):
        """
        Build advanced team statistics using detailed box scores
        """
        features = {}
        
        for season in seasons_to_use:
            try:
                # Filter men's detailed results for this season
                m_detailed = self.kaggle_data['MRegularSeasonDetailedResults']
                season_detailed = m_detailed[m_detailed['Season'] == season]
                
                # Get all unique teams for this season
                all_teams = set(season_detailed['WTeamID'].unique()) | set(season_detailed['LTeamID'].unique())
                
                season_features = {}
                for team_id in all_teams:
                    # Games won by this team
                    wins = season_detailed[season_detailed['WTeamID'] == team_id]
                    # Games lost by this team
                    losses = season_detailed[season_detailed['LTeamID'] == team_id]
                    
                    games_played = len(wins) + len(losses)
                    if games_played == 0:
                        continue
                    
                    # Advanced stats when team won
                    win_stats = {
                        'FGM': wins['WFGM'].sum(),
                        'FGA': wins['WFGA'].sum(),
                        'FGM3': wins['WFGM3'].sum(),
                        'FGA3': wins['WFGA3'].sum(),
                        'FTM': wins['WFTM'].sum(),
                        'FTA': wins['WFTA'].sum(),
                        'OR': wins['WOR'].sum(),
                        'DR': wins['WDR'].sum(),
                        'Ast': wins['WAst'].sum(),
                        'TO': wins['WTO'].sum(),
                        'Stl': wins['WStl'].sum(),
                        'Blk': wins['WBlk'].sum(),
                        'PF': wins['WPF'].sum()
                    }
                    
                    # Advanced stats when team lost
                    loss_stats = {
                        'FGM': losses['LFGM'].sum(),
                        'FGA': losses['LFGA'].sum(),
                        'FGM3': losses['LFGM3'].sum(),
                        'FGA3': losses['LFGA3'].sum(),
                        'FTM': losses['LFTM'].sum(),
                        'FTA': losses['LFTA'].sum(),
                        'OR': losses['LOR'].sum(),
                        'DR': losses['LDR'].sum(),
                        'Ast': losses['LAst'].sum(),
                        'TO': losses['LTO'].sum(),
                        'Stl': losses['LStl'].sum(),
                        'Blk': losses['LBlk'].sum(),
                        'PF': losses['LPF'].sum()
                    }
                    
                    # Combine stats from wins and losses
                    combined_stats = {}
                    for stat in win_stats:
                        combined_stats[stat] = win_stats[stat] + loss_stats[stat]
                    
                    # Calculate averages and percentages
                    season_features[team_id] = {
                        'FG_pct': combined_stats['FGM'] / combined_stats['FGA'] if combined_stats['FGA'] > 0 else 0,
                        'FG3_pct': combined_stats['FGM3'] / combined_stats['FGA3'] if combined_stats['FGA3'] > 0 else 0,
                        'FT_pct': combined_stats['FTM'] / combined_stats['FTA'] if combined_stats['FTA'] > 0 else 0,
                        'avg_assists': combined_stats['Ast'] / games_played,
                        'avg_rebounds': (combined_stats['OR'] + combined_stats['DR']) / games_played,
                        'avg_turnovers': combined_stats['TO'] / games_played,
                        'assist_to_turnover': combined_stats['Ast'] / combined_stats['TO'] if combined_stats['TO'] > 0 else 0,
                        'avg_steals': combined_stats['Stl'] / games_played,
                        'avg_blocks': combined_stats['Blk'] / games_played,
                        'avg_fouls': combined_stats['PF'] / games_played
                    }
                
                # Store all team advanced features for this season
                features[season] = season_features
            except KeyError:
                print(f"Detailed results not available for season {season}")
        
        self.advanced_stats = features
        return features
    
    def integrate_kenpom_data(self, seasons_to_use=range(2003, 2025)):
        """
        Integrate KenPom data if available
        """
        if self.kenpom_data is None:
            print("No KenPom data available to integrate")
            return {}
        
        # Process KenPom data - this will need to be adapted based on actual KenPom data structure
        # For now, assuming KenPom data has Season, TeamID, and various statistics columns
        features = {}
        
        for season in seasons_to_use:
            season_kenpom = self.kenpom_data[self.kenpom_data['Season'] == season] if 'Season' in self.kenpom_data.columns else None
            
            if season_kenpom is not None and not season_kenpom.empty:
                features[season] = {}
                for _, row in season_kenpom.iterrows():
                    team_id = row['TeamID']
                    
                    # Extract KenPom stats - adjust column names as needed
                    kenpom_features = {}
                    for col in season_kenpom.columns:
                        if col not in ['Season', 'TeamID', 'TeamName']:
                            kenpom_features[f'kenpom_{col}'] = row[col]
                    
                    features[season][team_id] = kenpom_features
            
        self.kenpom_features = features
        return features
    
    def combine_all_features(self, seasons_to_use=range(2003, 2025)):
        """
        Combine basic stats, advanced stats, and KenPom data
        """
        all_features = {}
        
        # Make sure we have built all feature types
        if not hasattr(self, 'basic_stats'):
            self.build_basic_stats(seasons_to_use)
        if not hasattr(self, 'advanced_stats'):
            self.build_advanced_stats(seasons_to_use)
        if not hasattr(self, 'kenpom_features'):
            self.integrate_kenpom_data(seasons_to_use)
        
        # Combine all features
        for season in seasons_to_use:
            all_features[season] = {}
            
            # Get all teams across all feature sets for this season
            all_teams = set()
            if season in self.basic_stats:
                all_teams.update(self.basic_stats[season].keys())
            if season in self.advanced_stats:
                all_teams.update(self.advanced_stats[season].keys())
            if season in self.kenpom_features:
                all_teams.update(self.kenpom_features[season].keys())
            
            # Combine features for each team
            for team_id in all_teams:
                team_features = {}
                
                # Add basic stats if available
                if season in self.basic_stats and team_id in self.basic_stats[season]:
                    team_features.update(self.basic_stats[season][team_id])
                
                # Add advanced stats if available
                if season in self.advanced_stats and team_id in self.advanced_stats[season]:
                    team_features.update(self.advanced_stats[season][team_id])
                
                # Add KenPom features if available
                if season in self.kenpom_features and team_id in self.kenpom_features[season]:
                    team_features.update(self.kenpom_features[season][team_id])
                
                all_features[season][team_id] = team_features
        
        self.all_features = all_features
        return all_features
    
    def save_features(self, output_path='/Volumes/MINT/projects/model/features'):
        """
        Save all feature sets to CSV files
        """
        if not hasattr(self, 'all_features'):
            print("No features to save. Run combine_all_features() first.")
            return
        
        # Convert nested dict to dataframe
        rows = []
        for season in self.all_features:
            for team_id in self.all_features[season]:
                row = {'Season': season, 'TeamID': team_id}
                row.update(self.all_features[season][team_id])
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save to CSV
        output_file = f"{output_path}/team_features.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved team features to {output_file}")
