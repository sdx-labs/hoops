import os
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
        teams_processed = 0
        
        print(f"Building basic stats for {len(list(seasons_to_use))} seasons")
        
        for season in seasons_to_use:
            # Filter men's results for this season
            if 'MRegularSeasonCompactResults' not in self.kaggle_data:
                print(f"ERROR: MRegularSeasonCompactResults not found in data for season {season}")
                continue
                
            m_results = self.kaggle_data['MRegularSeasonCompactResults']
            season_results = m_results[m_results['Season'] == season]
            
            if len(season_results) == 0:
                print(f"No games found for season {season}")
                continue
                
            print(f"Processing {len(season_results)} games for season {season}")
            
            # Get all unique teams for this season
            all_teams = set(season_results['WTeamID'].unique()) | set(season_results['LTeamID'].unique())
            
            season_features = {}
            teams_in_season = 0
            
            for team_id in all_teams:
                # Games won by this team
                team_wins = season_results[season_results['WTeamID'] == team_id]
                # Games lost by this team
                team_losses = season_results[season_results['LTeamID'] == team_id]
                
                # Calculate basic stats
                num_wins = len(team_wins)
                num_losses = len(team_losses)
                
                # Only process teams with sufficient games
                if (num_wins + num_losses) < 5:
                    continue
                
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
                
                teams_in_season += 1
                teams_processed += 1
            
            # Store all team features for this season
            features[season] = season_features
            print(f"Generated features for {teams_in_season} teams in season {season}")
        
        print(f"Total: processed {teams_processed} teams across all seasons")
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
        
        # Process KenPom data
        features = {}
        
        # Print column names to help diagnose issues
        print("KenPom data columns: ", self.kenpom_data.columns.tolist())
        
        # Try to load an existing team mapping file
        mapping_path = '/Volumes/MINT/projects/model/data/kenpom/team_mapping.csv'
        team_mapping = None
        if os.path.exists(mapping_path):
            try:
                mapping_df = pd.read_csv(mapping_path)
                team_mapping = dict(zip(mapping_df['KenPomTeam'], mapping_df['KaggleTeamID']))
                print(f"Loaded team mapping for {len(team_mapping)} teams")
            except Exception as e:
                print(f"Error loading team mapping: {str(e)}")
        
        # Check if TeamID column exists or try to create one
        team_id_col = None
        if 'TeamID' in self.kenpom_data.columns:
            team_id_col = 'TeamID'
            print("Found existing TeamID column in KenPom data")
        elif team_mapping is not None:
            # Find team name column to apply mapping - prioritize "Mapped ESPN Team Name"
            team_name_col = None
            if 'Mapped ESPN Team Name' in self.kenpom_data.columns:
                team_name_col = 'Mapped ESPN Team Name'
                print(f"Using 'Mapped ESPN Team Name' column for team ID mapping")
            elif 'Full Team Name' in self.kenpom_data.columns:
                team_name_col = 'Full Team Name'
                print(f"Using 'Full Team Name' column for team ID mapping")
            else:
                # Fall back to other columns
                for col in ['Team', 'TeamName', 'team', 'NAME', 'name']:
                    if col in self.kenpom_data.columns:
                        team_name_col = col
                        print(f"Falling back to '{col}' column for team ID mapping")
                        break
            
            if team_name_col:
                # Apply the mapping to create a TeamID column
                self.kenpom_data['TeamID'] = self.kenpom_data[team_name_col].map(team_mapping)
                team_id_col = 'TeamID'
                mapped_count = self.kenpom_data['TeamID'].notna().sum()
                print(f"Created TeamID column by mapping {mapped_count} out of {len(self.kenpom_data)} rows")
            else:
                print("Couldn't find a team name column to apply the mapping")
                print("Expected column name: 'Full Team Name'")
        
        # Handle the case where we still don't have a TeamID column
        if team_id_col is None:
            print("WARNING: No TeamID column in KenPom data. Trying to create a mapping...")
            
            # Create mapping from team names to TeamIDs - prioritize "Mapped ESPN Team Name"
            team_name_col = None
            if 'Mapped ESPN Team Name' in self.kenpom_data.columns:
                team_name_col = 'Mapped ESPN Team Name'
                print(f"Using 'Mapped ESPN Team Name' column for team ID mapping")
            elif 'Full Team Name' in self.kenpom_data.columns:
                team_name_col = 'Full Team Name'
                print(f"Using 'Full Team Name' column for team ID mapping")
            else:
                for col in ['Team', 'TeamName', 'team', 'NAME', 'name']:
                    if col in self.kenpom_data.columns:
                        team_name_col = col
                        print(f"Falling back to '{col}' column for team ID mapping")
                        break
            
            if team_name_col and 'MTeams' in self.kaggle_data:
                print(f"Creating mapping using {team_name_col} column")
                
                # Create a mapping dict
                name_to_id = {}
                for _, team_row in self.kaggle_data['MTeams'].iterrows():
                    if 'TeamName' in team_row and 'TeamID' in team_row:
                        name_to_id[team_row['TeamName'].lower()] = team_row['TeamID']
                
                # Add alternate spellings if available
                if 'MTeamSpellings' in self.kaggle_data:
                    for _, row in self.kaggle_data['MTeamSpellings'].iterrows():
                        if 'TeamNameSpelling' in row and 'TeamID' in row:
                            name_to_id[row['TeamNameSpelling'].lower()] = row['TeamID']
                
                # Create standardization function for team names
                def standardize_name(name):
                    if pd.isna(name):
                        return ""
                    name = str(name).lower()
                    # Remove common suffixes
                    for suffix in [" university", " college", " u", "univ ", "univ. "]:
                        name = name.replace(suffix, "")
                    return name.strip()
                
                # Apply mapping
                self.kenpom_data['StandardName'] = self.kenpom_data[team_name_col].apply(standardize_name)
                self.kenpom_data['TeamID'] = self.kenpom_data['StandardName'].map(
                    {standardize_name(k): v for k, v in name_to_id.items()}
                )
                
                team_id_col = 'TeamID'
                mapped_count = self.kenpom_data['TeamID'].notna().sum()
                print(f"Created TeamID column by direct matching for {mapped_count} out of {len(self.kenpom_data)} rows")
                
                # Try fuzzy matching for unmatched teams (if fuzzywuzzy is available)
                try:
                    from fuzzywuzzy import process, fuzz
                    
                    # Get unmatched teams
                    unmatched = self.kenpom_data[self.kenpom_data['TeamID'].isna()].copy()
                    
                    if not unmatched.empty:
                        print(f"Using fuzzy matching for {len(unmatched)} unmatched teams")
                        std_names = list(name_to_id.keys())
                        std_names = [standardize_name(name) for name in std_names]
                        
                        # Create lookup dict
                        std_to_id = {standardize_name(k): v for k, v in name_to_id.items()}
                        
                        # Apply fuzzy matching
                        for idx, row in unmatched.iterrows():
                            team_name = row['StandardName']
                            best_match, score = process.extractOne(team_name, std_names, scorer=fuzz.token_sort_ratio)
                            if score >= 85:  # Only accept high confidence matches
                                self.kenpom_data.at[idx, 'TeamID'] = std_to_id.get(best_match)
                        
                        new_mapped_count = self.kenpom_data['TeamID'].notna().sum()
                        print(f"After fuzzy matching: mapped {new_mapped_count} out of {len(self.kenpom_data)} rows")
                except ImportError:
                    print("fuzzywuzzy module not available for fuzzy team name matching")
            else:
                print("ERROR: Cannot map team names to IDs - required columns not found")
                return {}
        
        # Define possible team columns
        possible_team_cols = ['Team', 'TeamName', 'team', 'NAME', 'name']

        # Get the season column
        season_col = None
        possible_season_cols = ['Season', 'Year', 'season', 'SEASON', 'year']
        for col in possible_season_cols:
            if col in self.kenpom_data.columns:
                season_col = col
                print(f"Found season column: {col}")
                break
        
        if season_col is None:
            print("ERROR: No Season column found in KenPom data.")
            return {}
        
        # Now we can proceed with feature extraction
        for season in seasons_to_use:
            season_kenpom = self.kenpom_data[self.kenpom_data[season_col] == season]
            
            if not season_kenpom.empty:
                features[season] = {}
                for _, row in season_kenpom.iterrows():
                    if pd.isna(row[team_id_col]):
                        continue  # Skip rows without a mapped TeamID
                        
                    team_id = int(row[team_id_col])  # Convert to int to match other data
                    
                    # Extract KenPom stats (exclude non-feature columns)
                    kenpom_features = {}
                    exclude_cols = [team_id_col, season_col, 'StandardName'] + possible_team_cols
                    for col in season_kenpom.columns:
                        if col not in exclude_cols:
                            try:
                                # Only add non-null numeric values
                                if not pd.isna(row[col]) and pd.api.types.is_numeric_dtype(season_kenpom[col]):
                                    kenpom_features[f'kenpom_{col}'] = row[col]
                            except Exception as e:
                                print(f"Error processing column {col}: {str(e)}")
                    
                    features[season][team_id] = kenpom_features
                
                print(f"Added KenPom features for {len(features[season])} teams in season {season}")
            else:
                print(f"No KenPom data found for season {season}")
        
        print(f"Integrated KenPom data for {len(features)} seasons")
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
        feature_counts = {'basic': 0, 'advanced': 0, 'kenpom': 0}
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
                    basic_stats = self.basic_stats[season][team_id]
                    team_features.update(basic_stats)
                    if len(basic_stats) > feature_counts['basic']:
                        feature_counts['basic'] = len(basic_stats)
                
                # Add advanced stats if available
                if season in self.advanced_stats and team_id in self.advanced_stats[season]:
                    adv_stats = self.advanced_stats[season][team_id]
                    team_features.update(adv_stats)
                    if len(adv_stats) > feature_counts['advanced']:
                        feature_counts['advanced'] = len(adv_stats)
                
                # Add KenPom features if available
                if season in self.kenpom_features and team_id in self.kenpom_features[season]:
                    kp_stats = self.kenpom_features[season][team_id]
                    team_features.update(kp_stats)
                    if len(kp_stats) > feature_counts['kenpom']:
                        feature_counts['kenpom'] = len(kp_stats)
                
                all_features[season][team_id] = team_features
        
        self.all_features = all_features
        
        # Print summary of features
        print("\n--- Feature Generation Summary ---")
        print(f"Seasons processed: {len([s for s in seasons_to_use if s in all_features])}")
        print(f"Total teams: {sum(len(all_features[s]) for s in all_features)}")
        print(f"Feature counts:")
        print(f"  Basic stats: {feature_counts['basic']}")
        print(f"  Advanced stats: {feature_counts['advanced']}")
        print(f"  KenPom stats: {feature_counts['kenpom']}")
        print(f"Total features: {sum(feature_counts.values())}")
        
        # Print sample features for a random team
        if all_features:
            sample_season = list(all_features.keys())[0]
            if all_features[sample_season]:
                sample_team = list(all_features[sample_season].keys())[0]
                print(f"\nSample features for team {sample_team} in season {sample_season}:")
                
                # Print first 5 features
                sample_features = all_features[sample_season][sample_team]
                for i, (feature, value) in enumerate(list(sample_features.items())[:5]):
                    print(f"  {feature}: {value}")
                
                if len(sample_features) > 5:
                    print(f"  ... and {len(sample_features)-5} more features")
        
        return all_features
    
    def save_features(self, output_path='/Volumes/MINT/projects/model/features'):
        """
        Save all feature sets to CSV files.
        """
        # Ensure the output directory exists
        os.makedirs(output_path, exist_ok=True)
        
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
        output_file = f"{output_path}/team_features.csv"
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Saved team features to {output_file}")
