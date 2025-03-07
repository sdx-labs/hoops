import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Import our custom data collectors and processors
from src.data.womens_basketball_data import WomensBasketballDataCollector
from src.features.performance_tracker import TeamPerformanceTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasketballFeatureEngineer:
    def __init__(self, processed_data_dir="data/processed", features_dir="data/features", external_data_dir="data/external"):
        self.processed_data_dir = Path(processed_data_dir)
        self.features_dir = Path(features_dir)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.external_data_dir = Path(external_data_dir)
        
        # Initialize our data collectors and processors
        self.womens_data_collector = WomensBasketballDataCollector(
            data_dir=self.external_data_dir / "womens"
        )
        self.performance_tracker = TeamPerformanceTracker(
            window_sizes=[1, 3, 5, 10], 
            max_streak_for_normalization=15
        )
        
    def load_matchup_data(self, gender="M"):
        """Load processed matchup data"""
        try:
            file_path = self.processed_data_dir / f"{gender}_matchup_results.csv"
            if not file_path.exists():
                logger.error(f"Matchup file not found: {file_path}")
                return None
                
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {gender} matchup data with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading {gender} matchup data: {str(e)}")
            return None
            
    def load_kenpom_data(self):
        """Load KenPom ratings data"""
        try:
            # Try to load real KenPom data first
            kenpom_file = self.external_data_dir / "kenpom_historical_ratings.csv"
            
            if not kenpom_file.exists():
                # Fall back to sample data if real data is not available
                kenpom_file = self.external_data_dir / "kenpom_sample_data.csv"
                
            if not kenpom_file.exists():
                logger.warning("KenPom data not found")
                return None
                
            kenpom_df = pd.read_csv(kenpom_file)
            logger.info(f"Loaded KenPom data with {len(kenpom_df)} entries")
            return kenpom_df
            
        except Exception as e:
            logger.error(f"Error loading KenPom data: {str(e)}")
            return None
            
    def load_womens_ratings_data(self, season=None):
        """
        Load women's basketball ratings data as an alternative to KenPom
        
        Args:
            season: Specific season to load (None for all seasons)
        """
        try:
            # Get women's ratings from our collector
            womens_ratings = self.womens_data_collector.get_womens_ratings(season)
            
            if womens_ratings is None or womens_ratings.empty:
                logger.warning("No women's ratings data found")
                return None
                
            logger.info(f"Loaded women's ratings data with {len(womens_ratings)} entries")
            return womens_ratings
            
        except Exception as e:
            logger.error(f"Error loading women's ratings data: {str(e)}")
            return None
    
    def merge_advanced_metrics_with_team_data(self, team_stats_df, gender="M"):
        """
        Merge advanced metrics (KenPom for men, alternative for women) with team stats
        """
        try:
            if gender == "M":
                # For men's teams, use KenPom data
                kenpom_df = self.load_kenpom_data()
                
                if kenpom_df is None:
                    return team_stats_df
                
                # Check if we need to apply a team mapping
                mapping_file = self.external_data_dir / "kenpom_team_mapping.csv"
                
                if mapping_file.exists():
                    # Use mapping file to match teams
                    mapping_df = pd.read_csv(mapping_file)
                    # Apply mapping logic here
                    logger.info("Applied KenPom team mapping")
                
                # Merge KenPom data with team stats 
                merged_df = pd.merge(
                    team_stats_df,
                    kenpom_df[['Season', 'TeamID', 'AdjO', 'AdjD', 'AdjTempo', 'Luck', 'Strength_of_Schedule', 'KenPomRank']],
                    left_on=['Season', 'TeamID'],
                    right_on=['Season', 'TeamID'],
                    how='left'
                )
            else:
                # For women's teams, use our alternative ratings
                womens_df = self.load_womens_ratings_data()
                
                if womens_df is None:
                    return team_stats_df
                
                # Select the same columns for compatibility with men's data
                ratings_columns = ['Season', 'TeamID', 'AdjO', 'AdjD', 'AdjTempo', 'Luck', 'Strength_of_Schedule', 'KenPomRank']
                
                # Merge women's ratings with team stats
                merged_df = pd.merge(
                    team_stats_df,
                    womens_df[ratings_columns],
                    left_on=['Season', 'TeamID'],
                    right_on=['Season', 'TeamID'],
                    how='left'
                )
            
            # Fill NaN values for teams missing advanced metrics data
            advanced_columns = ['AdjO', 'AdjD', 'AdjTempo', 'Luck', 'Strength_of_Schedule', 'KenPomRank']
            for col in advanced_columns:
                if col in merged_df.columns:
                    merged_df[col].fillna(merged_df[col].mean(), inplace=True)
            
            logger.info(f"Merged advanced metrics with {gender} team stats")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error merging advanced metrics data: {str(e)}")
            return team_stats_df
    
    def calculate_team_stats(self, games_df, gender):
        """Calculate team statistics per season"""
        team_stats_by_season = {}
        
        for season in games_df['Season'].unique():
            season_games = games_df[games_df['Season'] == season]
            
            # Get all unique teams
            all_teams = set(season_games['Team1ID'].unique()) | set(season_games['Team2ID'].unique())
            
            team_stats = {team_id: {
                'wins': 0,
                'losses': 0,
                'points_scored': 0,
                'points_allowed': 0,
                'games_played': 0,
                'point_diff': 0
            } for team_id in all_teams}
            
            # Process each game to update team stats
            for _, game in season_games.iterrows():
                team1_id = game['Team1ID']
                team2_id = game['Team2ID']
                
                # Update Team1 stats
                team_stats[team1_id]['games_played'] += 1
                team_stats[team1_id]['points_scored'] += game['Team1Score']
                team_stats[team1_id]['points_allowed'] += game['Team2Score']
                team_stats[team1_id]['point_diff'] += game['ScoreDiff']
                
                if game['Result'] == 1:  # Team1 won
                    team_stats[team1_id]['wins'] += 1
                else:  # Team1 lost
                    team_stats[team1_id]['losses'] += 1
                
                # Update Team2 stats
                team_stats[team2_id]['games_played'] += 1
                team_stats[team2_id]['points_scored'] += game['Team2Score']
                team_stats[team2_id]['points_allowed'] += game['Team1Score']
                team_stats[team2_id]['point_diff'] -= game['ScoreDiff']  # Negative of Team1's diff
                
                if game['Result'] == 0:  # Team2 won
                    team_stats[team2_id]['wins'] += 1
                else:  # Team2 lost
                    team_stats[team2_id]['losses'] += 1
            
            # Calculate derived statistics
            for team_id, stats in team_stats.items():
                if stats['games_played'] > 0:
                    stats['win_rate'] = stats['wins'] / stats['games_played']
                    stats['avg_score'] = stats['points_scored'] / stats['games_played']
                    stats['avg_allowed'] = stats['points_allowed'] / stats['games_played']
                    stats['avg_point_diff'] = stats['point_diff'] / stats['games_played']
                else:
                    stats['win_rate'] = 0
                    stats['avg_score'] = 0
                    stats['avg_allowed'] = 0
                    stats['avg_point_diff'] = 0
            
            team_stats_by_season[season] = team_stats
            
        # Convert to DataFrame
        stats_data = []
        for season, season_stats in team_stats_by_season.items():
            for team_id, stats in season_stats.items():
                stats_row = {
                    'Season': season,
                    'TeamID': team_id,
                    'Gender': gender,
                    **stats  # Unpack all stats
                }
                stats_data.append(stats_row)
        
        team_stats_df = pd.DataFrame(stats_data)
        
        # Add advanced metrics (KenPom for men, alternative for women)
        enhanced_stats_df = self.merge_advanced_metrics_with_team_data(team_stats_df, gender)
        
        # Save team stats
        output_path = self.features_dir / f"{gender}_team_stats.csv"
        enhanced_stats_df.to_csv(output_path, index=False)
        logger.info(f"Saved {gender} team stats with advanced metrics to {output_path}")
        
        return enhanced_stats_df, team_stats_by_season
    
    def organize_games_by_team(self, games_df):
        """
        Organize games by team and season for performance tracking
        
        Args:
            games_df: DataFrame of games with chronological order
            
        Returns:
            Dictionary of games organized by season and team
        """
        logger.info("Organizing games by team and season")
        
        # Ensure games are sorted chronologically
        if 'DayNum' in games_df.columns:
            games_df = games_df.sort_values(['Season', 'DayNum'])
            
        games_by_team = {}
        
        for _, game in games_df.iterrows():
            season = game['Season']
            team1_id = game['Team1ID']
            team2_id = game['Team2ID']
            result = game['Result']  # 1 if Team1 won, 0 if Team2 won
            
            # Initialize season dictionary if needed
            if season not in games_by_team:
                games_by_team[season] = {}
                
            # Initialize team lists if needed
            if team1_id not in games_by_team[season]:
                games_by_team[season][team1_id] = []
            if team2_id not in games_by_team[season]:
                games_by_team[season][team2_id] = []
                
            # Add game to Team1's list (from Team1's perspective)
            team1_game = game.to_dict()
            team1_game['Result'] = result  # 1 if won, 0 if lost
            games_by_team[season][team1_id].append(team1_game)
            
            # Add game to Team2's list (from Team2's perspective)
            team2_game = game.to_dict()
            team2_game['Result'] = 1 - result  # 0 if lost, 1 if won
            # Swap team IDs for correct perspective
            team2_game['Team1ID'], team2_game['Team2ID'] = team2_game['Team2ID'], team2_game['Team1ID']
            team2_game['Team1Score'], team2_game['Team2Score'] = team2_game['Team2Score'], team2_game['Team1Score']
            team2_game['ScoreDiff'] = -team2_game['ScoreDiff']  # Reverse score diff
            games_by_team[season][team2_id].append(team2_game)
            
        return games_by_team
    
    def create_matchup_features(self, gender="M"):
        """Create features for matchups"""
        matchups = self.load_matchup_data(gender)
        if matchups is None:
            return None
        
        # Calculate team stats by season
        team_stats_df, team_stats_by_season = self.calculate_team_stats(matchups, gender)
        
        # Organize games by team for performance tracking
        games_by_team = self.organize_games_by_team(matchups)
        
        # Create features for each matchup
        feature_rows = []
        
        for _, matchup in matchups.iterrows():
            season = matchup['Season']
            team1_id = matchup['Team1ID']
            team2_id = matchup['Team2ID']
            
            # Skip if we don't have stats for this season
            if season not in team_stats_by_season:
                continue
                
            season_stats = team_stats_by_season[season]
            
            # Skip if we don't have stats for either team
            if team1_id not in season_stats or team2_id not in season_stats:
                continue
            
            team1_stats = season_stats[team1_id]
            team2_stats = season_stats[team2_id]
            
            # Create feature row
            feature_row = {
                'Season': season,
                'Team1ID': team1_id,
                'Team2ID': team2_id,
                'Gender': gender,
                
                # Team 1 features
                'Team1_WinRate': team1_stats['win_rate'],
                'Team1_AvgScore': team1_stats['avg_score'],
                'Team1_AvgAllowed': team1_stats['avg_allowed'],
                'Team1_AvgPointDiff': team1_stats['avg_point_diff'],
                'Team1_GamesPlayed': team1_stats['games_played'],
                
                # Team 2 features
                'Team2_WinRate': team2_stats['win_rate'],
                'Team2_AvgScore': team2_stats['avg_score'],
                'Team2_AvgAllowed': team2_stats['avg_allowed'],
                'Team2_AvgPointDiff': team2_stats['avg_point_diff'],
                'Team2_GamesPlayed': team2_stats['games_played'],
                
                # Comparative features
                'WinRate_Diff': team1_stats['win_rate'] - team2_stats['win_rate'],
                'AvgScore_Diff': team1_stats['avg_score'] - team2_stats['avg_score'],
                'AvgAllowed_Diff': team1_stats['avg_allowed'] - team2_stats['avg_allowed'],
                'AvgPointDiff_Diff': team1_stats['avg_point_diff'] - team2_stats['avg_point_diff'],
                
                # Advanced metrics features
                'Team1_AdjO': team1_stats.get('AdjO', None),
                'Team1_AdjD': team1_stats.get('AdjD', None),
                'Team1_AdjTempo': team1_stats.get('AdjTempo', None),
                'Team1_Luck': team1_stats.get('Luck', None),
                'Team1_SOS': team1_stats.get('Strength_of_Schedule', None),
                'Team1_AdvancedRank': team1_stats.get('KenPomRank', None),
                
                'Team2_AdjO': team2_stats.get('AdjO', None),
                'Team2_AdjD': team2_stats.get('AdjD', None),
                'Team2_AdjTempo': team2_stats.get('AdjTempo', None),
                'Team2_Luck': team2_stats.get('Luck', None),
                'Team2_SOS': team2_stats.get('Strength_of_Schedule', None),
                'Team2_AdvancedRank': team2_stats.get('KenPomRank', None),
                
                # Advanced metrics comparative features
                'AdjO_Diff': team1_stats.get('AdjO', 0) - team2_stats.get('AdjO', 0),
                'AdjD_Diff': team1_stats.get('AdjD', 0) - team2_stats.get('AdjD', 0),
                'AdvancedRank_Diff': team2_stats.get('KenPomRank', 0) - team1_stats.get('KenPomRank', 0),
                
                # Target variable
                'Result': matchup['Result']
            }
            
            # Add gender-specific features
            if gender == "W":
                # Add women's specific metrics if available
                if 'OffensiveRating' in team1_stats:
                    feature_row['Team1_OffensiveRating'] = team1_stats['OffensiveRating']
                    feature_row['Team2_OffensiveRating'] = team2_stats['OffensiveRating']
                    feature_row['OffensiveRating_Diff'] = team1_stats['OffensiveRating'] - team2_stats['OffensiveRating']
                
                if 'DefensiveRating' in team1_stats:
                    feature_row['Team1_DefensiveRating'] = team1_stats['DefensiveRating']
                    feature_row['Team2_DefensiveRating'] = team2_stats['DefensiveRating']
                    feature_row['DefensiveRating_Diff'] = team1_stats['DefensiveRating'] - team2_stats['DefensiveRating']
            
            feature_rows.append(feature_row)
        
        # Create base features dataframe
        features_df = pd.DataFrame(feature_rows)
        
        # Add performance metrics
        enhanced_df = self.performance_tracker.enhance_matchup_features(features_df, games_by_team)
        
        # Add estimated win probability for future performance metrics
        # This is a simple model based on existing features
        if len(enhanced_df) > 0:
            enhanced_df['WinProbability'] = self._estimate_win_probability(enhanced_df)
        
        # Save features
        output_path = self.features_dir / f"{gender}_matchup_features.csv"
        enhanced_df.to_csv(output_path, index=False)
        logger.info(f"Created {len(enhanced_df)} feature rows for {gender} matchups")
        
        return enhanced_df
        
    def _estimate_win_probability(self, features_df):
        """
        Estimate win probability based on team features
        This is a simple model for demonstration purposes
        
        Args:
            features_df: DataFrame with matchup features
            
        Returns:
            Series of win probabilities (probability that Team1 wins)
        """
        # Start with a base probability of 0.5
        probabilities = pd.Series([0.5] * len(features_df))
        
        # Adjust based on win rate difference (simple logistic-like model)
        if 'WinRate_Diff' in features_df.columns:
            win_rate_factor = features_df['WinRate_Diff'].clip(-0.5, 0.5) * 0.5
            probabilities += win_rate_factor
        
        # Adjust based on advanced metrics if available
        if 'AdjO_Diff' in features_df.columns and 'AdjD_Diff' in features_df.columns:
            # Scale adjustments to have reasonable impact
            adjo_factor = features_df['AdjO_Diff'].clip(-20, 20) / 100
            adjd_factor = features_df['AdjD_Diff'].clip(-20, 20) / 100
            probabilities += (adjo_factor + adjd_factor) * 0.1
        
        # Constrain to valid probabilities [0.05, 0.95]
        return probabilities.clip(0.05, 0.95)
        
    def prepare_submission_features(self):
        """Prepare features for all possible matchups in the submission format"""
        try:
            submission_format_path = self.processed_data_dir / "submission_format.csv"
            if not submission_format_path.exists():
                logger.error("Submission format file not found")
                return None
                
            submission_df = pd.read_csv(submission_format_path)
            logger.info(f"Loaded submission format with {len(submission_df)} rows")
            
            # Load team stats
            m_team_stats_df = pd.read_csv(self.features_dir / "M_team_stats.csv")
            w_team_stats_df = pd.read_csv(self.features_dir / "W_team_stats.csv")
            
            # Load matchup data for performance tracking
            m_matchups = self.load_matchup_data(gender="M")
            w_matchups = self.load_matchup_data(gender="W")
            
            # Organize games by team
            m_games_by_team = self.organize_games_by_team(m_matchups) if m_matchups is not None else {}
            w_games_by_team = self.organize_games_by_team(w_matchups) if w_matchups is not None else {}
            
            # Determine gender for each matchup in submission format
            # This is a simplification - you'll need to determine how to identify gender in your actual data
            # For example, you might have different ID ranges for men's and women's teams
            # As a placeholder, we'll assume IDs < 3000 are men's teams
            submission_df['Gender'] = submission_df['Team1ID'].apply(lambda x: 'M' if x < 3000 else 'W')
            
            # Create features for each potential matchup
            feature_rows = []
            
            for _, matchup in submission_df.iterrows():
                season = matchup['Season']
                team1_id = matchup['Team1ID']
                team2_id = matchup['Team2ID']
                gender = matchup['Gender']
                
                # Select appropriate team stats and games based on gender
                if gender == 'M':
                    team_stats_df = m_team_stats_df
                    games_by_team = m_games_by_team
                else:
                    team_stats_df = w_team_stats_df
                    games_by_team = w_games_by_team
                
                # Get stats for team1
                team1_stats = team_stats_df[(team_stats_df['Season'] == season) & 
                                           (team_stats_df['TeamID'] == team1_id)]
                
                # Get stats for team2
                team2_stats = team_stats_df[(team_stats_df['Season'] == season) & 
                                           (team_stats_df['TeamID'] == team2_id)]
                
                # Skip if we don't have stats for either team
                if team1_stats.empty or team2_stats.empty:
                    logger.warning(f"Missing stats for matchup: {season} {team1_id} vs {team2_id} ({gender})")
                    # Add row with missing values
                    feature_row = {
                        'ID': matchup['ID'],
                        'Season': season,
                        'Team1ID': team1_id,
                        'Team2ID': team2_id,
                        'Gender': gender
                    }
                    feature_rows.append(feature_row)
                    continue
                
                # Extract stats as Series
                team1_stats = team1_stats.iloc[0]
                team2_stats = team2_stats.iloc[0]
                
                # Create base feature row
                feature_row = {
                    'ID': matchup['ID'],
                    'Season': season,
                    'Team1ID': team1_id,
                    'Team2ID': team2_id,
                    'Gender': gender,
                    
                    # Team 1 features
                    'Team1_WinRate': team1_stats['win_rate'],
                    'Team1_AvgScore': team1_stats['avg_score'],
                    'Team1_AvgAllowed': team1_stats['avg_allowed'],
                    'Team1_AvgPointDiff': team1_stats['avg_point_diff'],
                    'Team1_GamesPlayed': team1_stats['games_played'],
                    
                    # Team 2 features
                    'Team2_WinRate': team2_stats['win_rate'],
                    'Team2_AvgScore': team2_stats['avg_score'],
                    'Team2_AvgAllowed': team2_stats['avg_allowed'],
                    'Team2_AvgPointDiff': team2_stats['avg_point_diff'],
                    'Team2_GamesPlayed': team2_stats['games_played'],
                    
                    # Comparative features
                    'WinRate_Diff': team1_stats['win_rate'] - team2_stats['win_rate'],
                    'AvgScore_Diff': team1_stats['avg_score'] - team2_stats['avg_score'],
                    'AvgAllowed_Diff': team1_stats['avg_allowed'] - team2_stats['avg_allowed'],
                    'AvgPointDiff_Diff': team1_stats['avg_point_diff'] - team2_stats['avg_point_diff'],
                }
                
                # Add advanced metrics features if available
                if 'AdjO' in team1_stats:
                    advanced_features = {
                        'Team1_AdjO': team1_stats.get('AdjO'),
                        'Team1_AdjD': team1_stats.get('AdjD'),
                        'Team1_AdjTempo': team1_stats.get('AdjTempo'),
                        'Team1_Luck': team1_stats.get('Luck'),
                        'Team1_SOS': team1_stats.get('Strength_of_Schedule'),
                        'Team1_AdvancedRank': team1_stats.get('KenPomRank'),
                        
                        'Team2_AdjO': team2_stats.get('AdjO'),
                        'Team2_AdjD': team2_stats.get('AdjD'),
                        'Team2_AdjTempo': team2_stats.get('AdjTempo'),
                        'Team2_Luck': team2_stats.get('Luck'),
                        'Team2_SOS': team2_stats.get('Strength_of_Schedule'),
                        'Team2_AdvancedRank': team2_stats.get('KenPomRank'),
                        
                        'AdjO_Diff': team1_stats.get('AdjO') - team2_stats.get('AdjO'),
                        'AdjD_Diff': team1_stats.get('AdjD') - team2_stats.get('AdjD'),
                        'AdvancedRank_Diff': team2_stats.get('KenPomRank') - team1_stats.get('KenPomRank'),
                    }
                    feature_row.update(advanced_features)
                
                # Add gender-specific features
                if gender == "W":
                    # Add women's specific metrics if available
                    if all(field in team1_stats for field in ['OffensiveRating', 'DefensiveRating']):
                        womens_features = {
                            'Team1_OffensiveRating': team1_stats['OffensiveRating'],
                            'Team2_OffensiveRating': team2_stats['OffensiveRating'],
                            'OffensiveRating_Diff': team1_stats['OffensiveRating'] - team2_stats['OffensiveRating'],
                            'Team1_DefensiveRating': team1_stats['DefensiveRating'],
                            'Team2_DefensiveRating': team2_stats['DefensiveRating'],
                            'DefensiveRating_Diff': team1_stats['DefensiveRating'] - team2_stats['DefensiveRating'],
                        }
                        feature_row.update(womens_features)
                
                # Get performance metrics if available
                if season in games_by_team:
                    # Estimate the number of games each team has played this season
                    team1_games_played = len(games_by_team[season].get(team1_id, []))
                    team2_games_played = len(games_by_team[season].get(team2_id, []))
                    
                    # Add streak features
                    if team1_id in games_by_team[season] and team1_games_played > 0:
                        recent_games = games_by_team[season][team1_id][-min(10, team1_games_played):]
                        team1_streak = sum(1 for g in recent_games if g['Result'] == 1)
                        feature_row['Team1_RecentWins'] = team1_streak
                    else:
                        feature_row['Team1_RecentWins'] = 0
                        
                    if team2_id in games_by_team[season] and team2_games_played > 0:
                        recent_games = games_by_team[season][team2_id][-min(10, team2_games_played):]
                        team2_streak = sum(1 for g in recent_games if g['Result'] == 1)
                        feature_row['Team2_RecentWins'] = team2_streak
                    else:
                        feature_row['Team2_RecentWins'] = 0
                        
                    # Add streak difference
                    feature_row['RecentWins_Diff'] = feature_row['Team1_RecentWins'] - feature_row['Team2_RecentWins']
                
                feature_rows.append(feature_row)
            
            # Create DataFrame with all features
            submission_features_df = pd.DataFrame(feature_rows)
            
            # Add performance metrics using TeamPerformanceTracker
            # First, create a submission-specific performance tracker
            tracker = TeamPerformanceTracker(window_sizes=[1, 3, 5, 10], max_streak_for_normalization=15)
            
            # For men's matchups
            m_submission = submission_features_df[submission_features_df['Gender'] == 'M']
            if not m_submission.empty and m_matchups is not None:
                m_enhanced = tracker.enhance_matchup_features(m_submission, m_games_by_team)
                
            # For women's matchups
            w_submission = submission_features_df[submission_features_df['Gender'] == 'W']
            if not w_submission.empty and w_matchups is not None:
                w_enhanced = tracker.enhance_matchup_features(w_submission, w_games_by_team)
            
            # Combine enhanced features
            enhanced_features = pd.concat([
                m_enhanced if not m_submission.empty else pd.DataFrame(),
                w_enhanced if not w_submission.empty else pd.DataFrame()
            ], ignore_index=True)
            
            # If we have enhanced features, use them; otherwise, use the original
            if not enhanced_features.empty:
                submission_features_df = enhanced_features
            
            # Save submission features
            output_path = self.features_dir / "submission_features.csv"
            submission_features_df.to_csv(output_path, index=False)
            logger.info(f"Created features for {len(submission_features_df)} potential matchups")
            
            return submission_features_df
            
        except Exception as e:
            logger.error(f"Error preparing submission features: {str(e)}")
            return None
    
    def process_all_features(self):
        """Process features for both genders and create combined dataset"""
        # Before processing, ensure we have women's ratings generated
        logger.info("Checking for women's basketball ratings data")
        womens_ratings = self.womens_data_collector.get_womens_ratings()
        if womens_ratings is None or womens_ratings.empty:
            logger.info("Generating women's basketball ratings data")
            self.womens_data_collector.create_derived_power_ratings()
        
        # Create matchup features for men's data
        m_features = self.create_matchup_features(gender="M")
        
        # Create matchup features for women's data
        w_features = self.create_matchup_features(gender="W")
        
        # Combine features for training
        if m_features is not None and w_features is not None:
            combined_features = pd.concat([m_features, w_features], ignore_index=True)
            combined_features.to_csv(self.features_dir / "combined_matchup_features.csv", index=False)
            logger.info(f"Created combined features dataset with {len(combined_features)} rows")
        else:
            combined_features = None
            
        # Create submission features
        submission_features = self.prepare_submission_features()
        
        return {
            "men_features": m_features,
            "women_features": w_features,
            "combined_features": combined_features,
            "submission_features": submission_features
        }

if __name__ == "__main__":
    engineer = BasketballFeatureEngineer()
    features = engineer.process_all_features()
