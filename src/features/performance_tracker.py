import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TeamPerformanceTracker:
    def __init__(self, window_sizes=[1, 3, 5, 10], max_streak_for_normalization=15):
        """
        Initialize the team performance tracker
        
        Args:
            window_sizes: List of game windows to track (e.g., last 1, 3, 5, 10 games)
            max_streak_for_normalization: Maximum streak length for normalizing win streaks to [0,1]
        """
        self.window_sizes = sorted(window_sizes)
        self.max_streak = max_streak_for_normalization
        
    def calculate_performance_metrics(self, games_df, model=None):
        """
        Calculate team performance metrics including actual vs expected and streaks
        
        Args:
            games_df: DataFrame with game results (must be chronologically sorted)
            model: Optional model to predict win probabilities if not already in games_df
            
        Returns:
            Dictionary of team performance metrics by team ID and season
        """
        logger.info("Calculating team performance metrics")
        
        # Ensure games are sorted chronologically
        if 'DayNum' in games_df.columns:
            games_df = games_df.sort_values(['Season', 'DayNum'])
        
        # Dictionary to store team metrics by season
        team_metrics = {}
        
        # Track team performance for each season separately
        for season in games_df['Season'].unique():
            season_games = games_df[games_df['Season'] == season].copy()
            
            # If win probabilities aren't provided, use a simple model
            if 'WinProbability' not in season_games.columns:
                season_games['WinProbability'] = self._estimate_win_probabilities(season_games, model)
            
            # Get unique teams
            all_teams = set(season_games['Team1ID'].unique()) | set(season_games['Team2ID'].unique())
            
            # Initialize metrics dictionary for this season
            team_metrics[season] = {team_id: {
                'games': [],
                'win_streak': 0,
                'performance_vs_expected': [],
                'window_metrics': {size: [] for size in self.window_sizes}
            } for team_id in all_teams}
            
            # Process games in chronological order
            for _, game in season_games.iterrows():
                team1_id = game['Team1ID']
                team2_id = game['Team2ID']
                team1_won = game['Result'] == 1
                prob_team1_wins = game['WinProbability']
                
                # Calculate performance vs expected
                team1_perf = team1_won - prob_team1_wins  # If win (1) - prob = positive value for overperformance
                team2_perf = (not team1_won) - (1 - prob_team1_wins)
                
                # Update Team 1 metrics
                self._update_team_metrics(team_metrics[season][team1_id], team1_won, team1_perf)
                
                # Update Team 2 metrics
                self._update_team_metrics(team_metrics[season][team2_id], not team1_won, team2_perf)
        
        return team_metrics
    
    def _update_team_metrics(self, team_data, won_game, performance):
        """
        Update a team's performance metrics after a game
        
        Args:
            team_data: Dictionary containing the team's metrics
            won_game: Boolean indicating if team won this game
            performance: Float indicating how much team over/under performed vs expectation
        """
        # Update win streak
        if won_game:
            team_data['win_streak'] += 1
        else:
            team_data['win_streak'] = 0
            
        # Add game result and performance
        team_data['games'].append(won_game)
        team_data['performance_vs_expected'].append(performance)
        
        # Update window metrics
        for window_size in self.window_sizes:
            # Get recent games limited to window size
            recent_games = team_data['games'][-window_size:] if len(team_data['games']) >= window_size else team_data['games']
            recent_perf = team_data['performance_vs_expected'][-window_size:] if len(team_data['performance_vs_expected']) >= window_size else team_data['performance_vs_expected']
            
            # Calculate metrics
            recent_win_pct = sum(recent_games) / len(recent_games) if recent_games else 0
            perf_vs_expected = sum(recent_perf) if recent_perf else 0
            
            # Store window metrics
            metric = {
                'win_pct': recent_win_pct,
                'performance_vs_expected': perf_vs_expected,
                'games_in_window': len(recent_games)
            }
            team_data['window_metrics'][window_size].append(metric)
    
    def _estimate_win_probabilities(self, games_df, model=None):
        """
        Estimate win probabilities for games if not provided
        
        Args:
            games_df: DataFrame with game results
            model: Optional model to predict win probabilities
            
        Returns:
            Series of win probabilities
        """
        if model:
            # Use provided model to predict
            try:
                # Create feature set for prediction
                # This will depend on what features your model uses
                features = games_df[['Team1_WinRate', 'Team2_WinRate', 'Team1_AvgPointDiff', 'Team2_AvgPointDiff']]
                return model.predict_proba(features)[:, 1]
            except Exception as e:
                logger.error(f"Error predicting with model: {str(e)}")
                
        # Fallback to a simple rating-based estimate
        # Use a naive approach based on win rates if available
        if 'Team1_WinRate' in games_df.columns and 'Team2_WinRate' in games_df.columns:
            # Simple probability based on relative win rates
            team1_rate = games_df['Team1_WinRate'].fillna(0.5)
            team2_rate = games_df['Team2_WinRate'].fillna(0.5)
            prob = (team1_rate + (1 - team2_rate)) / 2
            return prob.clip(0.05, 0.95)  # Constrain to reasonable range
        
        # If no good data, use a constant
        logger.warning("Using constant win probability of 0.5 (home team advantage)")
        return pd.Series([0.5] * len(games_df))
    
    def get_team_features(self, team_metrics, team_id, season, game_number):
        """
        Get a team's performance features for a specific game
        
        Args:
            team_metrics: Dictionary of team metrics from calculate_performance_metrics
            team_id: Team ID to get metrics for
            season: Season to get metrics for
            game_number: Game number in the season (0-indexed)
            
        Returns:
            Dictionary of team features
        """
        features = {}
        
        if season not in team_metrics or team_id not in team_metrics[season]:
            # No data available, return zeros
            features['win_streak'] = 0
            for size in self.window_sizes:
                features[f'last_{size}_perf_vs_exp'] = 0
                features[f'last_{size}_win_pct'] = 0
            return features
            
        # Get team data
        team_data = team_metrics[season][team_id]
        game_idx = min(game_number, len(team_data['games']) - 1) if team_data['games'] else 0
        
        # Normalized win streak (0 to 1)
        streak = min(team_data['win_streak'], self.max_streak) / self.max_streak if game_idx > 0 else 0
        features['win_streak'] = streak
        
        # Get window metrics
        for size in self.window_sizes:
            # If not enough games played, use metrics available so far
            if game_idx >= len(team_data['window_metrics'][size]):
                features[f'last_{size}_perf_vs_exp'] = 0
                features[f'last_{size}_win_pct'] = 0
            else:
                metrics = team_data['window_metrics'][size][game_idx]
                features[f'last_{size}_perf_vs_exp'] = metrics['performance_vs_expected']
                features[f'last_{size}_win_pct'] = metrics['win_pct']
                
        return features

    def enhance_matchup_features(self, matchups_df, games_by_team):
        """
        Add performance metrics to matchup features
        
        Args:
            matchups_df: DataFrame of matchup features
            games_by_team: Dictionary of games organized by team and season
            
        Returns:
            Enhanced matchup features DataFrame
        """
        logger.info("Enhancing matchup features with performance metrics")
        
        # Calculate performance metrics based on all games
        all_games = []
        for season in games_by_team:
            for team_id in games_by_team[season]:
                for game in games_by_team[season][team_id]:
                    all_games.append(game)
        
        games_df = pd.DataFrame(all_games)
        
        # Calculate performance metrics
        team_metrics = self.calculate_performance_metrics(games_df)
        
        # Enhance matchup features
        enhanced_df = matchups_df.copy()
        
        # For each matchup, find the game number for each team
        for idx, matchup in enhanced_df.iterrows():
            season = matchup['Season']
            team1_id = matchup['Team1ID']
            team2_id = matchup['Team2ID']
            
            # Get game counts for teams in this season
            if season in games_by_team:
                team1_games = len(games_by_team[season].get(team1_id, []))
                team2_games = len(games_by_team[season].get(team2_id, []))
                
                # Get team features
                team1_features = self.get_team_features(team_metrics, team1_id, season, team1_games)
                team2_features = self.get_team_features(team_metrics, team2_id, season, team2_games)
                
                # Add team1 features
                for key, value in team1_features.items():
                    enhanced_df.at[idx, f'Team1_{key}'] = value
                    
                # Add team2 features
                for key, value in team2_features.items():
                    enhanced_df.at[idx, f'Team2_{key}'] = value
                    
                # Add comparative features
                for size in self.window_sizes:
                    perf_diff = team1_features[f'last_{size}_perf_vs_exp'] - team2_features[f'last_{size}_perf_vs_exp']
                    enhanced_df.at[idx, f'last_{size}_perf_diff'] = perf_diff
                    
                # Win streak difference (normalized)
                streak_diff = team1_features['win_streak'] - team2_features['win_streak']
                enhanced_df.at[idx, 'win_streak_diff'] = streak_diff
        
        return enhanced_df
