import pandas as pd
import numpy as np
from pathlib import Path
import logging
import requests
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WomensBasketballDataCollector:
    def __init__(self, data_dir="data/external/womens"):
        """
        Initialize the women's basketball data collector.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def collect_ncaa_stats(self, season=None):
        """
        Collect team statistics from NCAA website
        
        Args:
            season: Basketball season year (e.g., 2023 for 2022-2023 season)
        """
        if season is None:
            season = datetime.now().year
            if datetime.now().month < 6:  # If before June, use previous season
                season -= 1
                
        logger.info(f"Collecting NCAA women's basketball stats for {season}-{season+1} season")
        
        # NCAA women's basketball stats URL patterns
        # Team stats: https://stats.ncaa.org/rankings/ranking_summary/12920
        # - 12920 is the ranking_id for 2022-2023 season
        
        # This is a placeholder - would need to determine correct ranking_id for each season
        ranking_id = self._get_ranking_id_for_season(season)
        if not ranking_id:
            logger.error(f"Could not determine NCAA ranking ID for {season} season")
            return self._generate_sample_ncaa_data(season)
        
        url = f"https://stats.ncaa.org/rankings/ranking_summary/{ranking_id}"
        try:
            response = self.session.get(url, headers=self.headers)
            if response.status_code != 200:
                logger.error(f"Failed to fetch NCAA data: Status code {response.status_code}")
                return self._generate_sample_ncaa_data(season)
                
            # Parse the HTML - this would be custom for NCAA's format
            # NCAA's site is complex and may require handling JavaScript
            # For now, we'll return sample data
            return self._generate_sample_ncaa_data(season)
            
        except Exception as e:
            logger.error(f"Error collecting NCAA stats: {str(e)}")
            return self._generate_sample_ncaa_data(season)
    
    def _get_ranking_id_for_season(self, season):
        """Determine the NCAA ranking ID for a given season"""
        # This would need to be implemented based on NCAA's system
        # For now, return dummy values for testing
        season_to_id = {
            2023: 12920,  # 2022-2023 season
            2024: 13140,  # 2023-2024 season
            # Add more mappings as determined
        }
        return season_to_id.get(season)
    
    def collect_her_hoop_stats(self, season=None):
        """
        Collect women's basketball data from Her Hoop Stats
        This would require an account/API key for full access
        
        Args:
            season: Basketball season year
        """
        if season is None:
            season = datetime.now().year
            if datetime.now().month < 6:  # If before June, use previous season
                season -= 1
                
        logger.info(f"Collecting Her Hoop Stats data for {season}-{season+1} season")
        
        # Her Hoop Stats requires authentication for most data
        # For development purposes, generate sample data
        return self._generate_sample_her_hoop_stats(season)
    
    def collect_sports_reference_data(self, season=None):
        """
        Collect women's basketball data from Sports-Reference
        
        Args:
            season: Basketball season year
        """
        if season is None:
            season = datetime.now().year
            if datetime.now().month < 6:  # If before June, use previous season
                season -= 1
                
        logger.info(f"Collecting Sports-Reference data for {season}-{season+1} season")
        
        # Sports-Reference URL pattern for women's college basketball
        # e.g., https://www.sports-reference.com/cbb/seasons/women/2023-ratings.html
        url = f"https://www.sports-reference.com/cbb/seasons/women/{season}-ratings.html"
        
        try:
            response = self.session.get(url, headers=self.headers)
            if response.status_code != 200:
                logger.error(f"Failed to fetch Sports-Reference data: Status code {response.status_code}")
                return self._generate_sample_sports_ref_data(season)
            
            # Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the ratings table - specific to Sports-Reference structure
            ratings_table = soup.find('table', {'id': 'ratings'})
            if not ratings_table:
                logger.warning("Could not find ratings table on Sports-Reference")
                return self._generate_sample_sports_ref_data(season)
            
            # Extract headers
            headers = []
            header_row = ratings_table.find('thead').find_all('tr')[-1]
            for th in header_row.find_all('th'):
                headers.append(th.text.strip())
            
            # Extract team data
            teams_data = []
            for row in ratings_table.find('tbody').find_all('tr'):
                if 'class' in row.attrs and 'thead' in row.attrs['class']:
                    continue  # Skip header rows within the table
                    
                cells = row.find_all(['td', 'th'])
                if len(cells) < len(headers):
                    continue  # Skip rows with insufficient data
                    
                team_dict = {}
                for i, cell in enumerate(cells):
                    if i < len(headers):
                        # Get cell text and clean it
                        cell_text = cell.text.strip()
                        # Handle team name special case
                        if i == 1 and cell.find('a'):
                            cell_text = cell.find('a').text.strip()
                        team_dict[headers[i]] = cell_text
                
                teams_data.append(team_dict)
            
            # Convert to DataFrame
            df = pd.DataFrame(teams_data)
            
            # Add season column
            df['Season'] = season
            
            # Save to CSV
            output_path = self.data_dir / f"sports_reference_{season}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved Sports-Reference data to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error collecting Sports-Reference data: {str(e)}")
            return self._generate_sample_sports_ref_data(season)
    
    def create_derived_power_ratings(self, raw_data_dir="data/raw", seasons=range(2015, 2025)):
        """
        Create our own power ratings based on historical tournament and regular season results
        
        Args:
            raw_data_dir: Directory with raw NCAA data
            seasons: Range of seasons to analyze
        """
        raw_data_dir = Path(raw_data_dir)
        
        logger.info("Creating women's basketball power ratings")
        
        all_teams_data = []
        
        for season in seasons:
            try:
                # Load regular season results
                reg_season_file = list(raw_data_dir.glob(f"*WRegularSeasonCompactResults_{season}.csv"))
                if not reg_season_file:
                    logger.warning(f"No regular season data found for {season}")
                    continue
                    
                reg_season_df = pd.read_csv(reg_season_file[0])
                
                # Load tournament results
                tourney_file = list(raw_data_dir.glob(f"*WNCAATourneyCompactResults_{season}.csv"))
                if not tourney_file:
                    logger.warning(f"No tournament data found for {season}")
                    continue
                    
                tourney_df = pd.read_csv(tourney_file[0])
                
                # Combine data and calculate team statistics
                all_games = pd.concat([reg_season_df, tourney_df])
                
                # Get all unique teams
                all_team_ids = set(all_games['WTeamID'].unique()) | set(all_games['LTeamID'].unique())
                
                # Calculate basic statistics for each team
                for team_id in all_team_ids:
                    # Games won
                    wins = all_games[all_games['WTeamID'] == team_id]
                    # Games lost
                    losses = all_games[all_games['LTeamID'] == team_id]
                    
                    total_games = len(wins) + len(losses)
                    win_rate = len(wins) / total_games if total_games > 0 else 0
                    
                    # Points scored and allowed
                    points_scored = wins['WScore'].sum() + losses['LScore'].sum()
                    points_allowed = wins['LScore'].sum() + losses['WScore'].sum()
                    
                    # Average point differential
                    point_diff = 0
                    if total_games > 0:
                        win_diff = wins['WScore'].sum() - wins['LScore'].sum()
                        loss_diff = losses['LScore'].sum() - losses['WScore'].sum()
                        point_diff = (win_diff + loss_diff) / total_games
                    
                    # Calculate strength of schedule based on opponent win rates
                    # This is a simplified approach and could be enhanced
                    opponent_ids = list(wins['LTeamID']) + list(losses['WTeamID'])
                    opponent_win_rates = []
                    
                    # This will be calculated in a second pass
                    
                    team_data = {
                        'Season': season,
                        'TeamID': team_id,
                        'Games': total_games,
                        'Wins': len(wins),
                        'Losses': len(losses),
                        'WinRate': win_rate,
                        'PointsScored': points_scored,
                        'PointsAllowed': points_allowed,
                        'AvgPointsScored': points_scored / total_games if total_games > 0 else 0,
                        'AvgPointsAllowed': points_allowed / total_games if total_games > 0 else 0,
                        'AvgPointDiff': point_diff,
                        'OpponentIDs': opponent_ids,
                        # SOS will be calculated later
                    }
                    
                    all_teams_data.append(team_data)
                    
            except Exception as e:
                logger.error(f"Error processing season {season}: {str(e)}")
        
        # Create DataFrame
        teams_df = pd.DataFrame(all_teams_data)
        
        # Calculate strength of schedule in a second pass
        teams_by_season = teams_df.groupby('Season')
        
        for season, season_teams in teams_by_season:
            win_rate_dict = dict(zip(season_teams['TeamID'], season_teams['WinRate']))
            
            for index, row in season_teams.iterrows():
                opponent_win_rates = [win_rate_dict.get(opp, 0) for opp in row['OpponentIDs']]
                sos = sum(opponent_win_rates) / len(opponent_win_rates) if opponent_win_rates else 0
                teams_df.at[index, 'SOS'] = sos
        
        # Drop the OpponentIDs column since it's a list and not needed anymore
        teams_df = teams_df.drop('OpponentIDs', axis=1)
        
        # Calculate a simple power rating
        teams_df['PowerRating'] = (
            teams_df['WinRate'] * 100 + 
            teams_df['AvgPointDiff'] * 3 + 
            teams_df['SOS'] * 20
        )
        
        # Save to CSV
        output_path = self.data_dir / "womens_power_ratings.csv"
        teams_df.to_csv(output_path, index=False)
        logger.info(f"Saved women's power ratings to {output_path}")
        
        # Normalize power ratings to be similar scale to KenPom
        scaler = StandardScaler()
        teams_df['NormalizedRating'] = scaler.fit_transform(teams_df[['PowerRating']])
        teams_df['AdjustedEfficiency'] = 100 + (teams_df['NormalizedRating'] * 10)
        
        # Create separate offensive and defensive ratings
        teams_df['OffensiveRating'] = 100 + (
            (teams_df['AvgPointsScored'] - teams_df['AvgPointsScored'].mean()) / 
            teams_df['AvgPointsScored'].std() * 10
        )
        teams_df['DefensiveRating'] = 100 - (
            (teams_df['AvgPointsAllowed'] - teams_df['AvgPointsAllowed'].mean()) / 
            teams_df['AvgPointsAllowed'].std() * 10
        )
        
        # Save normalized ratings
        output_path = self.data_dir / "womens_normalized_ratings.csv"
        teams_df.to_csv(output_path, index=False)
        logger.info(f"Saved normalized women's ratings to {output_path}")
        
        return teams_df
    
    def _generate_sample_ncaa_data(self, season):
        """Generate sample NCAA statistics for development purposes"""
        logger.info(f"Generating sample NCAA data for {season} season")
        
        # Create a DataFrame with realistic-looking college basketball stats
        # Get team data from raw directory
        try:
            raw_data_dir = Path("data/raw")
            teams_file = list(raw_data_dir.glob("*WTeams.csv"))
            
            if not teams_file:
                # Create random team IDs
                team_ids = range(3001, 3361)  # 360 teams
            else:
                # Use actual team IDs
                teams_df = pd.read_csv(teams_file[0])
                team_ids = teams_df['TeamID'].tolist()
            
            # Generate sample stats
            np.random.seed(season)  # for reproducibility
            
            stats_data = []
            for team_id in team_ids:
                # Randomize stats within realistic ranges
                points_per_game = round(np.random.normal(70, 8), 1)  # Points per game, normal distribution
                field_goal_pct = round(np.random.normal(0.425, 0.035), 3)  # FG%, normal distribution
                three_pt_pct = round(np.random.normal(0.330, 0.03), 3)  # 3PT%, normal distribution
                ft_pct = round(np.random.normal(0.700, 0.05), 3)  # FT%, normal distribution
                rebounds_per_game = round(np.random.normal(38, 4), 1)  # Rebounds per game
                assists_per_game = round(np.random.normal(14, 3), 1)  # Assists per game
                turnovers_per_game = round(np.random.normal(15, 3), 1)  # Turnovers per game
                steals_per_game = round(np.random.normal(7, 1.5), 1)  # Steals per game
                blocks_per_game = round(np.random.normal(3.5, 1), 1)  # Blocks per game
                
                stats_data.append({
                    'Season': season,
                    'TeamID': team_id,
                    'PointsPerGame': points_per_game,
                    'FGPct': field_goal_pct,
                    '3PTpct': three_pt_pct,
                    'FTPct': ft_pct,
                    'ReboundsPerGame': rebounds_per_game,
                    'AssistsPerGame': assists_per_game,
                    'TurnoversPerGame': turnovers_per_game,
                    'StealsPerGame': steals_per_game,
                    'BlocksPerGame': blocks_per_game,
                    'DataSource': 'NCAA_Sample'
                })
            
            # Create DataFrame
            df = pd.DataFrame(stats_data)
            
            # Save to CSV
            output_path = self.data_dir / f"ncaa_sample_stats_{season}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved sample NCAA stats to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating sample NCAA data: {str(e)}")
            return pd.DataFrame()  # Empty DataFrame
    
    def _generate_sample_her_hoop_stats(self, season):
        """Generate sample Her Hoop Stats data for development"""
        logger.info(f"Generating sample Her Hoop Stats data for {season} season")
        
        try:
            raw_data_dir = Path("data/raw")
            teams_file = list(raw_data_dir.glob("*WTeams.csv"))
            
            if not teams_file:
                # Create random team IDs
                team_ids = range(3001, 3361)  # 360 teams
            else:
                # Use actual team IDs
                teams_df = pd.read_csv(teams_file[0])
                team_ids = teams_df['TeamID'].tolist()
            
            # Generate sample stats
            np.random.seed(season + 1)  # different seed from NCAA
            
            stats_data = []
            for team_id in team_ids:
                # Advanced stats similar to those found on Her Hoop Stats
                offensive_rating = round(np.random.normal(100, 10), 1)
                defensive_rating = round(np.random.normal(100, 10), 1)
                net_rating = offensive_rating - defensive_rating
                tempo = round(np.random.normal(70, 5), 1)
                effective_fg_pct = round(np.random.normal(0.475, 0.04), 3)
                turnover_rate = round(np.random.normal(0.20, 0.03), 3)
                offensive_rebound_rate = round(np.random.normal(0.30, 0.05), 3)
                ft_rate = round(np.random.normal(0.25, 0.05), 3)
                
                stats_data.append({
                    'Season': season,
                    'TeamID': team_id,
                    'OffensiveRating': offensive_rating,
                    'DefensiveRating': defensive_rating,
                    'NetRating': net_rating,
                    'Tempo': tempo,
                    'EffectiveFGPct': effective_fg_pct,
                    'TurnoverRate': turnover_rate,
                    'OffensiveReboundRate': offensive_rebound_rate,
                    'FTRate': ft_rate,
                    'DataSource': 'HHS_Sample'
                })
            
            # Create DataFrame
            df = pd.DataFrame(stats_data)
            
            # Save to CSV
            output_path = self.data_dir / f"her_hoop_stats_sample_{season}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved sample Her Hoop Stats data to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating sample Her Hoop Stats data: {str(e)}")
            return pd.DataFrame()  # Empty DataFrame
    
    def _generate_sample_sports_ref_data(self, season):
        """Generate sample Sports-Reference data for development"""
        logger.info(f"Generating sample Sports-Reference data for {season} season")
        
        try:
            raw_data_dir = Path("data/raw")
            teams_file = list(raw_data_dir.glob("*WTeams.csv"))
            
            if not teams_file:
                # Create random team IDs
                team_ids = range(3001, 3361)  # 360 teams
            else:
                # Use actual team IDs
                teams_df = pd.read_csv(teams_file[0])
                team_ids = teams_df['TeamID'].tolist()
            
            # Generate sample stats
            np.random.seed(season + 2)  # different seed
            
            stats_data = []
            for i, team_id in enumerate(team_ids):
                # Stats similar to Sports-Reference format
                srs = round(np.random.normal(0, 8), 2)  # Simple Rating System
                sos = round(np.random.normal(0, 4), 2)  # Strength of Schedule
                offensive_rtg = round(np.random.normal(100, 10), 1)
                defensive_rtg = round(np.random.normal(100, 10), 1)
                pace = round(np.random.normal(70, 5), 1)
                
                stats_data.append({
                    'Season': season,
                    'TeamID': team_id,
                    'Rk': i+1,  # Rank
                    'SRS': srs,
                    'SOS': sos,
                    'ORtg': offensive_rtg,
                    'DRtg': defensive_rtg,
                    'Pace': pace,
                    'DataSource': 'SportsRef_Sample'
                })
            
            # Create DataFrame
            df = pd.DataFrame(stats_data)
            
            # Sort by SRS (highest first)
            df = df.sort_values('SRS', ascending=False).reset_index(drop=True)
            df['Rk'] = df.index + 1  # Reindex rankings
            
            # Save to CSV
            output_path = self.data_dir / f"sports_reference_sample_{season}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved sample Sports-Reference data to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating sample Sports-Reference data: {str(e)}")
            return pd.DataFrame()  # Empty DataFrame
    
    def merge_womens_data_sources(self, season=None):
        """
        Merge multiple women's basketball data sources into a single dataset
        that can be used as a KenPom equivalent for feature engineering
        
        Args:
            season: Basketball season year
        """
        if season is None:
            season = datetime.now().year
            if datetime.now().month < 6:  # If before June, use previous season
                season -= 1
        
        logger.info(f"Merging women's basketball data sources for {season} season")
        
        # Load power ratings (should always be available since we generate them)
        power_ratings_path = self.data_dir / "womens_normalized_ratings.csv"
        if not power_ratings_path.exists():
            # Create power ratings if they don't exist
            self.create_derived_power_ratings()
        
        if not power_ratings_path.exists():
            logger.error("Could not load or create women's power ratings")
            return None
            
        power_ratings = pd.read_csv(power_ratings_path)
        season_ratings = power_ratings[power_ratings['Season'] == season]
        
        # Try to load NCAA stats
        ncaa_path = self.data_dir / f"ncaa_sample_stats_{season}.csv"
        if not ncaa_path.exists():
            ncaa_stats = self.collect_ncaa_stats(season)
        else:
            ncaa_stats = pd.read_csv(ncaa_path)
        
        # Try to load Her Hoop Stats
        hhs_path = self.data_dir / f"her_hoop_stats_sample_{season}.csv"
        if not hhs_path.exists():
            hhs_stats = self.collect_her_hoop_stats(season)
        else:
            hhs_stats = pd.read_csv(hhs_path)
        
        # Try to load Sports-Reference data
        sports_ref_path = self.data_dir / f"sports_reference_sample_{season}.csv"
        if not sports_ref_path.exists():
            sports_ref_stats = self.collect_sports_reference_data(season)
        else:
            sports_ref_stats = pd.read_csv(sports_ref_path)
        
        # Start with power ratings as the base
        merged_data = season_ratings.copy()
        
        # Add NCAA stats if available
        if ncaa_stats is not None and not ncaa_stats.empty:
            # Filter for the current season
            ncaa_season = ncaa_stats[ncaa_stats['Season'] == season]
            if not ncaa_season.empty:
                # Merge on TeamID
                merged_data = pd.merge(
                    merged_data,
                    ncaa_season.drop(['Season'], axis=1),
                    on='TeamID',
                    how='left'
                )
        
        # Add Her Hoop Stats if available
        if hhs_stats is not None and not hhs_stats.empty:
            # Filter for the current season
            hhs_season = hhs_stats[hhs_stats['Season'] == season]
            if not hhs_season.empty:
                # Merge on TeamID
                merged_data = pd.merge(
                    merged_data,
                    hhs_season.drop(['Season'], axis=1),
                    on='TeamID',
                    how='left'
                )
        
        # Add Sports-Reference stats if available
        if sports_ref_stats is not None and not sports_ref_stats.empty:
            # Filter for the current season
            sr_season = sports_ref_stats[sports_ref_stats['Season'] == season]
            if not sr_season.empty:
                # Merge on TeamID
                merged_data = pd.merge(
                    merged_data,
                    sr_season.drop(['Season'], axis=1),
                    on='TeamID',
                    how='left'
                )
        
        # Create KenPom-like columns for compatibility with existing pipeline
        merged_data['AdjO'] = merged_data.get('OffensiveRating', merged_data.get('ORtg', merged_data['OffensiveRating']))
        merged_data['AdjD'] = merged_data.get('DefensiveRating', merged_data.get('DRtg', merged_data['DefensiveRating']))
        merged_data['AdjTempo'] = merged_data.get('Tempo', merged_data.get('Pace', 70.0))
        merged_data['Luck'] = np.random.normal(0, 0.05, size=len(merged_data))  # Random luck factor
        merged_data['Strength_of_Schedule'] = merged_data.get('SOS', 0)
        
        # Create a rank column sorted by PowerRating
        merged_data['KenPomRank'] = merged_data['PowerRating'].rank(ascending=False).astype(int)
        
        # Save merged data
        output_path = self.data_dir / f"womens_merged_stats_{season}.csv"
        merged_data.to_csv(output_path, index=False)
        logger.info(f"Saved merged women's basketball stats to {output_path}")
        
        # Create a historical collection of all seasons
        historical_path = self.data_dir / "womens_historical_ratings.csv"
        if historical_path.exists():
            historical_df = pd.read_csv(historical_path)
            # Remove existing data for this season
            historical_df = historical_df[historical_df['Season'] != season]
            # Append new season data
            historical_df = pd.concat([historical_df, merged_data], ignore_index=True)
        else:
            historical_df = merged_data
        
        # Save historical collection
        historical_df.to_csv(historical_path, index=False)
        logger.info(f"Updated women's historical ratings with {season} season")
        
        return merged_data
    
    def get_womens_ratings(self, season=None):
        """
        Get Women's basketball ratings (equivalent to KenPom) for a specific season
        
        Args:
            season: Basketball season year
        
        Returns:
            DataFrame with women's basketball ratings
        """
        if season is None:
            season = datetime.now().year
            if datetime.now().month < 6:  # If before June, use previous season
                season -= 1
        
        # Check if we already have merged data for this season
        merged_path = self.data_dir / f"womens_merged_stats_{season}.csv"
        if merged_path.exists():
            logger.info(f"Loading existing women's basketball ratings for {season}")
            return pd.read_csv(merged_path)
        
        # If not, check for historical data
        historical_path = self.data_dir / "womens_historical_ratings.csv"
        if historical_path.exists():
            historical_df = pd.read_csv(historical_path)
            season_data = historical_df[historical_df['Season'] == season]
            if not season_data.empty:
                logger.info(f"Found women's basketball ratings for {season} in historical data")
                return season_data
        
        # If still not found, create new data
        logger.info(f"Creating new women's basketball ratings for {season}")
        return self.merge_womens_data_sources(season)

if __name__ == "__main__":
    collector = WomensBasketballDataCollector()
    
    # Create derived power ratings from historical data
    power_ratings = collector.create_derived_power_ratings()
    
    # Get current season ratings (uses power ratings + other sources)
    current_ratings = collector.get_womens_ratings()
    
    logger.info("Women's basketball data collection completed")
