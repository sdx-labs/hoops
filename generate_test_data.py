import argparse
import os
import random
import pandas as pd
import numpy as np

def generate_test_kaggle_data(output_dir):
    """
    Generate minimal test data for the pipeline
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Teams data
    print("Generating team data...")
    men_teams = pd.DataFrame({
        'TeamID': list(range(1101, 1161)),
        'TeamName': [f'Team_{i}' for i in range(1101, 1161)],
        'FirstD1Season': [2000] * 60,
        'LastD1Season': [2025] * 60
    })
    
    women_teams = pd.DataFrame({
        'TeamID': list(range(3101, 3161)),
        'TeamName': [f'WTeam_{i}' for i in range(3101, 3161)],
        'FirstD1Season': [2000] * 60,
        'LastD1Season': [2025] * 60
    })
    
    # Generate team spellings
    team_spellings = []
    for team_id in men_teams['TeamID']:
        team_spellings.append({
            'TeamID': team_id,
            'TeamNameSpelling': f'team_{team_id}_alt'
        })
    
    # Generate seasons
    seasons = pd.DataFrame({
        'Season': list(range(2015, 2026)),
        'DayZero': ['10/31/2014', '10/30/2015', '10/28/2016', 
                   '10/27/2017', '10/26/2018', '10/25/2019',
                   '10/30/2020', '10/29/2021', '10/28/2022', 
                   '10/27/2023', '10/25/2024'],
        'RegionW': ['West'] * 11,
        'RegionX': ['East'] * 11,
        'RegionY': ['Midwest'] * 11,
        'RegionZ': ['South'] * 11
    })
    
    # Generate regular season results
    regular_results = []
    
    # Generate 2000 games for each season
    for season in range(2015, 2026):
        for _ in range(2000):
            # Randomly select teams
            team1 = random.choice(men_teams['TeamID'])
            team2 = random.choice([t for t in men_teams['TeamID'] if t != team1])
            
            # Generate scores
            score1 = random.randint(50, 100)
            score2 = random.randint(50, 100)
            
            # Determine winner and loser
            if score1 > score2:
                w_team, w_score = team1, score1
                l_team, l_score = team2, score2
            else:
                w_team, w_score = team2, score2
                l_team, l_score = team1, score1
            
            # Generate game location
            location = random.choice(['H', 'A', 'N'])
            
            # Generate day number
            day_num = random.randint(7, 132)
            
            regular_results.append({
                'Season': season,
                'DayNum': day_num,
                'WTeamID': w_team,
                'WScore': w_score,
                'LTeamID': l_team,
                'LScore': l_score,
                'WLoc': location,
                'NumOT': 0
            })
    
    # Create detailed results by adding box score stats
    detailed_results = []
    for game in regular_results:
        w_fgm = int(game['WScore'] * 0.4)
        w_fga = w_fgm + random.randint(10, 30)
        w_fgm3 = int(w_fgm * 0.3)
        w_fga3 = w_fgm3 + random.randint(5, 15)
        
        l_fgm = int(game['LScore'] * 0.4)
        l_fga = l_fgm + random.randint(10, 30)
        l_fgm3 = int(l_fgm * 0.3)
        l_fga3 = l_fgm3 + random.randint(5, 15)
        
        detailed_game = game.copy()
        detailed_game.update({
            'WFGM': w_fgm,
            'WFGA': w_fga,
            'WFGM3': w_fgm3,
            'WFGA3': w_fga3,
            'WFTM': game['WScore'] - 2*w_fgm - 3*w_fgm3,
            'WFTA': random.randint(10, 25),
            'WOR': random.randint(5, 15),
            'WDR': random.randint(15, 35),
            'WAst': random.randint(10, 25),
            'WTO': random.randint(5, 20),
            'WStl': random.randint(3, 12),
            'WBlk': random.randint(1, 8),
            'WPF': random.randint(10, 25),
            'LFGM': l_fgm,
            'LFGA': l_fga,
            'LFGM3': l_fgm3,
            'LFGA3': l_fga3,
            'LFTM': game['LScore'] - 2*l_fgm - 3*l_fgm3,
            'LFTA': random.randint(10, 25),
            'LOR': random.randint(5, 15),
            'LDR': random.randint(15, 35),
            'LAst': random.randint(10, 25),
            'LTO': random.randint(5, 20),
            'LStl': random.randint(3, 12),
            'LBlk': random.randint(1, 8),
            'LPF': random.randint(10, 25)
        })
        detailed_results.append(detailed_game)
    
    # Generate tournament results
    tourney_results = []
    for season in range(2015, 2025):  # No tournament results for 2025 yet
        for round_num in range(1, 7):  # 6 rounds in the tournament
            num_games = 2**(6-round_num)  # Number of games in this round
            day_num = 134 + round_num * 2  # Simple mapping for day numbers
            
            for _ in range(num_games):
                # Randomly select teams
                team1 = random.choice(men_teams['TeamID'])
                team2 = random.choice([t for t in men_teams['TeamID'] if t != team1])
                
                # Generate scores
                score1 = random.randint(60, 90)
                score2 = random.randint(50, 85)
                
                # Determine winner and loser
                if score1 > score2:
                    w_team, w_score = team1, score1
                    l_team, l_score = team2, score2
                else:
                    w_team, w_score = team2, score2
                    l_team, l_score = team1, score1
                
                tourney_results.append({
                    'Season': season,
                    'DayNum': day_num,
                    'WTeamID': w_team,
                    'WScore': w_score,
                    'LTeamID': l_team,
                    'LScore': l_score,
                    'WLoc': 'N',  # All tournament games are neutral
                    'NumOT': 0
                })
    
    # Generate tournament seeds
    tourney_seeds = []
    for season in range(2015, 2026):
        for region in ['W', 'X', 'Y', 'Z']:
            for seed_num in range(1, 17):
                seed_str = f"{region}{seed_num:02d}"
                team_id = random.choice(men_teams['TeamID'])
                
                tourney_seeds.append({
                    'Season': season,
                    'Seed': seed_str,
                    'TeamID': team_id
                })
    
    # Generate conference tournament data
    conf_tourney_games = []
    for season in range(2015, 2026):
        for conf in ['ACC', 'B10', 'B12', 'SEC', 'PAC']:
            # Each conference has 8 teams in tournament
            teams = random.sample(men_teams['TeamID'].tolist(), 8)
            
            # First round - 4 games
            day_num = 128
            round1_winners = []
            for i in range(0, 8, 2):
                team1, team2 = teams[i], teams[i+1]
                score1 = random.randint(60, 90)
                score2 = random.randint(50, 85)
                
                if score1 > score2:
                    w_team, w_score = team1, score1
                    l_team, l_score = team2, score2
                else:
                    w_team, w_score = team2, score2
                    l_team, l_score = team1, score1
                
                round1_winners.append(w_team)
                
                conf_tourney_games.append({
                    'ConfAbbrev': conf,
                    'Season': season,
                    'DayNum': day_num,
                    'WTeamID': w_team,
                    'WScore': w_score,
                    'LTeamID': l_team,
                    'LScore': l_score,
                    'WLoc': 'N',
                    'NumOT': 0
                })
            
            # Second round - 2 games
            day_num = 129
            round2_winners = []
            for i in range(0, 4, 2):
                team1, team2 = round1_winners[i], round1_winners[i+1]
                score1 = random.randint(60, 90)
                score2 = random.randint(50, 85)
                
                if score1 > score2:
                    w_team, w_score = team1, score1
                    l_team, l_score = team2, score2
                else:
                    w_team, w_score = team2, score2
                    l_team, l_score = team1, score1
                
                round2_winners.append(w_team)
                
                conf_tourney_games.append({
                    'ConfAbbrev': conf,
                    'Season': season,
                    'DayNum': day_num,
                    'WTeamID': w_team,
                    'WScore': w_score,
                    'LTeamID': l_team,
                    'LScore': l_score,
                    'WLoc': 'N',
                    'NumOT': 0
                })
            
            # Final round - 1 game
            day_num = 130
            team1, team2 = round2_winners
            score1 = random.randint(60, 90)
            score2 = random.randint(50, 85)
            
            if score1 > score2:
                w_team, w_score = team1, score1
                l_team, l_score = team2, score2
            else:
                w_team, w_score = team2, score2
                l_team, l_score = team1, score1
            
            conf_tourney_games.append({
                'ConfAbbrev': conf,
                'Season': season,
                'DayNum': day_num,
                'WTeamID': w_team,
                'WScore': w_score,
                'LTeamID': l_team,
                'LScore': l_score,
                'WLoc': 'N',
                'NumOT': 0
            })
    
    # Generate sample submission
    sample_submission = []
    for season in [2025]:
        # Generate all possible matchups between men's teams
        for i, team1 in enumerate(men_teams['TeamID']):
            for team2 in men_teams['TeamID'][i+1:]:
                sample_submission.append({
                    'ID': f"{season}_{min(team1, team2)}_{max(team1, team2)}",
                    'Pred': 0.5
                })
        
        # Generate all possible matchups between women's teams
        for i, team1 in enumerate(women_teams['TeamID']):
            for team2 in women_teams['TeamID'][i+1:]:
                sample_submission.append({
                    'ID': f"{season}_{min(team1, team2)}_{max(team1, team2)}",
                    'Pred': 0.5
                })
    
    # Save all data to CSV
    print("Saving test data...")
    men_teams.to_csv(os.path.join(output_dir, 'MTeams.csv'), index=False)
    women_teams.to_csv(os.path.join(output_dir, 'WTeams.csv'), index=False)
    pd.DataFrame(team_spellings).to_csv(os.path.join(output_dir, 'MTeamSpellings.csv'), index=False)
    
    seasons.to_csv(os.path.join(output_dir, 'MSeasons.csv'), index=False)
    seasons.to_csv(os.path.join(output_dir, 'WSeasons.csv'), index=False)
    
    pd.DataFrame(regular_results).to_csv(os.path.join(output_dir, 'MRegularSeasonCompactResults.csv'), index=False)
    pd.DataFrame(detailed_results).to_csv(os.path.join(output_dir, 'MRegularSeasonDetailedResults.csv'), index=False)
    pd.DataFrame(tourney_results).to_csv(os.path.join(output_dir, 'MNCAATourneyCompactResults.csv'), index=False)
    pd.DataFrame(tourney_seeds).to_csv(os.path.join(output_dir, 'MNCAATourneySeeds.csv'), index=False)
    pd.DataFrame(conf_tourney_games).to_csv(os.path.join(output_dir, 'MConferenceTourneyGames.csv'), index=False)
    pd.DataFrame(sample_submission).to_csv(os.path.join(output_dir, 'SampleSubmissionStage1.csv'), index=False)
    
    print(f"Test data generated successfully in {output_dir}")
    return {
        'MTeams': men_teams,
        'WTeams': women_teams,
        'MTeamSpellings': pd.DataFrame(team_spellings),
        'MSeasons': seasons,
        'WSeasons': seasons,
        'MRegularSeasonCompactResults': pd.DataFrame(regular_results),
        'MRegularSeasonDetailedResults': pd.DataFrame(detailed_results),
        'MNCAATourneyCompactResults': pd.DataFrame(tourney_results),
        'MNCAATourneySeeds': pd.DataFrame(tourney_seeds),
        'MConferenceTourneyGames': pd.DataFrame(conf_tourney_games),
        'SampleSubmissionStage1': pd.DataFrame(sample_submission)
    }

def generate_test_kenpom_data(output_dir, test_data):
    """
    Generate synthetic KenPom data that aligns with the test Kaggle data
    """
    kenpom_dir = os.path.join(output_dir, 'kenpom')
    os.makedirs(kenpom_dir, exist_ok=True)
    
    # Extract team IDs and seasons from the test data
    team_ids = test_data['MTeams']['TeamID'].tolist()
    seasons = range(2015, 2026)
    
    # Create a mapping from team name to team ID for later use
    team_name_to_id = dict(zip(test_data['MTeams']['TeamName'], test_data['MTeams']['TeamID']))
    
    # Create KenPom-like columns
    kenpom_rows = []
    for season in seasons:
        for team_id in team_ids[:60]:  # Use only a subset of teams
            team_name = test_data['MTeams'][test_data['MTeams']['TeamID'] == team_id]['TeamName'].iloc[0]
            
            # Create random efficiency metrics
            offensive_efficiency = random.uniform(95, 120)
            defensive_efficiency = random.uniform(90, 115)
            tempo = random.uniform(60, 80)
            
            # Create rankings based on the metrics
            offensive_rank = random.randint(1, 100)
            defensive_rank = random.randint(1, 100)
            overall_rank = random.randint(1, 100)
            
            kenpom_rows.append({
                'Season': season,
                'Full Team Name': team_name,
                'AdjO': offensive_efficiency,
                'AdjD': defensive_efficiency,
                'AdjT': tempo,
                'OE Rank': offensive_rank,
                'DE Rank': defensive_rank,
                'Overall Rank': overall_rank
            })
    
    # Create DataFrame
    kenpom_df = pd.DataFrame(kenpom_rows)
    
    # Save the KenPom data
    kenpom_path = os.path.join(kenpom_dir, 'historical_kenpom_data.csv')
    kenpom_df.to_csv(kenpom_path, index=False)
    print(f"Generated synthetic KenPom data at {kenpom_path}")
    
    # Create team mapping
    mapping_rows = []
    for team_name, team_id in team_name_to_id.items():
        mapping_rows.append({
            'KenPomTeam': team_name,
            'KaggleTeamID': team_id
        })
    
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_path = os.path.join(kenpom_dir, 'team_mapping.csv')
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Generated team mapping at {mapping_path}")
    
    # Create enhanced KenPom data with TeamIDs
    kenpom_df['TeamID'] = kenpom_df['Full Team Name'].map(team_name_to_id)
    enhanced_path = os.path.join(kenpom_dir, 'enhanced_kenpom_data.csv')
    kenpom_df.to_csv(enhanced_path, index=False)
    print(f"Generated enhanced KenPom data at {enhanced_path}")
    
    return kenpom_df

def run_test_pipeline():
    """
    Generate test data and run the pipeline with it
    """
    # Set up output directory
    output_dir = '/Volumes/MINT/projects/model/data/test-data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate test Kaggle data
    print("Generating test Kaggle data...")
    test_data = generate_test_kaggle_data(output_dir)
    
    # Generate test KenPom data
    print("\nGenerating test KenPom data...")
    generate_test_kenpom_data(output_dir, test_data)
    
    # Print instructions
    print("\nTest data generation complete!")
    print(f"Test data is available at: {output_dir}")
    print("\nTo run the pipeline with test data, use:")
    print(f"  python run_pipeline.py --kaggle-dir {output_dir} --kenpom-dir {output_dir}/kenpom")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test data for the March Madness pipeline")
    parser.add_argument('--output', type=str, default='/Volumes/MINT/projects/model/data/test-data',
                       help='Directory to save the generated test data')
    parser.add_argument('--run-pipeline', action='store_true', 
                       help='Run the pipeline with the generated test data')
    
    args = parser.parse_args()
    
    # Generate test data
    test_data = generate_test_kaggle_data(args.output)
    generate_test_kenpom_data(args.output, test_data)
    
    # Optionally run the pipeline
    if args.run_pipeline:
        print("\nRunning pipeline with test data...")
        
        # Import pipeline module
        from run_pipeline import run_full_pipeline
        
        # Run pipeline with test data
        run_full_pipeline(
            use_kenpom=True,
            tournament_season=2025,
            train_seasons=range(2015, 2025),
            prepare_kenpom=True,
            data_dir=args.output
        )
