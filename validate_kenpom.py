import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def validate_kenpom_integration(
    kenpom_path='/Volumes/MINT/projects/model/data/kenpom/enhanced_kenpom_data.csv',
    teams_path='/Volumes/MINT/projects/model/data/kaggle-data/MTeams.csv',
    output_dir='/Volumes/MINT/projects/model/evaluation/kenpom_validation'
):
    """Validate KenPom data integration by checking TeamID mappings"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(kenpom_path):
        print(f"ERROR: Enhanced KenPom file not found at {kenpom_path}")
        return False
    
    if not os.path.exists(teams_path):
        print(f"ERROR: MTeams file not found at {teams_path}")
        return False
    
    # Load data
    try:
        kenpom_data = pd.read_csv(kenpom_path)
        teams_data = pd.read_csv(teams_path)
        
        print(f"Loaded KenPom data with {len(kenpom_data)} rows")
        print(f"Loaded MTeams data with {len(teams_data)} rows")
        
        # Check if TeamID exists in KenPom data
        if 'TeamID' not in kenpom_data.columns:
            print("ERROR: No TeamID column in KenPom data")
            return False
        
        # Basic statistics
        mapped_count = kenpom_data['TeamID'].notna().sum()
        total_count = len(kenpom_data)
        mapping_rate = mapped_count / total_count
        
        print(f"\nMapping Statistics:")
        print(f"  Total KenPom rows: {total_count}")
        print(f"  Mapped rows: {mapped_count} ({mapping_rate:.1%})")
        
        # Check distribution of teams
        team_counts = kenpom_data['TeamID'].value_counts()
        print(f"\nNumber of unique teams in KenPom data: {len(team_counts)}")
        print(f"Teams with most entries:")
        for team_id, count in team_counts.nlargest(10).items():
            if pd.notna(team_id):
                team_name = teams_data[teams_data['TeamID'] == team_id]['TeamName'].iloc[0]
                print(f"  {team_name} (ID: {int(team_id)}): {count} rows")
        
        # Check seasons
        season_col = None
        for col in ['Season', 'Year', 'season', 'SEASON', 'year']:
            if col in kenpom_data.columns:
                season_col = col
                break
        
        if season_col:
            print(f"\nSeason distribution:")
            season_counts = kenpom_data[season_col].value_counts().sort_index()
            for season, count in season_counts.items():
                print(f"  Season {season}: {count} rows")
            
            # Plot season coverage
            plt.figure(figsize=(10, 6))
            sns.barplot(x=season_counts.index, y=season_counts.values)
            plt.title('KenPom Data Coverage by Season')
            plt.xlabel('Season')
            plt.ylabel('Number of Teams')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'kenpom_season_coverage.png'))
            
            # Check mapping rate by season
            mapping_by_season = kenpom_data.groupby(season_col)['TeamID'].notna().mean()
            print(f"\nMapping rate by season:")
            for season, rate in mapping_by_season.items():
                print(f"  Season {season}: {rate:.1%}")
            
            # Plot mapping rate by season
            plt.figure(figsize=(10, 6))
            sns.barplot(x=mapping_by_season.index, y=mapping_by_season.values)
            plt.title('KenPom TeamID Mapping Rate by Season')
            plt.xlabel('Season')
            plt.ylabel('Mapping Rate')
            plt.xticks(rotation=45)
            plt.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'kenpom_mapping_rate.png'))
        
        print(f"\nValidation results saved to {output_dir}")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        kenpom_path = sys.argv[1]
    else:
        kenpom_path = '/Volumes/MINT/projects/model/data/kenpom/enhanced_kenpom_data.csv'
    
    validate_kenpom_integration(kenpom_path)
