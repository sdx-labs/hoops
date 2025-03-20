import os
import pandas as pd
import sys
import argparse

def apply_team_mapping(kenpom_path, mapping_path, output_path=None):
    """Apply team mapping to KenPom data and save enhanced version"""
    
    # Load KenPom data
    if not os.path.exists(kenpom_path):
        print(f"ERROR: KenPom file not found at {kenpom_path}")
        return False
    
    # Load mapping
    if not os.path.exists(mapping_path):
        print(f"ERROR: Mapping file not found at {mapping_path}")
        return False
    
    try:
        kenpom_data = pd.read_csv(kenpom_path)
        print(f"Loaded KenPom data with {len(kenpom_data)} rows")
        
        mapping_df = pd.read_csv(mapping_path)
        print(f"Loaded mapping data with {len(mapping_df)} entries")
        
        # Convert mapping to dictionary
        mapping = dict(zip(mapping_df['KenPomTeam'], mapping_df['KaggleTeamID']))
        
        # Determine the team name column in KenPom data
        team_col = None
        if 'Mapped ESPN Team Name' in kenpom_data.columns:
            team_col = 'Mapped ESPN Team Name'
        elif 'Full Team Name' in kenpom_data.columns:
            team_col = 'Full Team Name'
        else:
            # Fall back to other possible column names
            for col in ['Team', 'TeamName', 'team', 'NAME', 'name']:
                if col in kenpom_data.columns:
                    team_col = col
                    break
        
        if team_col is None:
            print("ERROR: Could not find team name column in KenPom data")
            return False
        
        print(f"Using '{team_col}' as the team name column")
        
        # Apply the mapping
        kenpom_data['TeamID'] = kenpom_data[team_col].map(mapping)
        
        # Check mapping results
        mapped_count = kenpom_data['TeamID'].notna().sum()
        print(f"Applied mapping to {mapped_count} out of {len(kenpom_data)} rows ({mapped_count/len(kenpom_data)*100:.1f}%)")
        
        # Save enhanced data
        if output_path is None:
            output_path = os.path.join(os.path.dirname(kenpom_path), 'enhanced_kenpom_data.csv')
            
        kenpom_data.to_csv(output_path, index=False)
        print(f"Saved enhanced KenPom data to {output_path}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply team mapping to KenPom data")
    parser.add_argument('--kenpom', type=str, 
                      default='/Volumes/MINT/projects/model/data/kenpom/historical_kenpom_data.csv',
                      help='Path to the KenPom data file')
    parser.add_argument('--mapping', type=str, 
                      default='/Volumes/MINT/projects/model/data/kenpom/team_mapping.csv',
                      help='Path to the team mapping file')
    parser.add_argument('--output', type=str, 
                      default='/Volumes/MINT/projects/model/data/kenpom/enhanced_kenpom_data.csv',
                      help='Output path for enhanced KenPom data')
    
    args = parser.parse_args()
    apply_team_mapping(args.kenpom, args.mapping, args.output)
