import pandas as pd
import os
import glob

def load_kaggle_data(data_path='/Volumes/MINT/projects/model/data/kaggle-data'):
    """
    Load all CSV files from the Kaggle data directory into a dictionary
    """
    data = {}
    for file_path in glob.glob(os.path.join(data_path, '*.csv')):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        data[file_name] = pd.read_csv(file_path)
        print(f"Loaded {file_name} with shape {data[file_name].shape}")
    
    return data

def load_kenpom_data(data_path='/Volumes/MINT/projects/model/data/kenpom/historical_kenpom_data.csv'):
    """
    Load KenPom historical data
    """
    if os.path.exists(data_path):
        kenpom_data = pd.read_csv(data_path)
        print(f"Loaded KenPom data with shape {kenpom_data.shape}")
        return kenpom_data
    else:
        print("KenPom data file not found.")
        return None

def load_additional_data(data_path='/Volumes/MINT/projects/model/data/additional-data'):
    """
    Load all CSV files from the additional data directory into a dictionary
    """
    data = {}
    for file_path in glob.glob(os.path.join(data_path, '*.csv')):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        data[file_name] = pd.read_csv(file_path)
        print(f"Loaded additional data {file_name} with shape {data[file_name].shape}")
    
    return data

def explore_data_summary(data_dict):
    """
    Print summary statistics for each dataframe in the dictionary
    """
    for name, df in data_dict.items():
        print(f"\n--- Summary for {name} ---")
        print(f"Shape: {df.shape}")
        print("Column names:")
        print(df.columns.tolist())
        print("Data types:")
        print(df.dtypes)
        print("First 3 rows:")
        print(df.head(3))
        print("---" * 15)

if __name__ == "__main__":
    # Load and explore all datasets
    kaggle_data = load_kaggle_data()
    kenpom_data = load_kenpom_data()
    additional_data = load_additional_data()
    
    # Print summary of loaded data
    print("\nKaggle Data Summary:")
    explore_data_summary(kaggle_data)
    
    if kenpom_data is not None:
        print("\nKenPom Data Summary:")
        print(kenpom_data.head())
        print(kenpom_data.info())
    
    print("\nAdditional Data Summary:")
    explore_data_summary(additional_data)
