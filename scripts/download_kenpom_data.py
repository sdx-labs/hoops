#!/usr/bin/env python
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.kenpom_collector import KenPomScraper

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Download KenPom basketball data')
    parser.add_argument('--start-year', type=int, default=2020, 
                        help='First season to download (default: 2020)')
    parser.add_argument('--end-year', type=int, default=None, 
                        help='Last season to download (default: current year)')
    parser.add_argument('--frequency', choices=['daily', 'weekly', 'monthly', 'yearly'],
                        default='monthly', help='Data collection frequency (default: monthly)')
    parser.add_argument('--date', type=str, help='Specific date to download (format: YYYYMMDD)')
    parser.add_argument('--sleep', type=int, default=5,
                        help='Sleep time between requests in seconds (default: 5)')
    parser.add_argument('--output-dir', default="data/external/kenpom",
                        help='Directory to save KenPom data')
    
    return parser.parse_args()

def main():
    """Run KenPom data download"""
    args = parse_arguments()
    
    # Initialize the KenPom scraper
    # Credentials will be loaded from credentials.yaml or environment variables
    scraper = KenPomScraper(data_dir=args.output_dir)
    
    # If the scraper couldn't find credentials, exit
    if not scraper.username or not scraper.password:
        print("Error: KenPom credentials not found!")
        print("Please run: python scripts/setup_credentials.py")
        return 1
    
    # Login to KenPom
    if not scraper.login():
        print("Error: Failed to log in to KenPom. Check your credentials.")
        return 1
    
    # If specific date is provided, download that date only
    if args.date:
        print(f"Downloading KenPom data for specific date: {args.date}")
        ratings = scraper.scrape_ratings_for_date(args.date)
        if ratings is not None:
            print(f"Successfully downloaded data for {args.date}")
            return 0
        else:
            print(f"Failed to download data for {args.date}")
            return 1
    
    # Otherwise, download historical data for the specified years
    end_year = args.end_year or datetime.now().year
    print(f"Downloading KenPom historical data from {args.start_year} to {end_year}")
    ratings = scraper.scrape_historical_ratings(
        start_year=args.start_year,
        end_year=end_year,
        frequency=args.frequency,
        sleep_time=args.sleep
    )
    
    if ratings is not None:
        print(f"Successfully downloaded KenPom historical data")
        return 0
    else:
        print("Failed to download KenPom historical data")
        return 1

if __name__ == "__main__":
    sys.exit(main())
