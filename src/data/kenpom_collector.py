import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
import time
from pathlib import Path
import json
import re
from datetime import datetime, timedelta
import os
from src.utils.credentials import CredentialsManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KenPomScraper:
    def __init__(self, username=None, password=None, data_dir="data/external/kenpom"):
        """
        Initialize the KenPom scraper.
        
        Args:
            username: KenPom account username (subscription required)
            password: KenPom account password
            data_dir: Directory to store KenPom data
        """
        # If username/password not provided, try to load from credentials
        if username is None or password is None:
            creds = CredentialsManager()
            creds_username, creds_password = creds.get_kenpom_credentials()
            username = username or creds_username
            password = password or creds_password
            
            if not username or not password:
                logger.warning("KenPom credentials not provided and not found in credentials file")
                
        self.username = username
        self.password = password
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.logged_in = False
        
    def login(self):
        """Log in to KenPom website"""
        try:
            # Visit the login page first
            login_page = "https://kenpom.com/index.php"
            self.session.get(login_page, headers=self.headers)
            
            # Now attempt to login
            login_url = "https://kenpom.com/handler/login-handler.php"
            payload = {
                'email': self.username,
                'password': self.password,
                'remember': '1'
            }
            
            response = self.session.post(login_url, data=payload, headers=self.headers)
            
            # Check if login was successful by accessing a restricted page
            test_page = self.session.get("https://kenpom.com/index.php", headers=self.headers)
            
            # If we see "log out" in the response, we're logged in
            if "log out" in test_page.text.lower():
                logger.info("Successfully logged in to KenPom")
                self.logged_in = True
                return True
            else:
                logger.error("Failed to log in to KenPom. Check credentials.")
                return False
                
        except Exception as e:
            logger.error(f"Error logging in to KenPom: {str(e)}")
            return False
    
    def _save_html(self, content, year, page_type="rating"):
        """Save raw HTML content for archiving/debugging"""
        html_dir = self.data_dir / "raw_html"
        html_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{page_type}_{year}_{timestamp}.html"
        
        with open(html_dir / filename, "w", encoding="utf-8") as f:
            f.write(content)
            
        logger.info(f"Saved raw HTML to {filename}")
    
    def scrape_ratings_for_date(self, date, save_html=True):
        """
        Scrape ratings data for a specific date
        
        Args:
            date: Date in format 'YYYYMMDD' (str) or datetime object
            save_html: Whether to save raw HTML for debugging
        """
        if not self.logged_in:
            if not self.login():
                logger.error("Cannot scrape data without logging in")
                return None
                
        # Convert date to datetime if string
        if isinstance(date, str):
            try:
                date_obj = datetime.strptime(date, '%Y%m%d')
            except ValueError:
                logger.error(f"Invalid date format: {date}. Use 'YYYYMMDD'")
                return None
        else:
            date_obj = date
            
        # Format for KenPom URL: index.php?y=2023&d=20230212
        year = date_obj.year
        date_param = date_obj.strftime('%Y%m%d')
        
        url = f"https://kenpom.com/index.php?y={year}&d={date_param}"
        logger.info(f"Scraping KenPom data for date: {date_param}")
        
        try:
            response = self.session.get(url, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch data: Status code {response.status_code}")
                return None
                
            if "log out" not in response.text.lower():
                logger.warning("Session may have expired, attempting to re-login")
                if self.login():
                    # Try again after re-login
                    response = self.session.get(url, headers=self.headers)
                else:
                    return None
            
            # Save the raw HTML if requested
            if save_html:
                self._save_html(response.text, year, f"rating_{date_param}")
            
            # Process the HTML to extract ratings data
            return self._parse_ratings_page(response.text, date_obj)
            
        except Exception as e:
            logger.error(f"Error scraping data for {date_param}: {str(e)}")
            return None
    
    def _parse_ratings_page(self, html_content, date):
        """
        Parse the ratings page HTML to extract team ratings
        
        Args:
            html_content: Raw HTML content of the ratings page
            date: Date of the ratings (datetime object)
        """
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Find the ratings table
        ratings_table = soup.find('table', {'id': 'ratings-table'})
        if not ratings_table:
            logger.error("Could not find ratings table in the HTML")
            return None
            
        # Extract table headers
        headers = []
        header_row = ratings_table.find('thead').find_all('tr')[-1]  # Get the last header row
        
        for th in header_row.find_all('th'):
            # Clean up header text
            header_text = th.text.strip()
            if not header_text and th.get('aria-label'):
                header_text = th.get('aria-label')
            headers.append(header_text)
            
        # Process column headers
        clean_headers = []
        for h in headers:
            # Process headers for cleaner column names
            h = h.strip().replace('\n', ' ')
            if h == '':
                h = 'Rank'  # First column is typically rank
            elif h == 'Team':
                h = 'TeamName'
            elif h == 'Conf':
                h = 'Conference'
            elif h == 'AdjEM':
                h = 'AdjustedEfficiencyMargin'
            elif h == 'AdjO':
                h = 'AdjustedOffensiveEfficiency'
            elif h == 'AdjD':
                h = 'AdjustedDefensiveEfficiency'
            elif h == 'AdjT':
                h = 'AdjustedTempo'
            clean_headers.append(h)
            
        # Extract data rows
        data_rows = []
        
        # Get all rows in tbody
        for tr in ratings_table.find('tbody').find_all('tr'):
            row_data = []
            
            # Extract cell data
            for td in tr.find_all('td'):
                # Clean up the text
                cell_text = td.text.strip()
                
                # Handle team name special case - get the actual team name without the ranking
                if td.get('class') and 'team-name' in td.get('class'):
                    team_link = td.find('a')
                    if team_link:
                        cell_text = team_link.text.strip()
                        
                row_data.append(cell_text)
                
            # Create a dictionary for the row
            if len(row_data) == len(clean_headers):
                row_dict = dict(zip(clean_headers, row_data))
                row_dict['Date'] = date.strftime('%Y-%m-%d')
                row_dict['Season'] = date.year if date.month >= 7 else date.year - 1
                data_rows.append(row_dict)
            
        # Convert to DataFrame
        df = pd.DataFrame(data_rows)
        
        # Convert numeric columns
        numeric_columns = ['Rank', 'AdjustedEfficiencyMargin', 'AdjustedOffensiveEfficiency', 
                          'AdjustedDefensiveEfficiency', 'AdjustedTempo']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        logger.info(f"Extracted ratings for {len(df)} teams on {date.strftime('%Y-%m-%d')}")
        return df
    
    def scrape_historical_ratings(self, start_year=2002, end_year=None, 
                                  frequency='weekly', sleep_time=5):
        """
        Scrape historical KenPom ratings
        
        Args:
            start_year: First season to scrape (starting in the fall of that year)
            end_year: Last season to scrape (None means current year)
            frequency: 'daily', 'weekly', 'monthly', or 'yearly' 
            sleep_time: Seconds to wait between requests
            
        Returns:
            Combined DataFrame of all ratings
        """
        if end_year is None:
            end_year = datetime.now().year
            
        all_ratings = []
        
        for year in range(start_year, end_year + 1):
            logger.info(f"Processing season {year}")
            
            # Determine date range for this season (roughly October to April)
            season_start = datetime(year, 10, 1)  # October 1st
            season_end = datetime(year + 1, 4, 30)  # April 30th
            
            # If we're looking at the current season and it's before April 30,
            # use today's date as the end
            if year == datetime.now().year and datetime.now() < season_end:
                season_end = datetime.now()
            
            # Generate dates based on frequency
            dates_to_scrape = []
            
            if frequency == 'daily':
                current = season_start
                while current <= season_end:
                    dates_to_scrape.append(current)
                    current += timedelta(days=1)
            elif frequency == 'weekly':
                current = season_start
                while current <= season_end:
                    dates_to_scrape.append(current)
                    current += timedelta(days=7)
            elif frequency == 'monthly':
                for month in range(season_start.month, 13):  # Oct-Dec
                    dates_to_scrape.append(datetime(year, month, 15))
                for month in range(1, season_end.month + 1):  # Jan-Apr
                    dates_to_scrape.append(datetime(year + 1, month, 15))
            else:  # yearly - just get final ratings
                dates_to_scrape = [season_end - timedelta(days=7)]  # One week before end
            
            # Scrape each date
            for date in dates_to_scrape:
                # Check if we already have this date's data
                date_str = date.strftime('%Y-%m-%d')
                output_file = self.data_dir / f"ratings_{date_str}.csv"
                
                if output_file.exists():
                    logger.info(f"Already have data for {date_str}, skipping")
                    # Load and append to our collection
                    ratings_df = pd.read_csv(output_file)
                    all_ratings.append(ratings_df)
                    continue
                
                # Scrape the data
                ratings_df = self.scrape_ratings_for_date(date)
                
                if ratings_df is not None and not ratings_df.empty:
                    # Save individual date file
                    ratings_df.to_csv(output_file, index=False)
                    logger.info(f"Saved ratings for {date_str} to {output_file}")
                    all_ratings.append(ratings_df)
                
                # Be nice to the server
                time.sleep(sleep_time)
        
        # Combine all ratings into one DataFrame
        if all_ratings:
            combined_df = pd.concat(all_ratings, ignore_index=True)
            combined_output = self.data_dir / "kenpom_historical_ratings.csv"
            combined_df.to_csv(combined_output, index=False)
            logger.info(f"Saved combined historical ratings to {combined_output}")
            return combined_df
        else:
            logger.warning("No ratings data collected")
            return None
    
    def scrape_team_stats(self, year, team_id=None):
        """
        Scrape detailed stats for a team or all teams in a given year
        
        Args:
            year: Season year (int)
            team_id: KenPom team ID (None to scrape all teams)
        """
        # To be implemented - would scrape team stats pages
        pass
    
    def scrape_efficiency_data(self, year):
        """
        Scrape efficiency breakdowns for all teams in a given year
        
        Args:
            year: Season year (int)
        """
        # To be implemented - would scrape efficiency data
        pass

if __name__ == "__main__":
    # Example usage
    username = "your_kenpom_username"
    password = "your_kenpom_password"
    
    scraper = KenPomScraper(username, password)
    
    # 1. Scrape a single date
    # today = datetime.now()
    # ratings = scraper.scrape_ratings_for_date(today)
    
    # 2. Scrape historical data (careful with this!)
    # ratings = scraper.scrape_historical_ratings(start_year=2020, end_year=2023, frequency='monthly')
    
    # 3. For testing, just scrape last season end ratings
    # last_season = datetime.now().year - 1
    # end_date = datetime(last_season + 1, 4, 15)  # April 15th of the following year
    # ratings = scraper.scrape_ratings_for_date(end_date)
    
    logger.info("KenPom scraper completed.")
