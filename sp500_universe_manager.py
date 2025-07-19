#!/usr/bin/env python3
"""
S&P 500 Universe Manager
========================

Comprehensive S&P 500 stock universe management system for mlTrainer.
Scrapes and updates S&P 500 constituents quarterly from Wikipedia.
Provides sector analysis, market cap weighting, and index rebalancing handling.
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
import time
from dataclasses import dataclass, asdict
import yfinance as yf

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class SP500Stock:
    """S&P 500 stock information"""
    symbol: str
    company_name: str
    sector: str
    sub_industry: str
    headquarters: str
    date_added: str
    cik: str
    founded: str
    market_cap: Optional[float] = None
    weight: Optional[float] = None
    last_updated: Optional[str] = None


@dataclass
class SP500Universe:
    """S&P 500 universe data"""
    stocks: List[SP500Stock]
    last_updated: str
    total_stocks: int
    sectors: Dict[str, int]
    market_cap_total: Optional[float] = None


class SP500UniverseManager:
    """
    S&P 500 Universe Manager

    Features:
    - Quarterly scraping from Wikipedia
    - Sector analysis and rotation
    - Market cap weighting
    - Index rebalancing handling
    - Real-time data integration
    """

    def __init__(self):
        self.wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        self.data_dir = Path("data/sp500")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Quarterly update schedule
        self.update_months = [3, 6, 9, 12]  # March, June, September, December
        self.update_days = [31, 30, 30, 31]  # End of each quarter

        # Cache settings
        self.cache_duration_days = 90  # 3 months
        self.universe_data = None

        logger.info("S&P 500 Universe Manager initialized")

    def should_update_universe(self) -> bool:
        """Check if universe should be updated based on quarterly schedule"""
        now = datetime.now()

        # Check if we're at the end of a quarter
        if now.month in self.update_months:
            day_index = self.update_months.index(now.month)
            if now.day >= self.update_days[day_index]:
                return True

        # Check cache age
        cache_file = self.data_dir / "sp500_universe.json"
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.days > self.cache_duration_days:
                return True
        else:
            return True

        return False

    def scrape_sp500_wikipedia(self) -> List[SP500Stock]:
        """Scrape S&P 500 constituents from Wikipedia"""
        try:
            logger.info("Scraping S&P 500 constituents from Wikipedia...")

            # Add headers to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(self.wiki_url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the main table with S&P 500 constituents
            tables = soup.find_all('table', {'class': 'wikitable'})

            stocks = []
            for table in tables:
                rows = table.find_all('tr')[1:]  # Skip header row

                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 8:  # Ensure we have enough columns
                        try:
                            # Extract data from cells
                            symbol = cells[0].get_text(strip=True)
                            company_name = cells[1].get_text(strip=True)
                            sector = cells[2].get_text(strip=True)
                            sub_industry = cells[3].get_text(strip=True)
                            headquarters = cells[4].get_text(strip=True)
                            date_added = cells[5].get_text(strip=True)
                            cik = cells[6].get_text(strip=True)
                            founded = cells[7].get_text(strip=True)

                            # Create stock object
                            stock = SP500Stock(
                                symbol=symbol,
                                company_name=company_name,
                                sector=sector,
                                sub_industry=sub_industry,
                                headquarters=headquarters,
                                date_added=date_added,
                                cik=cik,
                                founded=founded
                            )

                            stocks.append(stock)

                        except Exception as e:
                            logger.warning(f"Error parsing row: {e}")
                            continue

            logger.info(f"Successfully scraped {len(stocks)} S&P 500 stocks")
            return stocks

        except Exception as e:
            logger.error(f"Error scraping S&P 500 data: {e}")
            return []

    def get_market_cap_data(self, symbols: List[str]) -> Dict[str, float]:
        """Get market cap data for symbols using yfinance"""
        market_caps = {}

        try:
            logger.info("Fetching market cap data for S&P 500 stocks...")

            # Process in batches to avoid rate limiting
            batch_size = 50
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]

                for symbol in batch:
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info

                        if 'marketCap' in info and info['marketCap'] is not None:
                            market_caps[symbol] = info['marketCap']
                        else:
                            market_caps[symbol] = 0.0

                    except Exception as e:
                        logger.warning(
                            f"Error getting market cap for {symbol}: {e}")
                        market_caps[symbol] = 0.0

                # Rate limiting
                time.sleep(1)

            logger.info(
                f"Retrieved market cap data for {len(market_caps)} stocks")
            return market_caps

        except Exception as e:
            logger.error(f"Error fetching market cap data: {e}")
            return {}

    def calculate_weights(self,
                          stocks: List[SP500Stock],
                          market_caps: Dict[str,
                                            float]) -> List[SP500Stock]:
        """Calculate market cap weights for stocks"""
        total_market_cap = sum(market_caps.values())

        for stock in stocks:
            if stock.symbol in market_caps:
                stock.market_cap = market_caps[stock.symbol]
                stock.weight = market_caps[stock.symbol] / \
                    total_market_cap if total_market_cap > 0 else 0
            else:
                stock.market_cap = 0.0
                stock.weight = 0.0

            stock.last_updated = datetime.now().isoformat()

        return stocks

    def analyze_sectors(self, stocks: List[SP500Stock]) -> Dict[str, int]:
        """Analyze sector distribution"""
        sector_counts = {}
        for stock in stocks:
            sector = stock.sector
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        return sector_counts

    def create_universe(self) -> SP500Universe:
        """Create complete S&P 500 universe"""
        try:
            # Scrape stock data
            stocks = self.scrape_sp500_wikipedia()

            if not stocks:
                logger.error("Failed to scrape S&P 500 data")
                return None

            # Get market cap data
            symbols = [stock.symbol for stock in stocks]
            market_caps = self.get_market_cap_data(symbols)

            # Calculate weights
            stocks = self.calculate_weights(stocks, market_caps)

            # Analyze sectors
            sectors = self.analyze_sectors(stocks)

            # Calculate total market cap
            total_market_cap = sum(
                stock.market_cap for stock in stocks if stock.market_cap)

            universe = SP500Universe(
                stocks=stocks,
                last_updated=datetime.now().isoformat(),
                total_stocks=len(stocks),
                sectors=sectors,
                market_cap_total=total_market_cap
            )

            logger.info(f"Created S&P 500 universe with {len(stocks)} stocks")
            return universe

        except Exception as e:
            logger.error(f"Error creating S&P 500 universe: {e}")
            return None

    def save_universe(self, universe: SP500Universe):
        """Save universe data to file"""
        try:
            # Convert to serializable format
            universe_dict = {
                'last_updated': universe.last_updated,
                'total_stocks': universe.total_stocks,
                'sectors': universe.sectors,
                'market_cap_total': universe.market_cap_total,
                'stocks': [asdict(stock) for stock in universe.stocks]
            }

            # Save to JSON
            universe_file = self.data_dir / "sp500_universe.json"
            with open(universe_file, 'w') as f:
                json.dump(universe_dict, f, indent=2)

            # Save to CSV for easy analysis
            csv_file = self.data_dir / "sp500_universe.csv"
            df = pd.DataFrame([asdict(stock) for stock in universe.stocks])
            df.to_csv(csv_file, index=False)

            logger.info(
                f"Saved S&P 500 universe data to {universe_file} and {csv_file}")

        except Exception as e:
            logger.error(f"Error saving universe data: {e}")

    def load_universe(self) -> Optional[SP500Universe]:
        """Load universe data from file"""
        try:
            universe_file = self.data_dir / "sp500_universe.json"

            if not universe_file.exists():
                logger.info("No cached universe data found")
                return None

            with open(universe_file, 'r') as f:
                data = json.load(f)

            # Reconstruct universe object
            stocks = [SP500Stock(**stock_data)
                      for stock_data in data['stocks']]

            universe = SP500Universe(
                stocks=stocks,
                last_updated=data['last_updated'],
                total_stocks=data['total_stocks'],
                sectors=data['sectors'],
                market_cap_total=data.get('market_cap_total')
            )

            logger.info(f"Loaded S&P 500 universe with {len(stocks)} stocks")
            return universe

        except Exception as e:
            logger.error(f"Error loading universe data: {e}")
            return None

    def get_current_universe(self) -> SP500Universe:
        """Get current S&P 500 universe, updating if necessary"""
        # Check if we need to update
        if self.should_update_universe():
            logger.info("Updating S&P 500 universe...")
            universe = self.create_universe()
            if universe:
                self.save_universe(universe)
                self.universe_data = universe
            else:
                logger.warning(
                    "Failed to create new universe, loading cached data")
                self.universe_data = self.load_universe()
        else:
            # Load cached data
            if self.universe_data is None:
                self.universe_data = self.load_universe()

                # If no cached data, create new universe
                if self.universe_data is None:
                    logger.info(
                        "No cached data available, creating new universe...")
                    universe = self.create_universe()
                    if universe:
                        self.save_universe(universe)
                        self.universe_data = universe

        return self.universe_data

    def get_stocks_by_sector(self, sector: str) -> List[SP500Stock]:
        """Get all stocks in a specific sector"""
        universe = self.get_current_universe()
        if not universe:
            return []

        return [stock for stock in universe.stocks if stock.sector == sector]

    def get_top_stocks_by_market_cap(self, n: int = 50) -> List[SP500Stock]:
        """Get top N stocks by market cap"""
        universe = self.get_current_universe()
        if not universe:
            return []

        sorted_stocks = sorted(
            universe.stocks,
            key=lambda x: x.market_cap or 0,
            reverse=True)
        return sorted_stocks[:n]

    def get_sector_weights(self) -> Dict[str, float]:
        """Get sector weights based on market cap"""
        universe = self.get_current_universe()
        if not universe:
            return {}

        sector_weights = {}
        for stock in universe.stocks:
            sector = stock.sector
            weight = stock.weight or 0
            sector_weights[sector] = sector_weights.get(sector, 0) + weight

        return sector_weights

    def get_stock_info(self, symbol: str) -> Optional[SP500Stock]:
        """Get information for a specific stock"""
        universe = self.get_current_universe()
        if not universe:
            return None

        for stock in universe.stocks:
            if stock.symbol == symbol.upper():
                return stock

        return None

    def get_universe_summary(self) -> Dict[str, Any]:
        """Get comprehensive universe summary"""
        universe = self.get_current_universe()
        if not universe:
            return {}

        sector_weights = self.get_sector_weights()
        top_stocks = self.get_top_stocks_by_market_cap(10)

        return {
            'total_stocks': universe.total_stocks,
            'last_updated': universe.last_updated,
            'total_market_cap': universe.market_cap_total,
            'sector_distribution': universe.sectors,
            'sector_weights': sector_weights,
            'top_stocks': [stock.symbol for stock in top_stocks],
            'update_schedule': {
                'next_update': self._get_next_update_date(),
                'update_months': self.update_months
            }
        }

    def _get_next_update_date(self) -> str:
        """Get the next scheduled update date"""
        now = datetime.now()

        for month in self.update_months:
            if month > now.month:
                day = self.update_days[self.update_months.index(month)]
                next_date = datetime(now.year, month, day)
                return next_date.isoformat()

        # If we're past all months this year, next update is first month next
        # year
        next_date = datetime(
            now.year + 1,
            self.update_months[0],
            self.update_days[0])
        return next_date.isoformat()

    def export_for_mltrainer(self) -> Dict[str, Any]:
        """Export universe data in mlTrainer-compatible format"""
        universe = self.get_current_universe()
        if not universe:
            return {}

        # Create mlTrainer-compatible format
        mltrainer_data = {
            'universe_name': 'S&P 500',
            'last_updated': universe.last_updated,
            'total_stocks': universe.total_stocks,
            'stocks': {
                stock.symbol: {
                    'company_name': stock.company_name,
                    'sector': stock.sector,
                    'sub_industry': stock.sub_industry,
                    'market_cap': stock.market_cap,
                    'weight': stock.weight,
                    'headquarters': stock.headquarters,
                    'date_added': stock.date_added,
                    'cik': stock.cik,
                    'founded': stock.founded
                }
                for stock in universe.stocks
            },
            'sectors': universe.sectors,
            'sector_weights': self.get_sector_weights(),
            'update_schedule': {
                'quarterly_updates': True,
                'update_months': self.update_months,
                'next_update': self._get_next_update_date()
            }
        }

        return mltrainer_data


# Global instance
_sp500_manager = None


def get_sp500_universe_manager() -> SP500UniverseManager:
    """Get global S&P 500 universe manager instance"""
    global _sp500_manager
    if _sp500_manager is None:
        _sp500_manager = SP500UniverseManager()
    return _sp500_manager


def get_sp500_universe() -> SP500Universe:
    """Get current S&P 500 universe"""
    manager = get_sp500_universe_manager()
    return manager.get_current_universe()


def get_sp500_stocks() -> List[str]:
    """Get list of S&P 500 stock symbols"""
    universe = get_sp500_universe()
    if not universe:
        return []
    return [stock.symbol for stock in universe.stocks]


def get_sp500_sectors() -> Dict[str, int]:
    """Get S&P 500 sector distribution"""
    universe = get_sp500_universe()
    if not universe:
        return {}
    return universe.sectors


def get_sp500_sector_weights() -> Dict[str, float]:
    """Get S&P 500 sector weights by market cap"""
    manager = get_sp500_universe_manager()
    return manager.get_sector_weights()


def get_sp500_summary() -> Dict[str, Any]:
    """Get comprehensive S&P 500 summary"""
    manager = get_sp500_universe_manager()
    return manager.get_universe_summary()


def export_sp500_for_mltrainer() -> Dict[str, Any]:
    """Export S&P 500 data for mlTrainer integration"""
    manager = get_sp500_universe_manager()
    return manager.export_for_mltrainer()


if __name__ == "__main__":
    # Test the S&P 500 universe manager
    manager = get_sp500_universe_manager()

    print("S&P 500 Universe Manager Test")
    print("=" * 40)

    # Get universe
    universe = manager.get_current_universe()
    if universe:
        print(f"Universe loaded: {universe.total_stocks} stocks")
        print(f"Last updated: {universe.last_updated}")
        print(f"Sectors: {universe.sectors}")

        # Get summary
        summary = manager.get_universe_summary()
        print(f"\nSummary: {summary}")

        # Test sector weights
        sector_weights = manager.get_sector_weights()
        print(f"\nSector Weights: {sector_weights}")

        # Test top stocks
        top_stocks = manager.get_top_stocks_by_market_cap(5)
        print(f"\nTop 5 stocks by market cap:")
        for stock in top_stocks:
            print(
                f"  {stock.symbol}: {stock.company_name} (${stock.market_cap:,.0f})")

    else:
        print("Failed to load S&P 500 universe")
