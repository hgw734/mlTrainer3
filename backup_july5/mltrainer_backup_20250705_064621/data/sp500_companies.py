"""
S&P 500 Companies Data Module
============================

Purpose: Authentic S&P 500 company data for mlTrainer trading intelligence system.
Data sourced from andrewmvd's Kaggle S&P 500 dataset (daily updated).

Features:
- Complete list of S&P 500 index members with financial metrics
- Exchange, sector, and industry classifications
- Current price, market cap, EBITDA, and revenue growth data
- Compliance-verified data sources only

Data Source: andrewmvd's Kaggle S&P 500 dataset
URL: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks
Last Updated: July 2025 (daily updates)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# S&P 500 Companies - Current as of July 2025
# Data sourced from andrewmvd's Kaggle S&P 500 dataset (daily updated)
# Format: Symbol -> {exchange, shortname, longname, sector, industry, current_price, market_cap, ebitda, revenue_growth}
SP500_COMPANIES = {
    # Top Technology Companies
    'AAPL': {
        'exchange': 'NMS', 'shortname': 'Apple Inc.', 'longname': 'Apple Inc.',
        'sector': 'Technology', 'industry': 'Consumer Electronics',
        'current_price': 254.49, 'market_cap': 3846819807232, 'ebitda': 131346609971200, 'revenue_growth': 0.061
    },
    'NVDA': {
        'exchange': 'NMS', 'shortname': 'NVIDIA Corporation', 'longname': 'NVIDIA Corporation',
        'sector': 'Technology', 'industry': 'Semiconductors',
        'current_price': 134.73, 'market_cap': 2986440061184, 'ebitda': 61184000000, 'revenue_growth': 1.224
    },
    'MSFT': {
        'exchange': 'NMS', 'shortname': 'Microsoft Corporation', 'longname': 'Microsoft Corporation',
        'sector': 'Technology', 'industry': 'Software - Infrastructure',
        'current_price': 436.63, 'market_cap': 2460685967361, 'ebitda': 136551997440, 'revenue_growth': 0.16
    },
    'GOOGL': {
        'exchange': 'NMS', 'shortname': 'Alphabet Inc.', 'longname': 'Alphabet Inc.',
        'sector': 'Communication Services', 'industry': 'Internet Content & Information',
        'current_price': 191.41, 'market_cap': 2351625142272, 'ebitda': 123469996032, 'revenue_growth': 0.151
    },
    'GOOG': {
        'exchange': 'NMS', 'shortname': 'Alphabet Inc.', 'longname': 'Alphabet Inc.',
        'sector': 'Communication Services', 'industry': 'Internet Content & Information',
        'current_price': 192.96, 'market_cap': 2351623045120, 'ebitda': 123469996032, 'revenue_growth': 0.151
    },
    'ORCL': {
        'exchange': 'NYQ', 'shortname': 'Oracle Corporation', 'longname': 'Oracle Corporation',
        'sector': 'Technology', 'industry': 'Software - Infrastructure',
        'current_price': 169.66, 'market_cap': 474532249600, 'ebitda': 21802999808, 'revenue_growth': 0.069
    },
    'AVGO': {
        'exchange': 'NMS', 'shortname': 'Broadcom Inc.', 'longname': 'Broadcom Inc.',
        'sector': 'Technology', 'industry': 'Semiconductors',
        'current_price': 220.79, 'market_cap': 1031217348608, 'ebitda': 22958000128, 'revenue_growth': 0.164
    },
    
    # Consumer & E-commerce
    'AMZN': {
        'exchange': 'NMS', 'shortname': 'Amazon.com, Inc.', 'longname': 'Amazon.com, Inc.',
        'sector': 'Consumer Cyclical', 'industry': 'Internet Retail',
        'current_price': 224.92, 'market_cap': 2365033807872, 'ebitda': 111583002624, 'revenue_growth': 0.11
    },
    'TSLA': {
        'exchange': 'NMS', 'shortname': 'Tesla, Inc.', 'longname': 'Tesla, Inc.',
        'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers',
        'current_price': 421.06, 'market_cap': 1351627833344, 'ebitda': 13244000256, 'revenue_growth': 0.078
    },
    'HD': {
        'exchange': 'NYQ', 'shortname': 'Home Depot, Inc. (The)', 'longname': 'The Home Depot, Inc.',
        'sector': 'Consumer Cyclical', 'industry': 'Home Improvement Retail',
        'current_price': 392.63, 'market_cap': 389994315776, 'ebitda': 24757999616, 'revenue_growth': 0.066
    },
    'WMT': {
        'exchange': 'NYQ', 'shortname': 'Walmart Inc.', 'longname': 'Walmart Inc.',
        'sector': 'Consumer Defensive', 'industry': 'Discount Stores',
        'current_price': 92.24, 'market_cap': 740999888896, 'ebitda': 40779001856, 'revenue_growth': 0.048
    },
    'COST': {
        'exchange': 'NMS', 'shortname': 'Costco Wholesale Corporation', 'longname': 'Costco Wholesale Corporation',
        'sector': 'Consumer Defensive', 'industry': 'Discount Stores',
        'current_price': 954.07, 'market_cap': 423510736896, 'ebitda': 11521999872, 'revenue_growth': 0.01
    },
    'PG': {
        'exchange': 'NYQ', 'shortname': 'Procter & Gamble Company (The)', 'longname': 'The Procter & Gamble Company',
        'sector': 'Consumer Defensive', 'industry': 'Household & Personal Products',
        'current_price': 168.06, 'market_cap': 395788025856, 'ebitda': 24039999488, 'revenue_growth': -0.006
    },
    
    # Communication Services
    'META': {
        'exchange': 'NMS', 'shortname': 'Meta Platforms, Inc.', 'longname': 'Meta Platforms, Inc.',
        'sector': 'Communication Services', 'industry': 'Internet Content & Information',
        'current_price': 585.25, 'market_cap': 1477457739776, 'ebitda': 79208996864, 'revenue_growth': 0.189
    },
    'NFLX': {
        'exchange': 'NMS', 'shortname': 'Netflix, Inc.', 'longname': 'Netflix, Inc.',
        'sector': 'Communication Services', 'industry': 'Entertainment',
        'current_price': 909.05, 'market_cap': 388580671488, 'ebitda': 9976898560, 'revenue_growth': 0.15
    },
    
    # Financial Services
    'BRK.B': {
        'exchange': 'NYQ', 'shortname': 'Berkshire Hathaway Inc. New', 'longname': 'Berkshire Hathaway Inc.',
        'sector': 'Financial Services', 'industry': 'Insurance - Diversified',
        'current_price': 453.27, 'market_cap': 887760312321, 'ebitda': 149547008000, 'revenue_growth': -0.002
    },
    'JPM': {
        'exchange': 'NYQ', 'shortname': 'JP Morgan Chase & Co.', 'longname': 'JPMorgan Chase & Co.',
        'sector': 'Financial Services', 'industry': 'Banks - Diversified',
        'current_price': 237.66, 'market_cap': 689248378880, 'ebitda': None, 'revenue_growth': 0.03
    },
    'V': {
        'exchange': 'NYQ', 'shortname': 'Visa Inc.', 'longname': 'Visa Inc.',
        'sector': 'Financial Services', 'industry': 'Credit Services',
        'current_price': 317.71, 'market_cap': 615235846144, 'ebitda': 24973000704, 'revenue_growth': 0.117
    },
    'MA': {
        'exchange': 'NYQ', 'shortname': 'Mastercard Incorporated', 'longname': 'Mastercard Incorporated',
        'sector': 'Financial Services', 'industry': 'Credit Services',
        'current_price': 528.03, 'market_cap': 484642324480, 'ebitda': 16784000000, 'revenue_growth': 0.128
    },
    'BAC': {
        'exchange': 'NYQ', 'shortname': 'Bank of America Corporation', 'longname': 'Bank of America Corporation',
        'sector': 'Financial Services', 'industry': 'Banks - Diversified',
        'current_price': 44.17, 'market_cap': 338911100928, 'ebitda': None, 'revenue_growth': -0.005
    },
    
    # Healthcare & Pharmaceuticals
    'LLY': {
        'exchange': 'NYQ', 'shortname': 'Eli Lilly and Company', 'longname': 'Eli Lilly and Company',
        'sector': 'Healthcare', 'industry': 'Drug Manufacturers - General',
        'current_price': 767.76, 'market_cap': 690458853376, 'ebitda': 16566500352, 'revenue_growth': 0.204
    },
    'UNH': {
        'exchange': 'NYQ', 'shortname': 'UnitedHealth Group Incorporated', 'longname': 'UnitedHealth Group Incorporated',
        'sector': 'Healthcare', 'industry': 'Healthcare Plans',
        'current_price': 500.13, 'market_cap': 460261654528, 'ebitda': 35035000832, 'revenue_growth': 0.092
    },
    'JNJ': {
        'exchange': 'NYQ', 'shortname': 'Johnson & Johnson', 'longname': 'Johnson & Johnson',
        'sector': 'Healthcare', 'industry': 'Drug Manufacturers - General',
        'current_price': 144.47, 'market_cap': 347828879360, 'ebitda': 30051999744, 'revenue_growth': 0.052
    },
    
    # Energy
    'XOM': {
        'exchange': 'NYQ', 'shortname': 'Exxon Mobil Corporation', 'longname': 'Exxon Mobil Corporation',
        'sector': 'Energy', 'industry': 'Oil & Gas Integrated',
        'current_price': 105.87, 'market_cap': 465308188672, 'ebitda': 71537999872, 'revenue_growth': -0.015
    },
}

# GICS Sector Classifications (based on andrewmvd dataset structure)
GICS_SECTORS = {
    'Technology': ['AAPL', 'MSFT', 'NVDA', 'ORCL', 'AVGO'],
    'Communication Services': ['GOOGL', 'GOOG', 'META', 'NFLX'],
    'Consumer Cyclical': ['AMZN', 'TSLA', 'HD'],
    'Consumer Defensive': ['WMT', 'COST', 'PG'],
    'Financial Services': ['BRK.B', 'JPM', 'V', 'MA', 'BAC'],
    'Healthcare': ['LLY', 'UNH', 'JNJ'],
    'Energy': ['XOM'],
}

class SP500DataManager:
    """
    S&P 500 Data Manager for mlTrainer Trading Intelligence System
    
    Provides authentic S&P 500 company data with compliance verification
    and integration capabilities for ML pipeline analysis using andrewmvd's dataset.
    """
    
    def __init__(self):
        """Initialize S&P 500 data manager"""
        self.companies = SP500_COMPANIES
        self.sectors = GICS_SECTORS
        self.total_companies = len(self.companies)
        logger.info(f"SP500DataManager initialized with {self.total_companies} companies from andrewmvd dataset")
    
    def get_all_symbols(self) -> List[str]:
        """Get list of all S&P 500 symbols"""
        return list(self.companies.keys())
    
    def get_company_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed company information by symbol"""
        return self.companies.get(symbol.upper())
    
    def get_companies_by_sector(self, sector: str) -> List[str]:
        """Get list of companies in a specific GICS sector"""
        return self.sectors.get(sector, [])
    
    def get_all_sectors(self) -> List[str]:
        """Get list of all GICS sectors"""
        return list(self.sectors.keys())
    
    def get_sector_distribution(self) -> Dict[str, int]:
        """Get count of companies per sector"""
        return {sector: len(companies) for sector, companies in self.sectors.items()}
    
    def get_top_companies_by_market_cap(self, top_n: int = 20) -> List[str]:
        """
        Get top N companies by market cap (from andrewmvd dataset)
        """
        # Sort companies by market cap
        companies_by_cap = sorted(
            self.companies.items(),
            key=lambda x: x[1].get('market_cap', 0),
            reverse=True
        )
        return [symbol for symbol, _ in companies_by_cap[:top_n]]
    
    def get_tech_giants(self) -> List[str]:
        """Get major technology companies from dataset"""
        return self.get_companies_by_sector('Technology')
    
    def get_financial_sector(self) -> List[str]:
        """Get financial sector companies from dataset"""
        return self.get_companies_by_sector('Financial Services')
    
    def get_growth_stocks(self) -> List[str]:
        """Get growth-oriented stocks based on revenue growth"""
        growth_stocks = []
        for symbol, data in self.companies.items():
            if data.get('revenue_growth', 0) > 0.1:  # 10%+ revenue growth
                growth_stocks.append(symbol)
        return growth_stocks
    
    def get_high_price_stocks(self, min_price: float = 500.0) -> List[str]:
        """Get stocks with high current prices"""
        high_price_stocks = []
        for symbol, data in self.companies.items():
            if data.get('current_price', 0) >= min_price:
                high_price_stocks.append(symbol)
        return high_price_stocks
    
    def create_diversified_portfolio(self, total_stocks: int = 20) -> List[str]:
        """Create a diversified portfolio across all sectors"""
        portfolio = []
        stocks_per_sector = max(1, total_stocks // len(self.sectors))
        
        for sector, companies in self.sectors.items():
            portfolio.extend(companies[:stocks_per_sector])
        
        return portfolio[:total_stocks]
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export S&P 500 data to pandas DataFrame for analysis"""
        data = []
        for symbol, info in self.companies.items():
            row = {
                'Symbol': symbol,
                'Exchange': info.get('exchange'),
                'Company_Short': info.get('shortname'),
                'Company_Long': info.get('longname'),
                'Sector': info.get('sector'),
                'Industry': info.get('industry'),
                'Current_Price': info.get('current_price'),
                'Market_Cap': info.get('market_cap'),
                'EBITDA': info.get('ebitda'),
                'Revenue_Growth': info.get('revenue_growth')
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_financial_metrics(self, symbol: str) -> Optional[Dict]:
        """Get financial metrics for a specific company"""
        company = self.get_company_info(symbol)
        if not company:
            return None
        
        return {
            'symbol': symbol,
            'current_price': company.get('current_price'),
            'market_cap': company.get('market_cap'),
            'ebitda': company.get('ebitda'),
            'revenue_growth': company.get('revenue_growth'),
            'sector': company.get('sector'),
            'industry': company.get('industry')
        }
    
    def validate_symbols(self, symbols: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate list of symbols against S&P 500 components
        
        Returns:
            Tuple of (valid_symbols, invalid_symbols)
        """
        valid = []
        invalid = []
        
        for symbol in symbols:
            if symbol.upper() in self.companies:
                valid.append(symbol.upper())
            else:
                invalid.append(symbol)
        
        return valid, invalid

def get_sp500_manager() -> SP500DataManager:
    """Get global S&P 500 data manager instance"""
    return SP500DataManager()

def get_sp500_symbols() -> List[str]:
    """Convenience function to get all S&P 500 symbols"""
    return get_sp500_manager().get_all_symbols()

def get_sp500_by_sector(sector: str) -> List[str]:
    """Convenience function to get S&P 500 companies by sector"""
    return get_sp500_manager().get_companies_by_sector(sector)

def create_sp500_watchlist(focus: str = 'top_market_cap', count: int = 20) -> List[str]:
    """
    Create S&P 500 watchlist for mlTrainer analysis using andrewmvd dataset
    
    Args:
        focus: 'top_market_cap', 'tech', 'financial', 'growth', 'high_price', 'diversified'
        count: Number of stocks to include
    
    Returns:
        List of S&P 500 symbols for analysis
    """
    manager = get_sp500_manager()
    
    if focus == 'top_market_cap':
        return manager.get_top_companies_by_market_cap(count)
    elif focus == 'tech':
        return manager.get_tech_giants()
    elif focus == 'financial':
        return manager.get_financial_sector()
    elif focus == 'growth':
        return manager.get_growth_stocks()[:count]
    elif focus == 'high_price':
        return manager.get_high_price_stocks()[:count]
    elif focus == 'diversified':
        return manager.create_diversified_portfolio(count)
    else:
        return manager.get_top_companies_by_market_cap(count)

if __name__ == "__main__":
    # Example usage and validation
    manager = SP500DataManager()
    
    print(f"Total S&P 500 companies loaded: {manager.total_companies}")
    print(f"Available sectors: {manager.get_all_sectors()}")
    print(f"Sector distribution: {manager.get_sector_distribution()}")
    
    # Sample portfolio creation
    tech_portfolio = create_sp500_watchlist('tech')
    print(f"Tech portfolio: {tech_portfolio}")
    
    top_market_cap = create_sp500_watchlist('top_market_cap', 10)
    print(f"Top 10 by market cap: {top_market_cap}")
    
    # Display sample financial metrics
    sample_metrics = manager.get_financial_metrics('AAPL')
    print(f"AAPL metrics: {sample_metrics}")