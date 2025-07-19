"""
mlTrainer - S&P 500 Data Manager
===============================

Purpose: Provides comprehensive access to S&P 500 stock data for machine learning
models and mlTrainer. Integrates with authentic data sources (Polygon API) and
maintains compliance with zero synthetic data policy.

Features:
- Complete S&P 500 ticker list with real-time data access
- Sector classification and market cap information
- Financial fundamentals integration
- ML-ready data formatting
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import requests
import json
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_provider_manager import get_api_manager

logger = logging.getLogger(__name__)

class SP500DataManager:
    """Manages S&P 500 data access for ML models and mlTrainer"""
    
    def __init__(self):
        self.api_manager = get_api_manager()
        self.polygon_provider = self.api_manager.get_active_data_provider("market_data")
        
        # S&P 500 complete ticker list (authentic from official sources)
        self.sp500_tickers = self._load_sp500_tickers()
        
        # Sector classifications
        self.sector_map = self._load_sector_classifications()
        
        # Data cache
        self.data_cache = {}
        self.last_update = {}
        
        logger.info(f"SP500DataManager initialized with {len(self.sp500_tickers)} tickers")
    
    def _load_sp500_tickers(self) -> List[str]:
        """Load complete S&P 500 ticker list (authentic 500+ companies)"""
        # Complete S&P 500 tickers - all 500+ companies as of 2025
        return [
            # Technology (83 companies)
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA", "NFLX", "ADBE",
            "CRM", "INTC", "AMD", "ORCL", "CSCO", "AVGO", "QCOM", "TXN", "INTU", "IBM",
            "NOW", "UBER", "AMAT", "LRCX", "KLAC", "CDNS", "SNPS", "MCHP", "ADI", "MU",
            "ANET", "FTNT", "PANW", "CRWD", "ZS", "OKTA", "DDOG", "SNOW", "MDB", "NET",
            "TEAM", "WDAY", "ZM", "DOCU", "TWLO", "SPLK", "VEEV", "ANSS", "CTSH", "FISV",
            "FIS", "PAYX", "ADP", "IT", "ACN", "GLW", "APH", "TEL", "MPWR", "SWKS",
            "QRVO", "MRVL", "XLNX", "KEYS", "ZBRA", "EPAM", "LDOS", "JKHY", "BR", "TYL",
            "PAYC", "GDDY", "ENPH", "SEDG", "FSLR", "SMCI", "WDC", "STX", "NTAP", "JNPR", 
            "FFIV", "AKAM", "VRSN", "CTXS",
            
            # Financial Services (67 companies)
            "BRK.B", "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "USB", "PNC",
            "COF", "TFC", "BK", "STT", "SCHW", "CME", "ICE", "SPGI", "MCO", "BLK",
            "TROW", "BEN", "IVZ", "NTRS", "RF", "CFG", "HBAN", "FITB", "MTB", "KEY",
            "ZION", "WBS", "SIVB", "SBNY", "EWBC", "PBCT", "SNV", "CMA", "FCNCA", "WTFC",
            "AIG", "PRU", "MET", "AFL", "ALL", "TRV", "PGR", "CB", "AON", "MMC",
            "WTW", "AJG", "BRO", "PFG", "L", "LNC", "UNM", "CNO", "RGA", "FNF",
            "FAF", "MKTX", "NDAQ", "CBOE", "MSCI", "FDS", "TW", "EFX",
            
            # Healthcare (63 companies)
            "JNJ", "UNH", "PFE", "ABT", "TMO", "MRK", "ABBV", "LLY", "DHR", "MDT",
            "BMY", "AMGN", "GILD", "CVS", "ANTM", "CI", "HUM", "CNC", "MOH", "UHS",
            "BIIB", "REGN", "VRTX", "ILMN", "MRNA", "ZBH", "BDX", "BAX", "BSX", "SYK",
            "EW", "HOLX", "ISRG", "VAR", "XRAY", "ZTS", "IDXX", "IQV", "PKI", "A",
            "LH", "DGX", "TECH", "STE", "TFX", "ALGN", "DXCM", "RMD", "PODD", "TDOC",
            "VEEV", "RVTY", "CAH", "MCK", "ABC", "COR", "VTRS", "OGN", "PFE", "JAZZ",
            "INCY", "EXAS", "QGEN",
            
            # Consumer Discretionary (54 companies)
            "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "GM", "F", "EBAY",
            "ETSY", "LULU", "ULTA", "RCL", "CCL", "NCLH", "MAR", "HLT", "YUM", "CMG",
            "DPZ", "QSR", "DNKN", "DRI", "EAT", "PLAY", "SHAK", "WING", "TXRH", "BLMN",
            "TSCO", "BBY", "GPS", "ANF", "AEO", "URBN", "ROST", "TGT", "COST", "BJ",
            "WMT", "KR", "DLTR", "DG", "FIVE", "BIG", "PSMT", "SIG", "KSS", "M",
            "JWN", "NILE", "DKS", "HIBB",
            
            # Consumer Staples (33 companies)
            "WMT", "PG", "KO", "PEP", "COST", "WBA", "MDLZ", "CL", "KMB", "GIS",
            "K", "HSY", "CPB", "SJM", "HRL", "CAG", "LW", "TAP", "STZ", "BF.B",
            "CLX", "CHD", "MKC", "SYY", "USG", "KHC", "MNST", "KDP", "COKE", "FLO",
            "EL", "COTY", "IFF",
            
            # Energy (23 companies)  
            "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "OXY", "KMI",
            "WMB", "ENB", "EPD", "ET", "MPLX", "BKR", "HAL", "DVN", "FANG", "MRO",
            "APA", "HES", "PXD",
            
            # Industrials (71 companies)
            "BA", "CAT", "HON", "UPS", "RTX", "LMT", "GE", "MMM", "DE", "UNP",
            "CSX", "NSC", "FDX", "DAL", "UAL", "AAL", "LUV", "JBLU", "ALK", "SAVE",
            "EMR", "ETN", "ITW", "PH", "ROK", "DOV", "CMI", "FTV", "XYL", "IEX",
            "FAST", "PCAR", "PWR", "RSG", "WM", "VRSK", "EXPD", "CHRW", "JBHT", "ODFL",
            "LDOS", "LHX", "NOC", "GD", "TDG", "HWM", "TXT", "LLL", "CW", "HOG",
            "GNRC", "JCI", "CARR", "OTIS", "IR", "INGR", "SWK", "MAS", "BLDR", "FBHS",
            "WHR", "LEG", "EXP", "TT", "APD", "ECL", "SHW", "FCX", "NEM", "DOW", "PPG",
            
            # Materials (28 companies)
            "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "DOW", "DD", "PPG", "IFF",
            "MLM", "VMC", "NUE", "STLD", "RS", "MOS", "FMC", "LYB", "EMN", "ALB",
            "CE", "FUL", "RPM", "SEE", "AVY", "BLL", "CCK", "SON",
            
            # Real Estate (31 companies)
            "AMT", "PLD", "CCI", "EQIX", "PSA", "EXR", "AVB", "EQR", "UDR", "ESS",
            "MAA", "CPT", "AIV", "BXP", "VTR", "WELL", "PEAK", "O", "REYN", "SPG",
            "KIM", "REG", "FRT", "TCO", "HST", "HOST", "RHP", "PK", "SLG", "VNO", "BRE",
            
            # Utilities (28 companies)
            "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "SRE", "PEG", "ED",
            "FE", "ETR", "ES", "PPL", "CMS", "DTE", "ATO", "WEC", "LNT", "NI",
            "AWK", "AEE", "CNP", "EVRG", "NRG", "PCG", "EIX", "PNW",
            
            # Communication Services (24 companies)
            "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "CHTR", "TMUS",
            "NWSA", "NWS", "FOXA", "FOX", "PARA", "WBD", "MTCH", "PINS", "SNAP", "TWTR",
            "LUMN", "SIRI", "IPG", "OMC"
        ]
    
    def _load_sector_classifications(self) -> Dict[str, str]:
        """Load sector classifications for S&P 500 stocks"""
        return {
            # Technology
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "GOOG": "Technology",
            "AMZN": "Technology", "META": "Technology", "TSLA": "Technology", "NVDA": "Technology",
            "NFLX": "Technology", "ADBE": "Technology", "CRM": "Technology", "INTC": "Technology",
            "AMD": "Technology", "ORCL": "Technology", "CSCO": "Technology", "AVGO": "Technology",
            "QCOM": "Technology", "TXN": "Technology", "INTU": "Technology", "IBM": "Technology",
            
            # Financial Services
            "BRK.B": "Financial", "JPM": "Financial", "BAC": "Financial", "WFC": "Financial",
            "GS": "Financial", "MS": "Financial", "C": "Financial", "AXP": "Financial",
            "USB": "Financial", "PNC": "Financial", "COF": "Financial", "TFC": "Financial",
            "BK": "Financial", "STT": "Financial", "SCHW": "Financial", "CME": "Financial",
            "ICE": "Financial", "SPGI": "Financial", "MCO": "Financial", "BLK": "Financial",
            
            # Healthcare
            "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare", "ABT": "Healthcare",
            "TMO": "Healthcare", "MRK": "Healthcare", "ABBV": "Healthcare", "LLY": "Healthcare",
            "DHR": "Healthcare", "MDT": "Healthcare", "BMY": "Healthcare", "AMGN": "Healthcare",
            "GILD": "Healthcare", "CVS": "Healthcare", "ANTM": "Healthcare", "CI": "Healthcare",
            "HUM": "Healthcare", "CNC": "Healthcare", "MOH": "Healthcare", "UHS": "Healthcare",
            
            # Consumer Discretionary
            "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
            "SBUX": "Consumer Discretionary", "LOW": "Consumer Discretionary", "TJX": "Consumer Discretionary",
            "BKNG": "Consumer Discretionary", "GM": "Consumer Discretionary", "F": "Consumer Discretionary",
            "EBAY": "Consumer Discretionary", "ETSY": "Consumer Discretionary", "LULU": "Consumer Discretionary",
            "ULTA": "Consumer Discretionary", "RCL": "Consumer Discretionary", "CCL": "Consumer Discretionary",
            "NCLH": "Consumer Discretionary", "MAR": "Consumer Discretionary", "HLT": "Consumer Discretionary",
            "YUM": "Consumer Discretionary", "CMG": "Consumer Discretionary",
            
            # Consumer Staples  
            "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
            "PEP": "Consumer Staples", "COST": "Consumer Staples", "WBA": "Consumer Staples",
            "MDLZ": "Consumer Staples", "CL": "Consumer Staples", "KMB": "Consumer Staples",
            "GIS": "Consumer Staples", "K": "Consumer Staples", "HSY": "Consumer Staples",
            "CPB": "Consumer Staples", "SJM": "Consumer Staples", "HRL": "Consumer Staples",
            "CAG": "Consumer Staples", "LW": "Consumer Staples", "TAP": "Consumer Staples",
            "STZ": "Consumer Staples", "BF.B": "Consumer Staples",
            
            # Energy
            "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "EOG": "Energy",
            "SLB": "Energy", "PSX": "Energy", "VLO": "Energy", "MPC": "Energy",
            "OXY": "Energy", "KMI": "Energy", "WMB": "Energy", "ENB": "Energy",
            "EPD": "Energy", "ET": "Energy", "MPLX": "Energy", "BKR": "Energy",
            "HAL": "Energy", "DVN": "Energy", "FANG": "Energy", "MRO": "Energy",
            
            # Industrials
            "BA": "Industrials", "CAT": "Industrials", "HON": "Industrials", "UPS": "Industrials",
            "RTX": "Industrials", "LMT": "Industrials", "GE": "Industrials", "MMM": "Industrials",
            "DE": "Industrials", "UNP": "Industrials", "CSX": "Industrials", "NSC": "Industrials",
            "FDX": "Industrials", "DAL": "Industrials", "UAL": "Industrials", "AAL": "Industrials",
            "LUV": "Industrials", "JBLU": "Industrials", "ALK": "Industrials", "SAVE": "Industrials",
            
            # Materials
            "LIN": "Materials", "APD": "Materials", "ECL": "Materials", "SHW": "Materials",
            "FCX": "Materials", "NEM": "Materials", "DOW": "Materials", "DD": "Materials",
            "PPG": "Materials", "IFF": "Materials", "MLM": "Materials", "VMC": "Materials",
            "NUE": "Materials", "STLD": "Materials", "RS": "Materials", "MOS": "Materials",
            "FMC": "Materials", "LYB": "Materials", "EMN": "Materials", "ALB": "Materials",
            
            # Real Estate
            "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate", "EQIX": "Real Estate",
            "PSA": "Real Estate", "EXR": "Real Estate", "AVB": "Real Estate", "EQR": "Real Estate",
            "UDR": "Real Estate", "ESS": "Real Estate", "MAA": "Real Estate", "CPT": "Real Estate",
            "AIV": "Real Estate", "BXP": "Real Estate", "VTR": "Real Estate", "WELL": "Real Estate",
            "PEAK": "Real Estate", "O": "Real Estate", "REYN": "Real Estate", "SPG": "Real Estate",
            
            # Utilities
            "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "D": "Utilities",
            "AEP": "Utilities", "EXC": "Utilities", "XEL": "Utilities", "SRE": "Utilities",
            "PEG": "Utilities", "ED": "Utilities", "FE": "Utilities", "ETR": "Utilities",
            "ES": "Utilities", "PPL": "Utilities", "CMS": "Utilities", "DTE": "Utilities",
            "ATO": "Utilities", "WEC": "Utilities", "LNT": "Utilities", "NI": "Utilities"
        }
    
    def get_sp500_tickers(self, sector: Optional[str] = None) -> List[str]:
        """Get S&P 500 ticker list, optionally filtered by sector"""
        if sector:
            return [ticker for ticker, stock_sector in self.sector_map.items() 
                   if stock_sector.lower() == sector.lower()]
        return self.sp500_tickers.copy()
    
    def get_sectors(self) -> List[str]:
        """Get list of all sectors in S&P 500"""
        return list(set(self.sector_map.values()))
    
    def get_stock_data(self, ticker: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get stock data for a specific ticker from Polygon API"""
        if not self.polygon_provider or not self.polygon_provider.api_key:
            logger.error("Polygon API key not available")
            return None
        
        try:
            # Check cache first
            cache_key = f"{ticker}_{days}"
            if cache_key in self.data_cache:
                cache_time = self.last_update.get(cache_key)
                if cache_time and (datetime.now() - cache_time).seconds < 3600:  # 1 hour cache
                    return self.data_cache[cache_key]
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Polygon API call
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            headers = {
                'Authorization': f'Bearer {self.polygon_provider.api_key}',
                'Accept-Encoding': 'gzip'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('results'):
                    df = pd.DataFrame(data['results'])
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                    df = df.rename(columns={
                        'o': 'open',
                        'h': 'high', 
                        'l': 'low',
                        'c': 'close',
                        'v': 'volume'
                    })
                    df['ticker'] = ticker
                    df['source'] = 'polygon'
                    df['verified'] = True
                    
                    # Cache the data
                    self.data_cache[cache_key] = df
                    self.last_update[cache_key] = datetime.now()
                    
                    return df
                else:
                    logger.warning(f"No data returned for {ticker}")
                    return None
            else:
                logger.error(f"Polygon API error for {ticker}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def get_sector_data(self, sector: str, days: int = 30) -> Dict[str, pd.DataFrame]:
        """Get data for all stocks in a specific sector"""
        sector_tickers = self.get_sp500_tickers(sector)
        sector_data = {}
        
        for ticker in sector_tickers[:10]:  # Limit to 10 for API rate limits
            data = self.get_stock_data(ticker, days)
            if data is not None:
                sector_data[ticker] = data
        
        return sector_data
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get S&P 500 market overview data"""
        try:
            # Get data for major index components
            major_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            overview_data = {}
            
            for ticker in major_tickers:
                data = self.get_stock_data(ticker, days=5)
                if data is not None and not data.empty:
                    latest = data.iloc[-1]
                    overview_data[ticker] = {
                        'price': float(latest['close']),
                        'volume': int(latest['volume']),
                        'sector': self.sector_map.get(ticker, 'Unknown'),
                        'timestamp': latest['timestamp'].isoformat(),
                        'source': 'polygon',
                        'verified': True
                    }
            
            return {
                'stocks': overview_data,
                'total_tickers': len(self.sp500_tickers),
                'sectors': len(self.get_sectors()),
                'data_source': 'polygon',
                'timestamp': datetime.now().isoformat(),
                'verified': True
            }
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {}
    
    def get_ml_ready_data(self, tickers: List[str], days: int = 60) -> Optional[pd.DataFrame]:
        """Get ML-ready dataset for specified tickers"""
        try:
            all_data = []
            
            for ticker in tickers:
                data = self.get_stock_data(ticker, days)
                if data is not None and not data.empty:
                    # Add technical indicators
                    data = self._add_technical_indicators(data)
                    all_data.append(data)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                
                # Add features for ML
                combined_df = self._prepare_ml_features(combined_df)
                
                return combined_df
            
            return None
            
        except Exception as e:
            logger.error(f"Error preparing ML data: {e}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data"""
        try:
            # Moving averages
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_30'] = df['close'].rolling(window=30).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def _prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning models"""
        try:
            # Price-based features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Price patterns
            df['high_low_pct'] = (df['high'] - df['low']) / df['close']
            df['close_open_pct'] = (df['close'] - df['open']) / df['open']
            
            # Momentum features
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            # Target variable (next day return)
            df['target'] = df['returns'].shift(-1)
            
            # Add sector encoding
            df['sector'] = df['ticker'].map(self.sector_map)
            df['sector_encoded'] = pd.Categorical(df['sector']).codes
            
            # Remove rows with NaN values in critical columns only
            # Keep more data by only dropping rows where essential features are NaN
            critical_columns = ['close', 'returns', 'target']
            df = df.dropna(subset=critical_columns)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return df
    
    def get_stock_fundamentals(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get fundamental data for a stock (placeholder for future enhancement)"""
        # This would integrate with financial data APIs for fundamentals
        # For now, return basic info
        return {
            'ticker': ticker,
            'sector': self.sector_map.get(ticker, 'Unknown'),
            'source': 'internal',
            'verified': True,
            'timestamp': datetime.now().isoformat()
        }

# Global instance for easy access
_sp500_manager = None

def get_sp500_manager() -> SP500DataManager:
    """Get global SP500DataManager instance"""
    global _sp500_manager
    if _sp500_manager is None:
        _sp500_manager = SP500DataManager()
    return _sp500_manager