import sqlite3
import pandas as pd
import json
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class ScanDatabase:
    """
    Database manager for storing scan results, historical data,
    and analytical metrics for institutional-grade stock scanning.
    """
    
    def __init__(self, db_path: str = "data/scanner.db"):
        """
        Initialize database connection and create tables
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"ScanDatabase initialized: {db_path}")
    
    def _init_database(self):
        """Initialize all database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Scan results table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS scan_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        scan_date DATE NOT NULL,
                        scan_timestamp DATETIME NOT NULL,
                        total_score REAL,
                        technical_score REAL,
                        fundamental_score REAL,
                        sentiment_score REAL,
                        price REAL,
                        volume INTEGER,
                        market_regime TEXT,
                        timeframe TEXT,
                        momentum_3d REAL,
                        momentum_5d REAL,
                        momentum_10d REAL,
                        momentum_20d REAL,
                        momentum_50d REAL,
                        rsi_14 REAL,
                        rsi_2 REAL,
                        macd_signal REAL,
                        bb_position REAL,
                        adx REAL,
                        volatility REAL,
                        sharpe_ratio REAL,
                        max_drawdown REAL,
                        var_95 REAL,
                        beta REAL,
                        analyst_rating REAL,
                        eps_surprise REAL,
                        revenue_surprise REAL,
                        guidance_direction TEXT,
                        institutional_ownership REAL,
                        reddit_sentiment REAL,
                        twitter_sentiment REAL,
                        news_sentiment REAL,
                        risk_regime TEXT,
                        institutional_grade TEXT,
                        raw_data TEXT  -- JSON string for additional data
                    )
                """)
                
                # Historical prices table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS historical_prices (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date DATE NOT NULL,
                        open_price REAL,
                        high_price REAL,
                        low_price REAL,
                        close_price REAL,
                        volume INTEGER,
                        adj_close REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, date)
                    )
                """)
                
                # Company information table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS company_info (
                        symbol TEXT PRIMARY KEY,
                        name TEXT,
                        sector TEXT,
                        industry TEXT,
                        market_cap REAL,
                        shares_outstanding REAL,
                        website TEXT,
                        description TEXT,
                        employees INTEGER,
                        exchange TEXT,
                        currency TEXT,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Earnings data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS earnings_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        quarter TEXT NOT NULL,
                        fiscal_year INTEGER,
                        report_date DATE,
                        eps_estimate REAL,
                        eps_actual REAL,
                        eps_surprise REAL,
                        revenue_estimate REAL,
                        revenue_actual REAL,
                        revenue_surprise REAL,
                        guidance_direction TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, quarter, fiscal_year)
                    )
                """)
                
                # News sentiment table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS news_sentiment (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date DATE NOT NULL,
                        title TEXT,
                        source TEXT,
                        url TEXT,
                        sentiment_score REAL,
                        sentiment_label TEXT,
                        confidence REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Social media sentiment table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS social_sentiment (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        platform TEXT NOT NULL,  -- reddit, twitter, etc.
                        date DATE NOT NULL,
                        mentions INTEGER,
                        sentiment_score REAL,
                        engagement_score REAL,
                        trending_status TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Performance tracking table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_tracking (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        entry_date DATE NOT NULL,
                        entry_price REAL,
                        entry_score REAL,
                        exit_date DATE,
                        exit_price REAL,
                        return_pct REAL,
                        holding_period INTEGER,  -- days
                        max_gain REAL,
                        max_loss REAL,
                        outcome TEXT,  -- winner, loser, ongoing
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Scanner settings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS scanner_settings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        setting_name TEXT UNIQUE NOT NULL,
                        setting_value TEXT,
                        setting_type TEXT,
                        description TEXT,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_scan_results_symbol_date ON scan_results(symbol, scan_date)",
                    "CREATE INDEX IF NOT EXISTS idx_scan_results_score ON scan_results(total_score DESC)",
                    "CREATE INDEX IF NOT EXISTS idx_historical_prices_symbol_date ON historical_prices(symbol, date)",
                    "CREATE INDEX IF NOT EXISTS idx_earnings_symbol_date ON earnings_data(symbol, report_date)",
                    "CREATE INDEX IF NOT EXISTS idx_news_symbol_date ON news_sentiment(symbol, date)",
                    "CREATE INDEX IF NOT EXISTS idx_social_symbol_date ON social_sentiment(symbol, date, platform)"
                ]
                
                for index_sql in indexes:
                    cursor.execute(index_sql)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def save_scan_results(self, results_df: pd.DataFrame) -> bool:
        """
        Save scan results to database
        
        Args:
            results_df: DataFrame with scan results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if results_df.empty:
                return True
            
            scan_timestamp = datetime.now()
            scan_date = scan_timestamp.date()
            
            with sqlite3.connect(self.db_path) as conn:
                for _, row in results_df.iterrows():
                    # Prepare raw data as JSON
                    raw_data = row.to_dict()
                    
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO scan_results (
                            symbol, scan_date, scan_timestamp, total_score, technical_score,
                            fundamental_score, sentiment_score, price, volume, market_regime,
                            timeframe, momentum_3d, momentum_5d, momentum_10d, momentum_20d,
                            momentum_50d, rsi_14, rsi_2, macd_signal, bb_position, adx,
                            volatility, sharpe_ratio, max_drawdown, var_95, beta,
                            analyst_rating, eps_surprise, revenue_surprise, guidance_direction,
                            institutional_ownership, reddit_sentiment, twitter_sentiment,
                            news_sentiment, risk_regime, institutional_grade, raw_data
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                                 ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row.get('symbol', ''),
                        scan_date,
                        scan_timestamp,
                        row.get('total_score', 0),
                        row.get('technical_score', 0),
                        row.get('fundamental_score', 0),
                        row.get('sentiment_score', 0),
                        row.get('price', 0),
                        row.get('volume', 0),
                        row.get('market_regime', ''),
                        row.get('timeframe', ''),
                        row.get('momentum_3d', 0),
                        row.get('momentum_5d', 0),
                        row.get('momentum_10d', 0),
                        row.get('momentum_20d', 0),
                        row.get('momentum_50d', 0),
                        row.get('rsi_14', 50),
                        row.get('rsi_2', 50),
                        row.get('macd_signal', 0),
                        row.get('bb_position', 0.5),
                        row.get('adx', 0),
                        row.get('volatility', 0),
                        row.get('sharpe_ratio', 0),
                        row.get('max_drawdown', 0),
                        row.get('var_95', 0),
                        row.get('beta', 1.0),
                        row.get('analyst_rating', 0),
                        row.get('eps_surprise', 0),
                        row.get('revenue_surprise', 0),
                        row.get('guidance_direction', ''),
                        row.get('institutional_ownership', 0),
                        row.get('reddit_sentiment', 0),
                        row.get('twitter_sentiment', 0),
                        row.get('news_sentiment', 0),
                        row.get('risk_regime', ''),
                        row.get('institutional_grade', ''),
                        json.dumps(raw_data)
                    ))
                
                conn.commit()
            
            logger.info(f"Saved {len(results_df)} scan results to database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save scan results: {e}")
            return False
    
    def get_scan_results(self, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        symbols: Optional[List[str]] = None,
                        min_score: Optional[float] = None,
                        limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve scan results from database
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            symbols: List of symbols to filter
            min_score: Minimum score threshold
            limit: Maximum number of results
            
        Returns:
            DataFrame with scan results
        """
        try:
            query = "SELECT * FROM scan_results WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND scan_date >= ?"
                params.append(start_date.date())
            
            if end_date:
                query += " AND scan_date <= ?"
                params.append(end_date.date())
            
            if symbols:
                placeholders = ','.join(['?' for _ in symbols])
                query += f" AND symbol IN ({placeholders})"
                params.extend(symbols)
            
            if min_score:
                query += " AND total_score >= ?"
                params.append(min_score)
            
            query += " ORDER BY scan_timestamp DESC, total_score DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get scan results: {e}")
            return pd.DataFrame()
    
    def save_historical_prices(self, symbol: str, price_data: pd.DataFrame) -> bool:
        """
        Save historical price data
        
        Args:
            symbol: Stock symbol
            price_data: DataFrame with OHLCV data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                for date, row in price_data.iterrows():
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO historical_prices 
                        (symbol, date, open_price, high_price, low_price, close_price, volume, adj_close)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol.upper(),
                        date.date() if hasattr(date, 'date') else date,
                        row.get('open', 0),
                        row.get('high', 0),
                        row.get('low', 0),
                        row.get('close', 0),
                        row.get('volume', 0),
                        row.get('close', 0)  # Use close as adj_close if not available
                    ))
                
                conn.commit()
            
            logger.info(f"Saved {len(price_data)} price records for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save historical prices for {symbol}: {e}")
            return False
    
    def get_historical_prices(self, symbol: str, 
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get historical price data for a symbol
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with historical prices
        """
        try:
            query = "SELECT * FROM historical_prices WHERE symbol = ?"
            params = [symbol.upper()]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date.date())
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date.date())
            
            query += " ORDER BY date"
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical prices for {symbol}: {e}")
            return pd.DataFrame()
    
    def save_company_info(self, symbol: str, company_data: Dict[str, Any]) -> bool:
        """
        Save company information
        
        Args:
            symbol: Stock symbol
            company_data: Dictionary with company information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO company_info 
                    (symbol, name, sector, industry, market_cap, shares_outstanding,
                     website, description, employees, exchange, currency, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol.upper(),
                    company_data.get('name', ''),
                    company_data.get('sector', ''),
                    company_data.get('industry', ''),
                    company_data.get('market_cap', 0),
                    company_data.get('shares_outstanding', 0),
                    company_data.get('website', ''),
                    company_data.get('description', ''),
                    company_data.get('employees', 0),
                    company_data.get('exchange', ''),
                    company_data.get('currency', 'USD'),
                    datetime.now()
                ))
                
                conn.commit()
            
            logger.info(f"Saved company info for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save company info for {symbol}: {e}")
            return False
    
    def get_top_performers(self, days: int = 7, limit: int = 20) -> pd.DataFrame:
        """
        Get top performing stocks from recent scans
        
        Args:
            days: Number of days to look back
            limit: Maximum number of results
            
        Returns:
            DataFrame with top performers
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = """
                SELECT symbol, AVG(total_score) as avg_score, COUNT(*) as scan_count,
                       MAX(total_score) as max_score, MIN(total_score) as min_score,
                       AVG(price) as avg_price, MAX(scan_date) as last_scan
                FROM scan_results 
                WHERE scan_date >= ?
                GROUP BY symbol
                HAVING scan_count >= 2
                ORDER BY avg_score DESC
                LIMIT ?
            """
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=[cutoff_date.date(), limit])
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get top performers: {e}")
            return pd.DataFrame()
    
    def get_performance_analytics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance analytics for the scanner
        
        Args:
            symbol: Optional symbol to filter (None for all symbols)
            
        Returns:
            Dictionary with analytics
        """
        try:
            analytics = {}
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Base query conditions
                where_clause = "WHERE 1=1"
                params = []
                
                if symbol:
                    where_clause += " AND symbol = ?"
                    params.append(symbol.upper())
                
                # Total scans
                cursor.execute(f"SELECT COUNT(*) FROM scan_results {where_clause}", params)
                analytics['total_scans'] = cursor.fetchone()[0]
                
                # Unique symbols scanned
                cursor.execute(f"SELECT COUNT(DISTINCT symbol) FROM scan_results {where_clause}", params)
                analytics['unique_symbols'] = cursor.fetchone()[0]
                
                # Average scores
                cursor.execute(f"""
                    SELECT AVG(total_score), AVG(technical_score), 
                           AVG(fundamental_score), AVG(sentiment_score)
                    FROM scan_results {where_clause}
                """, params)
                
                result = cursor.fetchone()
                analytics['avg_total_score'] = result[0] or 0
                analytics['avg_technical_score'] = result[1] or 0
                analytics['avg_fundamental_score'] = result[2] or 0
                analytics['avg_sentiment_score'] = result[3] or 0
                
                # Score distribution
                cursor.execute(f"""
                    SELECT 
                        SUM(CASE WHEN total_score >= 80 THEN 1 ELSE 0 END) as excellent,
                        SUM(CASE WHEN total_score >= 70 AND total_score < 80 THEN 1 ELSE 0 END) as good,
                        SUM(CASE WHEN total_score >= 60 AND total_score < 70 THEN 1 ELSE 0 END) as fair,
                        SUM(CASE WHEN total_score < 60 THEN 1 ELSE 0 END) as poor
                    FROM scan_results {where_clause}
                """, params)
                
                distribution = cursor.fetchone()
                analytics['score_distribution'] = {
                    'excellent': distribution[0] or 0,
                    'good': distribution[1] or 0,
                    'fair': distribution[2] or 0,
                    'poor': distribution[3] or 0
                }
                
                # Recent activity (last 7 days)
                week_ago = (datetime.now() - timedelta(days=7)).date()
                cursor.execute(f"""
                    SELECT COUNT(*) FROM scan_results 
                    {where_clause} AND scan_date >= ?
                """, params + [week_ago])
                
                analytics['scans_last_week'] = cursor.fetchone()[0]
                
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get performance analytics: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """
        Clean up old data from database
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).date()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old scan results
                cursor.execute("DELETE FROM scan_results WHERE scan_date < ?", (cutoff_date,))
                scan_deleted = cursor.rowcount
                
                # Delete old news sentiment
                cursor.execute("DELETE FROM news_sentiment WHERE date < ?", (cutoff_date,))
                news_deleted = cursor.rowcount
                
                # Delete old social sentiment
                cursor.execute("DELETE FROM social_sentiment WHERE date < ?", (cutoff_date,))
                social_deleted = cursor.rowcount
                
                conn.commit()
                
                total_deleted = scan_deleted + news_deleted + social_deleted
                
            logger.info(f"Cleaned up {total_deleted} old records (keeping {days_to_keep} days)")
            return total_deleted
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0
    
    def export_data(self, table_name: str, filepath: str) -> bool:
        """
        Export table data to CSV
        
        Args:
            table_name: Name of table to export
            filepath: Export file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            
            df.to_csv(filepath, index=False)
            
            logger.info(f"Exported {len(df)} records from {table_name} to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export {table_name}: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dictionary with database statistics
        """
        try:
            stats = {}
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get table sizes
                tables = [
                    'scan_results', 'historical_prices', 'company_info',
                    'earnings_data', 'news_sentiment', 'social_sentiment',
                    'performance_tracking', 'scanner_settings'
                ]
                
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                
                # Database file size
                stats['db_size_bytes'] = os.path.getsize(self.db_path)
                stats['db_size_mb'] = stats['db_size_bytes'] / (1024 * 1024)
                
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
