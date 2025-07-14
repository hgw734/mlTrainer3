"""
Test Data Connections - Polygon and FRED
=========================================

Test script to verify Polygon and FRED API connections are working.
"""

import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_polygon_connection():
    """Test Polygon API connection"""
    print("\n=== Testing Polygon Connection ===")
    try:
        from polygon_connector import get_polygon_connector

        connector = get_polygon_connector()
        print("✓ Polygon connector initialized")

        # Test 1: Get quote for AAPL
        print("\nTest 1: Getting quote for AAPL...")
        quote = connector.get_quote("AAPL")
        if quote:
            print(f"✓ AAPL Quote:")
            print(f"  Price: ${quote.price:.2f}")
            print(f"  Change: ${quote.change:.2f} ({quote.change_percent:+.2f}%)")
            print(f"  Volume: {quote.volume:,}")
            print(f"  High/Low: ${quote.high:.2f}/${quote.low:.2f}")
            else:
                print("✗ Failed to get AAPL quote")

                # Test 2: Get historical data
                print("\nTest 2: Getting historical data for MSFT...")
                hist = connector.get_historical_data("MSFT", days=30)
                if hist:
                    print(f"✓ MSFT Historical Data:")
                    print(f"  Records: {len(hist.data)}")
                    print(f"  Date range: {hist.start_date.strftime('%Y-%m-%d')} to {hist.end_date.strftime('%Y-%m-%d')}")
                    print(f"  Latest close: ${hist.data['close'].iloc[-1]:.2f}")
                    print(f"  30-day avg volume: {hist.data['volume'].mean():,.0f}")
                    else:
                        print("✗ Failed to get MSFT historical data")

                        # Test 3: Check API quality metrics
                        print("\nTest 3: API Quality Metrics...")
                        metrics = connector.get_quality_metrics()
                        print(f"✓ Quality Metrics:")
                        print(f"  Success rate: {metrics['success_rate']:.1%}")
                        print(f"  Dropout rate: {metrics['dropout_rate']:.1%}")
                        print(f"  Avg response time: {metrics['avg_response_time']:.2f}s")
                        print(f"  Total requests: {metrics['total_requests']}")
                        print(f"  Circuit open: {metrics['circuit_open']}")
                        print(f"  Rate limit active: {metrics['rate_limit_active']}")

                        print("\n✅ Polygon connection tests completed!")
                        return True

                        except Exception as e:
                            print(f"\n❌ Polygon connection test failed: {e}")
                            logger.error(f"Polygon test error: {e}", exc_info=True)
                            return False


                            def test_fred_connection():
                                """Test FRED API connection"""
                                print("\n=== Testing FRED Connection ===")
                                try:
                                    from fred_connector import get_fred_connector

                                    connector = get_fred_connector()
                                    print("✓ FRED connector initialized")

                                    # Test 1: Get GDP data
                                    print("\nTest 1: Getting GDP data...")
                                    gdp = connector.get_series_data("GDP", start_date="2023-01-01")
                                    if gdp:
                                        print(f"✓ GDP Data:")
                                        print(f"  Series: {gdp.name}")
                                        print(f"  Units: {gdp.units}")
                                        print(f"  Frequency: {gdp.frequency}")
                                        print(f"  Records: {len(gdp.data)}")
                                        print(f"  Latest value: ${gdp.data['value'].iloc[-1]:,.2f} billion")
                                        else:
                                            print("✗ Failed to get GDP data")

                                            # Test 2: Get unemployment rate
                                            print("\nTest 2: Getting unemployment rate...")
                                            unrate = connector.get_series_data("UNRATE", start_date="2024-01-01")
                                            if unrate:
                                                print(f"✓ Unemployment Rate:")
                                                print(f"  Series: {unrate.name}")
                                                print(f"  Latest rate: {unrate.data['value'].iloc[-1]:.1f}%")
                                                print(f"  3-month avg: {unrate.data['value'].tail(3).mean():.1f}%")
                                                else:
                                                    print("✗ Failed to get unemployment data")

                                                    # Test 3: Search for inflation series
                                                    print("\nTest 3: Searching for inflation series...")
                                                    results = connector.search_series("inflation", limit=5)
                                                    if results:
                                                        print(f"✓ Found {len(results)} inflation-related series:")
                                                        for i, series in enumerate(results[:3], 1):
                                                            print(f"  {i}. {series['id']}: {series['title']}")
                                                            print(f"     Frequency: {series['frequency']}, Units: {series['units']}")
                                                            else:
                                                                print("✗ Failed to search series")

                                                                # Test 4: Get popular series list
                                                                print("\nTest 4: Popular economic indicators...")
                                                                popular = connector.get_popular_series()
                                                                print(f"✓ Available popular series: {len(popular)}")
                                                                for code, name in list(popular.items())[:5]:
                                                                    print(f"  {code}: {name}")

                                                                    print("\n✅ FRED connection tests completed!")
                                                                    return True

                                                                    except Exception as e:
                                                                        print(f"\n❌ FRED connection test failed: {e}")
                                                                        logger.error(f"FRED test error: {e}", exc_info=True)
                                                                        return False


                                                                        def main():
                                                                            """Run all connection tests"""
                                                                            print(("=" * 60))
                                                                            print("DATA CONNECTION TESTS")
                                                                            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                                                            print(("=" * 60))

                                                                            polygon_ok = test_polygon_connection()
                                                                            fred_ok = test_fred_connection()

                                                                            print(("\n" + "=" * 60))
                                                                            print("SUMMARY")
                                                                            print(("=" * 60))
                                                                            print(f"Polygon API: {'✅ PASS' if polygon_ok else '❌ FAIL'}")
                                                                            print(f"FRED API: {'✅ PASS' if fred_ok else '❌ FAIL'}")

                                                                            if polygon_ok and fred_ok:
                                                                                print("\n🎉 All data connections working!")
                                                                                else:
                                                                                    print("\n⚠️  Some connections failed. Check logs for details.")

                                                                                    return polygon_ok and fred_ok


                                                                                    if __name__ == "__main__":
                                                                                        success = main()
                                                                                        exit(0 if success else 1)
