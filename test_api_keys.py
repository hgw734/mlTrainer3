"""
Test API Keys - Simple verification without pandas
==================================================
"""

import json
import requests
from config.api_config import POLYGON_API_KEY, FRED_API_KEY


def test_polygon_key():
    """Test Polygon API key"""
    print("\n=== Testing Polygon API Key ===")
    try:
        url = "https://api.polygon.io/v2/aggs/ticker/AAPL/prev"
        params = {"apikey": POLYGON_API_KEY}

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if "results" in data:
                result = data["results"][0]
                print(f"✓ Polygon API key valid!")
                print(f"  AAPL previous close: ${result['c']:.2f}")
                print(f"  Volume: {result['v']:,}")
                return True
                else:
                    print(f"✗ Unexpected response: {data}")
                    return False
                    else:
                        print(
                            f"✗ API error: {response.status_code} - {response.text}")
                        return False

                        except Exception as e:
                            print(f"✗ Error: {e}")
                            return False

                            def test_fred_key():
                                """Test FRED API key"""
                                print("\n=== Testing FRED API Key ===")
                                try:
                                    url = "https://api.stlouisfed.org/fred/series"
                                    params = {
                                        "series_id": "GDP",
                                        "api_key": FRED_API_KEY,
                                        "file_type": "json"}

                                    response = requests.get(
                                        url, params=params, timeout=10)

                                    if response.status_code == 200:
                                        data = response.json()
                                        if "seriess" in data and data["seriess"]:
                                            series = data["seriess"][0]
                                            print(f"✓ FRED API key valid!")
                                            print(
                                                f"  Series: {series['title']}")
                                            print(
                                                f"  Units: {series['units']}")
                                            print(
                                                f"  Last updated: {series['last_updated']}")
                                            return True
                                            else:
                                                print(
                                                    f"✗ Unexpected response: {data}")
                                                return False
                                                else:
                                                    print(
                                                        f"✗ API error: {response.status_code} - {response.text}")
                                                    return False

                                                    except Exception as e:
                                                        print(f"✗ Error: {e}")
                                                        return False

                                                        def test_rate_limiter():
                                                            """Test rate limiter functionality"""
                                                            print(
                                                                "\n=== Testing Rate Limiter ===")
                                                            try:
                                                                from polygon_rate_limiter import get_polygon_rate_limiter

                                                                limiter = get_polygon_rate_limiter()
                                                                print(
                                                                    "✓ Rate limiter initialized")

                                                                # Make a test
                                                                # request
                                                                result = limiter.make_request(
                                                                    "https://api.polygon.io/v2/aggs/ticker/MSFT/prev",
                                                                    params={
                                                                        "apikey": POLYGON_API_KEY})

                                                                if result["success"]:
                                                                    data = result["data"]
                                                                    print(
                                                                        "✓ Rate-limited request successful")
                                                                    print(
                                                                        f"  Response time: {result['response_time']:.2f}s")
                                                                    print(
                                                                        f"  MSFT previous close: ${data['results'][0]['c']:.2f}")

                                                                    # Show
                                                                    # quality
                                                                    # metrics
                                                                    metrics = result["quality_metrics"]
                                                                    print(
                                                                        f"\nQuality Metrics:")
                                                                    print(
                                                                        f"  Success rate: {metrics['success_rate']:.1%}")
                                                                    print(
                                                                        f"  Dropout rate: {metrics['dropout_rate']:.1%}")
                                                                    print(
                                                                        f"  Circuit open: {metrics['circuit_open']}")
                                                                    return True
                                                                    else:
                                                                        print(
                                                                            f"✗ Request failed: {result['error']}")
                                                                        return False

                                                                        except Exception as e:
                                                                            print(
                                                                                f"✗ Error: {e}")
                                                                            return False

                                                                            def main():
                                                                                """Run all tests"""
                                                                                print(
                                                                                    ("=" * 60))
                                                                                print(
                                                                                    "API KEY VERIFICATION")
                                                                                print(
                                                                                    ("=" * 60))

                                                                                # Display
                                                                                # API
                                                                                # key
                                                                                # info
                                                                                # (masked)
                                                                                print(
                                                                                    f"\nPolygon API Key: {POLYGON_API_KEY[:10]}...{POLYGON_API_KEY[-4:]}")
                                                                                print(
                                                                                    f"FRED API Key: {FRED_API_KEY[:10]}...{FRED_API_KEY[-4:]}")

                                                                                polygon_ok = test_polygon_key()
                                                                                fred_ok = test_fred_key()
                                                                                limiter_ok = test_rate_limiter()

                                                                                print(
                                                                                    ("\n" + "=" * 60))
                                                                                print(
                                                                                    "SUMMARY")
                                                                                print(
                                                                                    ("=" * 60))
                                                                                print(
                                                                                    f"Polygon API: {'✅ PASS' if polygon_ok else '❌ FAIL'}")
                                                                                print(
                                                                                    f"FRED API: {'✅ PASS' if fred_ok else '❌ FAIL'}")
                                                                                print(
                                                                                    f"Rate Limiter: {'✅ PASS' if limiter_ok else '❌ FAIL'}")

                                                                                if polygon_ok and fred_ok and limiter_ok:
                                                                                    print(
                                                                                        "\n🎉 All API connections verified!")
                                                                                    else:
                                                                                        print(
                                                                                            "\n⚠️  Some tests failed.")

                                                                                        if __name__ == "__main__":
                                                                                            main()
