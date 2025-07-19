
import logging
from core.configuration import get_api_key
from data_sources.polygon_api import fetch_polygon_ohlcv
from data_sources.fred_api import fetch_vix
import anthropic
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_anthropic_api():
    """Test Anthropic/Claude API"""
    try:
        api_key = get_api_key("claude")
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say 'API test successful' in exactly 3 words."}]
        )

        result = response.content[0].text.strip()
        logger.info(f"✅ ANTHROPIC API: {result}")
        return True

    except Exception as e:
        logger.error(f"❌ ANTHROPIC API FAILED: {e}")
        return False


def test_polygon_api():
    """Test Polygon API"""
    try:
        # Test with SPY data for last 5 days
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        df = fetch_polygon_ohlcv("SPY", start_date, end_date)

        if not df.empty:
            latest_close = df['close'].iloc[-1]
            logger.info(f"✅ POLYGON API: SPY latest close ${latest_close}")
            return True
        else:
            logger.error("❌ POLYGON API: No data returned")
            return False

    except Exception as e:
        logger.error(f"❌ POLYGON API FAILED: {e}")
        return False


def test_fred_api():
    """Test FRED API"""
    try:
        # Test with VIX data for last 30 days
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        vix_data = fetch_vix(start_date, end_date)

        if not vix_data.empty:
            latest_vix = vix_data.iloc[-1]
            logger.info(f"✅ FRED API: Latest VIX {latest_vix}")
            return True
        else:
            logger.error("❌ FRED API: No VIX data returned")
            return False

    except Exception as e:
        logger.error(f"❌ FRED API FAILED: {e}")
        return False


def main():
    """Test all three APIs"""
    print("🔍 Testing API connections...\n")

    results = {
        "Anthropic/Claude": test_anthropic_api(),
        "Polygon": test_polygon_api(),
        "FRED": test_fred_api()
    }

    print(f"\n📊 API Test Results:")
    print("=" * 40)

    working_count = 0
    for api_name, status in results.items():
        status_emoji = "✅" if status else "❌"
        status_text = "WORKING" if status else "FAILED"
        print(f"{status_emoji} {api_name}: {status_text}")
        if status:
            working_count += 1

    print("=" * 40)
    print(f"🎯 Summary: {working_count}/3 APIs working")

    if working_count == 3:
        print("🚀 All APIs are ready! Your mlTrainer can start.")
    else:
        print("⚠️  Some APIs need attention before starting mlTrainer.")


if __name__ == "__main__":
    main()
