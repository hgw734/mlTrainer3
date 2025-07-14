"""
Polygon API Rate Limiter and Data Validation
============================================

Purpose: Ensures compliance with Polygon's 100 requests/second limit while maintaining
data quality through dropout monitoring and automatic retry mechanisms.

Features:
    - Rate limiting to stay well below 100 RPS (default: 50 RPS max)
    - Exponential backoff retry system
    - Data dropout rate monitoring and validation
    - Circuit breaker pattern for API protection
    - Comprehensive error handling and logging
    """

import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
import requests
from threading import Lock
import statistics

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Track data quality metrics for validation"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    dropout_rate: float = 0.0
    avg_response_time: float = 0.0
    last_reset: datetime = None

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def failure_rate(self) -> float:
        return 1.0 - self.success_rate


class PolygonRateLimiter:
    """Advanced rate limiter with data quality monitoring for Polygon API"""

    def __init__(
        self, max_requests_per_second: int = 50, dropout_threshold: float = 0.15, circuit_breaker_threshold: int = 5
    ):
        # Rate limiting configuration
        self.max_rps = max_requests_per_second  # Stay well below 100 RPS limit
        self.request_window = 1.0  # 1 second window
        self.request_times = deque()
        self.lock = Lock()

        # Data quality monitoring
        self.dropout_threshold = dropout_threshold  # 15% max dropout rate
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.consecutive_failures = 0
        self.circuit_open = False
        self.circuit_open_time = None
        self.circuit_cooldown = 60  # 60 seconds

        # Metrics tracking
        self.metrics = DataQualityMetrics()
        self.response_times = deque(maxlen=100)  # Keep last 100 response times

        # Retry configuration
        self.max_retries = 3
        self.base_delay = 1.0  # Base delay for exponential backoff
        self.max_delay = 30.0  # Maximum delay between retries

        logger.info(
            f"PolygonRateLimiter initialized: {max_requests_per_second} RPS, {dropout_threshold*100}% dropout threshold"
        )

    def _cleanup_old_requests(self):
        """Remove request times older than the window"""
        current_time = time.time()
        while self.request_times and current_time - self.request_times[0] > self.request_window:
            self.request_times.popleft()

    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits"""
        with self.lock:
            self._cleanup_old_requests()

            if len(self.request_times) >= self.max_rps:
                # Calculate wait time to next available slot
                wait_time = self.request_window - (time.time() - self.request_times[0])
                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                    time.sleep(wait_time)
                    self._cleanup_old_requests()

                # Record this request
                self.request_times.append(time.time())

    def _check_circuit_breaker(self):
        """Check if circuit breaker should open or close"""
        if self.circuit_open:
            if self.circuit_open_time and time.time() - self.circuit_open_time > self.circuit_cooldown:
                self.circuit_open = False
                self.consecutive_failures = 0
                logger.info("Circuit breaker closed - resuming API calls")
            else:
                raise Exception("Circuit breaker OPEN - API temporarily disabled due to consecutive failures")

    def _record_success(self, response_time: float):
        """Record successful API call"""
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        self.consecutive_failures = 0
        self.response_times.append(response_time)

        # Update average response time
        if self.response_times:
            self.metrics.avg_response_time = statistics.mean(self.response_times)

        # Update dropout rate
        self.metrics.dropout_rate = self.metrics.failure_rate

        logger.debug(f"API success recorded: {response_time:.3f}s response time")

    def _record_failure(self, error: str):
        """Record failed API call"""
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        self.consecutive_failures += 1

        # Update dropout rate
        self.metrics.dropout_rate = self.metrics.failure_rate

        # Check circuit breaker
        if self.consecutive_failures >= self.circuit_breaker_threshold:
            self.circuit_open = True
            self.circuit_open_time = time.time()
            logger.error(f"Circuit breaker OPENED after {self.consecutive_failures} consecutive failures")

        logger.warning(f"API failure recorded: {error}")

    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay"""
        delay = min(self.base_delay * (2**attempt), self.max_delay)
        # Add jitter to prevent thundering herd
        jitter = delay * 0.1 * (0.5 - time.time() % 1)
        return delay + jitter

    def validate_data_quality(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate current data quality metrics"""
        is_valid = True
        issues = []

        # Check dropout rate
        if self.metrics.dropout_rate > self.dropout_threshold:
            is_valid = False
            issues.append(
                f"Dropout rate {self.metrics.dropout_rate:.1%} exceeds threshold {self.dropout_threshold:.1%}"
            )

        # Check circuit breaker
        if self.circuit_open:
            is_valid = False
            issues.append("Circuit breaker is OPEN - API temporarily disabled")

        # Check response times
        if self.metrics.avg_response_time > 10.0:  # 10 second threshold
            issues.append(f"High response times: {self.metrics.avg_response_time:.1f}s average")

        quality_report = {
            "is_valid": is_valid,
            "issues": issues,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "success_rate": self.metrics.success_rate,
                "dropout_rate": self.metrics.dropout_rate,
                "avg_response_time": self.metrics.avg_response_time,
                "circuit_open": self.circuit_open,
                "consecutive_failures": self.consecutive_failures,
            },
        }

        return is_valid, quality_report

    def make_request(
        self, url: str, headers: Dict[str, str] = None, params: Dict[str, Any] = None, timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Make rate-limited API request with automatic retries and quality monitoring

        Args:
            url: API endpoint URL
            headers: HTTP headers
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            API response data or error information
            """
        # Check circuit breaker
        self._check_circuit_breaker()

        # Validate data quality before making request
        is_valid, quality_report = self.validate_data_quality()
        if not is_valid and self.circuit_open:
            return {"success": False, "error": "Data quality validation failed", "quality_report": quality_report}

        for attempt in range(self.max_retries + 1):
            try:
                # Rate limiting
                self._wait_for_rate_limit()

                # Make API request
                start_time = time.time()
                response = requests.get(url, headers=headers, params=params, timeout=timeout)
                response_time = time.time() - start_time

                if response.status_code == 200:
                    self._record_success(response_time)
                    return {
                        "success": True,
                        "data": response.json(),
                        "response_time": response_time,
                        "quality_metrics": self.get_quality_summary(),
                    }
                elif response.status_code == 429:  # Rate limited
                    logger.warning(f"Rate limited by Polygon API, attempt {attempt + 1}")
                    if attempt < self.max_retries:
                        delay = self._exponential_backoff(attempt + 1)
                        logger.info(f"Retrying after {delay:.1f} seconds")
                        time.sleep(delay)
                        continue
                    else:
                        error_msg = f"HTTP {response.status_code}: {response.text}"
                        self._record_failure(error_msg)
                        if attempt < self.max_retries:
                            delay = self._exponential_backoff(attempt)
                            logger.info(f"Retrying after {delay:.1f} seconds")
                            time.sleep(delay)
                            continue
                        else:
                            return {"success": False, "error": error_msg, "quality_metrics": self.get_quality_summary()}
            except requests.exceptions.Timeout:
                error_msg = f"Request timeout after {timeout} seconds"
                self._record_failure(error_msg)
                if attempt < self.max_retries:
                    delay = self._exponential_backoff(attempt)
                    logger.info(f"Timeout - retrying after {delay:.1f} seconds")
                    time.sleep(delay)
                    continue
                else:
                    return {"success": False, "error": error_msg, "quality_metrics": self.get_quality_summary()}
            except Exception as e:
                error_msg = f"Request failed: {str(e)}"
                self._record_failure(error_msg)
                if attempt < self.max_retries:
                    delay = self._exponential_backoff(attempt)
                    logger.info(f"Error - retrying after {delay:.1f} seconds")
                    time.sleep(delay)
                    continue
                else:
                    return {"success": False, "error": error_msg, "quality_metrics": self.get_quality_summary()}

        # All retries exhausted
        return {
            "success": False,
            "error": f"All {self.max_retries} retries exhausted",
            "quality_metrics": self.get_quality_summary(),
        }

    def get_quality_summary(self) -> Dict[str, Any]:
        """Get current data quality summary"""
        return {
            "success_rate": self.metrics.success_rate,
            "dropout_rate": self.metrics.dropout_rate,
            "avg_response_time": self.metrics.avg_response_time,
            "total_requests": self.metrics.total_requests,
            "circuit_open": self.circuit_open,
            "rate_limit_active": len(self.request_times) >= self.max_rps * 0.8,  # 80% of limit
        }

    def reset_metrics(self):
        """Reset all metrics (useful for testing or daily resets)"""
        self.metrics = DataQualityMetrics()
        self.response_times.clear()
        self.consecutive_failures = 0
        self.circuit_open = False
        self.circuit_open_time = None
        logger.info("Polygon rate limiter metrics reset")


# Global rate limiter instance
_polygon_limiter = None


def get_polygon_rate_limiter() -> PolygonRateLimiter:
    """Get global Polygon rate limiter instance"""
    global _polygon_limiter
    if _polygon_limiter is None:
        _polygon_limiter = PolygonRateLimiter()
    return _polygon_limiter
