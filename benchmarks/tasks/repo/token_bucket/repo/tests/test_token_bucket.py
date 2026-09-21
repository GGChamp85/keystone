import time

from token_bucket import TokenBucket


def test_consume_within_capacity():
    bucket = TokenBucket(capacity=5, refill_rate=1.0)
    assert bucket.consume(3) is True
    assert bucket.consume(2) is True
    assert bucket.consume(1) is False


def test_tokens_never_exceed_capacity_after_long_idle():
    bucket = TokenBucket(capacity=5, refill_rate=1000.0)
    bucket.tokens = 0
    bucket.last_refill = time.monotonic() - 100
    bucket._refill()
    assert bucket.tokens <= bucket.capacity
