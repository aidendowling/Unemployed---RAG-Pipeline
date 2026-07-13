from datetime import datetime, timezone, timedelta
from unemployed_rag_pipeline.obsolescence.freshness import FreshnessChecker


def test_freshness_score_recent():
    fc = FreshnessChecker(days_half_life=10)
    now = datetime.now(timezone.utc).isoformat()
    doc = {"updated_at": now}
    s = fc.score(doc)
    assert s > 0.5


def test_freshness_score_old():
    fc = FreshnessChecker(days_half_life=10)
    old = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    doc = {"updated_at": old}
    s = fc.score(doc)
    assert s < 0.1
from datetime import datetime, timedelta, timezone

from unemployed_rag_pipeline.obsolescence import freshness


def test_freshness_score_and_obsolescence():
    now = datetime.now(tz=timezone.utc)
    recent = (now - timedelta(days=10)).isoformat()
    old = (now - timedelta(days=200)).isoformat()

    meta_recent = {"updated_at": recent}
    meta_old = {"updated_at": old}

    score_recent = freshness.freshness_score(meta_recent, now=now, max_days=365)
    score_old = freshness.freshness_score(meta_old, now=now, max_days=365)

    assert 0.0 < score_recent <= 1.0
    assert 0.0 <= score_old < score_recent

    assert not freshness.is_obsolete(meta_recent, threshold_days=180)
    assert freshness.is_obsolete(meta_old, threshold_days=180)
