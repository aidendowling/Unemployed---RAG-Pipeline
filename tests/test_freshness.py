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


def test_query_intent_vintage_extraction():
    """Test that queries are parsed for year intent."""
    cases = [
        ("unemployment in 2020", (2020, 2020)),
        ("data from 2015 to 2020", (2015, 2020)),
        ("before 2018", (1900, 2018)),
        ("no years here", (None, None)),
    ]
    for query, expected in cases:
        result = freshness._extract_years_from_query(query)
        assert result == expected, f"Query '{query}' should extract {expected}, got {result}"


def test_query_intent_vintage_score():
    """Test that documents are scored based on vintage alignment with query."""
    # Perfect match
    assert freshness.query_intent_vintage_score(
        {"vintage_year": 2020}, "unemployment in 2020"
    ) == 1.0
    
    # Within range
    assert freshness.query_intent_vintage_score(
        {"vintage_year": 2017}, "unemployment from 2015 to 2020"
    ) == 1.0
    
    # Out of range gets decay
    score = freshness.query_intent_vintage_score(
        {"vintage_year": 2010}, "unemployment in 2020"
    )
    assert 0.5 < score < 1.0, f"Out-of-range document should get decay, got {score}"
    
    # No vintage defaults to full relevance
    assert freshness.query_intent_vintage_score(
        {}, "unemployment in 2020"
    ) == 1.0

