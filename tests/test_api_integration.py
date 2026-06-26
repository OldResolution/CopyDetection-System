from src.api import routes
from src.main import create_app


class FakeDetector:
    def get_stats(self):
        return {
            "documents": 1,
            "chunks_faiss": 2,
            "total_words": 100,
        }

    def analyze_text(self, submission_text):
        return {
            "feature_names": ["Avg Word Len"],
            "submission_features": [4.2],
            "top_source_work": "No significant source-work match found",
            "combined_score": 0.0,
            "risk_level": "no actionable similarity detected",
            "legal_risk_code": "NO_ACTIONABLE_SIMILARITY",
            "legal_rationale": "The submission does not show enough overlap for a copyright-focused concern.",
            "matched_sources": [],
        }


def test_analyze_and_report_endpoints(monkeypatch):
    monkeypatch.setattr(routes, "detector", FakeDetector())
    app = create_app()
    app.config.update(TESTING=True)
    client = app.test_client()

    fixture_text = (
        "This original fixture text is intentionally longer than fifty characters "
        "so the analyzer accepts it for integration testing."
    )

    analyze_response = client.post("/analyze", json={"submission_text": fixture_text})

    assert analyze_response.status_code == 200
    analysis = analyze_response.get_json()
    assert analysis["risk_level"] == "no actionable similarity detected"
    assert analysis["extracted_text"] == fixture_text

    report_response = client.post(
        "/report",
        json={"text": fixture_text, "analysis": analysis},
    )

    assert report_response.status_code == 200
    report = report_response.get_json()
    assert "summary" in report
    assert "ngram_frequency" in report


def test_analyze_rejects_short_text(monkeypatch):
    monkeypatch.setattr(routes, "detector", FakeDetector())
    app = create_app()
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.post("/analyze", json={"submission_text": "too short"})

    assert response.status_code == 400
    assert "Text too short" in response.get_json()["error"]
