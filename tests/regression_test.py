"""Deterministic regression checks and an opt-in live-model quality gate.

Run the offline checks with:
    pytest tests/regression_test.py

Run the live quality gate separately with:
    $env:RUN_LIVE_CLASSIFIER_REGRESSION = "1"
    pytest tests/regression_test.py -k live

The live check calls the configured provider for every evaluation ticket and
uses the thresholds below as release guardrails. It is intentionally excluded
from ordinary CI runs.
"""

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

if os.getenv("RUN_LIVE_CLASSIFIER_REGRESSION") != "1":
    os.environ.setdefault("TYPESAFE_API_KEY", "offline-regression-test-key")

import graph
import main
from production_modules.fallback_retry import SAFE_CLASSIFICATION
from production_modules.prompt_injection import InjectionCheckResult
from schemas.output_schema import (
    ChoiceOutput,
    HumanReviewOutput,
    ScoreOutput,
    TicketClassificationOutput,
)
from schemas.schema import IssueCategory, Priority, Sentiment, TeamOwner


def _known_classification() -> TicketClassificationOutput:
    return TicketClassificationOutput(
        issue_output=ChoiceOutput[IssueCategory](
            choice=IssueCategory.ORDER,
            confidence=0.9,
            entropy=0.2,
            normalized_entropy=0.1,
        ),
        team_output=ChoiceOutput[TeamOwner](
            choice=TeamOwner.FULFILLMENT,
            confidence=0.8,
            entropy=0.4,
            normalized_entropy=0.2,
        ),
        priority_output=ScoreOutput[Priority](
            score=1.0,
            confidence=0.85,
            key=Priority.MEDIUM,
            approximate=Priority.LOW,
        ),
        sentiment_output=ScoreOutput[Sentiment](
            score=1.0,
            confidence=0.75,
            key=Sentiment.NEUTRAL,
        ),
        hr_review_output=HumanReviewOutput(
            requires_human_review=False,
            requires_hr_score=0.2,
        ),
    )


def test_classify_api_serializes_known_pipeline_output(monkeypatch):
    classification = _known_classification()
    monkeypatch.setattr(
        main,
        "run_pipeline",
        lambda ticket, channel: {
            "classification": classification,
            "pii_detected": True,
            "injection_blocked": False,
        },
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/classify",
            json={"ticket_text": "My order has not arrived.", "channel": "email"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "issue_category": "order_issue",
        "issue_confidence": 0.9,
        "issue_entropy": 0.2,
        "issue_normalized_entropy": 0.1,
        "assigned_team": "fulfillment_team",
        "team_confidence": 0.8,
        "team_entropy": 0.4,
        "team_normalized_entropy": 0.2,
        "priority": "medium",
        "priority_score": 1.0,
        "priority_confidence": 0.85,
        "priority_approximate": "low",
        "user_sentiment": "neutral",
        "sentiment_score": 1.0,
        "sentiment_confidence": 0.75,
        "sentiment_approximate": None,
        "requires_human_review": False,
        "requires_hr_score": 0.2,
        "pii_detected": True,
        "injection_blocked": False,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"ticket_text": "No", "channel": "web_form"},
        {"ticket_text": "My order has not arrived.", "channel": "chat"},
        {"ticket_text": "x" * 4001, "channel": "web_form"},
    ],
)
def test_classify_api_rejects_invalid_requests(payload):
    with TestClient(main.app) as client:
        response = client.post("/classify", json=payload)

    assert response.status_code == 422


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_CLASSIFIER_REGRESSION") != "1",
    reason="Set RUN_LIVE_CLASSIFIER_REGRESSION=1 to call the real provider.",
)
def test_live_classifier_quality_does_not_regress():
    """Run the labelled evaluation and enforce minimum release guardrails."""
    from evaluation.evaluate_jev_classifier import evaluate, run_predictions
    from production_modules.all_question import get_classification

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "support_ticket_eval_dataset.json"
    with dataset_path.open(encoding="utf-8") as dataset_file:
        gold = json.load(dataset_file)

    predictions = run_predictions(gold, get_classification, repeats=1)
    metrics, _ = evaluate(predictions, gold)

    assert metrics["runs_failed"] == 0
    assert metrics["issue_category"]["accuracy"] >= float(
        os.getenv("REGRESSION_MIN_ISSUE_ACCURACY", "0.75")
    )
    assert metrics["assigned_team"]["accuracy"] >= float(
        os.getenv("REGRESSION_MIN_TEAM_ACCURACY", "0.75")
    )
    assert metrics["priority"]["accuracy"] >= float(
        os.getenv("REGRESSION_MIN_PRIORITY_ACCURACY", "0.60")
    )
    assert metrics["user_sentiment"]["accuracy"] >= float(
        os.getenv("REGRESSION_MIN_SENTIMENT_ACCURACY", "0.65")
    )
    assert metrics["joint"]["critical_predicted_below_high"] == 0
    assert metrics["requires_human_review"]["recall"] >= float(
        os.getenv("REGRESSION_MIN_REVIEW_RECALL", "0.80")
    )
    assert metrics["latency_s"]["p95"] <= float(
        os.getenv("REGRESSION_MAX_P95_LATENCY_S", "5.0")
    )
