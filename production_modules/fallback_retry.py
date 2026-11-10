

from schemas.output_schema import (TicketClassificationOutput, ChoiceOutput, IssueCategory, TeamOwner,
                                    ScoreOutput, Priority, Sentiment, HumanReviewOutput)

safe_issue = ChoiceOutput(
    choice=IssueCategory.OTHER,
    confidence=0.0,
    entropy=0.0,
    normalized_entropy=0.0
)

safe_team = ChoiceOutput(
    choice=TeamOwner.CUSTOMER_SUPPORT,
    confidence=0.0,
    entropy=0.0,
    normalized_entropy=0.0
)

safe_priority = ScoreOutput(
    score = 0.0,
    confidence=0.0,
    key = Priority.LOW
)

safe_sentiment = ScoreOutput(
    score=0.0,
    confidence=0.0,
    key = Sentiment.NEUTRAL
)

safe_hr = HumanReviewOutput(
    requires_human_review=True,
    requires_hr_score=0.0
)

SAFE_CLASSIFICATION = TicketClassificationOutput(
    issue_output=safe_issue,
    team_output= safe_team,
    priority_output = safe_priority,
    sentiment_output = safe_sentiment,
    hr_review_output = safe_hr,
)