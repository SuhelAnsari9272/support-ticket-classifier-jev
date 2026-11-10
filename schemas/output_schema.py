from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from schemas.schema import IssueCategory, Priority, Sentiment, TeamOwner

T = TypeVar("T")


class ChoiceOutput(BaseModel, Generic[T]):
    choice: T
    confidence: float = Field(ge=0.0, le=1.0)
    entropy: float = Field(ge=0.0)
    normalized_entropy: float = Field(ge=0.0, le=1.0)


class ScoreOutput(BaseModel, Generic[T]):
    score: float
    confidence: float = Field(ge=0.0, le=1.0)
    key: T
    approximate: T | None = None


class HumanReviewOutput(BaseModel):
    requires_human_review: bool
    requires_hr_score: float = Field(ge=0.0, le=1.0)


class TicketClassificationOutput(BaseModel):
    issue_output: ChoiceOutput[IssueCategory]
    team_output: ChoiceOutput[TeamOwner]
    priority_output: ScoreOutput[Priority]
    sentiment_output: ScoreOutput[Sentiment]
    hr_review_output: HumanReviewOutput
