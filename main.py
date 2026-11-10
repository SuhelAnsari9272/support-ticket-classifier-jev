
import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from graph import run_pipeline


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# Request / Response models
# ---------------------------------------------------------------------------
class ClassifyRequest(BaseModel):
    ticket_text: str = Field(..., min_length=5, max_length=4000)
    channel: str = Field(default="web_form", pattern="^(web_form|email)$")


class ClassifyResponse(BaseModel):
    issue_category: str
    issue_confidence: float = Field(ge=0.0, le=1.0)
    issue_entropy: float = Field(ge=0.0)
    issue_normalized_entropy: float = Field(ge=0.0, le=1.0)
    assigned_team: str
    team_confidence: float = Field(ge=0.0, le=1.0)
    team_entropy: float = Field(ge=0.0)
    team_normalized_entropy: float = Field(ge=0.0, le=1.0)
    priority: str
    priority_score: float
    priority_confidence: float = Field(ge=0.0, le=1.0)
    priority_approximate: str | None = None
    user_sentiment: str
    sentiment_score: float
    sentiment_confidence: float = Field(ge=0.0, le=1.0)
    sentiment_approximate: str | None = None
    requires_human_review: bool
    requires_hr_score: float = Field(ge=0.0, le=1.0)
    pii_detected: bool
    injection_blocked: bool



jev_model = "typesafe/jev-1.14-20261014"

# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------
def print_banner():

    pii_enabled = True   # always on in this build
    cost_tracking = os.getenv("LOG_COSTS", "true").lower() == "true"

    banner = f"""
╔══════════════════════════════════════════════════════════╗
║       AI-Powered Support Ticket Classifier               ║
╠══════════════════════════════════════════════════════════╣
║  Model          : {jev_model:<38} ║
║  PII Redaction  : {'enabled' if pii_enabled else 'disabled':<38} ║
║  Cost Tracking  : {'enabled' if cost_tracking else 'disabled':<38} ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print_banner()
    yield
    # summary = session_tracker.summary
    # logger.info("Session ended. Total cost: $%.6f over %d calls", summary["total_cost_usd"], summary["calls"])

app = FastAPI(
    title="Support Ticket Classifier",
    description="LangGraph + FastAPI ticket classification pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
# @traceable(run_type="chain", name="pipeline", hide_inputs=True, hide_outputs=True)
async def classify(request: ClassifyRequest):
    try:
        state = run_pipeline(request.ticket_text, request.channel)
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    classification = state.get("classification")
    if classification is None:
        raise HTTPException(status_code=422, detail=state.get("error", "Classification failed"))

    return ClassifyResponse(
        issue_category=classification.issue_output.choice.value,
        issue_confidence=classification.issue_output.confidence,
        issue_entropy=classification.issue_output.entropy,
        issue_normalized_entropy=classification.issue_output.normalized_entropy,
        assigned_team=classification.team_output.choice.value,
        team_confidence=classification.team_output.confidence,
        team_entropy=classification.team_output.entropy,
        team_normalized_entropy=classification.team_output.normalized_entropy,
        priority=classification.priority_output.key.value,
        priority_score=classification.priority_output.score,
        priority_confidence=classification.priority_output.confidence,
        priority_approximate=(
            classification.priority_output.approximate.value
            if classification.priority_output.approximate is not None
            else None
        ),
        user_sentiment=classification.sentiment_output.key.value,
        sentiment_score=classification.sentiment_output.score,
        sentiment_confidence=classification.sentiment_output.confidence,
        sentiment_approximate=(
            classification.sentiment_output.approximate.value
            if classification.sentiment_output.approximate is not None
            else None
        ),
        requires_human_review=classification.hr_review_output.requires_human_review,
        requires_hr_score=classification.hr_review_output.requires_hr_score,
        pii_detected=state.get("pii_detected", False),
        injection_blocked=state.get("injection_blocked", False),
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 