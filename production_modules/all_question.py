from typesafe_sdk import TypeSafeClient, Noul, Choice, Score
import httpx2

from settings import JEV_API_KEY, JEV_BASE_URL, JEV_MODEL, VERIFY_SSL

from schemas.output_schema import (
    ChoiceOutput,
    HumanReviewOutput,
    ScoreOutput,
    TicketClassificationOutput,
)
from schemas.schema import IssueCategory, Priority, Sentiment, TeamOwner
from utils.utils import extract_choice_interpretation, extract_score_interpretation

if not JEV_API_KEY:
    raise RuntimeError(
        "Missing API key. Set TYPESAFE_API_KEY to a valid OpenRouter API key "
        "before running the classifier."
    )


jev = TypeSafeClient(api_key=JEV_API_KEY, base_url=JEV_BASE_URL, model=JEV_MODEL,
                        http_client=httpx2.Client(verify=VERIFY_SSL))

def get_classification(ticket: str) -> TicketClassificationOutput:
    questions = {
        "issue": Choice(
                instructions="Which category best describes the customer's issue?",
                criteria={
                    IssueCategory.ORDER.value: "Issues related to placing, modifying, or tracking an order.",
                    IssueCategory.PAYMENT.value: "Payment failures, declined transactions, duplicate charges, or money deducted despite a failed transaction.",
                    IssueCategory.DELIVERY.value: "Issues related to shipping, delivery delays, or missing deliveries.",
                    IssueCategory.PRODUCT.value: "Issues related to product quality, defects, damage, or incorrect items.",
                    IssueCategory.ACCOUNT.value: "Issues related to login, registration, account access, or profile details.",
                    IssueCategory.REFUND.value: "Requests for refunds, refund delays, or the status of an already initiated refund.",
                    IssueCategory.OTHER.value: "Issues that do not fit any of the categories above.",
                }
            ),

        "team": Choice(
                instructions="Select the team best suited to resolve the ticket based on the issue described.",
                criteria={
                    TeamOwner.FULFILLMENT.value: "Handles order processing, preparation, packing, and fulfillment issues.",
                    TeamOwner.PAYMENTS.value: "Handles payment failures, transaction issues, and unexpected charges.",
                    TeamOwner.LOGISTICS.value: "Handles shipping, delivery delays, tracking, and missing deliveries.",
                    TeamOwner.CUSTOMER_SUPPORT.value: "Handles general customer inquiries, complaints, and issues not covered by other teams.",
                    TeamOwner.TECH.value: "Handles technical problems, application bugs, system errors, and service outages.",
                },
            ),

        "priority": Score(
                instructions="Determine the ticket's priority based on its urgency, impact on the customer, and severity of the issue.",
                criteria=[priority.value for priority in Priority],
            ),

        "sentiment": Score(
                instructions="Determine the customer's sentiment from the ticket text, considering their tone and expressed emotions.",
                criteria=[sentiment.value for sentiment in Sentiment],
            ),

        "is_required_HR" : Noul(
                instructions="Determine whether this ticket requires human review due to ambiguity, insufficient information, or a complex or high-risk issue.",
                criteria={
                        "true": """
                            Return true if:
                            - The ticket is ambiguous or lacks essential information.
                            - Multiple issue categories are equally plausible.
                            - The issue involves financial risk or a potentially serious customer impact.
                            - The decision requires human judgment.

                            """,
                        "false": """
                            Return false if:
                            - The issue is clear and unambiguous.
                            - Sufficient information is available.
                            - The issue can be handled using standard business rules.
                            """
                    },
                )
    }

    outputs = jev.system_one(ticket, questions)
    answers = outputs.answers

    issue = extract_choice_interpretation(answers["issue"])
    team = extract_choice_interpretation(answers["team"])
    priority = extract_score_interpretation(answers["priority"])
    sentiment = extract_score_interpretation(answers["sentiment"])
    human_review_probability = answers["is_required_HR"].noul

    return TicketClassificationOutput(
        issue_output=ChoiceOutput[IssueCategory](
            choice=issue["choice"],
            confidence=issue["confidence"],
            entropy=issue["entropy"],
            normalized_entropy=issue["normalized_entropy"],
        ),
        team_output=ChoiceOutput[TeamOwner](
            choice=team["choice"],
            confidence=team["confidence"],
            entropy=team["entropy"],
            normalized_entropy=team["normalized_entropy"],
        ),
        priority_output=ScoreOutput[Priority](
            score=priority["score"],
            confidence=priority["confidence"],
            key=priority["key"],
            approximate=priority.get("approximate_key"),
        ),
        sentiment_output=ScoreOutput[Sentiment](
            score=sentiment["score"],
            confidence=sentiment["confidence"],
            key=sentiment["key"],
            approximate=sentiment.get("approximate_key"),
        ),
        hr_review_output=HumanReviewOutput(
            requires_human_review=human_review_probability >= 0.5,
            requires_hr_score=human_review_probability,
        ),
    )

if __name__=="__main__":

    ticket = "I was charged twice for order #9981. Please refund immediately!"
    outputs = get_classification(ticket)

    print(outputs)