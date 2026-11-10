from typesafe_sdk import TypeSafeClient, Noul, Choice, Score
import httpx2

from settings import JEV_API_KEY, JEV_BASE_URL, JEV_MODEL, VERIFY_SSL


if not JEV_API_KEY:
    raise RuntimeError(
        "Missing API key. Set TYPESAFE_API_KEY to a valid OpenRouter API key "
        "before running the classifier."
    )

jev = TypeSafeClient(api_key=JEV_API_KEY, base_url=JEV_BASE_URL, model=JEV_MODEL,
                        http_client=httpx2.Client(verify=VERIFY_SSL))

def get_issue_category(ticket):
    question = {
        "issue": Choice(
            instructions="Which category best describes the customer's issue?",
            criteria={
            "order": "Issues related to placing, modifying, or tracking an order.",
            "payment": "Payment failures, declined transactions, duplicate charges, or money deducted despite a failed transaction.",
            "delivery": "Issues related to shipping, delivery delays, or missing deliveries.",
            "product": "Issues related to product quality, defects, damage, or incorrect items.",
            "account": "Issues related to login, registration, account access, or profile details.",
            "refund": "Requests for refunds, refund delays, or the status of an already initiated refund.",
            "other": "Issues that do not fit any of the categories above."
            }
        )
    }

    output = jev.system_one(ticket, question)
    result_issue = output.answers["issue"]

    return {"issue" : result_issue.choice , "confidence" : result_issue.confidence}

def get_assigned_team(ticket):
    question = {
        "team": Choice(
            instructions="Select the team best suited to resolve the ticket based on the issue described.",
            criteria={
                "fulfillment_team": "Handles order processing, preparation, packing, and fulfillment issues.",
                "payments_team": "Handles payment failures, transaction issues, and unexpected charges.",
                "logistics_team": "Handles shipping, delivery delays, tracking, and missing deliveries.",
                "customer_support": "Handles general customer inquiries, complaints, and issues not covered by other teams.",
                "tech_team": "Handles technical problems, application bugs, system errors, and service outages.",
            },
        )
    }

    output = jev.system_one(ticket, question)
    result_team = output.answers["team"]

    return {"team" : result_team.choice , "confidence" : result_team.confidence}

def get_priority(ticket):
    question = {
        "priority": Score(
            instructions="Determine the ticket's priority based on its urgency, impact on the customer, and severity of the issue.",
            criteria=["low", "medium", "high", "critical"],
        )
    }

    output = jev.system_one(ticket, question)
    result = output.answers["priority"]

    return  {"score" : result.score , "confidence" : result.confidence}

def get_user_sentiment(ticket):
    question = {
        "sentiment": Score(
            instructions="Determine the customer's sentiment from the ticket text, considering their tone and expressed emotions.",
            criteria=["positive", "neutral", "negative", "angry",],
        )
    }
    output = jev.system_one(ticket, question)
    result = output.answers["sentiment"]

    return  {"score" : result.score , "confidence" : result.confidence}

def get_human_review(ticket):
    question = { 
        "is_required_HR" : Noul(
            instructions="Determine whether this ticket requires human review due to ambiguity, insufficient information, or a complex or high-risk issue."
        )
    }
    output = jev.system_one(ticket, question)
    result = output.answers["is_required_HR"]

    return {"hr_prob" : result.noul}


if __name__=="__main__":

    # test any of the above
    ticket = "I was charged twice for order #9981. Please refund immediately!"

    issue_category = get_issue_category(ticket)
    team = get_assigned_team(ticket)
    priority = get_priority(ticket)
    sentiment = get_user_sentiment(ticket)
    human_review = get_human_review(ticket)

    print("Ticket : ", ticket)
    print(f"Issue Category : {issue_category["issue"]}  |  conf : {issue_category["confidence"]}")
    print(f"Assigned Team: {team["team"]}  |  conf : {team["confidence"]}")
    print(f"Priority : {priority["score"]}  |  conf : {priority["confidence"]}")
    print(f"User Sentiment : {sentiment["score"]}  |  conf : {sentiment["confidence"]}")
    print(f"Is Human review required : {human_review["hr_prob"]}")
    print()