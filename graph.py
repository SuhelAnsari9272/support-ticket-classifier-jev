import logging

from langgraph.graph import StateGraph, START, END
from production_modules.all_question import get_classification
from production_modules.pii_redaction import redact_pii
from production_modules.prompt_injection import check_injection
from production_modules.fallback_retry import SAFE_CLASSIFICATION

logger = logging.getLogger(__name__)

### Nodes 

## pii redact node
def pii_redact_node(state:dict) -> dict :
    result = redact_pii(state['raw_ticket'])
    # print(result)
    return{
        **state,
        "redacted_ticket": result.redacted_text,
        "pii_detected" : result.pii_detected,
    }

## Injection Check Node

def injection_check_node(state:dict)-> dict:
    check = check_injection(state["redacted_ticket"])
    if not check.is_safe:
        logger.warning("Injection detected: %s", check.injection_prob)
        return {
            **state,
            "injection_blocked": True,
            "error": f"Injection detected: {check.injection_prob}",
            "classification": SAFE_CLASSIFICATION,
            # "validation_status": "blocked",
        }
    return {**state, "injection_blocked": False}


## Classify node

def classify(state:dict) -> dict:

    if state.get("injection_blocked"):
        return state


    ticket_text = state['redacted_ticket']
    
    try :
        classification = get_classification(ticket_text)
        return {**state, "classification": classification}

    except Exception as e:
        logger.error("classify_node failed: %s", e)
        return {**state, "error": str(e), "classification": None}


### Fallback Node
def fallback_node(state:dict) -> dict:

    if state.get("classification") is not None:
        return state

    logger.warning("Fallback node invoked due to error: %s", state.get("error"))

    last_error = state.get("error")
    for i in range(3) :
        try:
            classification = get_classification(state['redacted_ticket'])
            return {**state, "classification": classification}
        except Exception as e:
            logger.error("Fallback retry %d failed: %s", i + 1, e)
            last_error = e

    return {**state, "error": f"Fails after Fallback: {last_error}"}


def build_graph():
    builder = StateGraph(dict)

    builder.add_node("pii_redact", pii_redact_node)
    builder.add_node("injection_check", injection_check_node)
    builder.add_node("classify", classify)
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "pii_redact")
    builder.add_edge("pii_redact", "injection_check")
    builder.add_edge("injection_check", "classify")
    builder.add_edge("classify", "fallback")
    builder.add_edge("fallback", END)

    return builder.compile()

graph = build_graph()

def run_pipeline(ticket_text: str, channel: str = "web_form") -> dict:
    initial_state = {
        "raw_ticket": ticket_text,
        "channel": channel,
        "redacted_ticket": None,
        "classification": None,
        # "validation_status": None,
        # "cost_info": None,
        "error": None,
        "pii_detected": False,
        # "prompt_version": None,
        "injection_blocked": False,
    }
    return graph.invoke(initial_state)
