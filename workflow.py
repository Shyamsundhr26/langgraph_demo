from typing import TypedDict, Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()



# MODEL


model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
)


# APPLICATION STATE


class State(TypedDict):
    messages: list
    user_query: str

    request_type: str

    specialist_result: str

    final_response: str

    review_status: str

    human_escalation: bool

    current_step: str



# NODE — TRIAGE / ANALYZER


def triage_node(state: State):

    query = state["user_query"]

    prompt = f"""
You are a customer support triage agent.

Classify the following customer request into exactly one category:

technical
billing
general

Customer request:
{query}

Return ONLY one word:
technical
billing
general
"""

    response = model.invoke(prompt)

    classification = response.content.strip().lower()

    # Safety fallback if model returns unexpected text
    if classification not in {
        "technical",
        "billing",
        "general",
    }:
        classification = "general"

    return {
        "request_type": classification,
        "current_step": "triage_completed",
    }



# NODE  — TECHNICAL SPECIALIST


def technical_specialist(state: State):

    query = state["user_query"]

    # Mock technical lookup/tool
    technical_database = {
        "password": "Password reset instructions are available from the account settings page.",
        "login": "The login service is operational. Try resetting your password if authentication fails.",
        "error": "Please provide the error code so the technical team can investigate it."
    }

    result = "Technical support received your request."

    for keyword, answer in technical_database.items():
        if keyword in query.lower():
            result = answer
            break

    return {
        "specialist_result": result,
        "current_step": "technical_specialist_completed",
    }



# NODE  — BILLING SPECIALIST


def billing_specialist(state: State):

    query = state["user_query"]

    # Mock billing lookup/tool
    billing_database = {
        "refund": "Refund requests are normally reviewed within 5 business days.",
        "invoice": "Invoices can be downloaded from the billing section of your account.",
        "charge": "Please provide the transaction ID so the billing team can investigate the charge."
    }

    result = "Billing support received your request."

    for keyword, answer in billing_database.items():
        if keyword in query.lower():
            result = answer
            break

    return {
        "specialist_result": result,
        "current_step": "billing_specialist_completed",
    }


# NODE — GENERAL SPECIALIST


def general_specialist(state: State):

    query = state["user_query"]

    response = model.invoke(
        f"""
You are a general customer support assistant.

Answer this customer request clearly and briefly:

{query}
"""
    )

    return {
        "specialist_result": response.content,
        "current_step": "general_specialist_completed",
    }



# NODE — REVIEWER / GUARDRAIL


def reviewer_node(state: State):

    result = state["specialist_result"]

    # Simple deterministic guardrail for the demo.
    dangerous_terms = [
        "ignore previous instructions",
        "reveal system prompt",
        "api key",
        "password database",
        "secret key",
    ]

    failed = any(
        term in result.lower()
        for term in dangerous_terms
    )

    if failed:
        return {
            "review_status": "failed",
            "human_escalation": True,
            "final_response": "",
            "current_step": "review_failed",
        }

    return {
        "review_status": "passed",
        "human_escalation": False,
        "final_response": result,
        "current_step": "review_passed",
    }



# NODE 4 — HUMAN REVIEW


def human_review_node(state: State):

    # In a production application this could pause
    # for an actual human approval.
    #
    # For this demo we create an explicit escalation result.

    return {
        "final_response": (
            "This request has been escalated to a human "
            "support representative for review."
        ),
        "human_escalation": True,
        "current_step": "human_review",
    }


# CONDITIONAL ROUTING — TRIAGE

def route_after_triage(
    state: State,
) -> Literal[
    "technical_specialist",
    "billing_specialist",
    "general_specialist",
]:

    request_type = state["request_type"]

    if request_type == "technical":
        return "technical_specialist"

    if request_type == "billing":
        return "billing_specialist"

    return "general_specialist"


# CONDITIONAL ROUTING — REVIEW


def route_after_review(
    state: State,
) -> Literal[
    "human_review",
    "end",
]:

    if state["review_status"] == "failed":
        return "human_review"

    return "end"



# CREATE GRAPH


builder = StateGraph(State)


# Add nodes
builder.add_node(
    "triage",
    triage_node,
)

builder.add_node(
    "technical_specialist",
    technical_specialist,
)

builder.add_node(
    "billing_specialist",
    billing_specialist,
)

builder.add_node(
    "general_specialist",
    general_specialist,
)

builder.add_node(
    "reviewer",
    reviewer_node,
)

builder.add_node(
    "human_review",
    human_review_node,
)



# EDGES


builder.add_edge(
    START,
    "triage",
)


# Conditional routing after triage
builder.add_conditional_edges(
    "triage",
    route_after_triage,
)


# Specialists → reviewer
builder.add_edge(
    "technical_specialist",
    "reviewer",
)

builder.add_edge(
    "billing_specialist",
    "reviewer",
)

builder.add_edge(
    "general_specialist",
    "reviewer",
)


# Conditional routing after reviewer
builder.add_conditional_edges(
    "reviewer",
    route_after_review,
    {
        "human_review": "human_review",
        "end": END,
    },
)


# Human review → END
builder.add_edge(
    "human_review",
    END,
)



# CHECKPOINTER

checkpointer = MemorySaver()



# COMPILE GRAPH


app = builder.compile(
    checkpointer=checkpointer
)

if __name__ == "__main__":

    print("=================================")
    print("       AI Support Assistant")
    print("=================================")
    print("Type 'exit' to quit.\n")

    config = {
        "configurable": {
            "thread_id": "test-001"
        }
    }

    while True:

        user_query = input("You: ")

        if user_query.lower() == "exit":
            print("\nAI: Goodbye!")
            break

        result = app.invoke(
            {
                "messages": [],
                "user_query": user_query,
                "request_type": "",
                "specialist_result": "",
                "final_response": "",
                "review_status": "",
                "human_escalation": False,
                "current_step": "started",
            },
            config=config,
        )

        print(f"\nAI: {result['final_response']}\n")