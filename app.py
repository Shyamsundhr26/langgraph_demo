from fastapi import FastAPI
from pydantic import BaseModel

from langgraph_demo.workflow import app as graph


# Create the FastAPI application
api = FastAPI(
    title="LangGraph Support Assistant"
)


# Request structure
class ChatRequest(BaseModel):
    thread_id: str
    query: str


# Response structure
class ChatResponse(BaseModel):
    thread_id: str
    response: str
    request_type: str
    current_step: str
    human_escalation: bool


# POST /chat endpoint
@api.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    # Configuration used by LangGraph
    config = {
        "configurable": {
            "thread_id": request.thread_id
        }
    }

    # Send the user's request to LangGraph
    result = graph.invoke(
        {
            "messages":[],
            "user_query": request.query,
            "request_type": "",
            "specialist_result": "",
            "final_response": "",
            "review_status": "",
            "human_escalation": False,
            "current_step": "started",
        },
        config=config,
    )

    # Return selected information to the API client
    return ChatResponse(
        thread_id=request.thread_id,
        response=result["final_response"],
        request_type=result["request_type"],
        current_step=result["current_step"],
        human_escalation=result["human_escalation"],
    )