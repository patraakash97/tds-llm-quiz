import os
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from solver import solve_quiz_chain

# -------------------------------
# 1. Request Model
# -------------------------------

class QuizPayload(BaseModel):
    email: str
    secret: str
    url: str


# -------------------------------
# 2. Config
# -------------------------------

APP_SECRET = os.getenv("QUIZ_SECRET", "Akash123")
APP_EMAIL = os.getenv("QUIZ_EMAIL", "23f3004144@ds.study.iitm.ac.in")

app = FastAPI()


# -------------------------------
# 3. Root (Optional)
# -------------------------------

@app.get("/")
def home():
    return {
        "message": "TDS LLM Quiz Solver API Running",
        "endpoint": "/quiz",
        "version": "1.0"
    }


# -------------------------------
# 4. /quiz Endpoint
# -------------------------------

@app.post("/quiz")
def quiz_endpoint(payload: QuizPayload):
    """
    IITM will POST:
    {
        "email": "...",
        "secret": "...",
        "url": "quiz-start-url"
    }
    """

    # Verify secret
    if payload.secret != APP_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # Prepare processing window (3 minutes)
    start_time = datetime.utcnow()
    deadline = start_time + timedelta(minutes=3)

    try:
        final_result = solve_quiz_chain(
            email=payload.email,
            secret=payload.secret,
            first_url=payload.url,
            deadline=deadline
        )
    except Exception as e:
        # Catch ANY crash → prevent 500 errors
        return JSONResponse(
            status_code=200,
            content={
                "status": "solver-exception",
                "error": str(e)
            }
        )

    return {
        "status": "ok",
        "final_result": final_result
    }
