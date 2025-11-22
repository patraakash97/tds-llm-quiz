# app.py
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
# 3. Health check
# -------------------------------

@app.get("/")
def root():
    return {"status": "alive"}


# -------------------------------
# 4. Quiz Endpoint (SYNC)
# -------------------------------

@app.post("/quiz")
def quiz_endpoint(payload: QuizPayload):
    """
    Main endpoint that IITM will call.
    """

    email = payload.email
    secret = payload.secret
    url = payload.url

    # Secret check – this is the ONLY place that can return non-200/ok
    if secret != APP_SECRET:
        # 403 is required by spec for wrong secret
        raise HTTPException(status_code=403, detail="Invalid secret")

    # Deadline (3 minutes) – as per assignment
    start_time = datetime.utcnow()
    deadline = start_time + timedelta(minutes=3)

    # Call solver; never let exceptions bubble out as top-level status="error"
    try:
        solver_result = solve_quiz_chain(
            email=email,
            secret=secret,
            first_url=url,
            deadline=deadline,
        )
        final_result = {
            "status": "solver-ok",
            "result": solver_result,
        }
    except Exception as e:
        # Even if solver explodes, we still return HTTP 200 and status="ok" at top
        final_result = {
            "status": "solver-exception",
            "error": str(e),
        }

    # ALWAYS status="ok" at top level if secret is valid
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "final_result": final_result,
        },
    )
