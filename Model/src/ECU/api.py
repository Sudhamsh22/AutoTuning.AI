from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from .recommender import TuningAgent

app = FastAPI(title="ECU Recommendation API", version="1.0")

agent = TuningAgent(cfg_path="config.yaml", model_path="models/xgb_model.joblib")


class RecommendRequest(BaseModel):
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    goal: Optional[str] = None


class RecommendResponse(BaseModel):
    current_score: float
    best_score: float
    deltas: Dict[str, float]
    new_config: Dict[str, float]
    rationale: str


class ScoreRequest(BaseModel):
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ScoreResponse(BaseModel):
    score: float
    target: str
    direction: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    return {
        "target": agent.cfg["train"]["target"],
        "direction": agent.cfg["train"]["direction"],
        "tunable_params": agent.tunable,
        "step_sizes": agent.step_sizes,
        "bounds": agent.bounds,
        "required_features": agent.feature_cols,
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if req.goal:
        goal = req.goal.strip().lower()
        if goal not in ("minimize", "maximize"):
            raise HTTPException(status_code=400, detail="Invalid goal")
        agent.direction = goal

    rec = agent.recommend(
        req.config or {},
        max_iters=5,
        num_candidates=10
    )

    return {
        "current_score": rec.current_score,
        "best_score": rec.best_score,
        "deltas": rec.deltas,
        "new_config": rec.new_config,
        "rationale": rec.rationale,
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    s = agent._score(req.config or {})
    return {
        "score": s,
        "target": agent.cfg["train"]["target"],
        "direction": agent.direction,
    }
