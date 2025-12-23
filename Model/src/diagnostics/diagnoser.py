from __future__ import annotations
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Any

from src.diagnostics.car_failure_knowledge import CAR_FAILURE_SYMPTOM_CAUSE
from src.diagnostics.bike_failure_knowledge import BIKE_FAILURE_SYMPTOM_CAUSE

KB = {
    "car": CAR_FAILURE_SYMPTOM_CAUSE,
    "bike": BIKE_FAILURE_SYMPTOM_CAUSE,
}

def _norm(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _score(query: str, symptoms: List[str], causes: List[str]) -> float:
    q = _norm(query)
    q_tokens = set(q.split())

    sym_tokens = set(_norm(" ".join(symptoms)).split())
    cause_tokens = set(_norm(" ".join(causes)).split())
    overlap = len(q_tokens & (sym_tokens | cause_tokens))

    best_sym = max((_similar(q, _norm(s)) for s in symptoms), default=0.0)
    best_cause = max((_similar(q, _norm(c)) for c in causes), default=0.0)

    return (0.25 * overlap) + (0.45 * best_sym) + (0.30 * best_cause)

def diagnose(vehicle_type: str, query: str, topk: int = 3) -> List[Dict[str, Any]]:
    vehicle_type = vehicle_type.lower().strip()
    if vehicle_type not in KB:
        raise ValueError("vehicle_type must be 'car' or 'bike'")

    if not query or not query.strip():
        return []

    kb = KB[vehicle_type]

    scored: List[Tuple[str, float]] = []
    for key, entry in kb.items():
        score = _score(query, entry["symptoms"], entry["causes"])
        scored.append((key, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for key, score in scored[:topk]:
        entry = kb[key]
        results.append({
            "issue_key": key,
            "score": round(float(score), 4),
            "symptoms": entry["symptoms"],
            "probable_causes": entry["causes"],
        })
    return results
