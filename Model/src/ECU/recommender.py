from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import yaml
import json


# ------------------ data structures ------------------

@dataclass
class Recommendation:
    current_score: float
    best_score: float
    deltas: Dict[str, float]
    new_config: Dict[str, float]
    rationale: str
    direction: str  # REQUIRED for goal validation


# ------------------ agent ------------------

class TuningAgent:
    def __init__(
        self,
        cfg_path: str = "config.yaml",
        model_path: str = "models/xgb_model.joblib",
    ):
        root = Path(__file__).resolve().parents[2]


        # load config & model
        self.cfg = yaml.safe_load((root / cfg_path).read_text(encoding="utf-8-sig"))
        self.model = joblib.load(root / model_path)

        # feature metadata
        self.feature_cols = (root / "models/feature_columns.txt").read_text(
            encoding="utf-8"
        ).splitlines()

        tcfg = self.cfg["tuning"]
        self.tunable = tcfg["tunable_params"]
        self.step_sizes = {k: float(v) for k, v in tcfg.get("step_sizes", {}).items()}
        self.bounds = tcfg.get("bounds", {})
        self.num_candidates = int(tcfg["search"]["num_candidates"])
        self.max_iters = int(tcfg["search"]["max_iters"])

        # optimization direction
        self.direction = self.cfg["train"]["direction"]  # minimize / maximize

        # medians for missing values
        self.medians = json.loads(
            (root / "models/feature_medians.json").read_text(encoding="utf-8")
        )

    # ------------------ helpers ------------------

    def _better(self, a: float, b: float) -> bool:
        return a < b if self.direction == "minimize" else a > b

    def _vectorize(self, config: Dict[str, Any]) -> pd.DataFrame:
        row = {}
        for c in self.feature_cols:
            v = config.get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = self.medians.get(c, 0.0)
            row[c] = v

        X = pd.DataFrame([row]).apply(pd.to_numeric, errors="coerce")
        return X.fillna(0.0)

    def _score(self, config: Dict[str, Any]) -> float:
        X = self._vectorize(config)
        return float(self.model.predict(X)[0])

    # ------------------ main logic ------------------

    def recommend(
        self,
        current_config: Dict[str, Any],
        *,
        max_iters: Optional[int] = None,
        num_candidates: Optional[int] = None,
    ) -> Recommendation:
        """
        Fast + full compatible recommender.
        Use small limits for API, large limits for offline tuning.
        """

        iters = max_iters if max_iters is not None else self.max_iters
        cands = num_candidates if num_candidates is not None else self.num_candidates

        base = dict(current_config)
        base_score = self._score(base)

        best = dict(base)
        best_score = base_score

        rng = np.random.default_rng(42)

        for _ in range(iters):
            for _ in range(cands):
                cand = dict(best)

                for p in self.tunable:
                    step = float(self.step_sizes.get(p, 1.0))
                    lo, hi = self.bounds.get(p, [-1e9, 1e9])

                    direction = rng.choice([-1, 0, 1], p=[0.40, 0.20, 0.40])
                    cur = float(cand.get(p, 0.0) or 0.0)

                    cand[p] = float(np.clip(cur + direction * step, lo, hi))

                score = self._score(cand)
                if self._better(score, best_score):
                    best = cand
                    best_score = score

        deltas = {
            p: float(best.get(p, 0.0) - base.get(p, 0.0))
            for p in self.tunable
        }
        changed = {p: d for p, d in deltas.items() if abs(d) > 1e-9}

        rationale = (
            f"Optimized to {self.direction} '{self.cfg['train']['target']}'. "
            f"Score {base_score:.4f} → {best_score:.4f}. "
            f"Changes: {changed if changed else 'none'}."
        )

        return Recommendation(
            current_score=base_score,
            best_score=best_score,
            deltas=changed,
            new_config={p: float(best.get(p, 0.0)) for p in self.tunable},
            rationale=rationale,
            direction=self.direction,  # ALWAYS present
        )
