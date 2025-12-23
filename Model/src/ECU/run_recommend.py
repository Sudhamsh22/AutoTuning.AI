from __future__ import annotations
from .recommender import TuningAgent

if __name__ == "__main__":
    agent = TuningAgent(cfg_path="config.yaml", model_path="models/xgb_model.joblib")

    # Example current config (change to your actual starting values)
    current = {
        "BaseFuel": 1000,
        "BaseIgnition": 320,
        "IgnitionTiming": 10,
        "RPM": 3000,
        "Load": 60,
        "ThrottlePosition": 40,
        "CoolantTemp": 85,
        "AirTemp": 30,
        "BatteryVoltage": 13.8,
    }

    # FAST mode (CLI-safe, <5s)
    rec = agent.recommend(
        current,
        max_iters=5,
        num_candidates=10
    )

    print("\n--- TUNING RECOMMENDATION ---")
    print("Current score:", rec.current_score)
    print("Best score:", rec.best_score)
    print("Deltas:", rec.deltas)
    print("New config:", rec.new_config)
    print("Why:", rec.rationale)
