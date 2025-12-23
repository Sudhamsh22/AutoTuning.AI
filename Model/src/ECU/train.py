from __future__ import annotations
from pathlib import Path
import glob
import yaml
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

from .datalog_parser import parse_ecu_manager_datalog
from .features import clean_and_build_xy

ROOT = Path(__file__).resolve().parent.parent


def load_config(cfg_path: str = "config.yaml") -> dict:
    cfg_file = ROOT / cfg_path
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config not found at {cfg_file}")
    return yaml.safe_load(cfg_file.read_text(encoding="utf-8-sig"))


def load_folder(folder: str, file_glob: str) -> pd.DataFrame:
    paths = sorted(glob.glob(str(Path(folder) / file_glob)))
    if not paths:
        raise FileNotFoundError(f"No files found in {folder} with glob {file_glob}")

    dfs = []
    for p in paths:
        parsed = parse_ecu_manager_datalog(p)
        df = parsed.df.copy()
        df["__source_file__"] = Path(p).name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def train(cfg_path: str = "config.yaml"):
    cfg = load_config(cfg_path)

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    feat_cfg = cfg["features"]

    df = load_folder(data_cfg["folder"], data_cfg["file_glob"])

    X, y = clean_and_build_xy(
        df=df,
        target=train_cfg["target"],
        drop_columns=feat_cfg.get("drop_columns", []),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=train_cfg.get("test_size", 0.2),
        random_state=train_cfg.get("random_state", 42),
    )

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=train_cfg.get("random_state", 42),
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    # ------------------ SAVE ARTIFACTS ------------------
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "xgb_model.joblib")
    (out_dir / "feature_columns.txt").write_text("\n".join(X.columns))

    # Save medians for inference-time imputation
    medians = X.median(numeric_only=True).to_dict()
    (out_dir / "feature_medians.json").write_text(
        json.dumps(medians, indent=2)
    )

    metrics = {
        "rmse": float(rmse),
        "r2": float(r2),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
    }

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/metrics.yaml").write_text(yaml.safe_dump(metrics))

    print("Saved model to models/xgb_model.joblib")
    print("Metrics:", metrics)


if __name__ == "__main__":
    train()
