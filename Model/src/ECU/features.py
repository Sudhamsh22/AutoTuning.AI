from __future__ import annotations
import pandas as pd
import numpy as np


def clean_and_build_xy(df: pd.DataFrame, target: str, drop_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Available columns: {list(df.columns)}")

    # drop obvious non-features
    drop_set = set(drop_columns)
    drop_set.discard(target)

    X = df.drop(columns=[c for c in df.columns if c in drop_set], errors="ignore")
    y = df[target].copy()

    # Keep only numeric columns in X
    X = X.select_dtypes(include=[np.number])

    # Basic cleanup: remove rows without target
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    # Fill remaining missing values in X
    X = X.fillna(X.median(numeric_only=True))

    return X, y
