from __future__ import annotations

from typing import Any, Dict, List, Optional


def missingness_report(df) -> Dict[str, int]:
    try:
        return {str(c): int(df[c].isna().sum()) for c in df.columns}
    except Exception:
        return {}


def drop_duplicates(df, subset: Optional[List[str]] = None) -> Any:
    try:
        return df.drop_duplicates(subset=subset)
    except Exception:
        return df


def join_tables(left, right, on: List[str], how: str = "inner") -> Any:
    try:
        return left.merge(right, on=on, how=how)
    except Exception:
        return left


def scale_columns(df, cols: List[str], method: str = "standard") -> Any:
    try:
        import numpy as np  # type: ignore
        X = df.copy()
        for c in cols:
            if method == "minmax":
                mn, mx = X[c].min(), X[c].max()
                X[c] = (X[c] - mn) / (mx - mn) if mx != mn else 0.0
            else:
                mu, sd = X[c].mean(), X[c].std() or 1.0
                X[c] = (X[c] - mu) / sd
        return X
    except Exception:
        return df


def unit_convert(df, col: str, factor: float) -> Any:
    try:
        X = df.copy()
        X[col] = X[col].astype(float) * float(factor)
        return X
    except Exception:
        return df


