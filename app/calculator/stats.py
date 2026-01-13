from __future__ import annotations

from typing import Any, Dict, List, Optional


def t_test_independent(df, group_col: str, value_col: str, group_a: Any, group_b: Any) -> Dict[str, Any]:
    try:
        from scipy import stats  # type: ignore
    except Exception:
        return {"type": "error", "error": "scipy not installed"}

    a = df[df[group_col] == group_a][value_col].dropna()
    b = df[df[group_col] == group_b][value_col].dropna()
    if a.empty or b.empty:
        return {"type": "error", "error": "empty groups"}
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"type": "stat", "test": "t_test_independent", "t": float(t), "p": float(p), "n_a": int(a.size), "n_b": int(b.size)}


def mann_whitney(df, group_col: str, value_col: str, group_a: Any, group_b: Any) -> Dict[str, Any]:
    try:
        from scipy import stats  # type: ignore
    except Exception:
        return {"type": "error", "error": "scipy not installed"}

    a = df[df[group_col] == group_a][value_col].dropna()
    b = df[df[group_col] == group_b][value_col].dropna()
    if a.empty or b.empty:
        return {"type": "error", "error": "empty groups"}
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"type": "stat", "test": "mann_whitney", "u": float(u), "p": float(p), "n_a": int(a.size), "n_b": int(b.size)}


def anova_oneway(df, group_col: str, value_col: str) -> Dict[str, Any]:
    try:
        from scipy import stats  # type: ignore
    except Exception:
        return {"type": "error", "error": "scipy not installed"}

    groups = [g[value_col].dropna() for _, g in df.groupby(group_col)]
    if len(groups) < 2:
        return {"type": "error", "error": "need >=2 groups"}
    f, p = stats.f_oneway(*groups)
    return {"type": "stat", "test": "anova_oneway", "f": float(f), "p": float(p), "k": int(len(groups))}


def kruskal_wallis(df, group_col: str, value_col: str) -> Dict[str, Any]:
    try:
        from scipy import stats  # type: ignore
    except Exception:
        return {"type": "error", "error": "scipy not installed"}

    groups = [g[value_col].dropna() for _, g in df.groupby(group_col)]
    if len(groups) < 2:
        return {"type": "error", "error": "need >=2 groups"}
    h, p = stats.kruskal(*groups)
    return {"type": "stat", "test": "kruskal_wallis", "h": float(h), "p": float(p), "k": int(len(groups))}


def fdr_bh(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    try:
        import statsmodels.stats.multitest as smm  # type: ignore
        reject, p_adj, _, _ = smm.multipletests(p_values, alpha=alpha, method="fdr_bh")
        return {"reject": reject.tolist(), "p_adj": [float(x) for x in p_adj]}
    except Exception:
        # Manual BH fallback
        m = len(p_values)
        sorted_idx = sorted(range(m), key=lambda i: p_values[i])
        p_sorted = [p_values[i] for i in sorted_idx]
        p_adj = [0.0] * m
        prev = 1.0
        for k in range(m - 1, -1, -1):
            rank = k + 1
            val = min(prev, p_sorted[k] * m / rank)
            p_adj[k] = val
            prev = val
        # Reorder back
        result = [0.0] * m
        for pos, idx in enumerate(sorted_idx):
            result[idx] = float(p_adj[pos])
        reject = [pa <= alpha for pa in result]
        return {"reject": reject, "p_adj": result}


def correlations(df, cols: Optional[List[str]] = None, method: str = "pearson") -> Dict[str, Any]:
    import pandas as pd  # type: ignore
    use = df[cols] if cols else df.select_dtypes(include=["number"])  # type: ignore
    corr = use.corr(method=method)
    return {"type": "matrix", "method": method, "corr": corr.to_dict()}


def linear_regression(df, y: str, X: List[str]) -> Dict[str, Any]:
    try:
        import statsmodels.api as sm  # type: ignore
    except Exception:
        return {"type": "error", "error": "statsmodels not installed"}
    Y = df[y]
    Xmat = sm.add_constant(df[X])
    model = sm.OLS(Y, Xmat, missing="drop").fit()
    return {"type": "regression", "model": "linear", "params": {k: float(v) for k, v in model.params.items()}, "r2": float(model.rsquared)}


