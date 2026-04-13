from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def differential_like(df, group_col: str, value_cols: List[str], group_a: Any, group_b: Any, *, method: str = "ttest", fdr_alpha: float = 0.05) -> Dict[str, Any]:
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        from .stats import t_test_independent, mann_whitney, fdr_bh
    except Exception:
        return {"type": "error", "error": "required libraries not installed"}

    results: List[Dict[str, Any]] = []
    pvals: List[float] = []
    for gene in value_cols:
        if method == "mannwhitney":
            res = mann_whitney(df, group_col, gene, group_a, group_b)
        else:
            res = t_test_independent(df, group_col, gene, group_a, group_b)
        if res.get("type") == "stat":
            a = df[df[group_col] == group_a][gene].dropna()
            b = df[df[group_col] == group_b][gene].dropna()
            mean_a = float(a.mean()) if a.size else float("nan")
            mean_b = float(b.mean()) if b.size else float("nan")
            fc = float(mean_a - mean_b)
            p = float(res.get("p", 1.0))
            pvals.append(p)
            results.append({"feature": gene, "effect": fc, "p": p})
        else:
            results.append({"feature": gene, "error": res.get("error", "calc failed")})

    # FDR correction
    if pvals:
        fdr = fdr_bh(pvals, alpha=fdr_alpha)
        p_adj = fdr.get("p_adj", [1.0] * len(results))
        for i, r in enumerate(results):
            if "p" in r:
                r["q"] = float(p_adj[i])

    # Volcano components
    out_df = None
    try:
        import pandas as pd  # type: ignore

        out_df = pd.DataFrame(results)
        out_df["-log10p"] = -np.log10(out_df["p"].clip(lower=1e-300)) if "p" in out_df else None
    except Exception:
        pass

    return {"type": "de", "results": results, "dataframe": out_df}


def ora_enrichment(gene_set: List[str], universe: List[str], term_map: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    try:
        from scipy.stats import fisher_exact  # type: ignore
    except Exception:
        return []

    gene_set = list({g.upper() for g in gene_set})
    universe_set = set(g.upper() for g in universe)
    results: List[Dict[str, Any]] = []
    for term, members in term_map.items():
        members_set = set(g.upper() for g in members)
        a = len(members_set & set(gene_set))
        b = len(set(gene_set)) - a
        c = len(members_set - set(gene_set))
        d = max(len(universe_set) - (a + b + c), 0)
        try:
            odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        except Exception:
            odds, p = 0.0, 1.0
        results.append({"term": term, "overlap": a, "odds": float(odds), "p": float(p), "size": len(members_set)})
    # Basic sorting
    results.sort(key=lambda r: r["p"])
    return results


def gene_id_map(symbols: List[str], mapping: Dict[str, str]) -> Dict[str, str]:
    # mapping is a small dictionary symbol->ensembl (or similar) provided/embedded
    out: Dict[str, str] = {}
    for s in symbols:
        key = s.upper()
        out[s] = mapping.get(key, s)
    return out


def dose_response_ic50(df, concentration_col: str, response_col: str) -> Dict[str, Any]:
    try:
        import numpy as np  # type: ignore
        from scipy.optimize import curve_fit  # type: ignore
    except Exception:
        return {"type": "error", "error": "scipy/numpy not installed"}

    def four_pl(x, bottom, top, logIC50, hill):
        return bottom + (top - bottom) / (1 + 10 ** ((logIC50 - x) * hill))

    try:
        x = np.log10(df[concentration_col].astype(float).values)
        y = df[response_col].astype(float).values
        p0 = [min(y), max(y), np.median(x), 1.0]
        bounds = ([min(y) - abs(min(y)), max(y) - abs(max(y)), min(x) - 3, 0.01], [max(y) + abs(max(y)), max(y) + abs(max(y)), max(x) + 3, 5])
        popt, pcov = curve_fit(four_pl, x, y, p0=p0, maxfev=20000, bounds=bounds)
        bottom, top, logIC50, hill = [float(p) for p in popt]
        ic50 = float(10 ** logIC50)
        return {"type": "ic50", "params": {"bottom": bottom, "top": top, "logIC50": logIC50, "hill": hill}, "IC50": ic50}
    except Exception as e:
        return {"type": "error", "error": str(e)}


