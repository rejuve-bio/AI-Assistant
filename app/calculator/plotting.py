from __future__ import annotations

from typing import Any, Dict, Optional
import os


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_histogram(df, column: str, path: str, *, bins: int = 30) -> Dict[str, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns  # type: ignore
    except Exception:
        return {"type": "error", "error": "matplotlib/seaborn not installed"}

    _ensure_dir(path)
    plt.figure(figsize=(6, 4))
    try:
        sns.histplot(df[column].dropna(), bins=bins, kde=False)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        return {"type": "figure", "name": "histogram", "path": path}
    except Exception as e:
        plt.close()
        return {"type": "error", "error": str(e)}


def save_scatter(df, x: str, y: str, path: str) -> Dict[str, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns  # type: ignore
    except Exception:
        return {"type": "error", "error": "matplotlib/seaborn not installed"}

    _ensure_dir(path)
    plt.figure(figsize=(6, 4))
    try:
        sns.scatterplot(data=df, x=x, y=y)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        return {"type": "figure", "name": "scatter", "path": path}
    except Exception as e:
        plt.close()
        return {"type": "error", "error": str(e)}


def save_correlation_heatmap(df, path: str) -> Dict[str, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns  # type: ignore
    except Exception:
        return {"type": "error", "error": "matplotlib/seaborn not installed"}

    _ensure_dir(path)
    plt.figure(figsize=(7, 6))
    try:
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, cmap="vlag", center=0)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        return {"type": "figure", "name": "correlation_heatmap", "path": path}
    except Exception as e:
        plt.close()
        return {"type": "error", "error": str(e)}


def save_boxplot(df, column: str, by: Optional[str], path: str) -> Dict[str, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns  # type: ignore
    except Exception:
        return {"type": "error", "error": "matplotlib/seaborn not installed"}

    _ensure_dir(path)
    plt.figure(figsize=(6, 4))
    try:
        if by:
            sns.boxplot(data=df, x=by, y=column)
        else:
            sns.boxplot(data=df[[column]].dropna())
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        return {"type": "figure", "name": "boxplot", "path": path}
    except Exception as e:
        plt.close()
        return {"type": "error", "error": str(e)}


def save_timeseries(df, x: str, y: str, path: str) -> Dict[str, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns  # type: ignore
    except Exception:
        return {"type": "error", "error": "matplotlib/seaborn not installed"}

    _ensure_dir(path)
    plt.figure(figsize=(7, 4))
    try:
        sns.lineplot(data=df, x=x, y=y)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        return {"type": "figure", "name": "timeseries", "path": path}
    except Exception as e:
        plt.close()
        return {"type": "error", "error": str(e)}


def save_pca_preview(df, path: str, *, n_components: int = 2) -> Dict[str, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns  # type: ignore
        from sklearn.decomposition import PCA  # type: ignore
    except Exception:
        return {"type": "error", "error": "matplotlib/seaborn/sklearn not installed"}

    _ensure_dir(path)
    plt.figure(figsize=(6, 5))
    try:
        X = df.select_dtypes(include=["number"]).dropna(axis=0, how="any")
        if X.shape[1] < 2 or X.shape[0] < 2:
            return {"type": "error", "error": "insufficient numeric data for PCA"}
        pca = PCA(n_components=min(n_components, X.shape[1]))
        comp = pca.fit_transform(X)
        sns.scatterplot(x=comp[:, 0], y=comp[:, 1])
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        return {"type": "figure", "name": "pca_preview", "path": path}
    except Exception as e:
        plt.close()
        return {"type": "error", "error": str(e)}


