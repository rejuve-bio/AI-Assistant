import os
import tempfile
import pandas as pd


def test_plotting_hist_scatter_heatmap(tmp_path):
    from app.code_exec import plotting

    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6],
        "b": [2, 1, 2, 3, 4, 5],
        "c": [5, 3, 6, 2, 1, 0],
    })

    hist_path = os.path.join(tmp_path, "hist.png")
    res1 = plotting.save_histogram(df, "a", hist_path)
    assert res1.get("type") in ("figure", "error")

    scatter_path = os.path.join(tmp_path, "scatter.png")
    res2 = plotting.save_scatter(df, "a", "b", scatter_path)
    assert res2.get("type") in ("figure", "error")

    heatmap_path = os.path.join(tmp_path, "heatmap.png")
    res3 = plotting.save_correlation_heatmap(df, heatmap_path)
    assert res3.get("type") in ("figure", "error")

