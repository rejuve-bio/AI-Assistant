import pandas as pd


def test_differential_like_outputs_keys():
    from app.code_exec.bio_ops import differential_like

    df = pd.DataFrame({
        "group": ["A"] * 10 + ["B"] * 10,
        "g1": list(range(10)) + list(range(10, 20)),
        "g2": list(range(5, 15)) + list(range(15, 25)),
    })
    res = differential_like(df, "group", ["g1", "g2"], "A", "B")
    assert res.get("type") == "de"
    assert isinstance(res.get("results"), list)


def test_ora_enrichment_runs():
    from app.code_exec.bio_ops import ora_enrichment

    genes = ["TP53", "EGFR", "MYC"]
    universe = genes + ["GATA3", "BRCA1"]
    term_map = {"pathway1": ["TP53", "GATA3"], "pathway2": ["EGFR", "MYC"]}
    out = ora_enrichment(genes, universe, term_map)
    assert isinstance(out, list)


def test_ic50_fitting_smoke():
    from app.code_exec.bio_ops import dose_response_ic50
    import numpy as np

    conc = np.logspace(-3, 1, 10)
    response = 100 / (1 + (1/conc))  # synthetic monotonic curve
    df = pd.DataFrame({"conc": conc, "resp": response})
    res = dose_response_ic50(df, "conc", "resp")
    assert isinstance(res, dict)

