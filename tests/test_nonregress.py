def test_main_imports_and_symbols_present():
    import app.main as m
    # Ensure core symbols are importable and added
    assert hasattr(m, "AiAssistance")
    assert hasattr(m, "AgentState")


def test_classifier_prompt_mentions_code_exec():
    import inspect
    import app.main as m
    src = inspect.getsource(m.AiAssistance._classify_query)
    assert "code_exec" in src

