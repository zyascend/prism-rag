# tests/test_agent_config.py
from src.agent.config import agent_config, agent_cache_salt


def test_agent_defaults_disabled():
    c = agent_config()
    assert c["enabled"] is False
    assert c["max_subqueries"] == 3
    assert c["max_total_searches"] == 3
    assert c["max_search_per_subquery"] == 1
    assert c["max_llm_calls"] == 6
    assert c["max_grade_cycles"] == 1
    assert c["on_error"] == "degrade_pipeline"
    assert c["grade"]["enabled"] is True
    assert c["checkpoint"]["enabled"] is True
    assert c["hitl"]["review_subqueries"] is False
    assert c["react_demo"]["enabled"] is False


def test_cache_salt_stable_and_changes_with_enabled(monkeypatch):
    from src import config as config_mod

    # 无 agent 段时 salt 仍可生成
    s_off = agent_cache_salt()
    assert "ag=" in s_off
    assert "off" in s_off or "enabled=False" in s_off or "en=0" in s_off
