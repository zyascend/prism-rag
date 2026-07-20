"""Visual 按需路由单测（无模型 / 无 PG）。"""

from src.retrieval.visual_router import VisualRouter, build_visual_router_from_config


def test_heuristic_skips_plain_definitional_query():
    r = VisualRouter(mode="heuristic")
    assert r.should_use_visual("What is the definition of hydraulic pressure?") is False


def test_heuristic_enables_table_or_figure_query():
    r = VisualRouter(mode="heuristic")
    assert r.should_use_visual("According to the table, what is the max torque?") is True
    assert r.should_use_visual("In the diagram, which port is the inlet?") is True
    assert r.should_use_visual("See figure 3 for the wiring.") is True


def test_always_and_never_modes():
    assert VisualRouter(mode="always").should_use_visual("hello") is True
    assert VisualRouter(mode="never").should_use_visual("see the figure") is False


def test_invalid_mode_raises():
    try:
        VisualRouter(mode="maybe")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_from_config_disabled():
    def get(path, default=None):
        if path == "retrieval.visual_routing.enabled":
            return False
        return default

    assert build_visual_router_from_config(get) is None


def test_build_from_config_enabled_heuristic():
    def get(path, default=None):
        if path == "retrieval.visual_routing.enabled":
            return True
        if path == "retrieval.visual_routing.mode":
            return "heuristic"
        return default

    r = build_visual_router_from_config(get)
    assert r is not None
    assert r.mode == "heuristic"
