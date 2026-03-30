"""Import-only checks for optional render backends (no display required)."""


def test_moderngl_3d_renderer_module_loads() -> None:
    from village_ai_war.rendering.moderngl_3d_renderer import Moderngl3DRenderer

    assert Moderngl3DRenderer.__name__ == "Moderngl3DRenderer"
