"""Testes da base de visualização (PanelApplication / VisualizerPort / PanelVisualizer)."""

from __future__ import annotations

import panel as pn
import pytest
from radioskysims_viz import PanelApplication, PanelVisualizer, VisualizerPort


class _DemoApp(PanelApplication):
    def build_sidebar(self):
        self.slider = pn.widgets.FloatSlider(name="x", start=0, end=10, value=5)
        return pn.Column(self.slider)

    def build_main(self):
        self.label = pn.pane.Markdown("demo")
        return pn.Column(self.label)


def test_panel_application_builds_layout() -> None:
    app = _DemoApp()
    assert hasattr(app, "sidebar")
    assert hasattr(app, "main")
    assert hasattr(app, "layout")


def test_panel_application_is_viewer() -> None:
    app = _DemoApp()
    assert isinstance(app, pn.viewable.Viewer)
    # __panel__ retorna o layout
    assert app.__panel__() is app.layout


def test_panel_application_abstract() -> None:
    class Incomplete(PanelApplication):
        pass

    with pytest.raises(NotImplementedError):
        Incomplete()


def test_visualizer_port_is_abstract() -> None:
    with pytest.raises(TypeError):
        VisualizerPort()  # type: ignore[abstract]


def test_panel_visualizer_wraps_application() -> None:
    app = _DemoApp()
    viz = PanelVisualizer(app)
    assert viz.visualize() is app.__panel__()
