"""
Pacote: radioskysims_viz

Base de visualização interativa do ecossistema RadioSkySim:
- panel_application.PanelApplication: classe base com ciclo de vida
  ``build_sidebar()`` / ``build_main()`` / ``serve()``.
- ports.VisualizerPort: contrato de visualização.
- panel_visualizer.PanelVisualizer: adapter concreto baseado em Panel.

Copyright BINGO Collaboration
"""

from __future__ import annotations

__version__ = "0.1.0"

from .panel_application import PanelApplication
from .panel_visualizer import PanelVisualizer
from .ports import VisualizerPort

__all__ = ["PanelApplication", "PanelVisualizer", "VisualizerPort"]
