"""\
Módulo: packages/radioskysims-viz/src/radioskysims_viz/panel_visualizer.py

Descrição:
    Adapter concreto do ``VisualizerPort`` baseado em Panel. Resolve o
    acoplamento "Panel hardcoded em application" identificado na auditoria:
    a aplicação agora consome o port e pode trocar a UI sem mudar a lógica.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from typing import Any

from .panel_application import PanelApplication
from .ports import VisualizerPort


class PanelVisualizer(VisualizerPort):
    """Adapter de visualização que delega para um ``PanelApplication``."""

    def __init__(self, application: PanelApplication) -> None:
        self.application = application

    def visualize(
        self,
        *,
        serve: bool = False,
        title: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if serve:
            return self.application.serve(title=title, show=True, **kwargs)
        return self.application.__panel__()
