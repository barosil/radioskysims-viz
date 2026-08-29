"""\
Módulo: packages/radioskysims-viz/src/radioskysims_viz/ports.py

Descrição:
    Port de visualização — ``VisualizerPort``. Contrato para que a camada
    de aplicação não dependa de um framework de UI concreto (Panel, Bokeh,
    Matplotlib, ...). Implementações concretas em ``panel_visualizer``.

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VisualizerPort(ABC):
    """Contrato para visualização de dados/painéis."""

    @abstractmethod
    def visualize(
        self,
        *,
        serve: bool = False,
        title: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Renderiza (ou serve) a visualização.

        Args:
            serve: se True, sobe o servidor interativo; senão retorna o
                objeto renderizável.
            title: título da aplicação.

        Returns:
            Objeto renderizável ou None quando ``serve=True``.
        """
        raise NotImplementedError
