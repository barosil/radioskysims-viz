"""\
Módulo: packages/radioskysims-viz/src/radioskysims_viz/panel_application.py

Descrição:
    Classe base ``PanelApplication`` sintetizada a partir dos padrões comuns
    dos painéis do monólito (beam_panel, gain_panel, radiotelescope_panel,
    tod_panel e o dashboard MVC de interfaces/gui/skymap).

    Define o ciclo de vida padrão de uma aplicação Panel:
    ``build_sidebar()`` + ``build_main()`` e ``serve()``, com layout
    padrão (sidebar à esquerda + área principal).

Copyright BINGO Collaboration
Last modified: 2026-08-29
"""

from __future__ import annotations

from typing import Any

import panel as pn


class PanelApplication(pn.viewable.Viewer):
    """Aplicação Panel com ciclo de vida padronizado.

    Subclasses implementam :meth:`build_sidebar` e :meth:`build_main`.
    O layout resultante é exposto via ``__panel__`` (padrão Panel Viewer)
    e a aplicação pode ser servida com :meth:`serve`.

    Exemplo::

        class MeuPainel(PanelApplication):
            def build_sidebar(self):
                self.slider = pn.widgets.FloatSlider(name="x", start=0, end=10, value=5)
                return pn.Column(self.slider)

            def build_main(self):
                return pn.pane.Markdown("Olá")
    """

    def __init__(
        self,
        *,
        sidebar_width: int = 320,
        layout: str = "row",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sidebar_width = sidebar_width
        self.sidebar = self.build_sidebar()
        self.main = self.build_main()
        if layout == "row":
            self.layout = pn.Row(
                pn.Column(self.sidebar, width=sidebar_width),
                pn.Column(self.main, sizing_mode="stretch_width"),
            )
        else:
            self.layout = pn.Column(self.sidebar, self.main)

    # ── ciclo de vida ────────────────────────────────────────────
    def build_sidebar(self) -> Any:
        """Constrói os controles/parâmetros da sidebar."""
        raise NotImplementedError

    def build_main(self) -> Any:
        """Constrói a área principal (visualização, tabelas, etc.)."""
        raise NotImplementedError

    def __panel__(self) -> Any:
        """Layout renderizável pelo Panel (padrão Viewer)."""
        return self.layout

    def serve(
        self,
        *,
        title: str | None = None,
        show: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Serve a aplicação com ``pn.serve``."""
        return pn.serve(self, title=title, show=show, **kwargs)

    def visualize(self, *, serve: bool = False, title: str | None = None) -> Any:
        """Retorna o painel renderizável ou serve a aplicação."""
        if serve:
            return self.serve(title=title, show=True)
        return self.__panel__()
