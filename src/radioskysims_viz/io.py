"""\
Módulo: packages/radioskysims-viz/src/radioskysims_viz/io.py

Descrição:
    Utilitários de saída/exportação de visualizações (holoviews).
    ``save_plot`` salva elementos holoviews em HTML interativo (bokeh) ou
    em imagem (PNG/SVG) via backend matplotlib — sem exigir selenium.

Copyright BINGO Collaboration
Last modified: 2026-08-30
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import holoviews as hv
from radioskysims_core.paths import get_project_root

__all__ = ["save_figure", "save_plot"]

_VALID_FORMATS = frozenset({"html", "png", "svg", "pdf"})


def _resolve_outdir(outdir: str | Path | None) -> Path:
    """Resolve o diretório de saída de forma independente do CWD.

    - ``None`` → ``<raiz>/notebooks/plots`` (default p/ outputs de notebook);
    - caminho absoluto → como está;
    - relativo → ancorado na raiz do projeto (:func:`get_project_root`).
    """
    if outdir is None:
        return get_project_root() / "notebooks" / "plots"
    path = Path(outdir)
    if path.is_absolute():
        return path
    return get_project_root() / path


def save_figure(
    fig: Any,
    name: str,
    *,
    fmt: str = "png",
    dpi: int = 150,
    outdir: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """Salva uma ``matplotlib.Figure`` (ex.: Mollweide de tracks).

    Args:
        fig: Figure matplotlib.
        name: Nome base do arquivo (sem extensão).
        fmt: Formato — ``"png"``/``"svg"``/``"pdf"``.
        dpi: Resolução de saída.
        outdir: Pasta de saída. Default: ``<raiz>/notebooks/plots``; relativo
            resolve contra a raiz do projeto (independe do CWD).
        verbose: Se True, imprime o caminho do arquivo salvo.

    Returns:
        ``Path`` do arquivo salvo.
    """
    if fmt not in {"png", "svg", "pdf"}:
        msg = f"Formato não suportado para figure: {fmt!r}. Use png/svg/pdf."
        raise ValueError(msg)
    outdir = _resolve_outdir(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if verbose:
        print(f"salvo: {path.resolve()}")
    return path


def save_plot(
    plot: Any,
    name: str,
    *,
    fmt: str = "html",
    dpi: int = 150,
    outdir: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """Salva um plot holoviews em arquivo.

    Args:
        plot: Elemento holoviews (ex.: retorno de ``observer.view()`` ou
            ``get_properties(...).view()``).
        name: Nome base do arquivo (sem extensão).
        fmt: Formato de saída — ``"html"`` (padrão, interativo via bokeh),
            ou ``"png"``/``"svg"``/``"pdf"`` (via backend matplotlib, sem
            selenium).
        dpi: Resolução usada na exportação de imagem (matplotlib).
        outdir: Pasta de saída. Default: ``<raiz>/notebooks/plots``; relativo
            resolve contra a raiz do projeto (independe do CWD).
        verbose: Se True, imprime o caminho do arquivo salvo.

    Returns:
        ``Path`` do arquivo salvo.

    Raises:
        ValueError: Se ``fmt`` não for um formato suportado.

    Notes:
        - ``fmt="html"`` funciona para qualquer plot, inclusive o mapa OSM do
          observador (bokeh-only).
        - ``fmt`` de imagem exporta via matplotlib e não suporta o mapa OSM.
    """
    if fmt not in _VALID_FORMATS:
        msg = f"Formato não suportado: {fmt!r}. Use um de: {sorted(_VALID_FORMATS)}"
        raise ValueError(msg)

    outdir = _resolve_outdir(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.{fmt}"
    if fmt == "html":
        hv.save(plot, path, backend="bokeh")
    else:
        hv.save(plot, path, backend="matplotlib", dpi=dpi)
    if verbose:
        print(f"salvo: {path.resolve()}")
    return path
