"""\
Módulo: packages/radioskysims-viz/src/radioskysims_viz/sources.py

Descrição:
    Visualizações de alto nível para fontes de rádio e fontes locais (Sol)
    — duck-typing, sem importar tipos de domínio do pacote sources:
    - ``radio_sources_mollweide``: mapa RA/Dec (Mollweide + continentes),
      tamanho/cor por fluxo;
    - ``radio_sources_flux``: curvas de fluxo (Jy) × frequência (MHz),
      uma linha por fonte + soma;
    - ``sun_skychart``: carta local ALT/AZ (polar) do Sol;
    - ``sun_mollweide``: RA/Dec do Sol no mapa Mollweide;
    - ``sun_flux_density``: fluxo solar × frequência.

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .satellite import _draw_coastlines

__all__ = [
    "radio_sources_flux",
    "radio_sources_mollweide",
    "sun_flux_density",
    "sun_mollweide",
    "sun_skychart",
]


def _wrap_ra(ra_rad: np.ndarray) -> np.ndarray:
    """Envolve RA para [-π, π] (projeção Mollweide do matplotlib)."""
    return (np.asarray(ra_rad, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _mollweide_figure(
    title: str,
    figsize: tuple[float, float],
    *,
    coastlines: bool = True,
) -> tuple[Any, Any]:
    """Cria figura Mollweide com grade e contornos dos continentes."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True, linestyle="--", alpha=0.4)
    if coastlines:
        _draw_coastlines(ax)
    ax.set_title(title)
    return fig, ax


# ======================================================================
# Radio sources
# ======================================================================
def radio_sources_mollweide(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    fluxes_jy: np.ndarray | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (9.0, 5.0),
    coastlines: bool = False,
) -> Any:
    """Mapa Mollweide RA/Dec das fontes de rádio.

    Coordenadas CELESTES (RA/Dec) — por padrão SEM continentes geográficos.

    Args:
        ra_deg, dec_deg: Ascensão reta e declinação em GRAUS.
        fluxes_jy: Fluxo de referência (Jy) — controla tamanho e cor.
        coastlines: Se True, sobrepõe continentes terrestres.

    Returns:
        matplotlib.Figure.
    """
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    fig, ax = _mollweide_figure(
        title or "Radio sources — Mollweide (RA/Dec)", figsize, coastlines=coastlines
    )

    x = _wrap_ra(np.radians(ra))
    y = np.radians(dec)

    if fluxes_jy is None or len(ra) == 0:
        ax.scatter(x, y, s=20, c="#333333", alpha=0.9)
        fig.tight_layout()
        return fig

    flux = np.asarray(fluxes_jy, dtype=float)
    fmax = float(flux.max()) if flux.size and flux.max() > 0 else 1.0
    sizes = 80.0 * (flux / fmax) + 8.0
    sc = ax.scatter(x, y, s=sizes, c=flux, cmap=cmap, alpha=0.85)
    fig.colorbar(
        sc,
        ax=ax,
        orientation="horizontal",
        pad=0.12,
        shrink=0.8,
        label="fluxo S0 (Jy)",
    )
    fig.subplots_adjust(bottom=0.16)
    return fig


def radio_sources_flux(
    freqs_mhz: np.ndarray,
    flux_by_source: dict[str, np.ndarray],
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 5.0),
    loglog: bool = True,
    sum_line: bool = True,
    cmap: str = "viridis",
    n_legend: int = 8,
) -> Any:
    """Curvas de fluxo (Jy) × frequência (MHz) — uma linha por fonte + soma.

    Args:
        freqs_mhz: Grade de frequências em MHz.
        flux_by_source: Dict {id da fonte: array de fluxo (Jy)}.

    Returns:
        matplotlib.Figure.
    """
    import matplotlib.pyplot as plt

    freqs = np.asarray(freqs_mhz, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)

    vals = np.concatenate(list(flux_by_source.values())) if flux_by_source else np.array([1.0])
    use_log = bool(loglog) and bool(np.all(vals > 0))
    if use_log:
        ax.set_xscale("log")
        ax.set_yscale("log")

    cm = plt.get_cmap(cmap)
    n = len(flux_by_source)
    for i, (sid, f) in enumerate(flux_by_source.items()):
        f = np.asarray(f, dtype=float)
        lbl = sid if i < n_legend else None
        ax.plot(freqs, f, color=cm(i / max(n - 1, 1)), lw=1.2, alpha=0.85, label=lbl)

    if sum_line and flux_by_source:
        total = np.sum(np.stack(list(flux_by_source.values())), axis=0)
        ax.plot(freqs, total, color="k", lw=2.2, ls="--", label="soma")

    ax.set_xlabel("frequência (MHz)")
    ax.set_ylabel("fluxo (Jy)")
    ax.set_title(title or "Flux density — radio sources")
    ax.grid(True, which="both", alpha=0.3)
    if len(flux_by_source) > 1 or sum_line:
        ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


# ======================================================================
# Local source (Sun)
# ======================================================================
def sun_skychart(
    az_rad: np.ndarray,
    el_rad: np.ndarray,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (7.0, 7.0),
) -> Any:
    """Carta do céu local do Sol em ALT/AZ (projeção polar).

    Args:
        az_rad: Azimute em radianos (0 = Norte, sentido horário).
        el_rad: Elevação em radianos.

    Returns:
        matplotlib.Figure.
    """
    import matplotlib.pyplot as plt

    az = np.asarray(az_rad, dtype=float)
    el = np.asarray(el_rad, dtype=float)
    zenith = np.pi / 2.0 - el  # 0 no zênite, π/2 no horizonte

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0.0, np.pi / 2.0)
    ax.set_rticks([0.0, np.pi / 6.0, np.pi / 3.0, np.pi / 2.0])
    ax.set_yticklabels(["90°", "60°", "30°", "0° (horizonte)"], fontsize=7)
    ax.plot(az, zenith, "-", color="#ff8c00", lw=2.0, alpha=0.9)
    ax.scatter(az, zenith, s=26, color="#ff8c00", zorder=5)
    ax.set_title(title or "Sun — local sky chart (ALT/AZ)")
    fig.tight_layout()
    return fig


def sun_mollweide(
    ra_rad: np.ndarray,
    dec_rad: np.ndarray,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (9.0, 5.0),
    coastlines: bool = False,
) -> Any:
    """RA/Dec do Sol no mapa Mollweide.

    Coordenadas CELESTES (RA/Dec) — por padrão SEM continentes geográficos.

    Args:
        ra_rad, dec_rad: Ascensão reta e declinação em RADIANOS.
        coastlines: Se True, sobrepõe continentes terrestres (não recomendado
            em mapa de céu).

    Returns:
        matplotlib.Figure.
    """
    ra = np.asarray(ra_rad, dtype=float)
    dec = np.asarray(dec_rad, dtype=float)
    fig, ax = _mollweide_figure(title or "Sun — Mollweide (RA/Dec)", figsize, coastlines=coastlines)
    x = _wrap_ra(ra)
    ax.plot(x, dec, "-", color="#ff8c00", lw=1.6, alpha=0.8)
    ax.scatter(x, dec, s=30, color="#ff8c00", zorder=5, label="Sun")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def sun_flux_density(
    freqs_mhz: np.ndarray,
    flux_jy: np.ndarray,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 5.0),
) -> Any:
    """Fluxo solar (Jy) × frequência (MHz)."""
    import matplotlib.pyplot as plt

    freqs = np.asarray(freqs_mhz, dtype=float)
    flux = np.asarray(flux_jy, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(freqs, flux, "-o", color="#ff8c00", lw=1.8, ms=4)
    ax.set_xlabel("frequência (MHz)")
    ax.set_ylabel("fluxo (Jy)")
    ax.set_title(title or "Sun — flux density")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
