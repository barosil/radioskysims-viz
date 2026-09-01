"""\
Módulo: packages/radioskysims-viz/src/radioskysims_viz/satellite_rinex.py

Descrição:
    Visualizações de alto nível para satélites criados a partir de RINEX
    (NAV broadcast). Recebem o dataset de tracks do ``gnsstools4astro``
    (coords ``sv``/``time``; vars ``alt_deg/az_deg/ra_deg/dec_deg``):
    - ``rinex_tracks_mollweide``: RA/Dec dos satélites no mapa Mollweide
      (com continentes), uma cor por satélite;
    - ``rinex_tracks_skychart``: carta local ALT/AZ (polar) dos satélites
      visíveis acima do horizonte.

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .sources import _mollweide_figure, _wrap_ra

__all__ = ["rinex_signals_plot", "rinex_tracks_mollweide", "rinex_tracks_skychart"]


def rinex_signals_plot(
    ds: Any,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 5.0),
    cmap: str = "viridis",
) -> Any:
    """Espectros de sinais RINEX (PSD dBW/Hz × MHz) — uma curva por satélite.

    As PSDs são **analíticas** (closed-form, mesmo código do TLE) avaliadas no
    grid de frequências: cada satélite aparece como uma curva contínua com
    picos nas portadoras da constelação (ex.: GPS L5/L2C dentro do FR) e piso
    baixo fora delas — sem artefatos de banda larga.

    Args:
        ds: Dataset com coords ``sv``/``freq`` (MHz), ``constellation`` e
            variável ``PSD`` (dBW/Hz, já com piso numérico).

    Returns:
        matplotlib.Figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel("frequência (MHz)")
    ax.set_ylabel("PSD (dBW/Hz)")
    if (
        "sv" not in ds.coords
        or ds.sizes.get("sv", 0) == 0
        or ds.sizes.get("freq", 0) == 0
    ):
        ax.set_title(title or "RINEX satellites — PSD analítica")
        fig.tight_layout()
        return fig

    freqs = np.asarray(ds["freq"].values, dtype=float)
    svs = list(ds["sv"].values)
    cons = (
        list(ds["constellation"].values)
        if "constellation" in ds.coords
        else [""] * len(svs)
    )

    cm = plt.get_cmap(cmap)
    unique_cons: list[str] = []
    for c in cons:
        if c not in unique_cons:
            unique_cons.append(c)
    cons_color = {
        c: cm(i / max(len(unique_cons) - 1, 1)) for i, c in enumerate(unique_cons)
    }

    for i, sv in enumerate(svs):
        psd = np.asarray(ds["PSD"].sel(sv=sv).values, dtype=float)
        color = cons_color.get(cons[i], cm(0.5))
        lbl = cons[i] if cons[i] else str(sv)
        ax.plot(freqs, psd, "-", color=color, lw=1.1, label=lbl)

    ax.set_title(title or "RINEX satellites — PSD analítica")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=6, ncol=3)
    fig.tight_layout()
    return fig


def rinex_tracks_mollweide(
    ds: Any,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (9.0, 5.0),
    cmap: str = "viridis",
    coastlines: bool = False,
) -> Any:
    """RA/Dec dos satélites RINEX no mapa Mollweide.

    Coordenadas CELESTES (RA/Dec) — por padrão SEM continentes geográficos.

    Args:
        ds: Dataset com coords ``sv``/``time`` e vars ``ra_deg``/``dec_deg``.

    Returns:
        matplotlib.Figure.
    """
    import matplotlib.pyplot as plt

    if "sv" not in ds.coords or ds.sizes.get("sv", 0) == 0:
        fig, ax = _mollweide_figure(
            title or "RINEX satellites — Mollweide (RA/Dec)",
            figsize,
            coastlines=coastlines,
        )
        fig.tight_layout()
        return fig

    fig, ax = _mollweide_figure(
        title or "RINEX satellites — Mollweide (RA/Dec)",
        figsize,
        coastlines=coastlines,
    )
    svs = list(ds["sv"].values)
    cm = plt.get_cmap(cmap)
    n = len(svs)
    drew = False
    for i, sv in enumerate(svs):
        ra = np.asarray(ds["ra_deg"].sel(sv=sv).values, dtype=float)
        dec = np.asarray(ds["dec_deg"].sel(sv=sv).values, dtype=float)
        mask = ~(np.isnan(ra) | np.isnan(dec))
        if not mask.any():
            continue
        color = cm(i / max(n - 1, 1))
        ax.plot(
            _wrap_ra(np.radians(ra[mask])),
            np.radians(dec[mask]),
            "-",
            color=color,
            lw=1.2,
            alpha=0.85,
            label=str(sv),
        )
        drew = True
    if drew:
        ax.legend(loc="upper left", fontsize=6, ncol=2)
    fig.tight_layout()
    return fig


def rinex_tracks_skychart(
    ds: Any,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (7.0, 7.0),
    cmap: str = "viridis",
) -> Any:
    """Carta local ALT/AZ (polar) dos satélites RINEX visíveis.

    Args:
        ds: Dataset com coords ``sv``/``time`` e vars ``az_deg``/``alt_deg``.

    Returns:
        matplotlib.Figure.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0.0, 90.0)
    ax.set_rticks([0, 30, 60, 90])
    ax.set_yticklabels(["90°", "60°", "30°", "0° (horizonte)"], fontsize=7)

    if "sv" not in ds.coords or ds.sizes.get("sv", 0) == 0:
        ax.set_title(title or "RINEX satellites — sky chart (ALT/AZ)")
        fig.tight_layout()
        return fig

    svs = list(ds["sv"].values)
    cm = plt.get_cmap(cmap)
    n = len(svs)
    drew = False
    for i, sv in enumerate(svs):
        az = np.asarray(ds["az_deg"].sel(sv=sv).values, dtype=float)
        alt = np.asarray(ds["alt_deg"].sel(sv=sv).values, dtype=float)
        mask = (alt > 0.0) & ~(np.isnan(az) | np.isnan(alt))
        if not mask.any():
            continue
        color = cm(i / max(n - 1, 1))
        ax.plot(
            np.radians(az[mask]),
            90.0 - alt[mask],
            "-",
            color=color,
            lw=1.2,
            alpha=0.85,
            label=str(sv),
        )
        drew = True
    ax.set_title(title or "RINEX satellites — sky chart (ALT/AZ)")
    if drew:
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0), fontsize=6, ncol=2)
    fig.tight_layout()
    return fig
