"""\
Módulo: packages/radioskysims-viz/src/radioskysims_viz/satellite.py

Descrição:
    Visualizações de alto nível para satélites (duck-typing — não importa
    tipos de domínio do pacote sources):
    - ``tracks_mollweide``: órbitas-terra (sub-satellite points) em projeção
      Mollweide (matplotlib), com tracks por constelação, contornos de
      densidade de linhas (KernelDensity do sklearn, métrica esférica
      haversine, 5 níveis), contornos dos continentes (cartopy) e colorbar
      horizontal; ``kde=True`` troca por heatmap de densidade;
    - ``signals_plot``: espectros PSD (dB) × frequência (MHz), uma curva por
      sinal (corrige linhas espúrias por agrupamento indevido);

    Os builders recebem datasets xarray com as coordenadas esperadas:
    - tracks: ``satellite``, ``time`` (TT JD), ``names``, ``tle_line1``,
      ``tle_line2``, e variável ``positions`` (satellite, time, xyz em km);
    - signals: ``satellite`` (NORAD repetido por sinal), ``freq`` (MHz),
      ``fc`` e ``modulation`` (coords), variável ``PSD``.

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["signals_plot", "tracks_mollweide"]


# ======================================================================
# Sub-satellite points (ground track)
# ======================================================================
def _subpoint_latlon(ds: Any) -> tuple[np.ndarray, np.ndarray]:
    """Lat/lon (graus) do sub-satellite point a partir do dataset de tracks.

    Reconstrói o ``EarthSatellite`` via skyfield (das linhas TLE nas coords)
    e converte a posição para o frame terrestre (ITRS/ECEF), dando
    lat/lon do ponto na superfície da Terra sob o satélite.

    Returns:
        tuple (lat, lon): arrays (satellite, time) em graus.
    """
    from skyfield.api import EarthSatellite, load
    from skyfield.framelib import itrs

    ts = load.timescale()
    tt_jd = np.asarray(ds.time.values, dtype=float)
    times = ts.tt_jd(tt_jd)

    has_names = "names" in ds.coords
    lats: list[np.ndarray] = []
    lons: list[np.ndarray] = []
    for i in range(ds.sizes["satellite"]):
        tle1 = str(ds.tle_line1.values[i]).strip()
        tle2 = str(ds.tle_line2.values[i]).strip()
        name = (
            str(ds.names.values[i])
            if has_names
            else str(ds.satellite.values[i])
        )
        sat = EarthSatellite(tle1, tle2, name, ts)
        ecef = sat.at(times).frame_xyz(itrs).m  # (3, N)
        x, y, z = ecef[0], ecef[1], ecef[2]
        lats.append(np.degrees(np.arctan2(z, np.sqrt(x * x + y * y))))
        lons.append(np.degrees(np.arctan2(y, x)))
    return np.stack(lats, axis=0), np.stack(lons, axis=0)


def _smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Suavização por média móvel centrada (roll) para remover ruído.

    Returns:
        Array 2D ``(satellite, time)`` (aceita entrada 1D também).
    """
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    pad = window // 2
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        padded = np.pad(arr[i], pad, mode="wrap")
        kernel = np.ones(window) / window
        out[i] = np.convolve(padded, kernel, mode="same")[pad:-pad]
    return out


def _segment_lon(
    lon: np.ndarray, lat: np.ndarray, max_gap_deg: float = 180.0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Quebra a track onde a longitude dá o salto de 180° (evita linha
    horizontal espúria atravessando o mapa)."""
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    dlon = np.abs(np.diff(lon, prepend=lon[0]))
    breaks = np.where(dlon > max_gap_deg)[0]
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(lon)]])
    for s, e in zip(starts, ends, strict=False):
        if e - s >= 2:
            segments.append((lon[s:e], lat[s:e]))
    return segments


def _draw_coastlines(
    ax: Any,
    color: str = "#555555",
    linewidth: float = 0.6,
    alpha: float = 0.85,
    zorder: float = 3.0,
) -> bool:
    """Desenha os contornos dos continentes (Natural Earth 110m) no eixo.

    Usa ``cartopy.feature.COASTLINE`` apenas para extrair as geometrias de
    costa (lon/lat em graus) e as plota como ``LineCollection`` em radianos,
    compatível com o eixo Mollweide do matplotlib. Segmentos que cruzam o
    antimeridiano são quebrados para evitar linhas espúrias. Falhas de
    download/dados são ignoradas silenciosamente.

    Returns:
        True se algum contorno foi desenhado, False caso contrário.
    """
    from matplotlib.collections import LineCollection

    try:
        from cartopy import feature as cfeature
    except Exception:
        return False

    try:
        geoms = cfeature.COASTLINE.geometries()
    except Exception:
        return False

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for geom in geoms:
        lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for line in lines:
            lon = np.asarray(line.xy[0], dtype=float)
            lat = np.asarray(line.xy[1], dtype=float)
            segments.extend(_segment_lon(lon, lat))
    if not segments:
        return False

    lc = LineCollection(
        [
            np.column_stack([np.radians(slon), np.radians(slat)])
            for slon, slat in segments
        ],
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_collection(lc)
    return True


def _scott_bandwidth(xy: np.ndarray) -> float:
    """Bandwidth isotrópico de Scott para o KernelDensity 2D (em graus).

    ``h = n^(-1/(d+4)) · (σ_x · σ_y)^(1/2)`` com ``d=2`` (mesma regra do
    ``scipy.stats.gaussian_kde``). Dimensões com desvio nulo (dados
    degenerados) são elevadas a 1° para evitar divisão por zero.
    """
    n = len(xy)
    std = np.std(xy, axis=0)
    std = np.where(std > 0, std, 1.0)
    geom = float(np.exp(np.mean(np.log(std))))
    return float(n ** (-1.0 / 6.0)) * geom


class _Kde2D:
    """KDE 2D (sklearn) com métrica plana ou esférica.

    Recebe lon/lat em GRAUS. Com ``metric="haversine"`` (ESFÉRICA, padrão),
    dados e ``bandwidth`` são convertidos para radianos — a distância é
    medida sobre a esfera terrestre, sem artefatos no antimeridiano/polos.
    """

    def __init__(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        bandwidth: float | None = None,
        metric: str = "haversine",
    ) -> None:
        from sklearn.neighbors import KernelDensity

        self._metric = metric
        mask = ~(np.isnan(lon) | np.isnan(lat))
        xy_deg = np.stack(
            [np.asarray(lon)[mask], np.asarray(lat)[mask]], axis=-1
        )
        if len(xy_deg) < 5:
            self._kde = None
            return
        if bandwidth is None:
            bandwidth = _scott_bandwidth(xy_deg)
        if metric == "haversine":
            self._bw = np.radians(bandwidth)
            self._xy = np.radians(xy_deg)
        else:
            self._bw = float(bandwidth)
            self._xy = xy_deg
        self._kde = KernelDensity(
            bandwidth=self._bw, kernel="gaussian", metric=metric
        )
        self._kde.fit(self._xy)

    def __bool__(self) -> bool:
        """False se não há pontos suficientes (sem estimador ajustado)."""
        return self._kde is not None

    def score_grid(self, lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
        """Log-densidade na grade (lon/lat em graus), shape da grade."""
        lon_g = np.asarray(lon_grid)
        lat_g = np.asarray(lat_grid)
        grid = np.stack([lon_g.ravel(), lat_g.ravel()], axis=-1)
        if self._metric == "haversine":
            grid = np.radians(grid)
        return self._kde.score_samples(grid).reshape(lon_g.shape)

    def score_points(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Densidade linear nos pontos dados (lon/lat em graus)."""
        pts = np.stack(
            [np.asarray(lon).ravel(), np.asarray(lat).ravel()], axis=-1
        )
        if self._metric == "haversine":
            pts = np.radians(pts)
        return np.exp(self._kde.score_samples(pts))


def _density_contours(
    ax: Any,
    lon: np.ndarray,
    lat: np.ndarray,
    n_levels: int = 5,
    cmap: str = "viridis",
    alpha: float = 0.35,
    bandwidth: float | None = None,
    metric: str = "haversine",
) -> Any:
    """Contornos de densidade KDE 2D sobre TODOS os tracks (global).

    A densidade de linhas é calculada com TODOS os pontos de todas as
    constelações juntos (independente do agrupamento), via
    ``sklearn.neighbors.KernelDensity``. A métrica é ESFÉRICA (haversine)
    por padrão — distância sobre a esfera, sem artefatos no antimeridiano.
    A densidade é normalizada com pico = 1 e os ``n_levels`` níveis são
    bandas log-espaçadas entre 5% e 100% do pico.

    Args:
        bandwidth: largura de banda do kernel (graus). ``None`` → regra de
            Scott automática.
        metric: ``"haversine"`` (esférica, padrão) ou ``"euclidean"``.

    Returns:
        QuadContourSet (mappable) para a colorbar, ou ``None`` se não
        houver pontos suficientes.
    """
    k = _Kde2D(lon, lat, bandwidth=bandwidth, metric=metric)
    if not k:
        return None

    grid_lon = np.linspace(-160.0, 160.0, 180)
    grid_lat = np.linspace(-75.0, 75.0, 120)
    lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)
    log_density = k.score_grid(lon_grid, lat_grid)
    # normaliza o pico para 1 → níveis como frações do pico (5% a 100%)
    z = np.exp(log_density - log_density.max())
    levels = np.geomspace(0.05, 1.0, max(int(n_levels), 1) + 1)
    levels = np.unique(levels)  # níveis estritamente crescentes
    if levels.size < 2:
        return None

    cs = ax.contourf(
        np.radians(lon_grid),
        np.radians(lat_grid),
        z,
        levels=levels,
        cmap=cmap,
        alpha=alpha,
    )
    ax.contour(
        np.radians(lon_grid),
        np.radians(lat_grid),
        z,
        levels=levels,
        colors="k",
        linewidths=0.5,
        alpha=0.4,
    )
    return cs


def tracks_mollweide(
    ds: Any,
    smooth: bool = True,
    kde: bool = False,
    tracks: bool = True,
    by: str = "constellation",
    contours: bool = True,
    n_levels: int = 5,
    figsize: tuple[float, float] = (9.0, 5.0),
    title: str | None = None,
    cmap: str = "viridis",
    bandwidth: float | None = None,
    metric: str = "haversine",
    coastlines: bool = True,
) -> Any:
    """Vista Mollweide das órbitas-terra (sub-satellite points).

    - ``smooth=True``: suaviza a track (média móvel) antes de plotar;
    - ``tracks=False``: não plota as tracks individuais — útil para ver
      apenas os contornos de densidade;
    - ``by="constellation"``: agrupa e colora as tracks por constelação
      (ex.: GPS, GLONASS) com legenda única; ``by="satellite"`` por satélite;
    - ``contours=True``: desenha contornos de densidade KDE 2D sobre TODOS
      os tracks (independente da constelação), com ``n_levels`` níveis e uma
      colorbar horizontal inferior ("densidade de linhas");
    - ``kde=True``: plota heatmap 2D da densidade de passagens em vez das
      tracks individuais (sobrepõe ``contours`` e ``tracks``);
    - ``bandwidth``: largura de banda do kernel KDE (graus). ``None`` → regra
      de Scott automática (``sklearn.neighbors.KernelDensity``);
    - ``metric``: métrica do KDE — ``"haversine"`` (esférica, padrão) ou
      ``"euclidean"`` (plana em graus);
    - ``coastlines=True``: desenha os contornos dos continentes (Natural
      Earth 110m, via cartopy) sobre o mapa.

    Returns:
        matplotlib.Figure (backend padrão do ambiente).
    """
    import matplotlib.pyplot as plt

    lat, lon = _subpoint_latlon(ds)
    if smooth:
        lat = np.asarray(_smooth(lat))
        lon = np.asarray(_smooth(lon))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True, linestyle="--", alpha=0.4)
    if coastlines:
        _draw_coastlines(ax)

    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    has_const = "constellation" in ds.coords
    use_const = by == "constellation" and has_const
    cm = plt.get_cmap(cmap)

    if kde:
        lon_flat = lon.ravel()
        lat_flat = lat.ravel()
        mask = ~(np.isnan(lon_flat) | np.isnan(lat_flat))
        k = _Kde2D(lon_flat, lat_flat, bandwidth=bandwidth, metric=metric)
        if k:
            density = k.score_points(lon_flat[mask], lat_flat[mask])
        else:
            density = np.ones(int(mask.sum()))
        ax.scatter(
            lon_rad.ravel()[mask],
            lat_rad.ravel()[mask],
            c=density,
            s=6,
            cmap=cm,
            alpha=0.85,
        )
        ax.set_title(title or "Satellite pass density (KDE) — Mollweide")
        fig.colorbar(ax.collections[0], ax=ax, label="density", shrink=0.8)
        fig.tight_layout()
        return fig

    # Contornos de densidade GLOBAIS — todos os tracks juntos, independente
    # da constelação (a densidade de linhas não é por grupo).
    density = None
    if contours and not kde:
        density = _density_contours(
            ax, lon, lat, n_levels=n_levels, cmap=cmap,
            bandwidth=bandwidth, metric=metric,
        )

    # Agrupamento (por constelação ou por satélite)
    if use_const:
        const_vals = np.asarray(ds.constellation.values, dtype=str)
        labels = sorted({str(v) for v in const_vals})
        groups = [(label, np.where(const_vals == label)[0]) for label in labels]
        legend_title = "constelação"
    else:
        names = (
            np.asarray(ds.names.values, dtype=str)
            if "names" in ds.coords
            else np.asarray(ds.satellite.values, dtype=str)
        )
        groups = [(str(names[i]), [i]) for i in range(lon.shape[0])]
        legend_title = "satélite"

    n_groups = len(groups)
    for gi, (label, idxs) in enumerate(groups):
        color = cm(gi / max(n_groups - 1, 1))
        drew = False

        if tracks:
            for i in idxs:
                for seg_lon, seg_lat in _segment_lon(lon[i], lat[i]):
                    ax.plot(
                        np.radians(seg_lon),
                        np.radians(seg_lat),
                        "-",
                        color=color,
                        linewidth=1.4,
                    )
            drew = True

        # legenda por constelação mantida mesmo sem tracks
        if use_const or drew:
            ax.plot([], [], "-", color=color, label=label)  # proxy p/ legenda

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        by_label = dict(zip(labels, handles, strict=False))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
            fontsize=8,
            ncol=min(4, len(by_label)),
            title=legend_title,
        )
    default_title = (
        "Satellite ground tracks + line density — Mollweide"
        if use_const
        else "Satellite ground tracks + line density — Mollweide"
    )
    ax.set_title(title or default_title)

    if density is not None:
        # colorbar horizontal na parte inferior da figura
        fig.colorbar(
            density,
            ax=ax,
            orientation="horizontal",
            pad=0.12,
            shrink=0.8,
            label="densidade de linhas",
        )
        fig.subplots_adjust(bottom=0.16)
    else:
        fig.tight_layout()
    return fig


# ======================================================================
# Signals
# ======================================================================
def signals_plot(
    signals_ds: Any,
    *,
    norad_names: dict[int, str] | None = None,
    title: str = "PSD (dB) — sinais",
    width: int = 900,
    height: int = 450,
) -> Any:
    """Vista holoviews dos espectros PSD (dB) × frequência (MHz).

    Uma curva por sinal (satélite × portadora × modulação) — corrige as
    linhas espúrias causadas por agrupar por satélite quando há múltiplos
    sinais (L1/L2/L5) por satélite.

    Returns:
        holoviews.Overlay (backend bokeh).
    """
    import holoviews as hv

    hv.extension("bokeh")

    ds = signals_ds
    sat_ids = np.asarray(ds.satellite.values)
    freq = np.asarray(ds["freq"].values)
    fc = (
        np.asarray(ds["fc"].values)
        if "fc" in ds.coords
        else np.full(sat_ids.shape, np.nan)
    )
    mod = (
        np.asarray(ds["modulation"].values, dtype=str)
        if "modulation" in ds.coords
        else np.full(sat_ids.shape, "")
    )
    psd = ds["PSD"].compute().values

    curves: list[Any] = []
    for i in range(ds.sizes["satellite"]):
        name = (norad_names or {}).get(int(sat_ids[i]), str(int(sat_ids[i])))
        label = f"{name} | {mod[i]} @ {fc[i]:.1f} MHz"
        curves.append(
            hv.Curve(
                (freq, psd[i]),
                kdims=["freq (MHz)"],
                vdims=["PSD (dB/Hz)"],
                label=label,
            ).opts(line_width=1.6)
        )

    overlay = hv.Overlay(curves).opts(
        title=title,
        width=width,
        height=height,
        legend_position="right",
        show_grid=True,
        fontsize={"title": 12},
    )
    return overlay
