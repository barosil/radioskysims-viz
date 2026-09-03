"""\
Módulo: packages/radioskysims-viz/src/radioskysims_viz/radiotelescope.py

Descrição:
    Visões interativas (holoviews/bokeh) para objetos de radiotelescópio:
    - ``observer_map``: mapa com tiles OSM e a localização do observador;
    - ``fwhm_plot``: FWHM × frequência (por antena);
    - ``gain_plot``: Ganho (dBi) × frequência (por antena);
    - ``beam_pattern_plot``: padrão de feixe (dB) × ângulo, para N
      frequências espaçadas na faixa da antena;
    - ``property_table``: tabela genérica antena → valor.

    Os builders recebem objetos de domínio por duck-typing (não importam
    tipos de core/radiotelescope), mantendo o pacote viz desacoplado.

Copyright BINGO Collaboration
Last modified: 2026-08-30
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import holoviews as hv
import numpy as np

# Carrega o backend bokeh para que ``.opts(...)`` funcione sem chamada manual
# de ``hv.extension`` (necessário para aplicar opções de plotagem).
hv.extension("bokeh")

__all__ = [
    "beam_pattern_plot",
    "fwhm_plot",
    "gain_plot",
    "observer_map",
    "property_table",
]

_SPEED_OF_LIGHT_M_S = 299_792_458.0
# Fator de ângulo sólido gaussiano: Omega ≈ 1.133 * theta**2
_GAUSSIAN_SOLID_ANGLE_FACTOR = 1.133

# Paleta fixa por antena (mantém linha e marcadores com a mesma cor).
_MARKER_COLORS = [
    "#30a2da",
    "#fc4f30",
    "#e5ae38",
    "#6d904f",
    "#8b8b8b",
    "#e47bd1",
    "#1c7e6a",
    "#dc143c",
]


def _to_value(x: Any, unit: str) -> float:
    """Converte astropy Quantity/Angle para float na unidade desejada."""
    if hasattr(x, "to_value"):
        return float(x.to_value(unit))
    return float(x)


def _materialize(x: Any) -> Any:
    """Computa objetos lazy (dask/xarray) antes do plot.

    O dataset de beam pode ser um array lazy (dask); ``.compute()``
    materializa os valores para conversão em listas/numpy.
    """
    compute = getattr(x, "compute", None)
    if callable(compute):
        return compute()
    return x


def _web_mercator(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    """Projeta lon/lat (EPSG:4326) para Web Mercator (EPSG:3857, metros).

    Os tiles OSM do holoviews/bokeh são renderizados em Web Mercator, então
    o ponto precisa ser projetado para ficar na posição correta do mapa.
    """
    r = 6378137.0
    x = r * math.radians(lon_deg)
    y = r * math.log(math.tan(math.pi / 4.0 + math.radians(lat_deg) / 2.0))
    return x, y


def _frequency_grid(frequency_range: Any, n: int = 17) -> list[float]:
    """Gera ``n`` frequências espaçadas na faixa (MHz) da antena."""
    if frequency_range is None:
        return []
    lo = float(frequency_range.min_mhz)
    hi = float(frequency_range.max_mhz)
    if n <= 1:
        return [lo]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def _load_beam(antenna: Any):
    """Resolve o ``Beam`` de uma antena (mesma lógica do orchestrator).

    - ``beam_pattern_file`` → ``CSVBeamLoader`` (resolução de caminho legada,
      incluindo o prefixo ``src/radioskysims``);
    - ``profile`` → ``AnalyticalBeamLoader`` (com N frequências na faixa).
    """
    from radioskysims_beam.infrastructure.beams.loaders.analytical import (
        AnalyticalBeamLoader,
        BeamProfile,
    )
    from radioskysims_beam.infrastructure.beams.loaders.csv import CSVBeamLoader

    beam_spec = getattr(antenna, "beam", None)
    if beam_spec is None:
        return None
    pattern_file = getattr(beam_spec, "beam_pattern_file", None)
    if pattern_file:
        return CSVBeamLoader(file=pattern_file).load()
    profile = getattr(beam_spec, "profile", None)
    if profile:
        fr = getattr(antenna, "frequency_range", None)
        if fr is not None:
            n = 5
            lo = float(fr.min_mhz) * 1e6
            hi = float(fr.max_mhz) * 1e6
            freqs_hz = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        else:
            freqs_hz = [1000.0e6]
        width = getattr(beam_spec, "beam_width_deg", None) or 10.0
        return AnalyticalBeamLoader(
            beam_profile=BeamProfile(profile),
            sigma=width,
            frequency=freqs_hz,
            angular_range=(-90.0, 90.0),
            angular_resolution=0.5,
        ).load()
    return None


def _ds_frequencies_mhz(beam: Any) -> list[float]:
    """Frequências do dataset do beam em MHz (converte Hz quando necessário)."""
    ds_freq = beam.dataset["frequency"]
    raw = [float(f) for f in _materialize(ds_freq.values)]
    units = str(ds_freq.attrs.get("units", "")).lower()
    if "hz" in units and "mhz" not in units:
        return [f / 1e6 for f in raw]
    if raw and min(raw) > 1e6:  # heurística: valores grandes = Hz
        return [f / 1e6 for f in raw]
    return raw


def _beam_frequencies(beam: Any, frequency_range: Any) -> list[float]:
    """Frequências (MHz) do beam, filtradas pela faixa da antena."""
    ds_freqs = _ds_frequencies_mhz(beam)
    if frequency_range is None:
        return ds_freqs
    lo = float(frequency_range.min_mhz)
    hi = float(frequency_range.max_mhz)
    inside = [f for f in ds_freqs if lo <= f <= hi]
    return inside or ds_freqs


def _beam_polarization(antenna: Any, beam: Any) -> str | None:
    """Polarização a usar: a 1ª pedida no beam spec, senão a 1ª do dataset."""
    beam_spec = getattr(antenna, "beam", None)
    requested = getattr(beam_spec, "polarizations", None) or []
    available = [str(p) for p in _materialize(beam.dataset["polarization"].values)]
    for pol in requested:
        if pol in available:
            return pol
    return available[0] if available else None


def _fwhm_vs_frequency(antenna: Any, beam: Any) -> dict[float, float]:
    """FWHM (deg) por frequência via ``BeamProperties.compute_properties``.

    O dataset do beam pode ser lazy (dask); ``compute_properties`` materializa
    ao acessar ``.values``. Devolve {freq_mhz: fwhm_deg} na faixa da antena.
    """
    from radioskysims_beam.domain.beam import BeamProperties

    ds_props = BeamProperties.compute_properties(beam.dataset)
    fwhm = _materialize(ds_props["fwhm"].values)  # shape (n_pol, n_freq)
    freqs_mhz = _ds_frequencies_mhz(beam)  # mesma ordem do dataset de props
    pols = [str(p) for p in _materialize(ds_props["polarization"].values)]
    pol = _beam_polarization(antenna, beam)
    i_pol = pols.index(pol) if pol in pols else 0
    wanted = set(_beam_frequencies(beam, getattr(antenna, "frequency_range", None)))
    return {f: float(fwhm[i_pol, j]) for j, f in enumerate(freqs_mhz) if f in wanted}


def _pick_beam_frequencies(
    beam: Any,
    frequency_range: Any,
    n_freqs: int,
) -> list[float]:
    """Escolhe ``n_freqs`` frequências espaçadas na faixa (mais próximas do beam)."""
    available = np.asarray(_beam_frequencies(beam, frequency_range))
    if available.size == 0:
        return []
    if n_freqs <= 1:
        return [float(available[0])]
    lo, hi = float(available.min()), float(available.max())
    targets = [lo + (hi - lo) * i / (n_freqs - 1) for i in range(n_freqs)]
    picked: list[float] = []
    for t in targets:
        idx = int(np.argmin(np.abs(available - t)))
        f = float(available[idx])
        if f not in picked:
            picked.append(f)
    return picked


def _beam_response_angle(
    beam: Any,
    freqs_mhz: list[float],
    angles_deg: Iterable[float],
    *,
    polarization: str | None = None,
) -> np.ndarray:
    """Resposta do beam (n_freq, n_angle) via ``Beam.evaluate`` (lazy → compute).

    ``Beam.evaluate`` retorna todas as polarizações quando ``polarization`` é
    ``None``; por isso selecionamos a polarização antes de fatiar por frequência.
    """
    import xarray as xr

    angle_list = list(angles_deg)
    freq_da = xr.DataArray(freqs_mhz, dims="frequency", attrs={"units": "MHz"})
    angle_da = xr.DataArray(angle_list, dims="angle", attrs={"units": "deg"})
    amp = beam.evaluate(
        frequency=freq_da,
        angle=angle_da,
        polarization=polarization,
    )
    amp = _materialize(amp)  # dask -> numpy; dims (frequency, angle)
    # Seleção por posição (isel): o resultado mantém a ordem das frequências
    # pedidas, evitando mismatch de unidades (MHz no CSV vs Hz no analítico).
    out = np.empty((len(freqs_mhz), len(angle_list)), dtype=float)
    for i in range(len(freqs_mhz)):
        out[i] = np.asarray(amp.isel(frequency=i).values, dtype=float)
    return out


# ── Observador ────────────────────────────────────────────────────────
def observer_map(observer: Any) -> hv.Overlay:
    """Mapa com overlay OSM e a localização do observador.

    Os tiles OSM são Web Mercator; o ponto é projetado (lon/lat → EPSG:3857)
    para aparecer na posição correta sobre o mapa.

    Args:
        observer: ``Observer`` (core) — usa ``lat``/``lon``/``height``.

    Returns:
        ``hv.Overlay`` (tiles OSM * pontos) renderizável no notebook.
    """
    lat = _to_value(observer.lat, "deg")
    lon = _to_value(observer.lon, "deg")
    height = _to_value(observer.height, "m")
    x, y = _web_mercator(lon, lat)
    try:
        tile = hv.element.tiles.OSM()
    except Exception:
        tile = hv.Tiles("https://tile.openstreetmap.org/{X}/{Y}.png")
    point = hv.Points(
        [(x, y)],
        label=f"Observer ({lat:.4f}, {lon:.4f})",
    ).opts(size=12, color="blue", tools=["hover"])
    return (tile * point).opts(
        title=f"Observer — lat {lat:.6f}°, lon {lon:.6f}°, altitude {height:.0f} m",
        width=700,
        height=500,
    )


# ── Propriedades por antena ───────────────────────────────────────────
def _curve_with_markers(
    data: list[tuple[Any, Any]],
    kdims: list[str],
    vdims: list[str],
    label: str,
    color: str,
) -> hv.Overlay:
    """Curva com marcadores (Curve * Scatter), mesma cor e legenda unificada.

    ``hv.Curve`` do bokeh só renderiza linha; o overlay com ``hv.Scatter``
    adiciona os marcadores por ponto.
    """
    curve = hv.Curve(data, kdims=kdims, vdims=vdims, label=label).opts(
        color=color,
        line_width=2,
    )
    scatter = hv.Scatter(data, kdims=kdims, vdims=vdims, label=label).opts(
        color=color,
        marker="o",
        size=7,
    )
    return curve * scatter


def fwhm_plot(antennas: Any, *, n_freqs: int = 17) -> hv.Overlay:
    """Curvas FWHM × frequência (uma por antena), com marcadores.

    Carrega o ``Beam`` da antena (via loaders do pacote beam) e calcula o
    FWHM por frequência com ``BeamProperties.compute_properties`` (dataset
    lazy é computado). Sem beam, usa ``fwhm_deg`` (constante).
    """
    plots: list[hv.Overlay] = []
    for i, antenna in enumerate(antennas):
        name = getattr(antenna, "name", "antenna")
        fr = getattr(antenna, "frequency_range", None)
        beam = _load_beam(antenna)
        if beam is not None:
            fwhm_map = _fwhm_vs_frequency(antenna, beam)
            xs = list(fwhm_map)
            ys = [fwhm_map[f] for f in xs]
        else:
            xs = _frequency_grid(fr, n_freqs)
            fwhm = getattr(antenna, "fwhm_deg", None)
            ys = [fwhm if fwhm is not None else np.nan for _ in xs]
        color = _MARKER_COLORS[i % len(_MARKER_COLORS)]
        plots.append(
            _curve_with_markers(
                list(zip(xs, ys, strict=False)),
                kdims=["Frequency (MHz)"],
                vdims=["FWHM (deg)"],
                label=name,
                color=color,
            )
        )
    overlay = plots[0]
    for p in plots[1:]:
        overlay = overlay * p
    return overlay.opts(
        title="FWHM vs Frequency",
        legend_position="top_right",
        width=700,
        height=400,
    )


def gain_plot(antennas: Any, *, n_freqs: int = 17) -> hv.Overlay:
    """Curvas Ganho (dBi) × frequência (uma por antena), com marcadores.

    Com ``effective_area_m2`` usa ``G = 4π A / λ²``; senão estima pelo
    ângulo sólido do feixe (``G ≈ 4π / (1.133 θ²)``), com θ = FWHM(f)
    calculado a partir do Beam (lazy → compute).
    """
    plots: list[hv.Overlay] = []
    for i, antenna in enumerate(antennas):
        name = getattr(antenna, "name", "antenna")
        fr = getattr(antenna, "frequency_range", None)
        eff_area = getattr(antenna, "effective_area_m2", None)
        beam = _load_beam(antenna)
        fwhm_map = _fwhm_vs_frequency(antenna, beam) if beam is not None else {}
        if fwhm_map:
            xs = list(fwhm_map)
        else:
            xs = _frequency_grid(fr, n_freqs)
        ys: list[float] = []
        for f in xs:
            lam = _SPEED_OF_LIGHT_M_S / (f * 1e6)
            if eff_area is not None:
                gain = 4.0 * math.pi * eff_area / (lam * lam)
                ys.append(10.0 * math.log10(gain) if gain > 0.0 else np.nan)
                continue
            fwhm: float | None
            if fwhm_map:
                fwhm = fwhm_map.get(f, np.nan)
            else:
                fwhm = getattr(antenna, "fwhm_deg", None)
            if fwhm is None or fwhm <= 0.0 or (isinstance(fwhm, float) and math.isnan(fwhm)):
                ys.append(np.nan)
                continue
            theta = math.radians(fwhm)
            omega = _GAUSSIAN_SOLID_ANGLE_FACTOR * theta * theta
            gain = 4.0 * math.pi / omega
            ys.append(10.0 * math.log10(gain) if gain > 0.0 else np.nan)
        color = _MARKER_COLORS[i % len(_MARKER_COLORS)]
        plots.append(
            _curve_with_markers(
                list(zip(xs, ys, strict=False)),
                kdims=["Frequency (MHz)"],
                vdims=["Gain (dBi)"],
                label=name,
                color=color,
            )
        )
    overlay = plots[0]
    for p in plots[1:]:
        overlay = overlay * p
    return overlay.opts(
        title="Gain vs Frequency",
        legend_position="top_right",
        width=700,
        height=400,
    )


def beam_pattern_plot(antennas: Any, *, n_freqs: int = 5) -> hv.Overlay:
    """Padrão de feixe (dB) × ângulo, para ``n_freqs`` frequências.

    Carrega o ``Beam`` da antena e avalia em grade angular via
    ``Beam.evaluate`` (dataset lazy é computado antes do plot).
    """
    curves: list[hv.Curve] = []
    angles = np.linspace(-90.0, 90.0, 361)
    for antenna in antennas:
        name = getattr(antenna, "name", "antenna")
        fr = getattr(antenna, "frequency_range", None)
        beam = _load_beam(antenna)
        if beam is None:
            continue
        freqs = _pick_beam_frequencies(beam, fr, n_freqs)
        if not freqs:
            continue
        resp = _beam_response_angle(
            beam,
            freqs,
            angles,
            polarization=_beam_polarization(antenna, beam),
        )  # (n_freq, n_angle)
        for i, freq in enumerate(freqs):
            row = resp[i]
            p_max = float(row.max())
            db = 10.0 * np.log10(row / p_max) if p_max > 0.0 else np.full_like(row, -np.inf)
            curves.append(
                hv.Curve(
                    list(zip(angles, db, strict=False)),
                    kdims=["Angle (deg)"],
                    vdims=["Response (dB)"],
                    label=f"{name} — {freq:.0f} MHz",
                )
            )
    return hv.Overlay(curves).opts(
        title="Beam pattern vs Angle",
        legend_position="top_right",
        width=700,
        height=400,
    )


def property_table(name: str, values: dict[str, Any], unit: str | None = None) -> hv.Table:
    """Tabela genérica antena → valor (fallback para propriedades simples)."""
    rows = [{"antenna": k, "value": str(v)} for k, v in values.items()]
    vdim = f"{name} ({unit})" if unit else name
    return hv.Table(rows, kdims=["antenna"], vdims=[vdim]).opts(width=500)
