"""\
Módulo: tests/test_satellite_rinex_views.py

Descrição:
    Testes das views de satélites RINEX (``radioskysims_viz.satellite_rinex``).

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

matplotlib = pytest.importorskip("matplotlib")

from radioskysims_viz.satellite_rinex import (  # noqa: E402
    rinex_signals_plot,
    rinex_tracks_mollweide,
    rinex_tracks_skychart,
)


def _rinex_ds() -> xr.Dataset:
    times = np.array(["2024-06-10T00:00:00", "2024-06-10T01:00:00"], dtype="datetime64[ns]")
    svs = ["G01", "G02"]
    return xr.Dataset(
        {
            "alt_deg": (("sv", "time"), np.array([[20.0, 40.0], [30.0, 50.0]])),
            "az_deg": (("sv", "time"), np.array([[30.0, 60.0], [120.0, 150.0]])),
            "ra_deg": (("sv", "time"), np.array([[100.0, 110.0], [200.0, 210.0]])),
            "dec_deg": (("sv", "time"), np.array([[10.0, 12.0], [-20.0, -22.0]])),
        },
        coords={"sv": svs, "time": times},
    )


def test_rinex_tracks_mollweide_figure():
    fig = rinex_tracks_mollweide(_rinex_ds())
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_rinex_tracks_skychart_figure():
    fig = rinex_tracks_skychart(_rinex_ds())
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_rinex_tracks_mollweide_empty():
    ds = xr.Dataset(coords={"sv": [], "time": np.array([], dtype="datetime64[ns]")})
    fig = rinex_tracks_mollweide(ds)
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_rinex_tracks_skychart_empty():
    ds = xr.Dataset(coords={"sv": [], "time": np.array([], dtype="datetime64[ns]")})
    fig = rinex_tracks_skychart(ds)
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def _rinex_signals_ds() -> xr.Dataset:
    """Sinais RINEX analíticos: 2 SV GPS, PSD (dBW/Hz) sobre grid do FR."""
    svs = ["G01", "G02"]
    freqs = np.arange(1170.0, 1235.0, 1.0)
    psd = np.full((len(svs), len(freqs)), -300.0)
    for fc, peak in ((1176.45, -40.0), (1227.60, -35.0)):
        j = int(np.argmin(np.abs(freqs - fc)))
        psd[:, j] = peak
    return xr.Dataset(
        {"PSD": (("sv", "freq"), psd)},
        coords={
            "sv": svs,
            "freq": freqs,
            "constellation": ("sv", ["GPS", "GPS"]),
        },
    )


def test_rinex_signals_plot_figure():
    fig = rinex_signals_plot(_rinex_signals_ds())
    assert isinstance(fig, matplotlib.figure.Figure)
    assert "frequência (MHz)" == fig.axes[0].get_xlabel()
    matplotlib.pyplot.close(fig)


def test_rinex_signals_plot_continuous_curves():
    """Curvas contínuas de PSD por satélite (como o TLE), não marcadores."""
    ds = _rinex_signals_ds()
    fig = rinex_signals_plot(ds)
    ax = fig.axes[0]
    lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 0]
    assert len(lines) == 2  # uma curva por SV
    assert all(len(ln.get_xdata()) == len(ds["freq"]) for ln in lines)
    matplotlib.pyplot.close(fig)


def test_rinex_signals_plot_empty():
    ds = xr.Dataset(
        coords={
            "sv": [],
            "freq": np.array([], dtype=float),
            "constellation": np.array([], dtype=str),
        }
    )
    fig = rinex_signals_plot(ds)
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)
