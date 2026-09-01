"""\
Módulo: tests/test_sources_views.py

Descrição:
    Testes dos builders de visualização de fontes (``radioskysims_viz.sources``):
    mapas Mollweide (RA/Dec), curvas de fluxo e carta ALT/AZ.

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

from radioskysims_viz.sources import (  # noqa: E402
    radio_sources_flux,
    radio_sources_mollweide,
    sun_flux_density,
    sun_mollweide,
    sun_skychart,
)


def test_radio_sources_mollweide_returns_figure():
    ra = np.array([10.0, 150.0, 300.0])
    dec = np.array([-20.0, 30.0, 5.0])
    flux = np.array([0.1, 1.0, 5.0])
    fig = radio_sources_mollweide(ra, dec, fluxes_jy=flux)
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_radio_sources_mollweide_no_flux():
    fig = radio_sources_mollweide(np.array([10.0]), np.array([20.0]))
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_radio_sources_mollweide_empty():
    fig = radio_sources_mollweide(np.array([]), np.array([]))
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_radio_sources_flux_returns_figure():
    freqs = np.linspace(1000, 1500, 40)
    curves = {
        "A": 1.0 * (freqs / 1400.0) ** -0.7,
        "B": 2.0 * (freqs / 1400.0) ** -1.1,
    }
    fig = radio_sources_flux(freqs, curves)
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_radio_sources_flux_linear_fallback_nonpositive():
    # fluxo não positivo → não usa escala log (sem warning de log de 0)
    freqs = np.linspace(1000, 1500, 10)
    fig = radio_sources_flux(freqs, {"A": np.zeros(10)}, loglog=True)
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_sun_skychart_returns_figure():
    az = np.linspace(0, 2 * np.pi, 30)
    el = np.linspace(0.1, 1.2, 30)
    fig = sun_skychart(az, el)
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_sun_mollweide_returns_figure():
    ra = np.array([4.0, 4.01])
    dec = np.array([0.14, 0.16])
    fig = sun_mollweide(ra, dec)
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_sun_flux_density_returns_figure():
    freqs = np.linspace(1000, 1500, 40)
    fig = sun_flux_density(freqs, np.full_like(freqs, 1.5e6))
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)
