"""\
Módulo: tests/test_radiotelescope_views.py

Descrição:
    Testes dos builders holoviews do módulo ``radioskysims_viz.radiotelescope``
    (observer_map, fwhm_plot, gain_plot, beam_pattern_plot, property_table).

Copyright BINGO Collaboration
Last modified: 2026-08-30
"""

from __future__ import annotations

import pytest

from radioskysims_core.domain.common import FrequencyRange
from radioskysims_radiotelescope.domain.radiotelescope.antenna import (
    Antenna,
    Antennas,
    BeamSpec,
)

from radioskysims_viz.radiotelescope import (
    beam_pattern_plot,
    fwhm_plot,
    gain_plot,
    observer_map,
    property_table,
)

hv = pytest.importorskip("holoviews")


def _make_antenna(
    name: str = "horn",
    beam_width_deg: float = 15.0,
    pattern_file: str | None = None,
    profile: str | None = None,
) -> Antenna:
    return Antenna(
        name=name,
        frequency_range=FrequencyRange(min_mhz=1000.0, max_mhz=1250.0, resolution_mhz=0.05),
        beam=BeamSpec(
            name="beam",
            beam_width_deg=beam_width_deg,
            beam_pattern_file=pattern_file,
            profile=profile,
        ),
    )


def _make_observer():
    from radioskysims_core.domain.common import Observer, TimeRange

    return Observer(
        lat=-7.211637,
        lon=-35.90813,
        height=530.0,
        timeframe=TimeRange(
            start_jd=2460703.5,
            end_jd=2460704.5,
            resolution_sec=120.0,
        ),
    )


class TestObserverMap:
    def test_returns_overlay_with_point(self) -> None:
        import math

        plot = observer_map(_make_observer())
        assert isinstance(plot, hv.Overlay)
        # tiles OSM + points (overlay)
        assert len(plot) == 2
        point = list(plot)[1]
        assert isinstance(point, hv.Points)
        # O ponto é projetado para Web Mercator (EPSG:3857) para sobrepor os tiles
        x = 6378137.0 * math.radians(-35.90813)
        y = 6378137.0 * math.log(
            math.tan(math.pi / 4.0 + math.radians(-7.211637) / 2.0)
        )
        assert float(point["x"].data[0]) == pytest.approx(x)
        assert float(point["y"].data[0]) == pytest.approx(y)


class TestFwhmGainPlots:
    def test_fwhm_constant_from_beam_width(self) -> None:
        antennas = Antennas(antennas=[_make_antenna("horn", beam_width_deg=15.0)])
        plot = fwhm_plot(antennas)
        assert isinstance(plot, hv.Overlay)
        # linha + marcadores (Curve * Scatter)
        assert any(isinstance(e, hv.Scatter) for e in plot)
        curve = next(e for e in plot if isinstance(e, hv.Curve))
        # FWHM constante (sem arquivo de padrão): todos os valores = 15
        assert all(v == pytest.approx(15.0) for v in curve["FWHM (deg)"].data)

    def test_gain_vs_frequency(self) -> None:
        antennas = Antennas(antennas=[_make_antenna("horn", beam_width_deg=15.0)])
        plot = gain_plot(antennas)
        assert isinstance(plot, hv.Overlay)
        # linha + marcadores (Curve * Scatter)
        assert any(isinstance(e, hv.Scatter) for e in plot)
        curve = next(e for e in plot if isinstance(e, hv.Curve))
        ys = list(curve["Gain (dBi)"].data)
        assert len(ys) > 1
        # sem área efetiva, ganho vem do ângulo sólido (constante)
        assert all(y == pytest.approx(22.09, rel=1e-2) for y in ys)


class TestBeamPatternPlot:
    def test_analytical_five_frequencies(self) -> None:
        antennas = Antennas(
            antennas=[_make_antenna("horn", beam_width_deg=15.0, profile="gaussian")]
        )
        plot = beam_pattern_plot(antennas, n_freqs=5)
        assert isinstance(plot, hv.Overlay)
        assert len(plot) == 5  # 5 frequências espaçadas na faixa

    def test_pattern_file_five_frequencies(self) -> None:
        # Usa o padrão real do BINGO horn (resolução de caminho com src/)
        antennas = Antennas(
            antennas=[
                _make_antenna(
                    "BINGO Horn",
                    beam_width_deg=15.0,
                    pattern_file="radioskysims/resources/beams/beampattern_horn01.csv",
                )
            ]
        )
        plot = beam_pattern_plot(antennas, n_freqs=5)
        assert isinstance(plot, hv.Overlay)
        assert len(plot) == 5  # 5 frequências espaçadas na faixa 1000-1250 MHz


class TestFwhmFromPatternFile:
    def test_fwhm_varies_with_frequency(self) -> None:
        # Com o padrão real, o FWHM depende da frequência (não é constante)
        antennas = Antennas(
            antennas=[
                _make_antenna(
                    "BINGO Horn",
                    beam_width_deg=15.0,
                    pattern_file="radioskysims/resources/beams/beampattern_horn01.csv",
                )
            ]
        )
        plot = fwhm_plot(antennas)
        curve = next(e for e in plot if isinstance(e, hv.Curve))
        values = list(curve["FWHM (deg)"].data)
        assert len(values) > 1
        assert any(v != pytest.approx(values[0]) for v in values)  # varia com f


class TestPropertyTable:
    def test_returns_holoviews_table(self) -> None:
        table = property_table("Gain", {"horn": 22.09}, "dBi")
        assert isinstance(table, hv.Table)
        assert table.kdims[0].name == "antenna"
