"""\
Módulo: tests/test_satellite_views.py

Descrição:
    Testes dos builders de visualização de satélites
    (``radioskysims_viz.satellite``): ``signals_plot`` (holoviews) e
    ``tracks_mollweide`` (matplotlib).

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

import numpy as np
import pytest

hv = pytest.importorskip("holoviews")
matplotlib = pytest.importorskip("matplotlib")

from radioskysims_viz.satellite import signals_plot, tracks_mollweide  # noqa: E402

TLE_LINE1 = "1 25544U 98067A   21275.51857639  .00002182  00000-0  50362-4 0  9991"
TLE_LINE2 = "2 25544  51.6443  21.4184 0004295  35.2895 324.8335 15.48942522302788"


def _signals_ds():
    import xarray as xr

    return xr.Dataset(
        {"PSD": (("satellite", "freq"), np.random.default_rng(0).normal(-60, 5, (5, 20)))},
        coords={
            "satellite": [1, 1, 1, 2, 2],
            "freq": np.linspace(1000, 1250, 20),
            "fc": ("satellite", [1575.42, 1227.6, 1176.45, 1575.42, 1227.6]),
            "modulation": ("satellite", ["BPSK(1)"] * 5),
        },
    )


def _tracks_ds():
    import xarray as xr

    tt = np.linspace(2459480.0, 2459480.02, 60)
    return xr.Dataset(
        {"positions": (("satellite", "time", "xyz"), np.zeros((1, 60, 3)))},
        coords={
            "satellite": [25544],
            "time": tt,
            "xyz": ["x", "y", "z"],
            "names": ("satellite", ["ISS"]),
            "constellation": ("satellite", ["ISS"]),
            "tle_line1": ("satellite", [TLE_LINE1]),
            "tle_line2": ("satellite", [TLE_LINE2]),
        },
    )


def _tle_line1(epoch: str) -> str:
    base = "1 25544U 98067A   21275.51857639  .00002182  00000-0  50362-4 0  9991"
    return base.replace("21275.51857639", epoch)


def _tracks_ds_multi():
    """2 constelações (GPS/GLONASS), 1 satélite cada, TLEs distintos."""
    import xarray as xr

    tt = np.linspace(2459480.0, 2459480.02, 60)
    l1 = [_tle_line1("21275.50000000"), _tle_line1("21280.50000000")]
    return xr.Dataset(
        {"positions": (("satellite", "time", "xyz"), np.zeros((2, 60, 3)))},
        coords={
            "satellite": [25544, 28361],
            "time": tt,
            "xyz": ["x", "y", "z"],
            "names": ("satellite", ["NAVSTAR 63", "COSMOS 2504"]),
            "constellation": ("satellite", ["GPS", "GLONASS"]),
            "tle_line1": ("satellite", l1),
            "tle_line2": ("satellite", [TLE_LINE2] * 2),
        },
    )


def test_signals_plot_returns_overlay_with_one_curve_per_signal():
    ov = signals_plot(_signals_ds(), norad_names={1: "A", 2: "B"})
    assert isinstance(ov, hv.Overlay)
    assert len(ov) == 5


def test_signals_plot_curves_have_labels():
    ov = signals_plot(_signals_ds(), norad_names={1: "NAVSTAR A"})
    labels = [c.label for c in ov]
    assert any("NAVSTAR A" in lbl for lbl in labels)
    assert all("MHz" in lbl for lbl in labels)


def test_tracks_mollweide_returns_figure():
    fig = tracks_mollweide(_tracks_ds(), smooth=True)
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_tracks_mollweide_kde_returns_figure():
    fig = tracks_mollweide(_tracks_ds(), kde=True)
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_tracks_mollweide_by_constellation_legend():
    fig = tracks_mollweide(_tracks_ds(), by="constellation", contours=True)
    legend = fig.axes[0].get_legend()
    texts = [t.get_text() for t in legend.get_texts()]
    assert "ISS" in texts
    matplotlib.pyplot.close(fig)


def test_tracks_mollweide_tracks_false_keeps_contours_legend():
    # sem tracks individuais, mas com contornos → legenda por constelação mantida
    fig = tracks_mollweide(
        _tracks_ds(), by="constellation", contours=True, tracks=False
    )
    legend = fig.axes[0].get_legend()
    texts = [t.get_text() for t in legend.get_texts()]
    assert "ISS" in texts
    # nada de linhas de track desenhadas (sem Line2D de segmentos)
    lines = [
        ln for ln in fig.axes[0].get_lines() if ln.get_xdata().size > 0
    ]
    assert len(lines) == 0
    matplotlib.pyplot.close(fig)


def test_tracks_mollweide_by_satellite():
    fig = tracks_mollweide(_tracks_ds(), by="satellite", contours=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_tracks_mollweide_contours_global_horizontal_colorbar():
    # contornos de densidade GLOBAIS com colorbar horizontal inferior
    fig = tracks_mollweide(_tracks_ds(), contours=True, tracks=True)
    # colorbar horizontal → rótulo no eixo x
    assert any(a.get_xlabel() == "densidade de linhas" for a in fig.axes)
    matplotlib.pyplot.close(fig)


def _contour_sets(ax):
    """QuadContourSets (contourf + contorno de linha) do eixo Mollweide."""
    return [c for c in ax.collections if hasattr(c, "levels")]


def test_tracks_mollweide_contours_n_levels_controls_bands():
    # n_levels controla os limites (n_levels+1) → n_levels bandas de densidade
    fig = tracks_mollweide(_tracks_ds(), contours=True, n_levels=5)
    qcs = _contour_sets(fig.axes[0])
    assert len(qcs) == 2  # contourf + contorno de linha
    assert len(qcs[0].levels) - 1 == 5
    matplotlib.pyplot.close(fig)

    fig3 = tracks_mollweide(_tracks_ds(), contours=True, n_levels=3)
    qcs3 = _contour_sets(fig3.axes[0])
    assert len(qcs3[0].levels) - 1 == 3
    matplotlib.pyplot.close(fig3)


def test_tracks_mollweide_contours_global_single_kde_over_all_tracks():
    # Uma ÚNICA KDE global (1 contourf + 1 contorno), independente da
    # constelação: com 2 constelações → 2 QuadContourSets (não 4, como seria
    # se houvesse 1 footprint por grupo) + colorbar horizontal inferior.
    fig = tracks_mollweide(
        _tracks_ds_multi(), by="constellation", contours=True, n_levels=5
    )
    qcs = _contour_sets(fig.axes[0])
    assert len(qcs) == 2
    assert len(qcs[0].levels) - 1 == 5
    assert any(a.get_xlabel() == "densidade de linhas" for a in fig.axes)
    matplotlib.pyplot.close(fig)


def test_tracks_mollweide_contours_haversine_figure():
    # métrica ESFÉRICA (padrão) — contornos + colorbar horizontal
    fig = tracks_mollweide(_tracks_ds(), contours=True, metric="haversine")
    qcs = _contour_sets(fig.axes[0])
    assert len(qcs) == 2
    assert len(qcs[0].levels) - 1 == 5
    assert any(a.get_xlabel() == "densidade de linhas" for a in fig.axes)
    matplotlib.pyplot.close(fig)


def test_tracks_mollweide_contours_euclidean_figure():
    # opção plana preservada
    fig = tracks_mollweide(_tracks_ds(), contours=True, metric="euclidean")
    assert len(_contour_sets(fig.axes[0])) == 2
    matplotlib.pyplot.close(fig)


def test_kde_haversine_spherical_near_antimeridian():
    # A métrica esférica enxerga pontos próximos através do antimeridiano
    # (179.5° e -179.5° estão ~1° de distância na esfera).
    from radioskysims_viz.satellite import _Kde2D

    lon = np.array([179.5, 179.7, -179.8, -179.6, 179.6, -179.7, 179.9, -179.5])
    lat = np.array([10.0, 10.2, 10.1, 9.9, 10.3, 10.0, 10.1, 10.2])
    k = _Kde2D(lon, lat, bandwidth=3.0, metric="haversine")
    assert k
    d_near = k.score_points(np.array([179.9]), np.array([10.0]))[0]
    d_far = k.score_points(np.array([0.0]), np.array([10.0]))[0]
    assert d_near > d_far
    assert np.isfinite(d_near) and np.isfinite(d_far)


def test_kde2d_degenerate_returns_falsy():
    from radioskysims_viz.satellite import _Kde2D

    assert not _Kde2D(np.ones(3), np.ones(3), metric="haversine")
    assert not _Kde2D(np.ones(3), np.ones(3), metric="euclidean")


def test_tracks_mollweide_coastlines_toggle():
    from matplotlib.collections import LineCollection

    # sem coastlines → nenhuma LineCollection de contorno de continente
    fig_off = tracks_mollweide(_tracks_ds(), contours=False, coastlines=False)
    has_coast = any(
        isinstance(c, LineCollection) and len(c.get_segments()) > 0
        for c in fig_off.axes[0].collections
    )
    assert not has_coast
    matplotlib.pyplot.close(fig_off)

    # com coastlines → renderiza normalmente (cartopy extrai a costa)
    fig_on = tracks_mollweide(_tracks_ds(), contours=False, coastlines=True)
    assert isinstance(fig_on, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig_on)
