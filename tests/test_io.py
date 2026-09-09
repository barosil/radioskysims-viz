"""\
Módulo: tests/test_io.py

Descrição:
    Testes de ``radioskysims_viz.io.save_plot`` (exportação de plots
    holoviews em HTML ou imagem).

Copyright BINGO Collaboration
Last modified: 2026-08-30
"""

from __future__ import annotations

import holoviews as hv
import pytest
from radioskysims_viz.io import _resolve_outdir, save_plot


def _simple_plot():
    return hv.Curve([(1, 2), (2, 3), (3, 5)], kdims=["x"], vdims=["y"])


def test_resolve_outdir_default_is_notebooks_plots(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("radioskysims_viz.io.get_project_root", lambda: tmp_path)
    assert _resolve_outdir(None) == tmp_path / "notebooks" / "plots"


def test_resolve_outdir_relative_is_root_anchored(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("radioskysims_viz.io.get_project_root", lambda: tmp_path)
    assert _resolve_outdir("plots") == tmp_path / "plots"


def test_resolve_outdir_absolute_is_kept(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("radioskysims_viz.io.get_project_root", lambda: tmp_path)
    target = tmp_path / "outra" / "pasta"
    assert _resolve_outdir(target) == target


def test_save_html(tmp_path) -> None:
    path = save_plot(_simple_plot(), "curva", outdir=tmp_path, verbose=False)
    assert path == tmp_path / "curva.html"
    assert path.exists()
    assert path.stat().st_size > 0


def test_save_png_matplotlib(tmp_path) -> None:
    path = save_plot(_simple_plot(), "curva", fmt="png", outdir=tmp_path, verbose=False)
    assert path == tmp_path / "curva.png"
    assert path.exists()
    assert path.stat().st_size > 0


def test_save_creates_outdir(tmp_path) -> None:
    outdir = tmp_path / "sub" / "plots"
    path = save_plot(_simple_plot(), "curva", outdir=outdir, verbose=False)
    assert path.parent == outdir
    assert path.exists()


def test_save_invalid_format(tmp_path) -> None:
    with pytest.raises(ValueError, match="Formato não suportado"):
        save_plot(_simple_plot(), "curva", fmt="gif", outdir=tmp_path, verbose=False)
