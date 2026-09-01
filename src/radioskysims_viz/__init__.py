"""
Pacote: radioskysims_viz

Base de visualização interativa do ecossistema RadioSkySim:
- panel_application.PanelApplication: classe base com ciclo de vida
  ``build_sidebar()`` / ``build_main()`` / ``serve()``.
- ports.VisualizerPort: contrato de visualização.
- panel_visualizer.PanelVisualizer: adapter concreto baseado em Panel.
- radiotelescope: visões holoviews de objetos de radiotelescópio
  (observer_map, fwhm_plot, gain_plot, beam_pattern_plot, property_table).
- io: exportação de plots (``save_plot`` → HTML ou imagem).

Copyright BINGO Collaboration
"""

from __future__ import annotations

__version__ = "0.1.0"

from .io import save_figure, save_plot
from .panel_application import PanelApplication
from .panel_visualizer import PanelVisualizer
from .ports import VisualizerPort
from .radiotelescope import (
    beam_pattern_plot,
    fwhm_plot,
    gain_plot,
    observer_map,
    property_table,
)
from .satellite import signals_plot, tracks_mollweide
from .satellite_rinex import (
    rinex_signals_plot,
    rinex_tracks_mollweide,
    rinex_tracks_skychart,
)
from .sources import (
    radio_sources_flux,
    radio_sources_mollweide,
    sun_flux_density,
    sun_mollweide,
    sun_skychart,
)

__all__ = [
    "PanelApplication",
    "PanelVisualizer",
    "VisualizerPort",
    "beam_pattern_plot",
    "fwhm_plot",
    "gain_plot",
    "observer_map",
    "property_table",
    "radio_sources_flux",
    "radio_sources_mollweide",
    "rinex_signals_plot",
    "rinex_tracks_mollweide",
    "rinex_tracks_skychart",
    "save_figure",
    "save_plot",
    "signals_plot",
    "sun_flux_density",
    "sun_mollweide",
    "sun_skychart",
    "tracks_mollweide",
]
