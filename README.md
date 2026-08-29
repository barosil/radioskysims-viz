# radioskysims-viz

Base de visualização interativa do ecossistema **RadioSkySim**.

- `PanelApplication` — classe base com ciclo de vida `build_sidebar()` /
  `build_main()` / `serve()` (padrão Panel Viewer).
- `VisualizerPort` — contrato de visualização.
- `PanelVisualizer` — adapter concreto baseado em Panel.

## Testes

```bash
uv run --package radioskysims-viz pytest
```
