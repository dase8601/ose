# Workspace Layout

This repository root is the active `SystemABM` project.

Core project files stay at the top level:

- `abm/` — environments, learning modules, controllers, planning loops
- `abm_experiment.py` — main experiment entrypoint
- `setup_cloud.sh`, `setup_habitat.sh` — environment setup
- `tests/` — saved experiment snapshots
- `results/` — active project reports and metrics

Local-only materials are kept under `_workspace/` and ignored by git:

- `_workspace/backups/` — backup clones or one-off repo copies
- `_workspace/research/` — notes, transcripts, reading material
- `_workspace/archives/` — large source zips and downloads
- `_workspace/reference-images/` — screenshots and loose image files
- `_workspace/legacy/vjepa_poc/` — older standalone V-JEPA prototype code and outputs

`env/` remains at the root on purpose. Moving a Python virtual environment usually
breaks its internal paths, so it is better left where it was created.
