# Docs Index

## Entry points

- `index.html` — static orbital cockpit preview intended for GitHub Pages
- `OMEGA_COCKPIT_1_0.md` — architectural specification for the Omega cockpit refactor
- `ORBITAL_PREVIEW.md` — operational guide for local launch and web publication

## Current documentation map

### Cockpit architecture
- `OMEGA_COCKPIT_1_0.md`
  - attractor-centered layout
  - orbital model
  - screen topology
  - badges, inspector, event strip
  - refactor phases

### Preview surfaces
- `index.html`
  - static web preview
- `ORBITAL_PREVIEW.md`
  - local launch instructions
  - GitHub Pages publication steps

## Runtime entrypoints in code

- `main/apps/omega_app.py` — legacy local runtime cockpit
- `main/apps/omega_orbital_app.py` — orbital cockpit preview
- `main/apps/orbital_cockpit.py` — orbital topology model
- `main/apps/orbital_panels.py` — navigation and identity/event builders

## Launch scripts

### Legacy
- `scripts/run_ui.sh`

### Orbital preview
- `scripts/run_orbital_ui.sh`
- `scripts/run_orbital_ui.ps1`
- `scripts/run_orbital_ui.cmd`

## Practical note

For a website, the critical file is:
- `docs/index.html`

For humans navigating the repository, the documentation entry file is now:
- `docs/INDEX.md`
