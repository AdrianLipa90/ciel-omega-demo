# Local Debug Report — 2026-03-25

## Scope

This report documents a local reconstruction and smoke-test pass performed on the repository snapshot from `main`.

Input artifact:
- `ciel-omega-demo-main.zip`
- upstream snapshot commit: `7bc59bc5848780b257a0cb42e03841ec90bb9e3e`

---

## What was tested locally

### Packaging / install
- editable install with `pip install -e .`
- dependency resolution through NiceGUI stack
- import smoke for:
  - `main.apps.orbital_cockpit`
  - `main.apps.orbital_panels`
  - `main.apps.orbital_manifest_export`
  - `main.apps.omega_orbital_app`

### Runtime smoke
- `python -m main.apps.orbital_manifest_export`
- `python -m main.apps.omega_orbital_app`
- `python -m main.apps.omega_app`
- script launchers:
  - `bash scripts/run_orbital_ui.sh`
  - `bash scripts/run_ui.sh`

Validation criterion:
- HTTP root `/` responds with **200** instead of **500**.

---

## Real bugs found

### 1. Root page failure in NiceGUI apps

Both `omega_app.py` and `omega_orbital_app.py` started the server, but the root URL returned **HTTP 500**.

Observed mechanism:
- NiceGUI entered script-mode fallback,
- root route fell through to the 404 handler,
- fallback attempted to re-run the script,
- the app crashed during that fallback path.

This affected:
- legacy runtime cockpit,
- orbital cockpit preview.

### 2. Orbital app relative import issue

`omega_orbital_app.py` originally used relative imports for its local app modules.
When the NiceGUI fallback re-executed the file as a path-based script, Python lost package context and raised:

- `ImportError: attempted relative import with no known parent package`

### 3. Missing official console entrypoints for new orbital surfaces

`pyproject.toml` exposed:
- `ciel-cli`
- `ciel-omega`

but did **not** expose:
- orbital cockpit launcher
- orbital manifest exporter

This did not block module execution, but it was a packaging gap.

---

## Fixes applied in this local snapshot

### A. `main/apps/omega_orbital_app.py`
- switched local app imports to absolute package imports
- reworked startup so the UI is built through a root-page function instead of relying on NiceGUI script fallback
- explicitly neutralized NiceGUI script-mode fallback before `ui.run(...)`
- fixed page rendering state so the orbital workspace has a stable holder object accessible from render callbacks

### B. `main/apps/omega_app.py`
- reworked startup to use a root-page function
- neutralized NiceGUI script-mode fallback before `ui.run(...)`

### C. `pyproject.toml`
Added package entrypoints:
- `ciel-omega-orbital = "main.apps.omega_orbital_app:main"`
- `ciel-orbital-manifest = "main.apps.orbital_manifest_export:main"`

---

## Post-fix verification

### Passed
- editable install succeeds
- orbital manifest export works
- orbital app import smoke passes
- legacy app import smoke passes
- `python -m main.apps.omega_orbital_app` -> HTTP root **200**
- `python -m main.apps.omega_app` -> HTTP root **200**
- `bash scripts/run_orbital_ui.sh` -> HTTP root **200**
- `bash scripts/run_ui.sh` -> HTTP root **200**
- `.venv/bin/ciel-omega-orbital` -> HTTP root **200**
- `.venv/bin/ciel-omega` -> HTTP root **200**

### Notes
The generated `docs/orbital_manifest.json` was refreshed locally from:
- `main.apps.orbital_manifest_export`

This snapshot therefore contains a runtime-generated manifest rather than only the original static seed.

---

## Files changed in this local snapshot

### Runtime / packaging
- `pyproject.toml`
- `main/apps/omega_app.py`
- `main/apps/omega_orbital_app.py`

### Generated / refreshed
- `docs/orbital_manifest.json`

### Documentation
- `docs/REPORT_LOCAL_DEBUG_2026-03-25.md`
- `docs/INDEX.md` (if present in this packaged snapshot, updated to reference this report)

---

## Recommended next commit message

```text
fix: make NiceGUI root pages load correctly for legacy and orbital apps
```

Optional follow-up commit:

```text
feat: add console scripts for orbital cockpit and manifest export
```

---

## Ready state

This packaged repository snapshot is intended to be:
- unpacked,
- reviewed,
- committed/pushed from a Linux environment.

It is no longer only a structural scaffold.
It has passed a real local smoke-check for the two main NiceGUI entry surfaces.
