# CIEL (local demo)

CIEL to lokalny, single-userowy klient demo (UI + CLI) do uruchamiania i testowania workflow oraz (opcjonalnie) lokalnej inferencji na modelach **GGUF**.

- UI: **NiceGUI** (przeglądarka, lokalny serwer HTTP)
- CLI: narzędzia pomocnicze
- Modele: pliki `.gguf` zarządzane lokalnie
- Backend LLM (opcjonalny): `llama-cpp-python` (CPU / opcjonalnie GPU zależnie od instalacji)

Projekt jest przygotowany pod **lokalne demo produkcyjne** (bezpieczniejsze domyślne ustawienia, limity, feature-flagi) — bez autoryzacji i bez trybu multi-user.

## Właściwości systemu

- **Lokalne demo single-user**
  - Domyślnie UI binduje na `127.0.0.1` (localhost).
- **Zarządzanie GGUF w UI**
  - Skanowanie katalogu modeli
  - Upload własnych `.gguf`
  - Pobieranie modeli z URL (z limitem rozmiaru)
  - Wybór aktywnego modelu + `Load/Unload`
- **Katalog rekomendowanych modeli**
  - Proponowane modele GGUF wraz z metadanymi (RAM/VRAM/uwagi).
- **Hardening pod demo**
  - Limity upload/download przez ENV
  - Bezpieczna lokalizacja danych w katalogu użytkownika
  - Instalacja backendu przez pip z poziomu UI jest **feature-flagowana** (domyślnie wyłączona)

## Ograniczenia / uwagi

- Brak wbudowanej autoryzacji — to jest świadomie **local demo**.
- `llama-cpp-python` jest **opcjonalne** i bywa trudne do “zabundlowania” w 1 binarkę (różne CPU/GPU/toolchain). Zalecany wariant dla GGUF inferencji: instalacja w venv.

## Struktura projektu

- Runtime kod: `main/` (pakiet Pythona)
- Entry-pointy:
  - UI: `ciel-omega` → `main.apps.omega_app:main`
  - CLI: `ciel-cli` → `main.apps.cli:main`
- Skrypty:
  - `scripts/install_local.*` — instalacja venv + zależności
  - `scripts/run_ui.*` / `scripts/run_cli.*` — start UI/CLI z venv
  - `scripts/build_bundle.*` — budowanie binarek PyInstaller

## Wymagania

- Python: **>= 3.11**
- Linux: bash
- Windows: PowerShell (PS1) lub `.cmd` wrapper
- (opcjonalnie) kompilator/toolchain pod `llama-cpp-python` jeśli chcesz inferencję GGUF

## Instalacja i uruchomienie (venv) — zalecane do demo

### Linux

1) Instalacja:

```bash
bash scripts/install_local.sh
```

2) Start UI:

```bash
bash scripts/run_ui.sh
```

3) Start CLI:

```bash
bash scripts/run_cli.sh list
```

### Windows

1) Instalacja (najprościej):

- uruchom `scripts\install_local.cmd`

albo PowerShell:

```powershell
scripts\install_local.ps1
```

2) Start UI:

- uruchom `scripts\run_ui.cmd`

3) Start CLI:

```powershell
scripts\run_cli.ps1 list
```

## (Opcjonalnie) instalacja backendu GGUF (llama-cpp-python)

### Linux

```bash
INSTALL_LLAMA=1 bash scripts/install_local.sh
```

Dla CUDA (jeśli wspierane w Twoim środowisku):

```bash
INSTALL_LLAMA=1 LLAMA_BACKEND=cuda bash scripts/install_local.sh
```

### Windows

```powershell
scripts\install_local.ps1 -InstallLlama 1 -LlamaBackend cpu
```

## Zmienne środowiskowe (ENV)

- `CIEL_HOST`
  - Domyślnie: `127.0.0.1`
- `CIEL_PORT`
  - Domyślnie: `8080`
- `CIEL_DATA_DIR`
  - Nadpisuje lokalizację katalogu danych aplikacji
- `CIEL_MAX_UPLOAD_MB`
  - Domyślnie: `4096`
- `CIEL_MAX_DOWNLOAD_MB`
  - Domyślnie: `8192`
- `CIEL_ALLOW_PIP_INSTALL`
  - `1` włącza przycisk/flow instalacji pip z UI (domyślnie `0`)

Przykład (Linux):

```bash
CIEL_PORT=9090 bash scripts/run_ui.sh
```

## Dane aplikacji

Domyślnie dane trzymane są w:

- Linux/Windows (katalog użytkownika): `~/.ciel/ciel_omega_data`
  - Modele GGUF: `~/.ciel/ciel_omega_data/models`
  - Pliki: `~/.ciel/ciel_omega_data/files`
  - Stan: `~/.ciel/ciel_omega_data/state.json`

## Bundle / “instalator bez Pythona” (PyInstaller)

Ten wariant tworzy jednoplikowe binarki, które można spakować i dystrybuować jako `.zip` / `.tar.gz`.

### Build lokalny (Linux)

1) Zainstaluj zależności bundla w `.venv` (albo użyj skryptu):

```bash
PYTHON_BIN=.venv/bin/python bash scripts/build_bundle.sh
```

2) Wynik znajdziesz w `dist_bundle/`:

- `dist_bundle/ciel-omega`
- `dist_bundle/ciel-cli`

### Build lokalny (Windows)

```powershell
scripts\build_bundle.ps1
```

### CI (GitHub Actions)

Workflow: `.github/workflows/build_bundles.yml`

- uruchom ręcznie: Actions → `build-bundles` → Run workflow
- lub przez tag `v*`

Publikowane artefakty:

- `ciel-bundle-windows.zip`
- `ciel-bundle-linux.tar.gz`

## Szybkie FAQ / troubleshooting

- UI nie startuje / brak zależności:
  - upewnij się, że instalacja poszła przez `scripts/install_local.*`
- GGUF nie działa:
  - sprawdź czy `llama-cpp-python` jest zainstalowane (`pip show llama-cpp-python` w venv)
  - UI pokaże komunikat jeśli backend jest niedostępny
- Zmiana portu:
  - ustaw `CIEL_PORT` i uruchom ponownie


