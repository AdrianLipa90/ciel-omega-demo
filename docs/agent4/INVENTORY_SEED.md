# AGENT4 — Provisional Inventory Seed

## Status
Provisional inventory seed.

This file records the first AGENT4 inventory pass for the demo repository.
It is intended to support indexation and cross-reference holonomy.
It is not yet a canonical registry of every object.

## Source basis
This seed was prepared from the locally available demo snapshot used by AGENT4 for analysis, not from a complete GitHub-side object-by-object crawl.
It should therefore be treated as:
- useful for index planning,
- valid for initial class separation,
- pending later validator-backed refinement.

## High-level file picture from the analyzed snapshot
Approximate observed file counts:
- total files: 130
- Python files: 86
- Markdown files: 16
- shell scripts: 7
- PowerShell scripts: 5
- cmd launchers: 4
- JSON files: 3

## Top-level sectors observed in the analyzed snapshot
- `.github/`
- `.workload/`
- `docs/`
- `main/`
- `raporty/`
- `scripts/`

## Main internal sectors observed in the analyzed snapshot
- `main/apps/`
- `main/adam/`
- `main/cognition/`
- `main/core/`
- `main/emotion/`
- `main/experiments/`
- `main/kernels/`
- `main/omega/`
- `main/spectral/`

## First AGENT4 interpretation
For AGENT4 purposes, the current working hypothesis is:
- `main/` is the runtime/cockpit shell body,
- `docs/` is the documentation and preview surface,
- `scripts/` is the launcher and operator surface,
- `.github/` is CI/distribution support,
- `.workload/` and `raporty/` require explicit status marking before they are treated as stable surfaces.

## First indexing consequences
The next inventory pass should explicitly distinguish:
- stable runtime objects,
- preview-only surfaces,
- documentation-only surfaces,
- manifest-bound or CI/distribution artifacts,
- shell-facing objects,
- future engine-facing objects,
- placeholders/import slots.

## Immediate AGENT4 next tasks in this repository
1. identify the actual global index authority already present in the repository,
2. identify layer indexes already present or missing,
3. seed an object-class registry for top-level demo sectors,
4. mark shell vs engine-facing boundaries,
5. separate stable runtime from preview-only and documentation-only surfaces.

## Warning
This file must not be used as a pretext for blind upstream import.
Its role is inventory-first and holonomy-first.
