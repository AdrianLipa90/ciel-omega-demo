# AGENT4 — Indexing and Cross-Reference Holonomy Plan

## Repository role
This repository is the Omega demo shell, cockpit migration workspace, and documentation/publication surface. It is downstream relative to first-principles theory repositories.

## AGENT4 immediate mission
AGENT4 does not begin with blind upstream transplantation. It begins with internal epistemic closure of the demo-side graph.

Priority order:
1. inventory existing demo objects
2. distinguish stable runtime from preview, placeholder, and documentation-only layers
3. identify shell-facing vs engine-facing vs surface-facing objects
4. establish index authority
5. build internal cross-reference holonomy
6. define validator conditions
7. only then prepare controlled upstream bindings

## Object classes to use initially
- `runtime_stable`
- `preview_only`
- `surface_bound`
- `engine_facing`
- `manifest_bound`
- `placeholder`
- `import_slot`

## Minimal object record requirements
Each indexed object should eventually have:
- stable ID
- name
- layer
- status
- class
- upstream links
- downstream links
- path
- runtime/test visibility
- manifest visibility
- provenance
- import readiness marker

## Immediate AGENT4 work packages
### Package A — inventory seed
Map currently visible runtime, docs, manifests, and cockpit sectors.

### Package B — index authority
Clarify the role of global repo index, docs index, layer indexes, and machine-readable registry.

### Package C — cross-reference holonomy
Record at least the following relation types:
- `upstream`
- `downstream`
- `couples_to`
- `imports_from`
- `surface_for`
- `manifest_for`
- `documents`

### Package D — validator requirements
Prepare a validator spec that can later reject:
- runtime claim without visible implementation state
- preview presented as stable runtime
- manifest node with no discoverable surface
- import slot with no upstream provenance
- missing shell/engine boundary marker where required

## Current caution
AGENT4 should treat the repo as a shell/cockpit/documentation surface with future engine-facing bindings. Shell and engine must remain distinguishable in the inventory and cross-reference mesh.

## Deliverable rule
AGENT4 should prefer small, index-oriented artifacts first:
- plans
- registry seeds
- mapping tables
- validator notes
- cross-reference manifests

Not uncontrolled imports.
