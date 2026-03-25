# AGENT.md

## Purpose

This file defines the working rules for anyone extending or maintaining the CIEL/Ω repository.

It should be read as the **operational steering document** for the repo.
It is not a marketing text.
It is not a generic contributor guide.
It is the current source of truth for:

- how changes should be made,
- what the project is currently trying to become,
- what order of work is considered coherent,
- what validation is mandatory before claiming progress.

---

## Project identity

This repository currently serves three simultaneous roles:

1. **working local runtime demo**,
2. **migration path toward the Omega system cockpit**,
3. **documentation / publication surface for architecture review**.

The repo is therefore not only an app.
It is also a cockpit migration workspace and an epistemic instrument in construction.

---

## Central architectural direction

The project is moving:

- from a flat runtime control panel,
- toward an **attractor-centered, orbit-organized cockpit**.

The target geometry is:

- **Identity Attractor** as center,
- constitutive layers,
- dynamic layers,
- interaction layers,
- observation layers,
- boundary layers,
- educational layers,
- epistemic object cards,
- truth-convergence surfaces.

This means every meaningful change should be evaluated against one question:

> Does this move the repo closer to a readable system cockpit rather than back toward a pile of tabs and tools?

---

## Current implemented layers

At the time of writing, the repository already contains:

### Runtime surfaces
- legacy local NiceGUI cockpit,
- CLI,
- local GGUF runtime support,
- orbital cockpit preview,
- manifest exporter,
- static web preview.

### Documentation layers
- cockpit architecture docs,
- orbital preview docs,
- documentation index,
- educational analogy layer,
- object-card methodology,
- local debug report.

### Epistemic structure
- object cards,
- child-node object cards,
- crossrefs in cards,
- evidence / boundary / educational layers,
- manifest-exported object card state.

---

## Non-negotiable working rules

### 1. Preserve reality over appearance
Never claim that something is tested, validated, exported, or wired unless it actually is.

### 2. No fake completion
If a layer exists only structurally, call it structural.
If it is runtime-tested, say runtime-tested.
If it is only documented, say documented.

### 3. Preserve legacy runtime unless replacement is real
The orbital cockpit preview is not allowed to silently break the working legacy runtime surface.

### 4. Update the whole layer, not only one fragment
When changing a meaningful cockpit layer, update the corresponding:
- code,
- manifest,
- documentation,
- index / discoverability surface,
- and validation note when appropriate.

### 5. Crossrefs are not decoration
If a new object is introduced and it materially depends on other nodes, its object card should eventually expose crossrefs.

### 6. Publication boundary is first-class
Do not blur private, runtime-only, draft, and public-exported content.
Statuses must remain explicit.

### 7. Never hide validation limits
If a smoke-check was not rerun after a new architectural change, say so.

---

## Ordered plan of action

The repo should now evolve in the following order.

### Phase 1 — Keep runtime stable
Goal:
- maintain working local runtime surfaces,
- preserve launcher integrity,
- preserve manifest export.

Required outcomes:
- legacy UI starts,
- orbital UI starts,
- root page loads,
- manifest export works.

### Phase 2 — Finish epistemic readability
Goal:
- make major nodes readable as explicit objects.

Required outcomes:
- object cards for major nodes,
- child-node coverage,
- crossrefs rendered,
- docs and manifest updated.

### Phase 3 — Add truth-convergence layer
Goal:
- expose operational distance from the truth attractor.

Required outcomes:
- explicit metric schema,
- manifest integration,
- inspector visibility,
- workspace visibility,
- documented limitations.

### Phase 4 — Make topology navigable
Goal:
- move from displayed graph to traversable graph.

Required outcomes:
- clickable crossrefs,
- object-to-object navigation,
- context-preserving inspector transitions.

### Phase 5 — Bind theory to runtime more explicitly
Goal:
- reduce the gap between architectural labels and executable or inspectable structure.

Required outcomes:
- clearer theory export binding,
- stronger operator / constant / constraint surfaces,
- future object manifests and sector views.

### Phase 6 — Strengthen evidence and validation
Goal:
- stop treating tests and audit as placeholders only.

Required outcomes:
- executable test hooks,
- stronger provenance surfaces,
- dirty-manifest / sanitization states,
- repeatable smoke-check procedure.

---

## Immediate next-priority candidates

The next priority should be chosen from this constrained set:

1. **Truth-convergence integration**
   - wire the metric into manifest + inspector + cockpit workspace
2. **Runtime validation after object-card expansion**
   - rerun local smoke-check after the newest object-card / crossref changes
3. **Clickable crossrefs**
   - turn crossrefs from visible labels into actual navigation

Preferred order:

1. runtime validation,
2. truth-convergence integration,
3. clickable crossrefs.

Reason:
- validation should precede further epistemic claims,
- truth-convergence should exist before richer navigation to avoid pretty-but-shallow graph growth,
- clickable graphing should come after the metric layer is visible.

---

## Validation loop

Any meaningful cockpit change should pass this loop:

1. **Code update**
2. **Manifest update**
3. **Documentation update**
4. **Discoverability update**
   - README / docs index / AGENT where appropriate
5. **Smoke-check**
6. **Explicit note of what was and was not tested**

If step 5 cannot be completed in the current environment, step 6 is mandatory.

---

## Required smoke-check surfaces

Minimum validation surfaces:

- `python -m main.apps.omega_app`
- `python -m main.apps.omega_orbital_app`
- `python -m main.apps.orbital_manifest_export`
- launcher scripts for legacy and orbital UI

Success criteria:
- root page returns HTTP 200,
- manifest exports successfully,
- no new import/runtime regression is introduced.

---

## Documentation obligations

When a new layer is introduced, at least one of the following must be updated:

- `README.md`
- `docs/INDEX.md`
- `AGENT.md`
- dedicated layer document in `docs/`

If the change introduces a new epistemic object or schema, a dedicated doc is preferred.

---

## Anti-patterns

Avoid these repo failure modes:

- adding UI labels without underlying semantic objects,
- adding docs that do not match runtime reality,
- adding manifest fields never used anywhere,
- claiming validation without rerunning smoke-checks,
- introducing new layers without index/discoverability updates,
- replacing working runtime paths with preview-only abstractions.

---

## Summary rule

The repository should evolve by this sequence:

**runtime stability -> epistemic readability -> truth convergence -> navigable topology -> deeper theory/runtime binding -> stronger evidence**

If a proposed change violates that order, the burden of proof is on the change.
