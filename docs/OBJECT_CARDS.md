# Object Cards

## Purpose

Object cards are the first explicit epistemic interface for major Omega cockpit nodes.

They are meant to stop the system from exposing only labels such as:
- Theory
- Operators
- Evidence
- Boundary
- Analogies

without explaining what those layers *are*.

Each card makes a selected node readable through a stable schema.

---

## Card schema

Each object card contains:

1. **Role**
   - what the object is for in the system
2. **Definition**
   - what the object means formally or operationally
3. **Derivation**
   - where it comes from in the architecture / logic / workflow
4. **Implementation**
   - where it lives in code or how it is surfaced
5. **Test**
   - what kind of verification should confirm its behavior
6. **Status**
   - epistemic / runtime / publication status
7. **Interpretation**
   - how the object should be read in Omega language

Optional:
- supporting docs
- future crossrefs
- provenance links

---

## Why this matters

The cockpit should not be a menu of names.
It should be an epistemic instrument.

That means every major node should become inspectable as an object with:
- meaning,
- source,
- code location,
- verification pathway,
- interpretive place in the whole system.

Object cards are the first compact implementation of that requirement.

---

## Initial coverage

The first registry covers:

- Identity Attractor
- Theory
- Operators
- Constants
- Constraints
- Memory Topology
- Execution
- Kernel
- Agent
- Evidence
- Publication Boundary
- Analogies

This is intentionally not complete.
It is a starter registry that can be expanded node by node.

---

## Relationship to other layers

Object cards should eventually connect to:

- orbital manifest export,
- inspector payload,
- runtime diagnostics,
- documentation index,
- public educational layers,
- future truth-convergence metrics.

---

## Operational rule

A node without an object card is still visible,
but it is not yet epistemically mature.

The long-term direction is:

**visible node -> object card -> test binding -> truth convergence record**
