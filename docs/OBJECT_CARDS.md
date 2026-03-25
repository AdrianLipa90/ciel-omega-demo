# Object Cards

## Purpose

Object cards are the first explicit epistemic interface for Omega cockpit nodes.

They are meant to stop the system from exposing only labels such as:
- Theory
- Operators
- Evidence
- Boundary
- Analogies

without explaining what those layers *are*, how they relate, and how they should be checked.

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
8. **Crossrefs**
   - explicit links to other nodes which co-define, constrain, explain, or validate it

Optional:
- supporting docs
- provenance links
- future truth-convergence metrics

---

## Why this matters

The cockpit should not be a menu of names.
It should be an epistemic instrument.

That means every major node should become inspectable as an object with:
- meaning,
- source,
- code location,
- verification pathway,
- interpretive place in the whole system,
- and visible relations to other objects.

Object cards are the first compact implementation of that requirement.

---

## Coverage

The current registry covers:

### Core / parent nodes
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

### Child nodes
- Planner
- Session Dynamics
- Command Routing
- Chat
- Tool Interface
- Files
- Models
- Observability
- Tests
- Audit
- Provenance
- Crossrefs
- Analogy Registry
- Truth Attractor Analogies
- Mnemonic Book For Kids

---

## Relationship to other layers

Object cards connect directly to:

- orbital manifest export,
- inspector payload,
- orbital workspace rendering,
- documentation index,
- public educational layers,
- white-thread interpretation,
- future truth-convergence metrics.

---

## Operational rule

A node without an object card is still visible,
but it is not yet epistemically mature.

The long-term direction is:

**visible node -> object card -> crossref network -> test binding -> truth convergence record**

---

## Meaning of crossrefs

Crossrefs are not decorative.
They answer a specific question:

> What other objects must be read together with this one to avoid semantic distortion?

Examples:
- `theory -> operators`
- `execution -> kernel`
- `evidence -> audit`
- `analogies -> truth_attractor_analogies`
- `boundary -> constraints`

Crossrefs therefore serve as the first readable layer of relational topology inside the cockpit.
