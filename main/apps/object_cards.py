from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ObjectCard:
    key: str
    title: str
    role: str
    definition: str
    derivation: str
    implementation: str
    test: str
    status: str
    interpretation: str
    docs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


OBJECT_CARDS: Dict[str, ObjectCard] = {
    'identity': ObjectCard(
        key='identity',
        title='Identity Attractor',
        role='Central organizing state of the Omega cockpit and its active relational configuration.',
        definition='The identity attractor is the center that keeps export state, ethics, memory, resonance, and operational coherence readable as one system.',
        derivation='Derived from the attractor-centered cockpit architecture where system geometry is organized around relation and identity rather than peer runtime tabs.',
        implementation='Rendered in main.apps.omega_orbital_app with identity snapshot, inspector state, event strip, and runtime anchors.',
        test='Check root workspace load, identity snapshot population, and inspector payload consistency.',
        status='runtime-only',
        interpretation='This is not chat or a model. It is the operational center of coherence for the current Omega surface.',
        docs=['docs/OMEGA_COCKPIT_1_0.md'],
    ),
    'theory': ObjectCard(
        key='theory',
        title='Theory Interface',
        role='Formal entry layer for axioms, definitions, derivations, and structured concepts.',
        definition='Theory is the constitutive layer from which operators, constants, constraints, and memory topology derive their formal status.',
        derivation='Follows the workflow Axiom -> Definition -> Derivation -> Implementation -> Test -> Status -> Interpretation.',
        implementation='Represented as a constitutive orbit in main.apps.orbital_cockpit and surfaced in the orbital app workspace.',
        test='Verify navigation, object card rendering, and manifest export for the constitutive orbit.',
        status='provisional',
        interpretation='Theory is the source layer that gives meaning and structure to the rest of the cockpit.',
        docs=['docs/OMEGA_COCKPIT_1_0.md'],
    ),
    'operators': ObjectCard(
        key='operators',
        title='Operators',
        role='Bridge formal theory to executable or inspectable actions.',
        definition='Operators are exported formal actions, couplings, or transformations that can later be bound to runtime behavior.',
        derivation='Operators arise from the theory layer and are linked by white threads to execution and kernel layers.',
        implementation='Tracked as constitutive nodes and cross-linked to kernel logic and future object manifests.',
        test='Verify navigation and manifest coverage; later bind to explicit code paths and tests.',
        status='provisional',
        interpretation='Operators are where abstract structure begins to become actionable.',
        docs=['docs/OMEGA_COCKPIT_1_0.md'],
    ),
    'constants': ObjectCard(
        key='constants',
        title='Constants',
        role='Hold active constant sets and runtime-imported parameter values.',
        definition='Constants are stable values or named parameter bundles used by the cockpit and runtime.',
        derivation='They are exposed as constitutive nodes because execution depends on them but does not define them.',
        implementation='Surfaced in topology, manifest export, and object card registry; later bind to explicit exported constant sets.',
        test='Confirm visibility in navigation, manifest, and inspector.',
        status='runtime-only',
        interpretation='Constants are the stable anchors that prevent the runtime from becoming semantically shapeless.',
        docs=['docs/OMEGA_COCKPIT_1_0.md'],
    ),
    'constraints': ObjectCard(
        key='constraints',
        title='Constraints',
        role='Limit valid motion and preserve coherent state-space boundaries.',
        definition='Constraints define what counts as a valid path, state, or transformation inside the Omega cockpit.',
        derivation='They are formalized as constitutive conditions and linked to execution by bounding white threads.',
        implementation='Currently represented in topology and object cards; later should map to explicit runtime and publication checks.',
        test='Verify object-card rendering and manifest inclusion; later bind to real validations.',
        status='provisional',
        interpretation='Constraints do not kill freedom; they shape meaningful possibility.',
        docs=['docs/OMEGA_COCKPIT_1_0.md', 'docs/analogies/MNEMONIC_BOOK_FOR_KIDS.md'],
    ),
    'memory_topology': ObjectCard(
        key='memory_topology',
        title='Memory Topology',
        role='Represent the structured arrangement of memory channels and dependencies.',
        definition='Memory topology is the pattern of relational memory rather than mere storage of isolated items.',
        derivation='It follows from the project preference relation -> identity -> memory, where memory is downstream of identity.',
        implementation='Currently represented as a constitutive node; future work should bind it to explicit channels and attractor logic.',
        test='Verify topology presence and card availability; later test channel graphs and persistence semantics.',
        status='draft',
        interpretation='Memory is a tensioned structure, not a bag of disconnected facts.',
        docs=['docs/analogies/ANALOGY_REGISTRY.md'],
    ),
    'execution': ObjectCard(
        key='execution',
        title='Execution',
        role='Coordinate runtime movement, orchestration, and task progression.',
        definition='Execution is the dynamic layer that turns a prepared configuration into actual process flow.',
        derivation='It sits between constitutive formalism and interaction-level agent behavior.',
        implementation='Rendered in the orbital app with runtime config, kernel registry, and latest metrics.',
        test='Verify HTTP startup, execution panel rendering, and metrics presence.',
        status='runtime-only',
        interpretation='Execution is where the system stops describing motion and starts moving.',
        docs=['docs/REPORT_LOCAL_DEBUG_2026-03-25.md'],
    ),
    'kernel': ObjectCard(
        key='kernel',
        title='Kernel',
        role='Provide the active computational engine for state updates.',
        definition='The kernel is the runtime engine that consumes configuration and performs operational state changes.',
        derivation='Execution depends on kernel availability but the kernel itself is still downstream of theory and constraints.',
        implementation='Backed by main.kernels.registry and surfaced in runtime config and event strip.',
        test='Check kernel registry rendering and launcher startup.',
        status='runtime-only',
        interpretation='The kernel turns system motion into work.',
        docs=['docs/analogies/ANALOGY_REGISTRY.md'],
    ),
    'agent': ObjectCard(
        key='agent',
        title='Agent',
        role='Host chat, tools, and user-facing interaction pathways.',
        definition='The agent layer is the interaction shell through which the system meets inputs, outputs, and tools.',
        derivation='It is downstream of execution and kernel state, and upstream of files, models, and tool routing.',
        implementation='Rendered in the orbital app with shell status and legacy compatibility notes.',
        test='Verify agent workspace rendering and chat/tool visibility.',
        status='runtime-only',
        interpretation='The agent is not the whole system; it is the interaction face of a deeper architecture.',
        docs=['docs/OMEGA_COCKPIT_1_0.md'],
    ),
    'evidence': ObjectCard(
        key='evidence',
        title='Evidence',
        role='Expose logs, tests, audit, provenance, and diagnostics as epistemic pressure.',
        definition='Evidence is the observation layer that supports, weakens, or constrains system claims.',
        derivation='Evidence is distinct from theory and execution because it measures rather than merely declares or runs.',
        implementation='Rendered with log tails and epistemic state badges; linked to publication boundary by validation threads.',
        test='Check evidence workspace rendering and log accessibility.',
        status='runtime-only',
        interpretation='Evidence is the footprint trail that keeps the system honest.',
        docs=['docs/analogies/ANALOGY_REGISTRY.md', 'docs/REPORT_LOCAL_DEBUG_2026-03-25.md'],
    ),
    'boundary': ObjectCard(
        key='boundary',
        title='Publication Boundary',
        role='Separate internal state from external release and publication surfaces.',
        definition='The publication boundary decides what remains private, what is exported, and what is fit for public release.',
        derivation='It is required by the project split between private canon, public theory export, and runtime demo surfaces.',
        implementation='Rendered in the orbital app with visibility classes and future sanitization placeholders.',
        test='Verify boundary workspace rendering and export-state visibility.',
        status='private-only',
        interpretation='The boundary is a careful gate, not merely a wall.',
        docs=['docs/OMEGA_COCKPIT_1_0.md', 'docs/analogies/MNEMONIC_BOOK_FOR_KIDS.md'],
    ),
    'analogies': ObjectCard(
        key='analogies',
        title='Analogies',
        role='Translate hard formal concepts into mnemonic images without replacing the formal layer.',
        definition='Analogies form an educational layer that supports teaching, memory, and public explanation.',
        derivation='They are downstream of theory and linked to agent and evidence as explanatory bridges.',
        implementation='Rendered in omega_orbital_app as an education orbit and exported through orbital_manifest_export.',
        test='Verify navigation, markdown rendering, and inspector payload for educational nodes.',
        status='public-exported',
        interpretation='Analogies are the pedagogical bridge between formalism and memory.',
        docs=['docs/analogies/README.md', 'docs/analogies/ANALOGY_REGISTRY.md'],
    ),
}


def get_object_card(key: str) -> Optional[Dict[str, object]]:
    card = OBJECT_CARDS.get(key)
    return None if card is None else card.to_dict()


def export_object_cards() -> Dict[str, Dict[str, object]]:
    return {key: card.to_dict() for key, card in OBJECT_CARDS.items()}
