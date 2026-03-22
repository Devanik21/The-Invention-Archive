# 🌌 The Invention Archive

> **A Long-Term, Hyper-Structured Archival System for Conceptual Constructs, Systems, and AI Architectures**

The **Invention Archive** is designed to act as an immutable, century-spanning ledger and data repository for 21 advanced theoretical and functional frameworks. It provides strict isolation, standardized taxonomy, and an automated synchronization pipeline mapping external active repositories into static, version-controlled archival snapshots.

The archive mandates an exceptionally strict structural paradigm to ensure data survivability, universal human-and-machine readability, and architectural coherence without relying on complex dependency chains or proprietary data structures.

---

## 🏛️ Architectural Philosophy

The repository enforces a **zero-entropy** paradigm: an uncompromisingly rigid structural and format constraint applied universally across all directories. This ensures that the context, metadata, and structural logic of the archive remain independently verifiable and readable for generations.

### Immutable File Format Protocol
To guarantee long-term accessibility, the archive strictly limits file formats to:
* **`*.md`** — Markdown: For human-readable theoretical context, evolutionary tracking, and logs.
* **`*.json`** — JSON: For deterministic, machine-readable metadata and telemetry.
* **`*.zip`** — ZIP: For isolated, immutable point-in-time source code snapshots.
* **`*.txt`** — Plaintext: For manifest data and filesystem tree snapshots.

**Constraint:** No arbitrary extensions, binaries, or complex artifacts are permitted outside of the designated `snapshot/` containers.

---

## 📂 Core Topography

The root filesystem is strictly divided into functional tracking domains and isolation modules:

* **`constructs/`** — The primary functional layer. Each conceptual framework or invention is isolated in a sequentially numbered directory (e.g., `001_HRF`, `013_HM`).
  * **`constructs/Misc/`** — A sub-layer containing 8 auxiliary or miscellaneous frameworks structurally equivalent to the core layer.
* **`docs/`** — System-level documentation, including `repo-tracker.json` and chronological `daily-log.md` files.
* **`standards/`** — The operational protocols that govern the archive (`naming.md`, `structure.md`, `versioning.md`).
* **`archive/`** — Designated storage for fully deprecated or legacy static assets.

### Master Tracking Vectors
* **`INDEX.md`** — The deterministic master registry mapping every construct's ID, Acronym, Type, Status, and Internal Path via a strict `| ID | Name | Type | Status | Path |` table schema.
* **`TIMELINE.md`** — The chronological timeline of construct inception formatted strictly as `YYYY → NAME`.
* **`RELATIONS.md`** — The dependency graph defining inter-construct architectural links.
* **`LEXICON.md`** — A unified theoretical glossary mapping the terminology shared across the diverse subsystems.

---

## 🧬 Construct Anatomy

Every isolated unit (a "Construct") within the archive adheres to an identical internal skeletal structure:

```text
constructs/001_HRF/
├── metadata.json       # Deterministic configuration and lifecycle state
├── README.md           # Cloned external context and overview
├── source_repo.md      # Absolute URI mapping to the external live repository
├── notes/
│   ├── evolution.md    # Log of paradigm shifts and theoretical adjustments
│   └── thoughts.md     # Unstructured theoretical exploration
├── versions/
│   └── v1.md           # Version-locked architectural design documents
└── snapshot/
    ├── README.md       # Immutable snapshot of the external README
    ├── tree.txt        # The exact filesystem tree of the repository at snapshot time
    └── [RepoName].zip  # Compressed, complete source code of the active project
⚙️ Automated Archival Synchronization
The Invention Archive utilizes continuous integration pipelines via GitHub Actions to systematically map and pull external codebases into their designated archival pods.

perfect_sync.yml
Domain: 13 Core Constructs (constructs/001_HRF ➔ 013_HM)
Mechanism: Reads the exact external source_repo.md mappings, retrieves the latest .zip archives, downloads the README.md files, and injects them directly into the isolated snapshot/ layer. It additionally mirrors the README up to the construct root, preserving structural integrity.
misc_sync.yml
Domain: 8 Miscellaneous Constructs (constructs/Misc/001_LIM ➔ 008_DU)
Mechanism: Operates on the same parameters but exclusively targets the auxiliary logic manifolds.
This dual-pipeline architecture ensures that active external repositories are continuously flattened into static, immutable archival data points without human intervention.

🌌 The 21 Constructs
Core Frameworks (13)
001_HRF — Harmonic-Resonance-Forest
002_NECF — Non-Equilibrium-Cognitive-Field
003_TSP — The-Schrodinger-Paradox
004_AS — Aether-SPARC
005_FRAE — FRAE
006_TCA — The-Cytomorphic-Architecture
007_DDG — Dreamer-Dark-Genesis
008_CS — causa-sui
009_RHO — Recursive-Hebbian-Organism
010_LDD — Lucid-Dark-Dreamer
011_BA — BSHDER-Architecture
012_GGEO — GENEVO-GENetic-EVolutionary-Organoid
013_HM — HAG-MoE
Auxiliary / Miscellaneous Frameworks (8)
M01_LIM — Latent-Inference-Manifold
M02_LCM — Latent-Consensus-Manifold
M03_XV — xylia-vision
M04_LB — Life-Beyond
M05_TM — Thermodynamic-Mind
M06_DTM — Dark-Thermodynamic-Mind
M07_AARGE — AION-Algorithmic-Reversal-of-Genomic-Entropy
M08_DU — Deep-Universe
