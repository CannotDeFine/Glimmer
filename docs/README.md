# Documentation Guide

This directory contains the project rules and technical records that define how
Glimmer is built and maintained. Each document has a specific purpose and
update trigger.

| Document | Purpose | Update when |
| --- | --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Defines module responsibilities, dependency direction, and runtime boundaries. | A module is added, removed, or given a new responsibility. |
| [CODING_GUIDELINES.md](CODING_GUIDELINES.md) | Defines source, ownership, error-handling, logging, concurrency, and quality rules. | A project-wide coding rule changes. |
| [CUDA_API_COVERAGE.md](CUDA_API_COVERAGE.md) | Defines planned CUDA Driver API interception coverage, quota semantics, and compatibility tests. | A covered CUDA API, symbol path, quota rule, or compatibility claim changes. |
| [DEPENDENCY_MANAGEMENT.md](DEPENDENCY_MANAGEMENT.md) | Defines how runtime, build, and development dependencies are selected and updated. | A dependency or its integration method changes. |
| [DEVELOPMENT_LIFECYCLE.md](DEVELOPMENT_LIFECYCLE.md) | Defines the proposal, design, implementation, verification, review, integration, and release gates. | The development or review workflow changes. |
| [GIT_GUIDELINES.md](GIT_GUIDELINES.md) | Defines the single-branch workflow and commit-message format. | The contribution workflow changes. |
| [TESTING.md](TESTING.md) | Defines required test coverage and test levels. | Test policy or infrastructure changes. |
| [decisions/](decisions/) | Records accepted architecture decisions and their consequences. | A decision affects module boundaries, public behavior, dependencies, or operations. |

Documentation describes intended behavior, not guesses. Keep current code and
documentation aligned; mark a planned component as planned until it exists.
