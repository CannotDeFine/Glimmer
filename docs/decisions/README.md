# Architecture Decision Records

An Architecture Decision Record (ADR) captures a decision that is expensive to
rediscover or reverse. ADRs describe the decision, not an implementation log.

Create an ADR when changing a module boundary, public contract, dependency
strategy, runtime process model, security model, or scheduler guarantee.

Use monotonically increasing identifiers and keep accepted records immutable.
If a decision changes, add a new record that supersedes the old one.

## Template

```text
# <number>: <decision title>

Status: Accepted | Superseded | Deprecated

## Context

What problem requires a decision?

## Decision

What will the project do?

## Consequences

What becomes easier, harder, required, or intentionally unsupported?
```
