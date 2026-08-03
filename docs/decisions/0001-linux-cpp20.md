# 0001: Target Linux and C++20

Status: Accepted

## Context

Glimmer will use Linux dynamic linking, CUDA-related SDKs, and Linux deployment
tools. Supporting additional operating systems would broaden the build and
runtime surface before the core design is established.

## Decision

The project targets Linux only and requires a C++20-capable compiler. CMake
configuration fails on non-Linux systems.

## Consequences

Build scripts, CI, dependencies, and runtime integrations may use Linux
facilities. Portability layers for other operating systems are out of scope
until a future ADR changes this decision.
