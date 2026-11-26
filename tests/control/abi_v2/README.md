# Control ABI fixture

The headers in `include/`, including their project-owned transitive dependencies,
are frozen copies from commit `db74ab6`. The consumer
translation unit uses those copies, while the provider uses the current public
headers. The test checks object layouts before invoking the baseline entry points
against the current library. This detects representative ABI regressions; it
does not prove compatibility with every compiler or standard-library version.

Do not update the frozen headers to make an incompatible change pass. Add a
separate, explicitly versioned fixture when a new compatibility baseline is
needed.

This fixture replaces the v1 baseline frozen from commit `55118d4`. Commit
`db74ab6` intentionally broke that baseline: it changed the size of
`TaskAdmissionService`, `TaskProtocolRequest`, `TaskProtocolResponse`, their
parse results, `ControlRequestTiming`, `UnixSocketControlClient` and
`UnixSocketControlServer`, and replaced several entry points with signatures
that take additional parameters. Binaries built against the v1 headers must be
rebuilt.
