# Control ABI fixture

The headers in `include/`, including their project-owned transitive dependencies,
are frozen copies from commit `55118d4`. The consumer
translation unit uses those copies, while the provider uses the current public
headers. The test checks object layouts before invoking the old entry points
against the current library. This detects representative ABI regressions; it
does not prove compatibility with every compiler or standard-library version.

Do not update the frozen headers to make an incompatible change pass. Add a
separate, explicitly versioned fixture when a new compatibility baseline is
needed.
