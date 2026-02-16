# Metal Discrete GPU — Architecture & Roadmap

This fork makes llama.cpp's Metal backend work on all GPUs, not just Apple Silicon.

## Documents

| Doc | What |
|-----|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Two memory paths, device profile struct, multi-GPU/RPC strategy |
| [BACKLOG.md](BACKLOG.md) | All work items — completed, active, backlog |
| [REFACTOR-005-PLAN.md](REFACTOR-005-PLAN.md) | Current refactor plan — shader reorg, capability tiers |

## Dependency Chain

```
REFACTOR-005 (profile struct + shader reorg)
    → FEAT-002 (adaptive dispatch using profile locally)
        → FEAT-003 (RPC profile exchange for remote devices)
```

## Work Items (detailed)

Detailed theory files, investigation logs, and task tickets live at:
`~/.knowledge/llama-cpp/work/<TYPE>-<NNN>-<slug>/`

These docs are the summary — the knowledge base has the full trail.
