# ggml-metal: the Metal backend, organized by what varies

## Why it is laid out this way
This backend is built to run on old GPUs as well as new ones. Every old card differs: in its capability
flags, in the kernels it needs, and in which form of a kernel is fast on it. If each difference is a branch
or a copied kernel inside a few very large files, the cost of one more card grows with every card already
supported, and at some point no further legacy hardware can be carried.

So each thing that varies has exactly one home. One more legacy card is then a data row, plus at most one
more variant of an existing template, and the cost per card stays flat.

## One home per axis
| What varies | Home | What a new member costs |
|---|---|---|
| Op family (matmul, flash-attn, norm, rope, copy, rows, softmax, ssm, unary, conv, ...) | `<family>/`: its kernels (`.metal`), its launch code (`ops-*.cpp` / `dispatch*.cpp`), its pipeline selection (`pipeline*.cpp`) and its answer to "can the GPU run this op" (`supports-<family>.m`) | files in one folder |
| Weight format (q4_K, q6_K, iq4_xs, bf16, ...) | the matvec and block-dot kernels in `matmul/`, the type list in `matmul/supports-matmul.m` | a kernel instantiation and a type-list entry |
| Variant of a kernel (rows per group, columns per launch, form) | compile-time template values and function constants | a value |
| Card | data rows (`data/device.tsv`, and the per-card sections of `data/matmul.tsv` and `data/arithmetic.tsv`) and the capability flags read in `device/init.m` | a row |
| Arithmetic lane (fp32 multiply-add, packed fp16 multiply, a dot instruction, a matrix unit) | `lanes/`: one file per lane, wired by `data/arithmetic.tsv` section `[lanes]`, allowed per card by its section `[capabilities]` | a file and a row |
| A copy of an op's activations at the model's precision, converted once per op | `act-copy/`: the scratch buffer and the conversion launch, allowed per family and card by `data/arithmetic.tsv` section `[copies]` | a family: its loader takes the element type, one `_a<type>` entry, rows with `applies` no |
| Batch route of a matmul per card (the activation column count from which the small-batch and the tiled kernel take it) | the thresholds of `data/matmul.tsv` section `[tiles]`, read by `matmul/batch-route.cpp`, which both routes ask | a row |
| Layout of a kernel on a card (tile sizes, threads per group, group memory) | a rule function over the card object (`card-object.h`: card facts read once from the device); first one `matmul/tiled-plan.cpp` for `matmul/kernel-tiled.metal`, which is compiled once per plan with the plan as compile-time constants | a rule line; a fact only when a rule reads it |
| Fusion pattern (several graph nodes run as one kernel) | `fusion/`: one file per pattern, the pattern table in `fusion/list.cpp` | a file and a table entry |
| Machine plumbing | `device/`, `memory/`, `context/` (below) | - |

## The file tree is the map
What exists and what is wired where can be read from the folders and the tables beside them, without
reading code. Each member of an axis is a named file or a row. Nothing that decides which code runs is
hidden as an `#if` per card or lane, an environment variable, a device-name check, or a helper inside
another job's file.

## Two paths: legacy GPUs and Apple Silicon
- The legacy path is the discrete AMD and Intel GPUs of older Macs. Their workarounds, their lanes and any
  code built outside Metal (LLVM code objects, `lanes/llvm/`) belong to this path only.
- The Apple Silicon unified-memory path is upstream's main path. Upstream llama.cpp already serves
  Apple Silicon well, so anyone on Apple Silicon should use upstream; this repository does not support it
  and is not tested on it. By default that path runs what upstream
  runs, and the negotiation below lands on upstream's choice there. A change made for the legacy path may
  still reach it; such a change must compile there and is marked as not run on Apple Silicon. Legacy behavior is selected by capability rows that Apple Silicon does not match,
  never by editing a shared default.
- One source serves both. A shared kernel has one file; the two paths differ by an instantiation line or a
  table row, never by a copy, and every change must compile and run on both.

## Which number format runs: a negotiation between the model and the hardware
- Between ops, activations are handed off as float32, for every model. Inside an op that reads them more
  than once (a tiled or batched matrix multiply), they are converted once, at the start of the op, to the
  model's own precision, and every re-read uses that copy. Never to a lower precision than the model's.
- The model states what it needs: its own precision (for example bfloat16 for a model
  trained in bfloat16, float32 for a recurrent state). That requirement must reach the kernel as the tensor's
  type; no layer in between replaces it with a format that is merely easier to code.
- The hardware states what it can do: what the chip has, what Metal reaches, and what is correct on it
  (`data/arithmetic.tsv` section `[capabilities]`).
- The two are balanced. In order: the model's format natively where the card has it; else the same format
  held exactly another way (for example bfloat16 stored as 16-bit words and widened to float32 by a shift,
  on a card whose Metal has no bfloat type); else a wider format, with its cost stated. Never a narrower one.
  Among the choices that keep the model's format, the cheapest the card runs well wins, by measurement.
- Inside an op the order is: convert once to the model's precision; then the lane selector uses an accelerator
  lane (a dot or packed instruction) if the card has it, Metal reaches it, it is correct, and it holds that
  precision exactly; else the scalar lane (widen to float32, multiply-add), which is always correct.
- The code accommodates the negotiation; it is not a party to it. It offers every way the two can meet and
  carries out the result, recorded as a row of `data/arithmetic.tsv` section `[lanes]` with its reason. A result with no code path
  yet is a coverage gap to close, not a reason to change the result.

## Folders
| Folder | Job |
|---|---|
| `device/` | `init.m`: GPU properties, capability flags, the card profile and hardware-row lookup. `library.m`: kernel library build and load, pipeline compile, function constants. `encoder.m`: the compute encoder wrapper. `profiler.m`: per-dispatch timing. `test-hooks.m`: test overrides and the dispatch receipt. `ggml-metal-device.cpp`: pipeline lookup by name. |
| `memory/` | `placement.m`: shared or private buffers for this device. `buffers.m`: buffer create / map / free, tensor set and get, weight preparation at load. `buffer-types.cpp`: the shared, private and mapped buffer types the backend registers. `residency.m`: residency sets. `transfer.m`: uploads, downloads and copies between contexts. `hazards.cpp`: memory ranges read and written by a group of nodes (what may run concurrently). |
| `context/` | `init.m`: create and free the run state. `sync.m`: waits and command-buffer error checks. `graph-compute.m`: split a graph into command buffers and submit them. `profile-per-op.m`: the one-command-buffer-per-op timing mode. `optimize.cpp`: node reordering. `events.m`: graph optimize entry, events, settings. |
| `fusion/` | pattern checks (`norm.cpp`, `gdn-cache.cpp`, `ssm-conv-silu.cpp`, `conv-state.cpp`, `small.cpp`) and the matcher (`list.cpp`). |
| `<family>/` | one folder per op family, as in the table above. |
| `kernels/` | `00-common.metal`: shared shader helpers, included first. |
| `ops/` | `ggml-metal-ops.cpp`: the switch from a graph node to its family's launch function. |

The files `ggml-metal-device.m`, `ggml-metal-context.m` and `ggml-metal-fusion.cpp` are facades: they
include the internal header of their folder and hold what is shared. `ggml_metal_device_supports_op` lives
in `ggml-metal-device.m`: a short common part, then one call per family.

## The tables
The tables live in `data/`. The `#` header of each `.tsv` file defines its columns, the code decides what a value does, and a mismatch between the
two is a defect. This section is the index: one row per table of this folder, with the question it answers and where
the code reads it. Every generator runs when CMake configures (see `CMakeLists.txt`), refuses a malformed row with a
message naming it, and writes a header into the build folder, so no table is read at run time.
`make -C tests check-tables-doc OUT=<dir>` fails when a `.tsv` of this folder is missing from this section, a cited
line no longer holds the named function, or a column of a table has no comment line in its own file.

| Table | The question it answers | How it reaches the code | Check | Does a row change change output? | Columns |
|---|---|---|---|---|---|
| `data/matmul.tsv` | Section `[formats]`: what does each weight type need from the matvec and matmul kernels, and which routes exist for it? Section `[tiles]`: on this card and for this weight type, from how many activation columns does the small-batch kernel, and then the tiled kernel, take a matmul, and which work plan does the tiled kernel run; and for which work plans is the tiled kernel compiled? Section `[matvec_defaults]`: on this card, how do the one-column kernels sum, load and group? Section `[matvec_shapes]`: on this card and for this K-quant weight shape, how many rows does a simdgroup work, and in which loop form? | `[formats]`: `matmul/gen-formats.cmake` → `mv-formats-table.h` → `matmul/formats.h:47` (`ggml_metal_mv_format`). `[tiles]`: `matmul/gen-tiles.cmake` → `batch-routes-table.h` → `matmul/batch-route.cpp:33` (`ggml_metal_batch_route_decide`), asked by both routes through `matmul/dispatch.h:82` (`ggml_metal_batch_route`); → `tiled-plan-table.h` → `matmul/tiled-plan.cpp:44` (`ggml_metal_tiled_tune_load`), into the card's object (`card-object.h`); → `tiled-plans.metal` (the kernel's plan blocks) and `tiled-compiled-table.h` → `matmul/tiled-plan.cpp:23` (`ggml_metal_tiled_compiled_count`). `[matvec_defaults]` and `[matvec_shapes]`: `matmul/gen-matvec.cmake` → `matvec-rows.h` → `device/init.m:460` (`ggml_metal_matvec_default_row_of`) and `device/init.m:509` (`ggml_metal_mv_rows_of`) into the device profile, whose shape rows `matmul/dispatch.h:16` (`ggml_metal_mv_hw_row`) picks from | `make -C tests check-batch-routes OUT=<dir>`, `make -C tests check-tiled-plan OUT=<dir>` and `make -C tests check-device-rows OUT=<dir>`, then `test-backend-ops -o MUL_MAT -p type_a=<type>` on the card | `[formats]` and the thresholds: yes, another kernel runs (the tiled kernel sums in another order, so text can change). A plan: no, the same products in the same order; it changes speed and the library's first compile time. `[matvec_defaults]` and `[matvec_shapes]`: another kernel form runs; `mv_reduce` changes the order of a row's sum, so text can change, the other cells were each measured with equal bytes. | see the header of `data/matmul.tsv` |
| `data/arithmetic.tsv` | Section `[capabilities]`: what can each card do per arithmetic capability: is the instruction in the chip, does the shader compiler reach it, is the result correct? Section `[lanes]`: which arithmetic lane is wired into which op family and route, for which operand types, and may it be picked? Section `[copies]`: for this card, op family and route, are an op's activations copied once to the model's precision before the kernel reads them? | `[capabilities]` and `[lanes]`: `lanes/gen-tables.cmake` → `lanes-table.h` → `lanes/select.cpp:17` (`ggml_metal_lane_ok_mask`) and `lanes/select.cpp:40` (`ggml_metal_negotiate_lane_in`). `[copies]`: `act-copy/gen-act-copy.cmake` → `act-copy-table.h` → `act-copy/decide.cpp:13` (`ggml_metal_act_copy_decide_in`) | `make -C tests check-lanes-table OUT=<dir>` and `make -C tests check-act-copy OUT=<dir>` | `[lanes]`: only a row that is on and not exact; the check refuses such a row unless its origin is upstream. `[capabilities]`: only together with a `[lanes]` row that is on and names the capability. `[copies]`: yes when a row applies: the kernel then reads the activations at the model's precision. Every shipped row is off. | see the header of `data/arithmetic.tsv` |
| `data/device.tsv` | How large is this card, how does the backend schedule work on it (graph fusions, command buffers, uploads, the logits staging buffer), and which timing probes work on it? | `device/gen-devices.cmake` → `device-rows.h` → `device/init.m:459` (`ggml_metal_device_row_of`), into the device profile | `make -C tests check-device-rows OUT=<dir>` | The sizes: no, they tune choices that keep the same products. The switches: other kernels and launches run; each was measured with equal logits on its card. | see the header of `data/device.tsv` |
| `data/attention.tsv` | Which flash attention route takes an op, given the card and the op's shape and types? | `flash-attn/gen-routes.cmake` → `routes-table.h` → the loop at `flash-attn/router.cpp:98` (`fa_route_row_matches`) | `make -C tests check-fa-routes OUT=<dir>`, then `test-backend-ops -o FLASH_ATTN_EXT` | Yes: another kernel runs. | see the header of `data/attention.tsv` |

## Rules
- A file does one job. When a second job moves in, it moves out.
- A kernel body is not copied to add a format, a column count or a card. Add a template value, a function
  constant or a row.
- `context/graph-compute.m` only submits work. Nothing model-specific goes there.
- "The GPU can run this" (`supports-<family>.m`) changes in the same commit as the kernel that makes it true.
  An op the backend does not accept runs on the CPU, so a wrong "yes" is a wrong result and a wrong "no" is
  a slow path nobody sees.

## How to add
- **An op:** the kernel in `<family>/*.metal`, the launch function in `<family>/ops-*.cpp` (declared in
  `ggml-metal-ops.h`, called from `ops/ggml-metal-ops.cpp`), the pipeline lookup in `<family>/pipeline*.cpp`,
  the argument struct in `<family>/kargs-<family>.h` (included by `ggml-metal-impl.h`), and its case in `<family>/supports-<family>.m` plus the call in
  `ggml_metal_device_supports_op`. Check with `test-backend-ops -o <OP>`.
- **A weight format:** the block layout comes from `ggml-common.h`; add the dequantize / block-dot function
  and the kernel instantiations in `matmul/` (matvec, small-batch, tiled), `rows/` (get_rows) and `copy/`,
  the tuning entry in `matmul/tuning.h`, and the type in `matmul/supports-matmul.m`.
- **A card:** read its capability flags in `device/init.m` (they come from the device, not from its name);
  add rows for it in `data/device.tsv` and the per-card sections of `data/matmul.tsv` only where a measured value differs from the default. No card
  name in a kernel or in launch code.
- **A model:** the graph builder lives outside this folder (`src/models/`). Here it needs only the ops and
  fusion patterns it uses; add those in their family folders and in `fusion/`.

### CPU and GPU timing probes

Build with `-DGGML_METAL_PROBES=ON` to include the timing probes. Set
`ZR_ON=1 ZR_OUT=/absolute/path/probe.tsv` to record a run. The recorder
writes the dump at exit. The `ZR_CAP` environment variable sets the event
capacity. The dump header reports dropped records.

CPU zones include graph construction, input and result transfer, per-family
encoding, submission and sampling. GPU rows keep the command-buffer span
and finer timestamp intervals where supported by the device. Resolve uses
the existing synchronization points. A mixed command buffer is labeled
`mixed`; its elapsed time cannot be attributed to one kernel.

The dump carries dictionaries for family (`F`), zone (`Z`), ggml op (`O`),
and kernel per device (`K`). Interval records carry the family, zone and op
in the event ID, with the top bit set for the end record. GPU interval
values reference the kernel dictionary. The invalid and dropped zone
values count GPU coverage gaps. CPU, GPU and nested intervals can overlap.
