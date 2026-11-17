# Why this repo exists, and how it works

## Why it exists

A lot of people have an older Mac with an AMD GPU and can't spend $2,000 on a new card. Those GPUs still have real memory bandwidth; most software just doesn't use it well. This project tries to get useful local AI speed out of hardware people already own.

## How this repo works

This repo is a snapshot of a one-person research lab, not a maintained fork.

The work behind it is mostly experiments on old Mac GPUs (FirePro D700, Radeon Pro Vega 64, Radeon Pro 5500M): measure where the time goes, try a change, keep it if it's faster and gives exactly the same output, throw it away if not. Most trials fail. A single day can produce dozens of probe branches, rejected kernels and dead ends, and the ones that work get reshaped several times before they settle.

Publishing that as git history would bury the useful part under the noise, and keeping branches, pull requests and issues in order on top of it is more than one person can manage while still doing the research. So:

- Each update adds one commit with the code that passed testing on all three cards.
- Pull requests aren't accepted. Issues may be read, but there's no promise of a reply.
- What was learned along the way, including why things are built the way they are, goes into the code comments where it matters.

## Why this isn't in llama.cpp

- **Every card wants different settings.** The same change can help one GPU, do nothing on another, and slow down a third. Reading the weights as 4-byte words gave the FirePro D700 about 44 % more tokens per second and the Vega 64 about 11 %, and did nothing on the Radeon Pro 5500M. Reading the shared input vector in 16-byte loads gave the D700 another 14 % and the Vega nothing. Running 8 rows per group only pays on the D700 above about 16,000 rows. So this repo keeps a measured table of settings per card, and a card that isn't in the table runs the plain defaults. The nuances are everywhere: GCN cards (D700, Vega) run 64 threads in lockstep where RDNA (5500M) runs 32; the D700 charges a whole 64-byte memory line for every read a group of threads touches; a kernel that needs a few more registers can halve how many threads a card keeps in flight; discrete cards copy results back over the bus where Apple Silicon shares memory. Generic code has to pick one middle path through all of that, and every card pays for the choices made for the others.
- **Each setting is found by measurement, not by reasoning.** A typical win comes after dozens of trials, and most of them lose. Ideas that looked right on paper (a read pattern that tested faster on its own, a fused kernel with fewer launches) were slower inside the real kernel, often because they used more registers and the GPU could run fewer threads at once.
- **Nothing here can be checked on the hardware most llama.cpp users have.** llama.cpp's Metal backend mostly serves Apple Silicon. A change upstream has to be shown not to slow down M1 to M4 chips, and this lab has none.
- **The output has to stay bit-for-bit the same.** Every change here is kept only if it gives exactly the same bytes as before, checked on every card, and fast-math is off for that reason. That rules out the usual shortcuts, like rounding inputs down to smaller types.
- **The changes reach outside the GPU code.** On the Vega 64 about a quarter of the time for each token was CPU work the GPU waited on (sampling, copying the logits back, how the work is split into command buffers). Fixing that touches shared code every backend uses.
- **llama.cpp can't move the way a lab can.** Thousands of apps, bindings and forks build on it, so its shape is held in place by what it promised years ago. In llama.cpp v1.0.0: history back to March 2023; 367 public `GGML_API` and 242 `LLAMA_API` declarations other projects link against; 18 GPU/CPU backends that all share one op set and one scheduler, so a change to how an op or a token is run has to hold on every one of them; and 176 hand-written model files, 156 of which repeat the same embed-norm-attention sequence (177 `build_attn` calls), because a new model is added by copying an old one. That stability is what makes it useful to everyone, and it is also why per-card tuning for a few old GPUs doesn't fit inside it.
- **The code had to be restructured to work on at all.** The Metal backend is built in a way that leads to copy-and-paste between kernels and backends, so the work started by reorganizing it. That alone puts this tree too far from upstream to rebase.

Turning any one of these into an upstream pull request means rebuilding it on current llama.cpp, proving it on Apple Silicon, and keeping it in step with every release. For a one-person lab that is a second full-time project. So this repo stays a snapshot: if a finding here turns out to be general, the place for it is a fresh, small pull request to llama.cpp, not a merge of this tree.
