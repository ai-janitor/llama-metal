# llama-metal

**Research, not a product.** A lab snapshot of llama.cpp's Metal backend tuned for **older Macs with AMD GPUs**: Mac Pro 2013 (FirePro D300/D500/D700), iMac and iMac Pro with Radeon Pro / Vega, MacBook Pro 2019 with Radeon Pro 5300M/5500M. One commit per update; pull requests not accepted.

Research and tuning: [ai-janitor](https://github.com/ai-janitor).

## Speed (tokens per second, this snapshot)

Tokens per second while generating, as llama.cpp prints it in its `eval time` line. Each cell is the middle of 3 runs of the "Check it on your Mac" command below (up to 128 tokens, greedy, `-fa off`, all layers on the GPU), on this snapshot's own build.

| GPU (Mac) | Qwen3.5-0.8B Q8_0 | Qwen3.5-4B Q4_K_M | Gemma 4 E4B Q4_K_M | Qwen3.5-0.8B Q8_0, `-fa on` |
|---|---|---|---|---|
| FirePro D700 (Mac Pro 2013) | 68.3 | 20.1 | 14.1 | 61.3 |
| Radeon Pro Vega 64 (iMac Pro 2017) | 160.9 | 56.5 | 38.6 | 142.8 |
| Radeon Pro 5500M (MacBook Pro 2019) | 118.8 | 38.6 | 32.3 | 116.7 |

Each column has its own command below, so every cell can be checked. Flash attention (last column) is slower than off on these GPUs.

### Check it on your Mac

After the Build steps below, with the model in `~/models`:

```bash
# models (about 0.8, 2.7 and 5.0 GB)
curl -L -o ~/models/Qwen3.5-0.8B-Q8_0.gguf https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf
curl -L -o ~/models/Qwen3.5-4B-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf
curl -L -o ~/models/gemma-4-E4B-it-Q4_K_M.gguf https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf

# column 1: Qwen3.5-0.8B
./build/bin/llama-completion -m ~/models/Qwen3.5-0.8B-Q8_0.gguf -ngl 99 -fa off --temp 0 -n 128 -no-cnv -p "The capital of France is" 2>&1 | grep "eval time"
# column 2: Qwen3.5-4B
./build/bin/llama-completion -m ~/models/Qwen3.5-4B-Q4_K_M.gguf -ngl 99 -fa off --temp 0 -n 128 -no-cnv -p "The capital of France is" 2>&1 | grep "eval time"
# column 3: Gemma 4 E4B
./build/bin/llama-completion -m ~/models/gemma-4-E4B-it-Q4_K_M.gguf -ngl 99 -fa off --temp 0 -n 128 --ignore-eos -no-cnv -p "The capital of France is" 2>&1 | grep "eval time"
# column 4: Qwen3.5-0.8B with flash attention on
./build/bin/llama-completion -m ~/models/Qwen3.5-0.8B-Q8_0.gguf -ngl 99 -fa on --temp 0 -n 128 -no-cnv -p "The capital of France is" 2>&1 | grep "eval time"
```

The number before `tokens per second` on the line that does NOT say `prompt` is the one in the table. On the same GPU, expect roughly the same figure; a much lower one usually means another app is using the GPU, the Mac is running hot, or your GPU isn't one of the three above.

## What to expect

- Tested on three GPUs only: FirePro D700, Radeon Pro Vega 64, Radeon Pro 5500M. Other AMD Macs get llama.cpp's defaults: they may work, or not.
- Not for Apple Silicon or Intel graphics: use upstream llama.cpp there.
- Measured with Qwen3.5 0.8B and 4B; other models run but weren't tuned. The model must fit in the GPU's memory.
- Research, not a product: no releases, no support, issues may go unanswered.

## Build (macOS 12 or newer, Xcode Command Line Tools, CMake)

Full Xcode is not needed, and neither is a Metal compiler: the GPU shaders are embedded as source and compiled on your Mac the first time a model runs. git comes with the Command Line Tools.

```bash
git clone https://github.com/ai-janitor/llama-metal.git
cd llama-metal
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_OPENSSL=OFF -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DGGML_BLAS=OFF
cmake --build build --target llama-completion llama-bench -j
```

On a Mac Pro 2013 keep `-DCMAKE_OSX_DEPLOYMENT_TARGET=12.0` (macOS 12 Monterey is the last macOS it supports). On newer Macs it can be your macOS version.

## Run

Get a model (the two measured above):

```bash
mkdir -p ~/models
curl -L -o ~/models/Qwen3.5-0.8B-Q8_0.gguf https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf
```

One prompt:

```bash
./build/bin/llama-completion -m ~/models/Qwen3.5-0.8B-Q8_0.gguf -ngl 99 -fa off -p "The capital of France is" -n 64 -no-cnv
```

Check that your GPU was found and which settings it got: `llama-bench` (below) prints a line starting `hw row 0x1002:0x…` with your card's name on stderr. No such line means your GPU isn't in the table yet and runs the defaults (it still works).

Speed on your card:

```bash
./build/bin/llama-bench -m ~/models/Qwen3.5-0.8B-Q8_0.gguf -ngl 99 -fa 0
```

## For AI assistants setting this up for someone

Follow these steps in order; each has a check. Stop and tell the person if a check fails.

1. `sw_vers -productVersion` → 12.0 or newer. `system_profiler SPDisplaysDataType | grep -E "Chipset Model|Metal"` → an AMD Radeon / FirePro GPU with Metal support. Apple Silicon (M1-M4): use upstream llama.cpp instead; this repo is not tuned for it.
2. `xcode-select -p` prints a path (else run `xcode-select --install` and wait for it to finish; it also installs git). `cmake --version` prints a version (tested with 4.4 and 4.5). If cmake is missing: download the macOS .dmg from cmake.org, drag CMake.app to /Applications, then run `export PATH="/Applications/CMake.app/Contents/bin:$PATH"`; to keep it, add that line to `~/.zshrc` (or `~/.bash_profile` if your shell is bash). Check again: `cmake --version`. (Homebrew no longer supports Intel Macs, so it is not used here.)
3. Clone and build with the two `cmake` lines under Build. Keep `-DGGML_METAL_EMBED_LIBRARY=ON`. Check: `./build/bin/llama-completion --version` prints a version.
4. Download one GGUF model into `~/models`. Start small (Qwen3.5-0.8B Q8_0, about 0.8 GB). Rough rule (not measured for every model): the model file should be smaller than the GPU's memory minus about 1 GB (D700: 6 GB, Vega 64: 16 GB, 5500M: 4 or 8 GB).
5. Run the one-prompt command under Run. Check: the text is sensible.
6. Always pass `-ngl 99` (all layers on the GPU) and `-fa off` (measured faster on these cards).
7. If the output is garbage or the run crashes, collect `sw_vers`, the GPU name, the exact command and the last 40 lines of stderr. Don't change kernel code or settings files to work around it.

## Research

Why this repo exists, how it works, why it isn't in llama.cpp, and what was measured: [research/](research/README.md).

## License

MIT, same as upstream llama.cpp. The llama.cpp and ggml code is © The ggml authors; the research and AMD Mac GPU tuning in this repository is © 2026 ai-janitor. If you reuse it, keep both notices.
