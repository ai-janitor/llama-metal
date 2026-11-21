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

After the Build steps below, from the `llama-metal` folder:

```bash
# models (about 0.8, 2.7 and 5.0 GB); -C - resumes a download that was cut off: run the same line again
mkdir -p ~/models
curl -L -C - -o ~/models/Qwen3.5-0.8B-Q8_0.gguf https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf
curl -L -C - -o ~/models/Qwen3.5-4B-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf
curl -L -C - -o ~/models/gemma-4-E4B-it-Q4_K_M.gguf https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf

# column 1: Qwen3.5-0.8B
./build/bin/llama-completion -m ~/models/Qwen3.5-0.8B-Q8_0.gguf -ngl 99 -fa off --temp 0 -n 128 -no-cnv -p "The capital of France is" 2>&1 | grep "eval time"
# column 2: Qwen3.5-4B
./build/bin/llama-completion -m ~/models/Qwen3.5-4B-Q4_K_M.gguf -ngl 99 -fa off --temp 0 -n 128 -no-cnv -p "The capital of France is" 2>&1 | grep "eval time"
# column 3: Gemma 4 E4B
./build/bin/llama-completion -m ~/models/gemma-4-E4B-it-Q4_K_M.gguf -ngl 99 -fa off --temp 0 -n 128 --ignore-eos -no-cnv -p "The capital of France is" 2>&1 | grep "eval time"
# column 4: Qwen3.5-0.8B with flash attention on
./build/bin/llama-completion -m ~/models/Qwen3.5-0.8B-Q8_0.gguf -ngl 99 -fa on --temp 0 -n 128 -no-cnv -p "The capital of France is" 2>&1 | grep "eval time"
```

The number before `tokens per second` on the line that does NOT say `prompt` is the one in the table. On the same GPU, expect roughly the same figure; a much lower one usually means another app is using the GPU, the Mac is running hot, or your GPU isn't one of the three above. Run each command a second time before reading it: the first run of each model also compiles GPU shaders and can read a few percent low.

## What to expect

- Tested on three GPUs only: FirePro D700, Radeon Pro Vega 64, Radeon Pro 5500M. Other AMD Macs get llama.cpp's defaults: they may work, or not.
- Not for Apple Silicon or Intel graphics: use upstream llama.cpp there.
- Measured with Qwen3.5 0.8B and 4B; other models run but weren't tuned. The model must fit in the GPU's memory.
- Research, not a product: no releases, no support, issues may go unanswered.

## Build (macOS 12 or newer, Xcode Command Line Tools, CMake)

Full Xcode is not needed, and neither is a Metal compiler: the GPU shaders are embedded as source and compiled on your Mac the first time a model runs. git comes with the Command Line Tools. Every step below is a terminal command, so it also works over ssh.

### Before you build

Keep the Mac awake. A Mac reached over ssh, or left alone during a long download, goes to sleep and drops the connection. Turn sleep off (needs an admin password), and check:

```bash
sudo pmset -a sleep 0 powernap 0
pmset -g | grep -E " sleep|powernap"     # both 0
```

Without an admin password, start each long command with `caffeinate -i` instead (for example `caffeinate -i cmake --build build …`).

Command Line Tools: `xcode-select -p` prints a path if they (or Xcode) are installed. If not, install them from the terminal (needs an admin password):

```bash
touch /tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
softwareupdate -l | grep "Command Line Tools"     # note the label after "Label: "
sudo softwareupdate -i "Command Line Tools for Xcode-14.2" --verbose   # use the label printed above
rm /tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
xcode-select -p
```

(`xcode-select --install` does the same through a dialog on the Mac's screen; it can't be answered over ssh.)

CMake: `cmake --version` prints a version if installed (tested with 4.4 and 4.5). If not, install it from the terminal; no admin password is needed when your user is an administrator:

```bash
curl -L -o /tmp/cmake.dmg https://github.com/Kitware/CMake/releases/download/v4.5.1/cmake-4.5.1-macos-universal.dmg
hdiutil attach -nobrowse -readonly -mountpoint /tmp/cmake-mnt /tmp/cmake.dmg
cp -R /tmp/cmake-mnt/CMake.app /Applications/
hdiutil detach /tmp/cmake-mnt
rm /tmp/cmake.dmg
export PATH="/Applications/CMake.app/Contents/bin:$PATH"
cmake --version
```

To keep `cmake` on your PATH, add the `export` line to `~/.zshrc` (or `~/.bash_profile` if your shell is bash). (Homebrew no longer supports Intel Macs, so it is not used here.)

### Build

```bash
git clone https://github.com/ai-janitor/llama-metal.git
cd llama-metal
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_OPENSSL=OFF -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DGGML_BLAS=OFF
cmake --build build --target llama-completion llama-bench -j
```

On a Mac Pro 2013 keep `-DCMAKE_OSX_DEPLOYMENT_TARGET=12.0` (macOS 12 Monterey is the last macOS it supports). On newer Macs it can be your macOS version.

The build takes about 2 minutes on a Mac Pro 2013. Warnings are expected ("OpenMP not found", "ccache not found", compiler warnings); the build worked if it ends with `Built target llama-bench` and `./build/bin/llama-completion --version` prints a version.

## Run

Get a model (the two measured above):

```bash
mkdir -p ~/models
curl -L -C - -o ~/models/Qwen3.5-0.8B-Q8_0.gguf https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf
```

One prompt:

```bash
./build/bin/llama-completion -m ~/models/Qwen3.5-0.8B-Q8_0.gguf -ngl 99 -fa off -p "The capital of France is" -n 64 -no-cnv
```

The first run compiles the GPU shaders: about 45 seconds on a FirePro D700 before any text appears. Later runs start in a few seconds. This command checks that the model works; its speed figure includes that first-run work, so use the "Check it on your Mac" commands or `llama-bench` for speed.

Chat (type a message and press Enter; Ctrl-D ends the chat). This repo has no `llama-cli`; `llama-completion` in conversation mode is the chat:

```bash
./build/bin/llama-completion -m ~/models/Qwen3.5-0.8B-Q8_0.gguf -ngl 99 -fa off -cnv
```

Qwen3.5 writes a thinking block ("Thinking Process: …" up to `</think>`) before its answer; don't set a small `-n` in chat, or it can stop before answering.

Check that your GPU was found and which settings it got: `llama-bench` (below) prints a line on stderr containing `hw row` and your card's name, for example `ggml_metal_device_init: hw row AMD Radeon HD - FirePro D700 (FirePro D700 (GCN 1.0, Tahiti)): CUs=32, …`. No such line means your GPU isn't in the table yet and runs the defaults (it still works). `llama-completion` doesn't print this line.

Mac Pro 2013 with two GPUs: llama.cpp lists and uses one of them (`./build/bin/llama-bench --list-devices` shows `MTL0`); nothing to set.

Speed on your card:

```bash
./build/bin/llama-bench -m ~/models/Qwen3.5-0.8B-Q8_0.gguf -ngl 99 -fa 0
```

## For AI assistants setting this up for someone

Follow these steps in order; each has a check. Stop and tell the person if a check fails.

0. Over ssh, keep the Mac awake first ("Before you build": `sudo pmset -a sleep 0 powernap 0`, or `caffeinate -i` before each long command). A Mac that sleeps drops the connection mid-download.
1. `sw_vers -productVersion` → 12.0 or newer. `system_profiler SPDisplaysDataType | grep -E "Chipset Model|Metal"` → an AMD Radeon / FirePro GPU with Metal support. Apple Silicon (M1-M4): use upstream llama.cpp instead; this repo is not tuned for it.
2. `xcode-select -p` prints a path; if not, install the Command Line Tools with the `softwareupdate` lines under "Before you build" (needs the person's admin password: ask them to type it; it also installs git). `cmake --version` prints a version (tested with 4.4 and 4.5); if not, run the CMake lines under "Before you build" (curl, hdiutil attach, cp, hdiutil detach, export PATH). Check again: `cmake --version`. No step needs a click on the Mac's screen.
3. Clone and build with the two `cmake` lines under Build. Keep `-DGGML_METAL_EMBED_LIBRARY=ON`. Warnings (OpenMP, ccache, compiler) are expected. Check: `./build/bin/llama-completion --version` prints a version.
4. Download one GGUF model into `~/models` (`mkdir -p ~/models` first; `curl -L -C -` so a cut download resumes when run again). Start small (Qwen3.5-0.8B Q8_0, about 0.8 GB). Rough rule (not measured for every model): the model file should be smaller than the GPU's memory minus about 1 GB (D700: 6 GB, Vega 64: 16 GB, 5500M: 4 or 8 GB).
5. Run the one-prompt command under Run (the first run waits about 45 s compiling shaders). Check: the text is sensible. Then the chat line under Run, one short message, Ctrl-D: check the reply after `</think>` is sensible.
6. Always pass `-ngl 99` (all layers on the GPU) and `-fa off` (measured faster on these cards).
7. If the output is garbage or the run crashes, collect `sw_vers`, the GPU name, the exact command and the last 40 lines of stderr. Don't change kernel code or settings files to work around it.

## Research

Why this repo exists, how it works, why it isn't in llama.cpp, and what was measured: [research/](research/README.md).

## License

MIT, same as upstream llama.cpp. The llama.cpp and ggml code is © The ggml authors; the research and AMD Mac GPU tuning in this repository is © 2026 ai-janitor. If you reuse it, keep both notices.
