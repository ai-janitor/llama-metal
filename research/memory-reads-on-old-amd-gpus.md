# What limits matrix-vector kernels on old AMD Mac GPUs

Measured on a FirePro D700 (Mac Pro 2013, GCN 1.0, rated 263 GB/s) while tuning the Q8_0 matrix-vector kernels that generate each token. Each finding is stated as a general rule first; the D700 numbers are the evidence. Other cards behave differently (see the per-card table in `ggml/src/ggml-metal/ggml-metal-hw-rows.h`).

## 1. Find the real ceiling with a probe that does not write
Reads and writes share one memory budget. A read test that also writes 4 bytes per thread measures the budget minus its own writes.
- D700: 158.5 GB/s with a write per thread, 207.1 GB/s with one write per group. The plain-read ceiling is 204-209 GB/s, 79 % of the rated 263. Thread count, loads in flight, access pattern and size did not move it.

## 2. The card charges whole 64-byte lines, not the bytes you use
If the threads of a group touch a 64-byte line, the whole line is paid for, however few bytes are read from it. Threads should read neighbouring bytes and use every line they touch.
- D700: 4-byte loads with threads 4 to 64 bytes apart all took the same time over the same 270 MB (1,435-1,613 µs), even when only 6 % of the bytes were used.

## 3. Two reads into the same lines cost about one; a third does not
A second read issued together with the first into lines already in flight is almost free; a third and fourth cost 35-80 % more; a read issued after the previous one was used pays for the line again.
- D700: 16 bytes per thread as one 16-byte load, two 8-byte loads, four 4-byte loads: 1,478 / 1,458 / 2,549 µs.

## 4. Read wide: a load is as wide as the compiler can prove it aligned
Q8_0 weights sit in 34-byte blocks, so a thread's weights start 2 bytes into a block and compiled to single-byte loads. Reading the covering aligned 4-byte words and shifting the offset out does the same multiplications in the same order (same output bytes).
- D700 about 44 % more tokens per second, Vega 64 about 11 %, Radeon Pro 5500M (RDNA 1) no change.

## 5. Count the reads of every input, including the small shared one
After the weights were read as words, the kernel was limited by the activation vector every row shares: 4 KB, read by each thread as 8 separate 4-byte loads. Reading it as two aligned 16-byte loads (same values, same order) fixed it.
- D700 Q8_0 kernels per token: arithmetic alone 4.31 ms; with the 8 single activation loads 13.63 ms; whole kernel 12.43 → 10.09 ms with 16-byte loads. In the full token: about 14 % more tokens per second. Vega 64: no change.

## 6. Re-measure what limits the kernel after every win
Split the time into reads and arithmetic again, one input at a time, after each change. The limit moves: with byte loads the weights were nearly all of it, after word loads about half, and the activations became the limit. Ideas aimed at the old limit (two 8-byte weight loads) lost in the kernel (14.4 vs 12.4 ms) even though a read-pattern test favoured them.

## 7. A gain measured alone is a candidate, not a result
Every change here was measured alone, then in the real token on every card, with byte-for-byte equal output. The D700 activation change predicted -1.89 ms per token alone and gave -1.88 in the token.
