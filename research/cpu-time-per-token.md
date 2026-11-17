# The CPU side of every token on a discrete AMD GPU

On these Macs the GPU and CPU have separate memory, and each generated token needs some CPU work the GPU waits for. Measured on a Radeon Pro Vega 64 with Qwen3.5-0.8B (about 1.95 ms of CPU work per token, about a quarter of the token's time):

| CPU work per token | ms |
|---|---|
| a separate CPU step just to look up the next token's embedding row | 0.69 |
| copying the logits (about 1 MB) back into a buffer made new for each token | 0.44 |
| the sampler building a list of all 248,320 candidate tokens | 0.39 |
| preparing the first batch of GPU work before the GPU has anything to do | 0.19 |
| waiting between steps | 0.13 |

That is why the GPU alone runs the 0.8B model at about 179 tokens per second on the Vega 64 while llama.cpp reports about 134. Work in progress moves these steps off the critical path: reusing one buffer, not waiting on each input upload, picking the token on the GPU, and keeping the embedding lookup on the GPU.
