IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## Code Style Preferences

- **Clean variable naming**: Rename ambiguous or overloaded variable names. Use descriptive names that make the code self-documenting. The cryptic single-letter conventions (`g_t`, `k_t`, `d`, `ls`) from the upstream codebase are not sacred — rename them when they cause confusion.
- **Call out variable overloading**: If the same variable name is reused for different purposes (e.g., `state` meaning both input and output, or `H` meaning different things in different contexts), flag it explicitly and rename to disambiguate.
- **Always validate model output quality**: Never trust benchmark throughput numbers alone. Always A/B test with actual text generation to verify correctness.
