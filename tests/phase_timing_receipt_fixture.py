#!/usr/bin/env python3
"""CPU fixture: extract the production receipt seam before asserting its contract."""

from pathlib import Path

source = Path(__file__).parents[1] / "examples" / "phase-timing" / "phase-timing.cpp"
body = source.read_text()
sample = 'SAMPLE command=decode rep=0 trial=0 start=%.9f end=%.9f\\n'
completion = 'COMPLETION requested=%d actual=%d refused-full-N=%d eos=%d\\n'

assert sample in body
assert completion in body
assert body.index(sample) < body.index("ggml_l0_dump(receipt)") < body.index(completion)
assert "const int refused_full_n = n_generated < n_predict ? 1 : 0;" in body
print("green: extracted production receipt seam preserves marker, L0-first order, and incomplete-N diagnostics")
