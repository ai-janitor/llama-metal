# lanes

A lane is the arithmetic that forms the products inside a kernel: fp32 multiply-add, packed fp16 multiply, a dot instruction, a matrix unit.

- One file per lane, named for its instruction: `metal/` holds lanes compiled by Metal's shader compiler, `llvm/` is the home for lanes compiled by LLVM for AMD cards (`llvm/toolchain.tsv`; no kernel there yet).
- `lanes.tsv` says which lane is wired into which op family and route, for which operand types, and whether it is on. `capability.tsv` says what each card can do, with the source of every cell.
- The choice is a negotiation between the tensor type of the op (the model's format) and the card's rows: the lane that holds the type natively, else exactly another way, else wider. A lane narrower than the tensor is dropped; no lane left is a coverage gap, not a fallback.
- Which table answers which question: "what multiplies inside this kernel on this card" is `lanes.tsv`; "what can this card do, and how do we know" is `capability.tsv`. Tensors between kernels are f32 on every card.
- `select.cpp` is the only code that chooses, and it reads only those tables. There is no environment variable and no device-name test.
- A card with no row in `capability.tsv` (every Apple Silicon GPU) gets the base lane of each route: the kernels upstream ships.

To add a lane: write its file, add its rows with `default` off, run its test on the card, then turn the row on.
