# act-copy

Tensors between ops are f32 on every card. An op whose kernel reads its activations more than once may read a copy at the model's precision instead, converted once at the start of the op. This folder owns that copy.

- `data/arithmetic.tsv` section `[copies]` says, per card (its Metal device name), op family and route, whether the copy applies and why. A card with no row gets none.
- `decide.cpp` reads the table; `act-copy.cpp` owns the scratch buffer (one per backend context, one source held at a time) and encodes the conversion with the existing copy kernels (`copy/copy.metal`).
- The copy is never narrower than the model's precision, and a model whose precision is f32 or not known gets none.
- After the copy the arithmetic lane is picked by `lanes/select.cpp` for the copy's type: a lane that holds it exactly, else f32 multiply-add.

A family joins in four steps: its activation loader takes the element type as a template parameter; one kernel entry per weight type named `<base>_a<type>`; its dispatch calls `ggml_metal_act_copy_plan`, then `ggml_metal_act_copy_encode`, and binds the plan's buffer; rows in `[copies]` with `applies` no, turned on per card after the test. It also adds its op to `act_copy_users` in `act-copy.cpp`.
