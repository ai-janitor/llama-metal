.PHONY: build clean test bench bench-fa0 bench-fa1 bench-long run

# Metal working tree — AMD Radeon Pro 5300M (discrete, non-UMA)
MODEL := $(HOME)/Library/Caches/llama.cpp/Qwen_Qwen2.5-1.5B-Instruct-GGUF_qwen2.5-1.5b-instruct-q4_k_m.gguf
PROMPT ?= Explain backpropagation:
N_TOKENS ?= 50

build:
	cmake -B build \
		-DGGML_METAL=ON \
		-DGGML_VULKAN=OFF \
		-DGGML_CUDA=OFF \
		-DGGML_SYCL=OFF \
		-DGGML_OPENCL=OFF \
		-DGGML_RPC=OFF \
		-DGGML_METAL_EMBED_LIBRARY=ON \
		-DLLAMA_BUILD_SERVER=OFF \
		-DLLAMA_BUILD_TESTS=ON
	cmake --build build -j8 -t llama-bench llama-completion test-backend-ops

clean:
	rm -rf build

test:
	@echo "==> Running Metal backend-ops test..."
	@./build/bin/test-backend-ops test -b Metal

bench: bench-fa0

bench-fa0:
	@echo "==> Metal benchmark (FA=0):"
	@./build/bin/llama-bench -m $(MODEL) -ngl 99 -p 128 -n 32 -t 1 -fa 0 -r 10

bench-fa1:
	@echo "==> Metal benchmark (FA=1):"
	@./build/bin/llama-bench -m $(MODEL) -ngl 99 -p 128 -n 32 -t 1 -fa 1 -r 10

bench-long:
	@echo "==> Metal benchmark long (FA=0):"
	@./build/bin/llama-bench -m $(MODEL) -ngl 99 -p 512 -n 128 -t 1 -fa 0 -r 5

run:
	@echo "==> Metal text generation:"
	@./build/bin/llama-completion -m $(MODEL) -ngl 99 -n $(N_TOKENS) -p "$(PROMPT)" -no-cnv
