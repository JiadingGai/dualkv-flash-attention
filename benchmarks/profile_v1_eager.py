"""Profile V1-eager in-process (no fork).

Key trick: VLLM_ENABLE_V1_MULTIPROCESSING=0 forces V1 engine to run
in the same process so torch.profiler can capture all CUDA kernels.

Usage:
    CUDA_VISIBLE_DEVICES=0 python profile_v1_eager.py
"""
import os
import sys
import time

# Force V1 engine, disable multiprocessing (run in-process)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-8B"
CONTEXT_LEN = 16384
NUM_SEQS = 60
MAX_TOKENS = 256
DTYPE = "float16"
OUT_PATH = "torch_trace_v1_eager_bs60_dec256.json"


def build_long_prompt(tokenizer, target_len):
    base = ("The following is a comprehensive analysis of modern deep learning "
            "architectures and their applications. ")
    filler = ("Neural networks process information through layers of "
              "interconnected nodes, each applying learned transformations. "
              "The attention mechanism allows models to focus on relevant "
              "parts of the input sequence. Gradient descent optimizes the "
              "model parameters by minimizing the loss function. "
              "Regularization techniques prevent overfitting by constraining "
              "model complexity. Batch normalization stabilizes training by "
              "normalizing layer inputs. Residual connections enable training "
              "of very deep networks by providing shortcut paths. ")
    prompt = base
    while True:
        tokens = tokenizer.encode(prompt)
        if len(tokens) >= target_len:
            tokens = tokens[:target_len]
            prompt = tokenizer.decode(tokens, skip_special_tokens=True)
            break
        prompt += filler
    return prompt, len(tokenizer.encode(prompt))


def main():
    print(f"V1-eager in-process profiling: {NUM_SEQS} seqs, {MAX_TOKENS} tokens")
    print(f"Model: {MODEL}, Context: {CONTEXT_LEN}, dtype: {DTYPE}")
    print(f"Output: {OUT_PATH}")
    print()

    llm = LLM(model=MODEL, max_model_len=CONTEXT_LEN + 256,
              dtype=DTYPE, gpu_memory_utilization=0.85,
              enforce_eager=True,
              max_num_seqs=min(NUM_SEQS, 256))

    tokenizer = llm.get_tokenizer()
    prompt, actual_len = build_long_prompt(tokenizer, CONTEXT_LEN)
    print(f"Prompt length: {actual_len} tokens")

    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    prompts = [prompt] * NUM_SEQS

    # Warmup
    print("Warmup...", flush=True)
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=2))
    torch.cuda.synchronize()

    # Profile
    print(f"Profiling...", flush=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with record_function("v1_eager_generate"):
            start = time.time()
            outputs = llm.generate(prompts, sp)
            elapsed = time.time() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"V1-eager: {total_tokens} tokens in {elapsed:.2f}s = {total_tokens/elapsed:.0f} tok/s")

    prof.export_chrome_trace(OUT_PATH)
    trace_size = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"Trace saved to {OUT_PATH} ({trace_size:.1f} MB)")

    # Print kernel summary
    print("\nTop 20 CUDA kernels:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
