"""Profile DualKV bypass decode with chrome trace export.

Requires vLLM with DualKV attention backend installed.

Usage:
    CUDA_VISIBLE_DEVICES=0 python profile_dualkv_bypass.py
"""
import os
import sys
import time

os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_USE_DUALKV"] = "1"
os.environ["VLLM_DUALKV_MAX_DECODE_LEN"] = "272"

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-8B"
CONTEXT_LEN = 16384
NUM_SEQS = 60
MAX_TOKENS = 256
DTYPE = "float16"
OUT_PATH = "torch_trace_bypass_bs60_dec256.json"


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
    from vllm.attention.backends.flash_attn_dualkv import _dualkv_states

    llm = LLM(model=MODEL, max_model_len=CONTEXT_LEN + 256,
              dtype=DTYPE, gpu_memory_utilization=0.80,
              enforce_eager=True, max_num_seqs=min(NUM_SEQS, 256))
    tokenizer = llm.get_tokenizer()
    prompt, actual_len = build_long_prompt(tokenizer, CONTEXT_LEN)
    eos_token_id = tokenizer.eos_token_id
    device = torch.device("cuda:0")

    # Prefill
    print("Prefill...", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    prefill_outputs = llm.generate([prompt], sp)

    first_token = list(prefill_outputs[0].outputs[0].token_ids)
    all_generated = [list(first_token) for _ in range(NUM_SEQS)]

    model_runner = llm.llm_engine.model_executor.driver_worker.model_runner

    captured = {k: s for k, s in _dualkv_states.items() if s.context_k is not None}
    for state in captured.values():
        if not state.initialized:
            nheads_k = state.context_k.shape[2]
            hdim = state.context_k.shape[3]
            kv_device = state.context_k.device
            state.decoded_k = torch.zeros(
                NUM_SEQS, MAX_TOKENS + 16, nheads_k, hdim,
                device=kv_device, dtype=state.context_k.dtype)
            state.decoded_v = torch.zeros(
                NUM_SEQS, MAX_TOKENS + 16, nheads_k, hdim,
                device=kv_device, dtype=state.context_k.dtype)
            state.decoded_seqlens = torch.zeros(
                NUM_SEQS, dtype=torch.int32, device=kv_device)
            state.initialized = True

    active = [True] * NUM_SEQS

    # Warmup
    print("Warmup...", flush=True)
    for _ in range(3):
        last_tokens = [all_generated[i][-1] for i in range(NUM_SEQS)]
        input_ids = torch.tensor(last_tokens, dtype=torch.long, device=device)
        positions = torch.tensor(
            [actual_len + len(all_generated[i]) - 1 for i in range(NUM_SEQS)],
            dtype=torch.long, device=device)
        logits = model_runner.dualkv_decode_step(input_ids, positions, NUM_SEQS)
        next_tokens = logits.argmax(dim=-1).tolist()
        for i in range(NUM_SEQS):
            if active[i]:
                all_generated[i].append(next_tokens[i])
                if next_tokens[i] == eos_token_id:
                    active[i] = False
    torch.cuda.synchronize()

    # Profile
    print(f"Profiling bypass: {NUM_SEQS} seqs, {MAX_TOKENS} steps...", flush=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with record_function("bypass_decode_loop"):
            start = time.time()
            for step in range(MAX_TOKENS - 1):
                if not any(active):
                    break
                last_tokens = [all_generated[i][-1] for i in range(NUM_SEQS)]
                input_ids = torch.tensor(last_tokens, dtype=torch.long, device=device)
                positions = torch.tensor(
                    [actual_len + len(all_generated[i]) - 1 for i in range(NUM_SEQS)],
                    dtype=torch.long, device=device)
                logits = model_runner.dualkv_decode_step(input_ids, positions, NUM_SEQS)
                next_tokens = logits.argmax(dim=-1).tolist()
                for i in range(NUM_SEQS):
                    if active[i]:
                        all_generated[i].append(next_tokens[i])
                        if next_tokens[i] == eos_token_id:
                            active[i] = False
            elapsed = time.time() - start

    torch.cuda.synchronize()
    total_tokens = sum(len(g) for g in all_generated)
    print(f"Bypass: {total_tokens} tokens in {elapsed:.2f}s = {total_tokens/elapsed:.0f} tok/s (decode only)")

    prof.export_chrome_trace(OUT_PATH)
    trace_size = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"Trace saved to {OUT_PATH} ({trace_size:.1f} MB)")

    print("\nTop 20 CUDA kernels:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
