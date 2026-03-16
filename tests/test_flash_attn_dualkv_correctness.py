"""
Correctness test: compare fwd_kvcache (original FA2 split-KV)
against a pure PyTorch reference implementation.

Minimal scenario: 2 context blocks + 1 decoded block.
  context_seqlen = 256 (2 × kBlockN=128)
  decoded_seqlen = 50  (fits in 1 block)
  seqlen_q       = 1   (single decode step)

Later we add the DualKV comparison on top.
"""

import math
import pytest
import torch
from einops import repeat
from flash_attn import flash_attn_with_kvcache


# ─────────────────────── PyTorch reference ───────────────────────

def attention_ref(
    q,          # (B, seqlen_q, nheads, hdim)
    k,          # (B, seqlen_k, nheads_k, hdim)
    v,          # (B, seqlen_k, nheads_k, hdim)
    causal=True,
    softmax_scale=None,
):
    """
    Pure PyTorch scaled dot-product attention with GQA support and
    bottom-right aligned causal mask.

    Returns: (B, seqlen_q, nheads, hdim)
    """
    B, seqlen_q, nheads, hdim = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert nheads % nheads_k == 0
    if softmax_scale is None:
        softmax_scale = hdim ** (-0.5)

    # Expand KV heads to match Q heads for GQA
    # (B, seqlen_k, nheads_k, hdim) -> (B, seqlen_k, nheads, hdim)
    groups = nheads // nheads_k
    k = repeat(k, "b s h d -> b s (h g) d", g=groups)
    v = repeat(v, "b s h d -> b s (h g) d", g=groups)

    # Transpose to (B, nheads, seqlen, hdim)
    q = q.transpose(1, 2).float()  # fp32 for reference accuracy
    k = k.transpose(1, 2).float()
    v = v.transpose(1, 2).float()

    # Scaled dot-product: (B, nheads, seqlen_q, seqlen_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    # Bottom-right aligned causal mask
    if causal:
        # Row i of Q can attend to columns [0 .. seqlen_k - seqlen_q + i]
        row_idx = torch.arange(seqlen_q, device=q.device).unsqueeze(1)  # (sq, 1)
        col_idx = torch.arange(seqlen_k, device=q.device).unsqueeze(0)  # (1, sk)
        # Causal: col_idx <= row_idx + (seqlen_k - seqlen_q)
        causal_mask = col_idx > (row_idx + seqlen_k - seqlen_q)
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)  # (B, nheads, seqlen_q, hdim)

    return out.transpose(1, 2).to(q.dtype)  # back to (B, seqlen_q, nheads, hdim)


# ─────────────────────── test ───────────────────────

class TestFwdKvcacheVsReference:
    """
    Compare flash_attn_with_kvcache (original, use_dualkv_attention=False)
    against pure PyTorch reference.

    Scenario: 2 context blocks + 1 decoded block, no rotary (for simplicity).
    """

    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("bs", [1, 2])
    @pytest.mark.parametrize("seqlen_q", [1, 5])
    @pytest.mark.parametrize("context_seqlen", [256])
    @pytest.mark.parametrize("decoded_seqlen", [50])
    @pytest.mark.parametrize("hdim", [128])
    @pytest.mark.parametrize("nheads_nheads_k", [(32, 4)])
    def test_fwd_kvcache_vs_pytorch_ref(
        self, dtype, bs, seqlen_q, context_seqlen, decoded_seqlen, hdim,
        nheads_nheads_k,
    ):
        device = "cuda"
        nheads, nheads_k = nheads_nheads_k

        torch.manual_seed(42)

        # Pre-allocated cache size (with headroom for new tokens)
        cache_seqlen_max = context_seqlen + decoded_seqlen + seqlen_q + 64

        # ─── Generate data ───

        q = torch.randn(bs, seqlen_q, nheads, hdim, device=device, dtype=dtype)
        k_new = torch.randn(bs, seqlen_q, nheads_k, hdim, device=device, dtype=dtype)
        v_new = torch.randn(bs, seqlen_q, nheads_k, hdim, device=device, dtype=dtype)

        # KV cache: fill with random data up to context+decoded
        k_cache = torch.randn(bs, cache_seqlen_max, nheads_k, hdim, device=device, dtype=dtype)
        v_cache = torch.randn(bs, cache_seqlen_max, nheads_k, hdim, device=device, dtype=dtype)

        # The "used" portion is [0, context_seqlen + decoded_seqlen)
        cache_seqlens = torch.full(
            (bs,), context_seqlen + decoded_seqlen, dtype=torch.int32, device=device
        )

        # ─── Build the PyTorch reference ───

        # The full K/V after appending new tokens:
        # k_full[b] = k_cache[b, :ctx+dec] || k_new[b]
        used = context_seqlen + decoded_seqlen
        k_full = torch.cat([k_cache[:, :used, :, :], k_new], dim=1)  # (B, used+sq, nheads_k, hdim)
        v_full = torch.cat([v_cache[:, :used, :, :], v_new], dim=1)

        out_ref = attention_ref(q, k_full, v_full, causal=True)

        # ─── Call FA2 fwd_kvcache ───

        k_cache_fa = k_cache.clone()
        v_cache_fa = v_cache.clone()

        out_fa = flash_attn_with_kvcache(
            q.clone(),
            k_cache=k_cache_fa,
            v_cache=v_cache_fa,
            k=k_new.clone(),
            v=v_new.clone(),
            cache_seqlens=cache_seqlens.clone(),
            causal=True,
            num_splits=0,
            use_dualkv_attention=False,
        )

        # ─── Compare ───

        # Convert reference to fp16 for comparison
        out_ref_fp16 = out_ref.to(dtype)

        max_diff = (out_fa - out_ref_fp16).abs().max().item()
        mean_diff = (out_fa - out_ref_fp16).abs().mean().item()

        print(f"\n[FA2 vs PyTorch ref] bs={bs}, q={seqlen_q}, ctx={context_seqlen}, "
              f"dec={decoded_seqlen}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        # Loose tolerance: PyTorch ref uses fp32 accumulation, FA2 uses fp16 tiled GEMM
        atol = 5e-2
        rtol = 1e-2
        assert torch.allclose(out_fa, out_ref_fp16, atol=atol, rtol=rtol), (
            f"FA2 fwd_kvcache output differs from PyTorch reference!\n"
            f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}\n"
            f"  params: bs={bs}, seqlen_q={seqlen_q}, context={context_seqlen}, "
            f"decoded={decoded_seqlen}"
        )


class TestDualKVSmallContext:
    """
    Test DualKV with small context lengths and various head configurations.

    Covers edge cases: ctx < kBlockN (128), decoded=0, MHA vs GQA, hdim=64 vs 128.
    """

    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("bs", [1, 2])
    @pytest.mark.parametrize("seqlen_q", [1])
    @pytest.mark.parametrize("context_seqlen", [1, 5, 16, 64, 127, 128, 129, 256])
    @pytest.mark.parametrize("decoded_seqlen", [0, 1, 10])
    @pytest.mark.parametrize("hdim", [64, 128])
    @pytest.mark.parametrize("nheads_nheads_k", [(32, 4), (12, 2), (12, 12)])
    def test_dualkv_small_context(
        self, dtype, bs, seqlen_q, context_seqlen, decoded_seqlen, hdim,
        nheads_nheads_k,
    ):
        """Test DualKV correctness for small context lengths."""
        device = "cuda"
        nheads, nheads_k = nheads_nheads_k

        torch.manual_seed(42)

        context_seqlen_max = max(context_seqlen + 64, 256)
        decoded_seqlen_max = max(decoded_seqlen + 64, 128)

        q = torch.randn(bs, seqlen_q, nheads, hdim, device=device, dtype=dtype)
        k_new = torch.randn(bs, seqlen_q, nheads_k, hdim, device=device, dtype=dtype)
        v_new = torch.randn(bs, seqlen_q, nheads_k, hdim, device=device, dtype=dtype)

        k_context_cache = torch.randn(bs, context_seqlen_max, nheads_k, hdim, device=device, dtype=dtype)
        v_context_cache = torch.randn(bs, context_seqlen_max, nheads_k, hdim, device=device, dtype=dtype)
        k_decoded_cache = torch.randn(bs, decoded_seqlen_max, nheads_k, hdim, device=device, dtype=dtype)
        v_decoded_cache = torch.randn(bs, decoded_seqlen_max, nheads_k, hdim, device=device, dtype=dtype)

        cache_seqlens_context = torch.full((bs,), context_seqlen, dtype=torch.int32, device=device)
        cache_seqlens_decoded = torch.full((bs,), decoded_seqlen, dtype=torch.int32, device=device)

        # PyTorch reference
        k_full = torch.cat([
            k_context_cache[:, :context_seqlen, :, :],
            k_decoded_cache[:, :decoded_seqlen, :, :],
            k_new
        ], dim=1)
        v_full = torch.cat([
            v_context_cache[:, :context_seqlen, :, :],
            v_decoded_cache[:, :decoded_seqlen, :, :],
            v_new
        ], dim=1)

        out_ref = attention_ref(q, k_full, v_full, causal=True)

        # DualKV kernel
        out_dualkv = flash_attn_with_kvcache(
            q.clone(),
            k_cache_context=k_context_cache.clone(),
            v_cache_context=v_context_cache.clone(),
            k_cache_decoded=k_decoded_cache.clone(),
            v_cache_decoded=v_decoded_cache.clone(),
            k=k_new.clone(),
            v=v_new.clone(),
            cache_seqlens_context=cache_seqlens_context.clone(),
            cache_seqlens_decoded=cache_seqlens_decoded.clone(),
            causal=True,
            use_dualkv_attention=True,
        )

        out_ref_fp16 = out_ref.to(dtype)
        max_diff = (out_dualkv - out_ref_fp16).abs().max().item()
        mean_diff = (out_dualkv - out_ref_fp16).abs().mean().item()

        print(f"\n[DualKV small ctx] bs={bs}, q={seqlen_q}, ctx={context_seqlen}, "
              f"dec={decoded_seqlen}, hdim={hdim}, heads={nheads}/{nheads_k}: "
              f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        atol = 5e-2
        rtol = 1e-2
        assert torch.allclose(out_dualkv, out_ref_fp16, atol=atol, rtol=rtol), (
            f"DualKV output differs from PyTorch reference!\n"
            f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}\n"
            f"  all_zero={int((out_dualkv == 0).all())}, any_nan={int(out_dualkv.isnan().any())}\n"
            f"  params: bs={bs}, seqlen_q={seqlen_q}, ctx={context_seqlen}, "
            f"dec={decoded_seqlen}, hdim={hdim}, heads={nheads}/{nheads_k}"
        )


class TestDualKVMaskBug:
    """
    Test to expose the mask initialization bug on line 479 of flash_fwd_kernel_dualkv.h.

    Bug: mask uses actual_n_block_max_context * kBlockN instead of actual_seqlen_k_cache_context
    This causes max_seqlen_k to be 123 tokens too large when context has partial blocks.

    Test case:
    - context: 389 tokens (4 blocks: 3 full + 1 partial, rounds to 512)
    - decoded: 145 tokens (actual)
    - new: 5 query tokens

    Bug: max_seqlen_k = 150 + 512 = 662
    Correct: max_seqlen_k = 150 + 389 = 539
    """

    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("bs", [1, 2])
    @pytest.mark.parametrize("seqlen_q", [5])
    @pytest.mark.parametrize("context_seqlen", [389])
    @pytest.mark.parametrize("decoded_seqlen", [145])
    @pytest.mark.parametrize("hdim", [128])
    @pytest.mark.parametrize("nheads_nheads_k", [(32, 4)])
    def test_dualkv_partial_context_blocks(
        self, dtype, bs, seqlen_q, context_seqlen, decoded_seqlen, hdim,
        nheads_nheads_k,
    ):
        """Test DualKV with partial context blocks to expose mask bug."""
        device = "cuda"
        nheads, nheads_k = nheads_nheads_k

        torch.manual_seed(42)

        # Large pre-allocated buffers (to match real usage scenarios)
        context_seqlen_max = 2048
        decoded_seqlen_max = 512

        # ─── Generate data ───

        q = torch.randn(bs, seqlen_q, nheads, hdim, device=device, dtype=dtype)
        k_new = torch.randn(bs, seqlen_q, nheads_k, hdim, device=device, dtype=dtype)
        v_new = torch.randn(bs, seqlen_q, nheads_k, hdim, device=device, dtype=dtype)

        # Context cache: 389 tokens in a 2048-capacity buffer
        k_context_cache = torch.randn(bs, context_seqlen_max, nheads_k, hdim, device=device, dtype=dtype)
        v_context_cache = torch.randn(bs, context_seqlen_max, nheads_k, hdim, device=device, dtype=dtype)

        # Decoded cache: 145 tokens in a 512-capacity buffer
        k_decoded_cache = torch.randn(bs, decoded_seqlen_max, nheads_k, hdim, device=device, dtype=dtype)
        v_decoded_cache = torch.randn(bs, decoded_seqlen_max, nheads_k, hdim, device=device, dtype=dtype)

        # Actual usage
        cache_seqlens_context = torch.full((bs,), context_seqlen, dtype=torch.int32, device=device)
        cache_seqlens_decoded = torch.full((bs,), decoded_seqlen, dtype=torch.int32, device=device)

        # ─── Build the PyTorch reference ───

        # Full K/V after appending: context || decoded || new
        k_full = torch.cat([
            k_context_cache[:, :context_seqlen, :, :],
            k_decoded_cache[:, :decoded_seqlen, :, :],
            k_new
        ], dim=1)
        v_full = torch.cat([
            v_context_cache[:, :context_seqlen, :, :],
            v_decoded_cache[:, :decoded_seqlen, :, :],
            v_new
        ], dim=1)

        out_ref = attention_ref(q, k_full, v_full, causal=True)

        # ─── Call DualKV flash attention ───

        out_dualkv = flash_attn_with_kvcache(
            q.clone(),
            k_cache_context=k_context_cache.clone(),
            v_cache_context=v_context_cache.clone(),
            k_cache_decoded=k_decoded_cache.clone(),
            v_cache_decoded=v_decoded_cache.clone(),
            k=k_new.clone(),
            v=v_new.clone(),
            cache_seqlens_context=cache_seqlens_context.clone(),
            cache_seqlens_decoded=cache_seqlens_decoded.clone(),
            causal=True,
            use_dualkv_attention=True,
        )

        # ─── Compare ───

        out_ref_fp16 = out_ref.to(dtype)
        max_diff = (out_dualkv - out_ref_fp16).abs().max().item()
        mean_diff = (out_dualkv - out_ref_fp16).abs().mean().item()

        print(f"\n[DualKV mask bug test] bs={bs}, q={seqlen_q}, "
              f"ctx={context_seqlen}, dec={decoded_seqlen}: "
              f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        # With the mask bug, causal masking boundaries are wrong by 123 tokens
        # This should cause visible errors in the output
        atol = 5e-2
        rtol = 1e-2
        assert torch.allclose(out_dualkv, out_ref_fp16, atol=atol, rtol=rtol), (
            f"DualKV output differs from PyTorch reference (mask bug?)!\n"
            f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}\n"
            f"  Bug: max_seqlen_k = 150 + 512 = 662 (uses rounded blocks)\n"
            f"  Correct: max_seqlen_k = 150 + 389 = 539 (uses actual tokens)\n"
            f"  params: bs={bs}, seqlen_q={seqlen_q}, context={context_seqlen}, "
            f"decoded={decoded_seqlen}"
        )


# ─────────────────────── standalone runner ───────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Testing FA2 fwd_kvcache vs PyTorch reference")
    print("=" * 60)
    test = TestFwdKvcacheVsReference()
    for bs in [1, 2]:
        for sq in [1, 5]:
            try:
                test.test_fwd_kvcache_vs_pytorch_ref(
                    dtype=torch.float16, bs=bs, seqlen_q=sq,
                    context_seqlen=256, decoded_seqlen=50, hdim=128,
                    nheads_nheads_k=(32, 4),
                )
                print("  ✓ PASSED")
            except Exception as e:
                print(f"  ✗ FAILED: {e}")

    print("\n" + "=" * 60)
    print("Testing DualKV mask bug (partial context blocks)")
    print("=" * 60)
    test_mask = TestDualKVMaskBug()
    for bs in [1, 2]:
        try:
            test_mask.test_dualkv_partial_context_blocks(
                dtype=torch.float16, bs=bs, seqlen_q=5,
                context_seqlen=389, decoded_seqlen=145, hdim=128,
                nheads_nheads_k=(32, 4),
            )
            print("  ✓ PASSED")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

    print("\n" + "=" * 60)
    print("Testing DualKV V2 (refactored kernel)")
    print("=" * 60)
    for bs in [1, 2]:
        try:
            # Use refactored kernel
            dtype = torch.float16
            seqlen_q = 5
            context_seqlen = 389
            decoded_seqlen = 145
            hdim = 128
            nheads, nheads_k = 32, 4

            device = "cuda"
            torch.manual_seed(42)

            context_seqlen_max = 2048
            decoded_seqlen_max = 512

            q = torch.randn(bs, seqlen_q, nheads, hdim, device=device, dtype=dtype)
            k_new = torch.randn(bs, seqlen_q, nheads_k, hdim, device=device, dtype=dtype)
            v_new = torch.randn(bs, seqlen_q, nheads_k, hdim, device=device, dtype=dtype)

            k_context_cache = torch.randn(bs, context_seqlen_max, nheads_k, hdim, device=device, dtype=dtype)
            v_context_cache = torch.randn(bs, context_seqlen_max, nheads_k, hdim, device=device, dtype=dtype)

            k_decoded_cache = torch.randn(bs, decoded_seqlen_max, nheads_k, hdim, device=device, dtype=dtype)
            v_decoded_cache = torch.randn(bs, decoded_seqlen_max, nheads_k, hdim, device=device, dtype=dtype)

            cache_seqlens_context = torch.full((bs,), context_seqlen, dtype=torch.int32, device=device)
            cache_seqlens_decoded = torch.full((bs,), decoded_seqlen, dtype=torch.int32, device=device)

            # Build PyTorch reference
            k_full = torch.cat([
                k_context_cache[:, :context_seqlen, :, :],
                k_decoded_cache[:, :decoded_seqlen, :, :],
                k_new
            ], dim=1)
            v_full = torch.cat([
                v_context_cache[:, :context_seqlen, :, :],
                v_decoded_cache[:, :decoded_seqlen, :, :],
                v_new
            ], dim=1)

            out_ref = attention_ref(q, k_full, v_full, causal=True)

            # Call DualKV V2 (refactored kernel)
            out_v2 = flash_attn_with_kvcache(
                q.clone(),
                k_cache_context=k_context_cache.clone(),
                v_cache_context=v_context_cache.clone(),
                k_cache_decoded=k_decoded_cache.clone(),
                v_cache_decoded=v_decoded_cache.clone(),
                k=k_new.clone(),
                v=v_new.clone(),
                cache_seqlens_context=cache_seqlens_context.clone(),
                cache_seqlens_decoded=cache_seqlens_decoded.clone(),
                causal=True,
                use_dualkv_attention=True,
                use_dualkv_v2=True,  # Use refactored kernel
            )

            out_ref_fp16 = out_ref.to(dtype)
            max_diff = (out_v2 - out_ref_fp16).abs().max().item()
            mean_diff = (out_v2 - out_ref_fp16).abs().mean().item()

            print(f"[DualKV V2] bs={bs}, q={seqlen_q}, ctx={context_seqlen}, "
                  f"dec={decoded_seqlen}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

            atol = 5e-2
            rtol = 1e-2
            assert torch.allclose(out_v2, out_ref_fp16, atol=atol, rtol=rtol), (
                f"DualKV V2 output differs from PyTorch reference!\n"
                f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}\n"
                f"  params: bs={bs}, seqlen_q={seqlen_q}, context={context_seqlen}, "
                f"decoded={decoded_seqlen}"
            )

            print("  ✓ PASSED")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
