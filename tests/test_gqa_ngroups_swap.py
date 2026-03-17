"""
Test: DualKV kernel with GQA and seqlenq_ngroups_swapped.

When nheads > nheads_k and seqlen_q == 1, the C++ layer swaps Q from
(bs, 1, nheads, hdim) to (bs, ngroups, nheads_k, hdim).  This inflates
seqlen_q to ngroups.  A previous bug in n_block_max used actual_seqlen_q
(= ngroups) instead of seqlen_knew (= 1) to size the decoded block range,
causing an OOB read before the decoded buffer for certain (bs, decoded)
combinations.

The test covers:
  - GQA ratios 6:1 and 8:1 (the ngroups_swap path)
  - Large batch size (100) where OOB access always crashes
  - Decoded lengths near and above kBlockN boundaries
"""
import pytest
import torch
from flash_attn import flash_attn_with_kvcache


@pytest.mark.parametrize("nheads,nheads_k", [(12, 2), (16, 2)])
@pytest.mark.parametrize("decoded_before", [64, 96, 120, 123, 124, 127, 128, 140])
@pytest.mark.parametrize("bs", [32, 100])
def test_gqa_ngroups_swap_no_crash(nheads, nheads_k, decoded_before, bs):
    """Kernel must not crash with GQA + large decoded counts."""
    context_len = 16384
    max_decoded = 160
    hdim = 128
    dtype = torch.float16
    device = "cuda"

    torch.manual_seed(42)
    ck = torch.randn(1, context_len, nheads_k, hdim, dtype=dtype, device=device) * 0.01
    cv = torch.randn(1, context_len, nheads_k, hdim, dtype=dtype, device=device) * 0.01
    dk = torch.randn(bs, max_decoded, nheads_k, hdim, dtype=dtype, device=device) * 0.01
    dv = torch.randn(bs, max_decoded, nheads_k, hdim, dtype=dtype, device=device) * 0.01
    sc = torch.full((bs,), context_len, dtype=torch.int32, device=device)
    sd = torch.full((bs,), decoded_before, dtype=torch.int32, device=device)

    q = torch.randn(bs, 1, nheads, hdim, dtype=dtype, device=device) * 0.1
    kn = torch.randn(bs, 1, nheads_k, hdim, dtype=dtype, device=device) * 0.01
    vn = torch.randn(bs, 1, nheads_k, hdim, dtype=dtype, device=device) * 0.01

    out = flash_attn_with_kvcache(
        q=q,
        k_cache_context=ck, v_cache_context=cv,
        k_cache_decoded=dk, v_cache_decoded=dv,
        k=kn, v=vn,
        cache_seqlens_context=sc,
        cache_seqlens_decoded=sd,
        softmax_scale=hdim ** -0.5,
        causal=True,
        num_splits=1,
        use_dualkv_attention=True,
    )
    torch.cuda.synchronize()
    assert out.shape == (bs, 1, nheads, hdim)
    assert torch.isfinite(out).all(), "Output contains non-finite values"
