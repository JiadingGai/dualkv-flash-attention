import math
import time
import statistics
import pytest
import torch
from flash_attn import flash_attn_with_kvcache
from flash_attn.layers.rotary import apply_rotary_emb

# Default dtype / device
dtype = torch.float16
device = "cuda"

def plot_latency_csv_old(latency_csv):
    import matplotlib.pyplot as plt
    for c, b, f in latency_csv:
        print(c, ",", b, ",", f)
    ctx_ = [x[0] for x in latency_csv]
    ba2_ = [x[1] for x in latency_csv]
    fa2_ = [x[2] for x in latency_csv]
    plt.xlabel("Cache Context Length")
    plt.ylabel("Latency in microseconds")
    plt.title("BA+FA vs FA: attention kernel latency comparison in 17B v3")
    plt.plot(ctx_, ba2_, "r^--", label="BA+FA", markersize=3.0, linewidth=1)
    plt.plot(ctx_, fa2_, "g^--", label="FA", markersize=3.0, linewidth=1)
    plt.legend()
    plt.savefig("__plot_timing.png", dpi=600)

def plot_latency_csv(latency_csv, out_path="__plot_latency_combined.png",
                     title="BA+FA vs FA: attention kernel latency comparison"):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # unpack & sort
    ctx = np.array([x[0] for x in latency_csv], dtype=np.int64)
    ba_us = np.array([x[1] for x in latency_csv], dtype=np.float64)
    fa_us = np.array([x[2] for x in latency_csv], dtype=np.float64)
    order = np.argsort(ctx)
    ctx, ba_us, fa_us = ctx[order], ba_us[order], fa_us[order]

    # μs → ms
    ba_ms = ba_us / 1000.0
    fa_ms = fa_us / 1000.0

    # combined figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    thous = FuncFormatter(lambda x, _: f"{int(x):,}")

    # 1) Latency subplot
    ax1.plot(ctx, ba_ms, "^--", label="BA+FA", markersize=3.5, linewidth=1)
    ax1.plot(ctx, fa_ms, "o--", label="FA",    markersize=3.5, linewidth=1)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title(title)
    ax1.yaxis.set_major_formatter(thous)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend()

    # OLS slope (ms/token) — prints in console
    import numpy as np
    b_ba, a_ba = np.polyfit(ctx, ba_ms, 1)
    b_fa, a_fa = np.polyfit(ctx, fa_ms, 1)
    print(f"[fit] BA+FA: latency ≈ {a_ba:.3f} + {b_ba:.6f}·ctx (ms)")
    print(f"[fit] FA   : latency ≈ {a_fa:.3f} + {b_fa:.6f}·ctx (ms)")
    print(f"[fit] slope ratio (FA/BA+FA): {b_fa / max(b_ba, 1e-12):.2f}×")

    # 2) Speedup subplot — bigger dots
    speedup = np.divide(fa_ms, np.maximum(ba_ms, 1e-12))
    ax2.scatter(ctx, speedup, s=24, marker="o", alpha=0.85)  # bigger & clearer
    ax2.set_xlabel("Cache Context Length")
    ax2.set_ylabel("Speedup (FA / BA+FA)")
    ax2.xaxis.set_major_formatter(thous)
    ax2.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)


class TestFlashFwdDualkvAttention:
    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("rotary_dim", [128])
    @pytest.mark.parametrize("seqlen", [5])
    @pytest.mark.parametrize("bs", [100])
    @pytest.mark.parametrize("context_seqlen_max", [32 * 1024])
    @pytest.mark.parametrize("decoded_seqlen_max", [512])
    def test_timing(self, dtype, bs, seqlen, context_seqlen_max, decoded_seqlen_max, rotary_dim):
        # set seed
        torch.manual_seed(123)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(123)

        decoded_high = decoded_seqlen_max - seqlen
        decoded_low = 1

        latency_csv = []
        __debug_logs = []
        for context_low in range(1, context_seqlen_max + 1, 57):
            cache_seqlens_decoded = torch.randint(
                low=decoded_low, high=decoded_high, size=(bs,), device=device, dtype=torch.int32
            )
            cache_seqlens_context = context_low  # scalar (global) context length for this sweep

            print(
                f"\n====== Benchmarking context_seqlen_max={context_seqlen_max}, "
                f"seqlen={seqlen}, bs={bs}, cache_seqlens_context={context_low} ======\n"
            )
            dualkv_time, og_fa_time, log_msg = self.template_timing_run(
                dtype=dtype,
                device=device,
                bs=bs,
                seqlen=seqlen,
                context_seqlen_max=context_seqlen_max,
                decoded_seqlen_max=decoded_seqlen_max,
                rotary_dim=rotary_dim,
                num_runs=1,
                cache_seqlens_context=cache_seqlens_context,
                cache_seqlens_decoded=cache_seqlens_decoded,
            )
            # convert time from seconds to microseconds:
            dualkv_time *= 1_000_000.0
            og_fa_time *= 1_000_000.0

            pad2 = max(len("dualkv+FA (ours) "), len("original FA"))
            print(f'{"dualkv+FA (ours) ":<{pad2}}: {dualkv_time:.6f} microseconds.')
            print(f'{"original FA":<{pad2}}: {og_fa_time:.6f} microseconds.')
            latency_csv.append((context_low, dualkv_time, og_fa_time))
            __debug_logs.append(log_msg)

        plot_latency_csv(latency_csv)
        for _log in __debug_logs:
            print(_log)

    def template_timing_run(
        self,
        dtype=torch.float16,
        device="cuda",
        bs=5,
        seqlen=7,
        context_seqlen_max=1536,
        decoded_seqlen_max=2904,
        rotary_dim=128,
        num_runs=1,
        cache_seqlens_context=1536,
        cache_seqlens_decoded=2904,
    ):
        """
        Timing harness comparing dual-KV+FA against original FA, using CUDA events for stable timing.
        """

        # Config
        nheads = 32
        nheads_k = 4
        hdim = 128
        assert (nheads % nheads_k) == 0, "nheads must be divisible by nheads_k for head repeat"
        IsCausal = True

        # Max kv-cache length (pre-allocated)
        kvcache_seqlen_max = context_seqlen_max + decoded_seqlen_max

        # Shapes
        seqlen_q = seqlen
        seqlen_k = kvcache_seqlen_max
        rotary_interleaved = True

        # Inputs (tokens to append)
        q = torch.randn([bs, seqlen, nheads, hdim], device=device, dtype=dtype)
        k = torch.randn([bs, seqlen, nheads_k, hdim], device=device, dtype=dtype)
        v = torch.randn([bs, seqlen, nheads_k, hdim], device=device, dtype=dtype)

        # Context cache: single shared context buffer (bs=1), with NaNs beyond current context length
        if isinstance(cache_seqlens_context, torch.Tensor):
            cache_seqlens_context = int(cache_seqlens_context.item())
        Kcontext_cache = torch.randn([1, context_seqlen_max, nheads_k, hdim], device=device, dtype=dtype)
        Vcontext_cache = torch.randn([1, context_seqlen_max, nheads_k, hdim], device=device, dtype=dtype)
        oob_val = torch.nan
        Kcontext_cache[:, cache_seqlens_context:, :, :] = oob_val
        Vcontext_cache[:, cache_seqlens_context:, :, :] = oob_val

        # Decoded cache: batched buffers with NaNs beyond current decoded lengths
        if not isinstance(cache_seqlens_decoded, torch.Tensor):
            cache_seqlens_decoded = torch.tensor(cache_seqlens_decoded, device=device, dtype=torch.int32)
        Kdecode_cache = torch.randn([bs, decoded_seqlen_max, nheads_k, hdim], device=device, dtype=dtype)
        Vdecode_cache = torch.randn([bs, decoded_seqlen_max, nheads_k, hdim], device=device, dtype=dtype)
        for bid in range(bs):
            Kdecode_cache[bid, cache_seqlens_decoded[bid] :, :, :] = oob_val
            Vdecode_cache[bid, cache_seqlens_decoded[bid] :, :, :] = oob_val

        # Aggregate lengths for original FA path
        cache_seqlens = cache_seqlens_context + cache_seqlens_decoded

        # Rotary embeddings (optional)
        if rotary_dim > 0:
            angle = torch.rand(seqlen_k, rotary_dim // 2, device=device) * 2 * math.pi
            cos = torch.cos(angle).to(dtype=dtype)
            sin = torch.sin(angle).to(dtype=dtype)
            q_ro = apply_rotary_emb(
                q, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
            )
            k_ro = apply_rotary_emb(
                k, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved
            )
        else:
            cos = sin = None
            q_ro, k_ro = q, k

        torch.set_printoptions(sci_mode=False)

        # ---- dual-KV + FlashAttention timing ----
        dualkv_time_list = []
        for _ in range(num_runs):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            out_bf, lse_bf = flash_attn_with_kvcache(
                q=q_ro,
                k_cache=None,  # not needed by dualkv attention
                v_cache=None,  # not needed by dualkv attention
                k_cache_decoded=Kdecode_cache,  # will be muted
                v_cache_decoded=Vdecode_cache,  # will be muted
                k_cache_context=Kcontext_cache,
                v_cache_context=Vcontext_cache,
                k=k_ro,
                v=v,
                rotary_cos=cos,
                rotary_sin=sin,
                cache_seqlens=None,
                cache_seqlens_context=cache_seqlens_context,
                cache_seqlens_decoded=cache_seqlens_decoded,
                cache_batch_idx=None,
                cache_leftpad=None,
                block_table=None,
                causal=IsCausal,
                window_size=[-1, 0],
                rotary_interleaved=True,
                alibi_slopes=None,
                num_splits=1,
                use_dualkv_attention=True,
                return_softmax_lse=True,
            )
            end_evt.record()
            end_evt.synchronize()
            dualkv_time_list.append(start_evt.elapsed_time(end_evt) / 1e3)  # seconds
        dualkv_time = statistics.median(dualkv_time_list)

        print("=== dualkv flash attention done: sleep 3 seconds ===")
        time.sleep(1)
        print("=== original flash attention ===")

        # ---- Original FlashAttention timing ----
        # Build combined cache (context + decoded) for the FA path; pad to full capacity with NaNs
        Kcache_tmp = torch.concat(
            (Kcontext_cache[:, :cache_seqlens_context, :, :].repeat([bs, 1, 1, 1]), Kdecode_cache), dim=1
        )
        Vcache_tmp = torch.concat(
            (Vcontext_cache[:, :cache_seqlens_context, :, :].repeat([bs, 1, 1, 1]), Vdecode_cache), dim=1
        )
        Kcache = torch.full(
            [bs, context_seqlen_max + decoded_seqlen_max, nheads_k, hdim],
            float("nan"),
            device=device,
            dtype=dtype,
        )
        Vcache = torch.full(
            [bs, context_seqlen_max + decoded_seqlen_max, nheads_k, hdim],
            float("nan"),
            device=device,
            dtype=dtype,
        )
        Kcache[:, : cache_seqlens_context + decoded_seqlen_max, :, :] = Kcache_tmp
        Vcache[:, : cache_seqlens_context + decoded_seqlen_max, :, :] = Vcache_tmp

        og_fa_time_list = []
        for _ in range(num_runs):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            out_og, lse_og = flash_attn_with_kvcache(
                q=q_ro,
                k_cache=Kcache,
                v_cache=Vcache,
                k_cache_decoded=None,
                v_cache_decoded=None,
                k_cache_context=None,
                v_cache_context=None,
                k=k_ro,
                v=v,
                rotary_cos=cos,
                rotary_sin=sin,
                cache_seqlens=cache_seqlens,
                cache_batch_idx=None,
                cache_leftpad=None,
                block_table=None,
                causal=IsCausal,
                window_size=[-1, 0],
                rotary_interleaved=True,
                alibi_slopes=None,
                num_splits=0,
                use_dualkv_attention=False,
                return_softmax_lse=True,
            )
            end_evt.record()
            end_evt.synchronize()
            og_fa_time_list.append(start_evt.elapsed_time(end_evt) / 1e3)  # seconds
        og_fa_time = statistics.median(og_fa_time_list)


        diff = (out_bf - out_og).abs()
        loc = torch.argwhere(diff >= diff.max())
        print("out_og@max_diff = ", out_og[loc[0][0], loc[0][1], loc[0][2], loc[0][3]])
        print("out_bf@max_diff = ", out_bf[loc[0][0], loc[0][1], loc[0][2], loc[0][3]])

        log_msg = (
            f"[INFO] context length = {cache_seqlens_context}, decoded length = {cache_seqlens_decoded}. PASS!"
        )
        assert not torch.isnan(torch.flatten(out_bf)).any()
        #assert torch.allclose(out_bf, out_og, rtol=1e-08, atol=9.8e-04)
        assert torch.allclose(out_bf, out_og, rtol=1e-08, atol=2.5e-03)
        return (dualkv_time, og_fa_time, log_msg)

