"""Test DualKV kernel correctness across all head dimensions (64, 128, 192, 256).
Each test config runs in a subprocess to isolate CUDA errors.
Context KV is shared (bs=1), decoded KV is per-batch.
All tests use num_splits=1."""

import subprocess
import sys
import json


def run_single_test(bs, ctx, dec, hdim, nheads, nheads_k):
    script = f"""
import torch, json
from einops import repeat
from flash_attn import flash_attn_with_kvcache

def attention_ref(q, k, v, causal=True, softmax_scale=None):
    B, sq, nh, hd = q.shape
    _, sk, nhk, _ = k.shape
    if softmax_scale is None: softmax_scale = hd ** (-0.5)
    g = nh // nhk
    k = repeat(k, 'b s h d -> b s (h g) d', g=g)
    v = repeat(v, 'b s h d -> b s (h g) d', g=g)
    q, k, v = q.transpose(1,2).float(), k.transpose(1,2).float(), v.transpose(1,2).float()
    scores = torch.matmul(q, k.transpose(-2,-1)) * softmax_scale
    if causal:
        ri = torch.arange(sq, device=q.device).unsqueeze(1)
        ci = torch.arange(sk, device=q.device).unsqueeze(0)
        scores.masked_fill_((ci > (ri + sk - sq)).unsqueeze(0).unsqueeze(0), float('-inf'))
    return torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1,2).half()

bs, sq, nh, nhk, hd = {bs}, 1, {nheads}, {nheads_k}, {hdim}
ctx, dec = {ctx}, {dec}
ctx_max = max(ctx+64, 256)
dec_max = max(dec+64, 128)
torch.manual_seed(42)
device = 'cuda'
q = torch.randn(bs,sq,nh,hd,device=device,dtype=torch.float16)
kn = torch.randn(bs,sq,nhk,hd,device=device,dtype=torch.float16)
vn = torch.randn(bs,sq,nhk,hd,device=device,dtype=torch.float16)
kc = torch.randn(1,ctx_max,nhk,hd,device=device,dtype=torch.float16)
vc = torch.randn(1,ctx_max,nhk,hd,device=device,dtype=torch.float16)
kd = torch.randn(bs,dec_max,nhk,hd,device=device,dtype=torch.float16)
vd = torch.randn(bs,dec_max,nhk,hd,device=device,dtype=torch.float16)
sc = torch.full((bs,),ctx,dtype=torch.int32,device=device)
sd = torch.full((bs,),dec,dtype=torch.int32,device=device)
kc_exp = kc.expand(bs,-1,-1,-1)
vc_exp = vc.expand(bs,-1,-1,-1)
kf = torch.cat([kc_exp[:,:ctx],kd[:,:dec],kn],dim=1)
vf = torch.cat([vc_exp[:,:ctx],vd[:,:dec],vn],dim=1)
ref = attention_ref(q, kf, vf, causal=True)
out = flash_attn_with_kvcache(q.clone(), k_cache_context=kc.clone(), v_cache_context=vc.clone(),
    k_cache_decoded=kd.clone(), v_cache_decoded=vd.clone(), k=kn.clone(), v=vn.clone(),
    cache_seqlens_context=sc.clone(), cache_seqlens_decoded=sd.clone(),
    causal=True, num_splits=1, use_dualkv_attention=True)
md = (out-ref).abs().max().item()
mn = (out-ref).abs().mean().item()
nan = out.isnan().any().item()
zero = (out==0).all().item()
print(json.dumps({{"max_diff": md, "mean_diff": mn, "nan": nan, "zero": zero}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        return None, result.stderr[-300:]
    try:
        return json.loads(result.stdout.strip()), None
    except:
        return None, result.stdout[-200:]


def main():
    # kBlockN by hdim: 64->256, 128->128, 192->64, 256->64
    configs = []

    for hdim in [64, 128, 192, 256]:
        # Basic configs per hdim
        configs += [
            # (bs, ctx, dec, hdim, nheads, nheads_k)
            (1,   5,   0, hdim, 8, 8),   # tiny ctx, no decoded
            (1,   5,   1, hdim, 8, 8),   # tiny ctx + 1 decoded
            (1,  64,   0, hdim, 8, 8),   # 1 block boundary
            (1,  64,  10, hdim, 8, 8),
            (1, 128,  50, hdim, 8, 8),   # larger
            (1, 200, 100, hdim, 8, 8),
            (1, 256,   0, hdim, 8, 8),   # 256 context, no decoded
            (2, 128,  50, hdim, 8, 8),   # multi-batch
            # GQA
            (1, 128,  50, hdim, 8, 2),
            (2, 200, 100, hdim, 8, 2),
        ]

    pass_count = 0
    fail_count = 0
    current_hdim = None

    for bs, ctx, dec, hdim, nh, nhk in configs:
        if hdim != current_hdim:
            kbn = {64: 256, 128: 128, 192: 64, 256: 64}[hdim]
            print(f"\n--- hdim={hdim} (kBlockN={kbn}) ---")
            current_hdim = hdim

        label = f"bs={bs:2d} ctx={ctx:4d} dec={dec:3d} h={nh}/{nhk}"
        r, err = run_single_test(bs, ctx, dec, hdim, nh, nhk)
        if r is None:
            fail_count += 1
            # Truncate error for readability
            err_short = err.split('\n')[-2] if err and '\n' in err else (err or 'unknown')[:80]
            print(f"  [ERROR] {label}  {err_short}")
        else:
            ok = r["max_diff"] < 0.05 and not r["nan"] and not r["zero"]
            status = "PASS" if ok else "FAIL"
            if ok:
                pass_count += 1
            else:
                fail_count += 1
            reason = ""
            if r["nan"]: reason = " [NaN]"
            elif r["zero"]: reason = " [ALL_ZERO]"
            elif r["max_diff"] >= 0.05: reason = f" [max_diff={r['max_diff']:.4f}]"
            print(f"  [{status}] {label}  max={r['max_diff']:.6f} mean={r['mean_diff']:.6f}{reason}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {pass_count} PASS, {fail_count} FAIL out of {pass_count + fail_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
