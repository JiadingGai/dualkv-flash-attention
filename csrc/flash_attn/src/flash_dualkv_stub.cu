#include "flash.h"
#include <cutlass/half.h>
#include <cutlass/bfloat16.h>

namespace FLASH_NAMESPACE {

template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_dualkv_(Flash_fwd_params &params, cudaStream_t stream) {
}

template<typename T, int Headdim, bool Is_causal>
void run_mha_bwd_dualkv_(Flash_bwd_params &params, cudaStream_t stream) {
}

// bf16 fwd stubs (all hdims)
template void run_mha_fwd_dualkv_<cutlass::bfloat16_t, 32, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_dualkv_<cutlass::bfloat16_t, 32, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_dualkv_<cutlass::bfloat16_t, 64, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_dualkv_<cutlass::bfloat16_t, 64, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_dualkv_<cutlass::bfloat16_t, 96, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_dualkv_<cutlass::bfloat16_t, 96, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_dualkv_<cutlass::bfloat16_t, 128, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_dualkv_<cutlass::bfloat16_t, 128, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_dualkv_<cutlass::bfloat16_t, 192, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_dualkv_<cutlass::bfloat16_t, 192, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_dualkv_<cutlass::bfloat16_t, 256, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_dualkv_<cutlass::bfloat16_t, 256, true>(Flash_fwd_params &, cudaStream_t);

// bf16 bwd stubs (all hdims)
template void run_mha_bwd_dualkv_<cutlass::bfloat16_t, 32, false>(Flash_bwd_params &, cudaStream_t);
template void run_mha_bwd_dualkv_<cutlass::bfloat16_t, 32, true>(Flash_bwd_params &, cudaStream_t);
template void run_mha_bwd_dualkv_<cutlass::bfloat16_t, 64, false>(Flash_bwd_params &, cudaStream_t);
template void run_mha_bwd_dualkv_<cutlass::bfloat16_t, 64, true>(Flash_bwd_params &, cudaStream_t);
template void run_mha_bwd_dualkv_<cutlass::bfloat16_t, 96, false>(Flash_bwd_params &, cudaStream_t);
template void run_mha_bwd_dualkv_<cutlass::bfloat16_t, 96, true>(Flash_bwd_params &, cudaStream_t);
template void run_mha_bwd_dualkv_<cutlass::bfloat16_t, 128, false>(Flash_bwd_params &, cudaStream_t);
template void run_mha_bwd_dualkv_<cutlass::bfloat16_t, 128, true>(Flash_bwd_params &, cudaStream_t);
template void run_mha_bwd_dualkv_<cutlass::bfloat16_t, 192, false>(Flash_bwd_params &, cudaStream_t);
template void run_mha_bwd_dualkv_<cutlass::bfloat16_t, 192, true>(Flash_bwd_params &, cudaStream_t);
template void run_mha_bwd_dualkv_<cutlass::bfloat16_t, 256, false>(Flash_bwd_params &, cudaStream_t);
template void run_mha_bwd_dualkv_<cutlass::bfloat16_t, 256, true>(Flash_bwd_params &, cudaStream_t);

// fp16 hdim32 stubs (not supported by DualKV kernels)
template void run_mha_fwd_dualkv_<cutlass::half_t, 32, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_dualkv_<cutlass::half_t, 32, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_bwd_dualkv_<cutlass::half_t, 32, false>(Flash_bwd_params &, cudaStream_t);
template void run_mha_bwd_dualkv_<cutlass::half_t, 32, true>(Flash_bwd_params &, cudaStream_t);

}  // namespace FLASH_NAMESPACE
