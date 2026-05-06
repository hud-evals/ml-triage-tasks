// nmoe: CUDA Polar Express orthogonalization (BF16 I/O, FP32 accumulate)
//
// Implements Polar Express sign method from https://arxiv.org/pdf/2505.16932
// by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.
//
// Coefficients match modded-nanogpt's Polar Express implementation (5 iterations).
//
// API (C):
//   void normuon_orth_bf16_inplace(void* x_bf16, long long B, int M, int N,
//                                  cudaStream_t stream);
//     x_bf16:   BF16 buffer (row-major) of size B*M*N elements, updated in place
//     B:        batch size (flattened leading dims)
//     M, N:     matrix dims
//     stream:   CUDA stream (may be nullptr)
//
// Notes:
// - Uses spectral norm estimate: X / (||X||_F * (1 + safety_factor))
// - GEMM math uses column-major semantics via cuBLAS.
// - Workspace for A (B, M, M) and T (B, M, M) is chunked over the batch to cap memory.
// - Data types: X is BF16 in/out; A/T are FP32; computeType is CUBLAS_COMPUTE_32F_FAST_TF32.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <stdint.h>
#include <stdio.h>

#define NMOE_CHECK_CUDA(cmd)                                                     \
  do {                                                                           \
    cudaError_t _e = (cmd);                                                      \
    if (_e != cudaSuccess) {                                                     \
      fprintf(stderr, "[normuon.cu] CUDA error %d: %s at %s:%d\n",               \
              int(_e), cudaGetErrorString(_e), __FILE__, __LINE__);              \
      return;                                                                    \
    }                                                                            \
  } while (0)

#define NMOE_CHECK_CUBLAS(cmd)                                                   \
  do {                                                                           \
    cublasStatus_t _s = (cmd);                                                   \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                           \
      fprintf(stderr, "[normuon.cu] cuBLAS error %d at %s:%d\n", int(_s),        \
              __FILE__, __LINE__);                                               \
      return;                                                                    \
    }                                                                            \
  } while (0)

#define NMOE_CHECK_CUBLASLT(cmd)                                                 \
  do {                                                                           \
    cublasStatus_t _s = (cmd);                                                   \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                           \
      fprintf(stderr, "[normuon.cu] cuBLASLt error %d at %s:%d\n", int(_s),      \
              __FILE__, __LINE__);                                               \
      return;                                                                    \
    }                                                                            \
  } while (0)

// -------------------------------
// Polar Express Coefficients (a, b, c)
// From modded-nanogpt: computed for num_iters=5, safety_factor=2e-2, cushion=2
// Reference: https://arxiv.org/pdf/2505.16932
// -------------------------------
struct ABC { float a, b, c; };

// 5-iteration Polar Express coefficients matching modded-nanogpt exactly
static const ABC POLAR_EXPRESS_COEFFS[5] = {
  {8.156554524902461f, -22.48329292557795f, 15.878769915207462f},
  {4.042929935166739f, -2.808917465908714f, 0.5000178451051316f},
  {3.8916678022926607f, -2.772484153217685f, 0.5060648178503393f},
  {3.285753657755655f, -2.3681294933425376f, 0.46449024233003106f},
  {2.3465413258596377f, -1.7097828382687081f, 0.42323551169305323f},
};

static const int POLAR_EXPRESS_ITERS = 5;
static const float SAFETY_FACTOR = 2e-2f;  // Matches modded-nanogpt

// -----------------------------------------
// Frobenius norm (BF16) and in-place scale
// -----------------------------------------
// Device helper: in-place x[i] = 1 / (sqrt(x[i]) * (1 + safety_factor) + eps)
// Matches modded-nanogpt: X / (X.norm() * (1 + 2e-2) + 1e-6)
__global__ void normuon_invsqrt_kernel(float* x, long long n) {
  long long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    float norm = sqrtf(fmaxf(v, 0.0f));
    x[i] = 1.0f / (norm * (1.0f + SAFETY_FACTOR) + 1e-6f);
  }
}
static __global__ void reduce_frob_bf16(const __nv_bfloat16* __restrict__ X,
                                 long long stride_elems,
                                 long long elems_per_mat,
                                 float* __restrict__ out_norm2, // sum of squares
                                 long long B) {
  long long b = blockIdx.x;
  if (b >= B) return;
  const __nv_bfloat16* base = X + b * stride_elems;
  float sum = 0.0f;
  for (long long i = threadIdx.x; i < elems_per_mat; i += blockDim.x) {
    float v = __bfloat162float(base[i]);
    sum += v * v;
  }
  __shared__ float smem[256];
  smem[threadIdx.x] = sum;
  __syncthreads();
  // reduction
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) out_norm2[b] = smem[0];
}

static __global__ void scale_frob_bf16(__nv_bfloat16* __restrict__ X,
                                long long stride_elems,
                                long long elems_per_mat,
                                const float* __restrict__ inv_norm,
                                long long B) {
  long long b = blockIdx.x;
  if (b >= B) return;
  __nv_bfloat16* base = X + b * stride_elems;
  float s = inv_norm[b];
  for (long long i = threadIdx.x; i < elems_per_mat; i += blockDim.x) {
    float v = __bfloat162float(base[i]);
    base[i] = __float2bfloat16(v * s);
  }
}

// -----------------------------------------
// Helpers
// -----------------------------------------
static inline int64_t div_up_int64(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

// Compute B = b*A + c*T  (A,T in FP32), batched over tileB. In-place write into A.
static __global__ void lincomb_fp32(float* __restrict__ A,
                             const float* __restrict__ T,
                             int64_t stride_a, int64_t stride_t,
                             int elems, float b, float c,
                             int tileB) {
  int bidx = blockIdx.y;
  if (bidx >= tileB) return;
  float* a = A + bidx * stride_a;
  const float* t = T + bidx * stride_t;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elems; i += blockDim.x * gridDim.x) {
    a[i] = b * a[i] + c * t[i];
  }
}

// Cast FP32 buffer to BF16 (batched), vectorized: 2x FP32 -> bf162
static __global__ void nmoe_cast_fp32_to_bf16_vec2(const float* __restrict__ in,
                                            __nv_bfloat16* __restrict__ out,
                                            int64_t stride_in, int64_t stride_out,
                                            int elems, int tileB) {
  int bidx = blockIdx.y;
  if (bidx >= tileB) return;
  const float* src = in + bidx * stride_in;
  __nv_bfloat16* dst = out + bidx * stride_out;

  // Convert pairs
  int64_t pairs = elems >> 1; // elems / 2
  for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x; p < pairs; p += blockDim.x * gridDim.x) {
    float2 f2 = reinterpret_cast<const float2*>(src)[p];
    __nv_bfloat162 b2 = __floats2bfloat162_rn(f2.x, f2.y);
    reinterpret_cast<__nv_bfloat162*>(dst)[p] = b2;
  }
  // Odd tail
  if (((elems & 1) != 0)) {
    int64_t i = pairs * 2 + threadIdx.x;
    if (i == elems - 1 && blockIdx.x == 0) {
      dst[i] = __float2bfloat16(src[i]);
    }
  }
}

// Cast BF16 buffer to FP32 (batched), vectorized: bf162 -> 2x FP32
static __global__ void nmoe_cast_bf16_to_fp32_vec2(const __nv_bfloat16* __restrict__ in,
                                            float* __restrict__ out,
                                            int64_t stride_in, int64_t stride_out,
                                            int elems, int tileB) {
  int bidx = blockIdx.y;
  if (bidx >= tileB) return;
  const __nv_bfloat16* src = in + bidx * stride_in;
  float* dst = out + bidx * stride_out;

  int64_t pairs = elems >> 1; // elems / 2
  for (int64_t p = blockIdx.x * blockDim.x + threadIdx.x; p < pairs; p += blockDim.x * gridDim.x) {
    __nv_bfloat162 b2 = reinterpret_cast<const __nv_bfloat162*>(src)[p];
    float2 f2 = __bfloat1622float2(b2);
    reinterpret_cast<float2*>(dst)[p] = f2;
  }
  if ((elems & 1) != 0) {
    int64_t i = pairs * 2 + threadIdx.x;
    if (i == elems - 1 && blockIdx.x == 0) {
      dst[i] = __bfloat162float(src[i]);
    }
  }
}

// -------------------------------
// Persistent plan (workspace + handles)
// -------------------------------
struct MuonPlan {
  int Bmax{0};
  int M{0};
  int N{0};
  int S{0};
  // cuBLAS handles
  cublasHandle_t h{nullptr};
  // Reserve Lt handle for next diff (gemm via Lt)
  cublasLtHandle_t lt{nullptr};
  // Workspaces sized for Bmax tiles
  float* d_A{nullptr};
  float* d_T{nullptr};
  float* d_X32{nullptr};
  float* d_norm2{nullptr};
  int64_t stride_a{0};
  int64_t bytes_per_A{0};
  int64_t stride_x{0};
  int64_t bytes_per_X32{0};
};

extern "C" void* normuon_plan_create(int Bmax, int M, int N) {
  if (Bmax <= 0 || M <= 0 || N <= 0) return nullptr;
  if ((((int64_t)M * (int64_t)N) & 1LL) != 0) {
    fprintf(stderr,
            "[normuon.cu] Unsupported shape for vec2 casts: M*N must be even, got M=%d N=%d\n",
            M, N);
    return nullptr;
  }
  MuonPlan* P = new MuonPlan();
  P->Bmax = Bmax;
  P->M = M;
  P->N = N;
  P->S = (M < N) ? M : N;
  P->stride_a = (int64_t)P->S * (int64_t)P->S;
  P->bytes_per_A = P->stride_a * (int64_t)sizeof(float);
  P->stride_x = (int64_t)N * (int64_t)M;
  P->bytes_per_X32 = P->stride_x * (int64_t)sizeof(float);

  // Handles
  if (cublasCreate(&P->h) != CUBLAS_STATUS_SUCCESS) { delete P; return nullptr; }
  // Use Tensor Core TF32 for FP32 GEMMs (B200 target).
  if (cublasSetMathMode(P->h, CUBLAS_TF32_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(P->h);
    delete P;
    return nullptr;
  }
  if (cublasLtCreate(&P->lt) != CUBLAS_STATUS_SUCCESS) { cublasDestroy(P->h); delete P; return nullptr; }

  // Workspaces (async allocation on default stream)
  if (cudaMalloc(&P->d_A, (size_t)(P->bytes_per_A * Bmax)) != cudaSuccess) { cublasDestroy(P->h); cublasLtDestroy(P->lt); delete P; return nullptr; }
  if (cudaMalloc(&P->d_T, (size_t)(P->bytes_per_A * Bmax)) != cudaSuccess) { cudaFree(P->d_A); cublasDestroy(P->h); cublasLtDestroy(P->lt); delete P; return nullptr; }
  if (cudaMalloc(&P->d_X32, (size_t)(P->bytes_per_X32 * Bmax)) != cudaSuccess) { cudaFree(P->d_T); cudaFree(P->d_A); cublasDestroy(P->h); cublasLtDestroy(P->lt); delete P; return nullptr; }
  if (cudaMalloc(&P->d_norm2, (size_t)(sizeof(float) * (size_t)Bmax)) != cudaSuccess) { cudaFree(P->d_X32); cudaFree(P->d_T); cudaFree(P->d_A); cublasDestroy(P->h); cublasLtDestroy(P->lt); delete P; return nullptr; }
  return P;
}

extern "C" void normuon_plan_destroy(void* plan) {
  if (!plan) return;
  MuonPlan* P = reinterpret_cast<MuonPlan*>(plan);
  if (P->d_norm2) cudaFree(P->d_norm2);
  if (P->d_X32) cudaFree(P->d_X32);
  if (P->d_T) cudaFree(P->d_T);
  if (P->d_A) cudaFree(P->d_A);
  if (P->lt) cublasLtDestroy(P->lt);
  if (P->h) cublasDestroy(P->h);
  delete P;
}

extern "C" void normuon_plan_run(void* plan,
                                 void* x_bf16,
                                 long long B,
                                 int M, int N,
                                 cudaStream_t stream) {
  if (!plan || x_bf16 == nullptr || B <= 0 || M <= 0 || N <= 0) return;
  MuonPlan* P = reinterpret_cast<MuonPlan*>(plan);
  // Shape guard (explicit, loud)
  if (P->M != M || P->N != N) {
    fprintf(stderr, "[normuon.cu] Plan shape mismatch: plan=(%d,%d), got=(%d,%d)\n", P->M, P->N, M, N);
    return;
  }
  if ((((int64_t)M * (int64_t)N) & 1LL) != 0) {
    fprintf(stderr,
            "[normuon.cu] Unsupported shape for vec2 casts: M*N must be even, got M=%d N=%d\n",
            M, N);
    return;
  }

  // Strides (row-major X: [B, M, N])
  const int64_t stride_x_elems = (int64_t)M * (int64_t)N;
  const int64_t elems_per_mat = stride_x_elems;

  // Polar Express iterations in batched tiles using cuBLAS GEMM (FP32 accumulate)
  // Set stream on persistent cuBLAS handle
  NMOE_CHECK_CUBLAS(cublasSetStream(P->h, stream));
  // Scalars (host mode)
  const float one = 1.0f, zero = 0.0f;

  // Workspace cap in bytes (A and T buffers). Keep modest to avoid OOM.
  // Tile size is bounded by persistent allocation
  int64_t tileB_max = P->Bmax;

  // Batched strides (column-major view): X_col is N x M with lda=N and stride=N*M
  const int lda_x = N; // leading dimension of X_col
  const int64_t stride_x = (int64_t)N * (int64_t)M; // elements
  const cudaDataType F32 = CUDA_R_32F;
  const cublasComputeType_t compute = CUBLAS_COMPUTE_32F_FAST_TF32;

  // Polar Express: 5 fixed iterations with specific coefficients
  const int num_iters = POLAR_EXPRESS_ITERS;

  for (int64_t start = 0; start < B; start += tileB_max) {
    int tileB = (int) ((start + tileB_max <= B) ? tileB_max : (B - start));

    // Base pointers for this tile
    __nv_bfloat16* X_tile = reinterpret_cast<__nv_bfloat16*>(x_bf16) + start * stride_x;

    // Frobenius normalization per matrix for this tile:
    // X /= (||X||_F * (1 + safety_factor) + eps)
    reduce_frob_bf16<<<(unsigned)tileB, 256, 0, stream>>>(
        X_tile, stride_x_elems, elems_per_mat, P->d_norm2, tileB);
    NMOE_CHECK_CUDA(cudaGetLastError());
    normuon_invsqrt_kernel<<<(unsigned)div_up_int64(tileB, 256), 256, 0, stream>>>(P->d_norm2, tileB);
    NMOE_CHECK_CUDA(cudaGetLastError());
    scale_frob_bf16<<<(unsigned)tileB, 256, 0, stream>>>(
        X_tile, stride_x_elems, elems_per_mat, P->d_norm2, tileB);
    NMOE_CHECK_CUDA(cudaGetLastError());

    for (int it = 0; it < num_iters; ++it) {
      const ABC c = POLAR_EXPRESS_COEFFS[it];

      // A = X * X^T in FP32: lift X(BF16) -> X32, then A based on orientation
      {
        const int threads = 256;
        const int64_t elems = (int64_t)M * (int64_t)N;
        const int blocks_x = (int)div_up_int64(elems, threads * 8LL);
        dim3 grid(blocks_x, (unsigned)tileB, 1);
        nmoe_cast_bf16_to_fp32_vec2<<<grid, threads, 0, stream>>>(
            X_tile, P->d_X32, stride_x, P->stride_x, (int)elems, tileB);
        NMOE_CHECK_CUDA(cudaGetLastError());
      }
      // Orientation selection
      const bool postmul = (M > N); // if true: X <- a*X + X*B ; else: X <- a*X + B*X
      const int S = postmul ? N : M;
      // A build: if !postmul -> A = X32^T @ X32 (M x M); else -> A = X32 @ X32^T (N x N)
      if (!postmul) {
        NMOE_CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            P->h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            M, M, N,
            &one,
            P->d_X32, F32, lda_x, P->stride_x,
            P->d_X32, F32, lda_x, P->stride_x,
            &zero,
            P->d_A, F32, M, P->stride_a,
            tileB,
            compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      } else {
        NMOE_CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            P->h,
            CUBLAS_OP_N, CUBLAS_OP_T,
            N, N, M,
            &one,
            P->d_X32, F32, lda_x, P->stride_x,
            P->d_X32, F32, lda_x, P->stride_x,
            &zero,
            P->d_A, F32, N, P->stride_a,
            tileB,
            compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }

      // (No additional preconditioning per blog baseline)

      // T = A @ A  (both FP32), with square S
      NMOE_CHECK_CUBLAS(cublasGemmStridedBatchedEx(
          P->h,
          CUBLAS_OP_N, CUBLAS_OP_N,
          S, S, S,
          &one,
          P->d_A, F32, S, P->stride_a,
          P->d_A, F32, S, P->stride_a,
          &zero,
          P->d_T, F32, S, P->stride_a,
          tileB,
          compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

      // B = b*A + c*T (in place over A)
      {
        const int threads = 256;
        const int64_t elems = (int64_t)S * (int64_t)S;
        const int blocks_x = (int)div_up_int64(elems, threads * 8LL);
        dim3 grid(blocks_x, (unsigned)tileB, 1);
        lincomb_fp32<<<grid, threads, 0, stream>>>(
            P->d_A, P->d_T, P->stride_a, P->stride_a, (int)elems, c.b, c.c, tileB);
        NMOE_CHECK_CUDA(cudaGetLastError());
      }

      // Update X32 in FP32: if !postmul -> X32 = a*X32 + X32 @ B^T
      //                      if  postmul -> X32 = a*X32 + B^T @ X32
      if (!postmul) {
        NMOE_CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            P->h,
            CUBLAS_OP_N, CUBLAS_OP_T,
            N, M, S,
            &one,
            P->d_X32, F32, lda_x, P->stride_x,
            P->d_A, F32, S, P->stride_a,
            &c.a,
            P->d_X32, F32, lda_x, P->stride_x,
            tileB,
            compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      } else {
        NMOE_CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            P->h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            S, M, S,
            &one,
            P->d_A, F32, S, P->stride_a,
            P->d_X32, F32, lda_x, P->stride_x,
            &c.a,
            P->d_X32, F32, lda_x, P->stride_x,
            tileB,
            compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
      // Cast X32 back to BF16 in place to X_tile
      {
        const int threads = 256;
        const int64_t elems = (int64_t)M * (int64_t)N;
        const int blocks_x = (int)div_up_int64(elems, threads * 8LL);
        dim3 grid(blocks_x, (unsigned)tileB, 1);
        nmoe_cast_fp32_to_bf16_vec2<<<grid, threads, 0, stream>>>(
            P->d_X32, X_tile, P->stride_x, (int64_t)M * (int64_t)N, (int)elems, tileB);
        NMOE_CHECK_CUDA(cudaGetLastError());
      }
    }
  }
}

// Single-shot entry point for convenience
extern "C" void normuon(void* x_bf16,
                        long long B,
                        int M, int N,
                        cudaStream_t stream) {
  // Create a tiny plan with Bmax= min(B, 32) and destroy after run.
  int Bmax = (int)((B < 32) ? B : 32);
  void* plan = normuon_plan_create(Bmax, M, N);
  if (!plan) return;
  normuon_plan_run(plan, x_bf16, B, M, N, stream);
  normuon_plan_destroy(plan);
}
