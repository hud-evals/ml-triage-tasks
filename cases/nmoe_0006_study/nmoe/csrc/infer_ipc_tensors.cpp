// SPDX-License-Identifier: Apache-2.0
// Torch-linked helper for allocating CUDA-IPC-friendly tensors via cudaMalloc.
//
// Why this exists:
// - nmoe.csrc.rdep is intentionally torch-independent (links only CUDA + pybind11).
// - CUDA IPC in rdep_infer_* expects raw cudaIpcMemHandle_t handles (64 bytes).
// - PyTorch tensors are typically sub-allocated from a caching allocator segment and
//   are not safe to pass directly to cudaIpcGetMemHandle/cudaIpcOpenMemHandle by data_ptr.
//
// This module allocates dedicated cudaMalloc regions and wraps them as torch::Tensor
// via from_blob, so data_ptr() is the allocation base and CUDA-IPC is reliable.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <tuple>

static inline void cuda_check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

static torch::Tensor cuda_malloc_tensor(
    at::IntArrayRef sizes,
    torch::TensorOptions options) {
  if (!options.device().is_cuda()) {
    throw std::runtime_error("cuda_malloc_tensor: expected CUDA device");
  }
  int64_t numel = 1;
  for (auto s : sizes) numel *= s;
  const auto scalar = options.dtype().toScalarType();
  const auto elsize = c10::elementSize(scalar);
  const size_t nbytes = static_cast<size_t>(numel) * static_cast<size_t>(elsize);
  void* ptr = nullptr;
  cuda_check(cudaMalloc(&ptr, nbytes), "cudaMalloc");
  // Zero-init for safety (barrier signals, routing buffers).
  cuda_check(cudaMemset(ptr, 0, nbytes), "cudaMemset");

  auto deleter = [](void* p) {
    cudaFree(p);
  };
  return torch::from_blob(ptr, sizes, deleter, options);
}

// NOTE: CUDA IPC has a practical limit on concurrently opened mem handles.
// For 2-channel inference (decode+prefill), opening per-tensor handles
// (recv_x_fp8/scale/topk/weights/ret_y/barrier) can exceed that limit.
//
// The slab API below allocates ONE cudaMalloc region per channel per rank and
// returns typed tensor views into it (with a no-op deleter). Only the slab base
// pointer is used for IPC handle exchange, so each rank only opens ~7 handles
// per channel (one per peer) instead of ~6*7.

static void noop_deleter(void*) {}

struct InferIpcSlabLayout {
  int64_t barrier_off = 0;
  int64_t recv_x_fp8_off = 0;
  int64_t recv_x_scale_off = 0;
  int64_t recv_topk_idx_off = 0;
  int64_t recv_topk_w_off = 0;
  int64_t ret_y_off = 0;
  int64_t total_bytes = 0;
};

static inline int64_t align_up(int64_t x, int64_t a) {
  return (x + (a - 1)) / a * a;
}

static InferIpcSlabLayout infer_slab_layout(int64_t world, int64_t T_cap, int64_t H, int64_t K) {
  const int64_t B_global = world * T_cap;
  const int64_t Hsf = H / 128;

  // Conservative alignment to keep all typed views naturally aligned.
  // BF16 and FP8 are 2/1 bytes, but we align to 256 for vectorized stores.
  const int64_t A = 256;

  InferIpcSlabLayout l;
  int64_t off = 0;

  l.barrier_off = off;
  off += align_up(static_cast<int64_t>(world) * static_cast<int64_t>(sizeof(int32_t)), A);

  l.recv_x_fp8_off = off;
  off += align_up(B_global * H * static_cast<int64_t>(1), A);  // fp8 bytes

  l.recv_x_scale_off = off;
  off += align_up(B_global * Hsf * static_cast<int64_t>(sizeof(float)), A);

  l.recv_topk_idx_off = off;
  off += align_up(B_global * K * static_cast<int64_t>(sizeof(int64_t)), A);

  l.recv_topk_w_off = off;
  off += align_up(B_global * K * static_cast<int64_t>(sizeof(float)), A);

  l.ret_y_off = off;
  off += align_up(world * T_cap * H * static_cast<int64_t>(sizeof(at::BFloat16)), A);

  l.total_bytes = off;
  return l;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
alloc_infer_ipc_slab_fp8(
    int64_t world,
    int64_t T_cap,
    int64_t H,
    int64_t K) {
  if (world <= 1) throw std::runtime_error("world must be > 1");
  if (world > 8) throw std::runtime_error("world must be <= 8 (single-node NVLink)");
  if (T_cap <= 0) throw std::runtime_error("T_cap must be > 0");
  if (H <= 0 || (H % 128) != 0) throw std::runtime_error("H must be > 0 and divisible by 128");
  if (K <= 0) throw std::runtime_error("K must be > 0");

  const int device = static_cast<int>(at::cuda::current_device());
  c10::cuda::CUDAGuard guard(device);

  const int64_t B_global = world * T_cap;
  const int64_t Hsf = H / 128;

  auto dev = torch::Device(torch::kCUDA, device);

  const auto layout = infer_slab_layout(world, T_cap, H, K);
  void* base = nullptr;
  cuda_check(cudaMalloc(&base, static_cast<size_t>(layout.total_bytes)), "cudaMalloc(slab)");
  cuda_check(cudaMemset(base, 0, static_cast<size_t>(layout.total_bytes)), "cudaMemset(slab)");

  auto slab = torch::from_blob(
      base,
      {layout.total_bytes},
      [](void* p) { cudaFree(p); },
      torch::TensorOptions().device(dev).dtype(torch::kUInt8));

  auto barrier = torch::from_blob(
      reinterpret_cast<uint8_t*>(base) + layout.barrier_off,
      {world},
      noop_deleter,
      torch::TensorOptions().device(dev).dtype(torch::kInt32));
  auto recv_x_fp8 = torch::from_blob(
      reinterpret_cast<uint8_t*>(base) + layout.recv_x_fp8_off,
      {B_global, H},
      noop_deleter,
      torch::TensorOptions().device(dev).dtype(c10::ScalarType::Float8_e4m3fn));
  auto recv_x_scale = torch::from_blob(
      reinterpret_cast<uint8_t*>(base) + layout.recv_x_scale_off,
      {B_global, Hsf},
      noop_deleter,
      torch::TensorOptions().device(dev).dtype(torch::kFloat32));
  auto recv_topk_idx = torch::from_blob(
      reinterpret_cast<uint8_t*>(base) + layout.recv_topk_idx_off,
      {B_global, K},
      noop_deleter,
      torch::TensorOptions().device(dev).dtype(torch::kInt64));
  auto recv_topk_w = torch::from_blob(
      reinterpret_cast<uint8_t*>(base) + layout.recv_topk_w_off,
      {B_global, K},
      noop_deleter,
      torch::TensorOptions().device(dev).dtype(torch::kFloat32));
  auto ret_y = torch::from_blob(
      reinterpret_cast<uint8_t*>(base) + layout.ret_y_off,
      {world, T_cap, H},
      noop_deleter,
      torch::TensorOptions().device(dev).dtype(torch::kBFloat16));

  auto offs = torch::tensor(
      {layout.barrier_off, layout.recv_x_fp8_off, layout.recv_x_scale_off, layout.recv_topk_idx_off, layout.recv_topk_w_off, layout.ret_y_off},
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64));

  return std::make_tuple(slab, barrier, recv_x_fp8, recv_x_scale, recv_topk_idx, recv_topk_w, ret_y, offs);
}

// Legacy API (per-tensor cudaMalloc). Kept for debugging/back-compat.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
alloc_infer_ipc_buffers_fp8(
    int64_t world,
    int64_t T_cap,
    int64_t H,
    int64_t K) {
  if (world <= 1) throw std::runtime_error("world must be > 1");
  if (world > 8) throw std::runtime_error("world must be <= 8 (single-node NVLink)");
  if (T_cap <= 0) throw std::runtime_error("T_cap must be > 0");
  if (H <= 0 || (H % 128) != 0) throw std::runtime_error("H must be > 0 and divisible by 128");
  if (K <= 0) throw std::runtime_error("K must be > 0");

  const int device = static_cast<int>(at::cuda::current_device());
  c10::cuda::CUDAGuard guard(device);

  const int64_t B_global = world * T_cap;
  const int64_t Hsf = H / 128;

  auto dev = torch::Device(torch::kCUDA, device);

  auto barrier = cuda_malloc_tensor({world}, torch::TensorOptions().device(dev).dtype(torch::kInt32));
  auto recv_x_fp8 = cuda_malloc_tensor({B_global, H}, torch::TensorOptions().device(dev).dtype(c10::ScalarType::Float8_e4m3fn));
  auto recv_x_scale = cuda_malloc_tensor({B_global, Hsf}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
  auto recv_topk_idx = cuda_malloc_tensor({B_global, K}, torch::TensorOptions().device(dev).dtype(torch::kInt64));
  auto recv_topk_w = cuda_malloc_tensor({B_global, K}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
  auto ret_y = cuda_malloc_tensor({world, T_cap, H}, torch::TensorOptions().device(dev).dtype(torch::kBFloat16));

  return std::make_tuple(barrier, recv_x_fp8, recv_x_scale, recv_topk_idx, recv_topk_w, ret_y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "alloc_infer_ipc_buffers_fp8",
      &alloc_infer_ipc_buffers_fp8,
      "Allocate cudaMalloc-backed inference IPC buffers (FP8 recv + BF16 return).");
  m.def(
      "alloc_infer_ipc_slab_fp8",
      &alloc_infer_ipc_slab_fp8,
      "Allocate ONE cudaMalloc slab and return typed views + byte offsets (reduces CUDA-IPC handle count).");
}
