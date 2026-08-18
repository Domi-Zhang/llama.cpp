# 精读：CUDA 后端内核（ggml-cuda）

> 日期：2026-08-13 | 核心文件：`ggml/src/ggml-cuda/`
> 前置：已读 `07`（后端调度）。CUDA 是 llama.cpp 最重要的 GPU 后端。
> 2026-08-17 修订：扩充为初学者详细版（CUDA 入门+策略动机）。

## CUDA 执行模型 5 分钟入门

写 CUDA 代码前，先建立三个心智模型：线程层级、显存分离、合并访存。

### 1. 线程层级：grid / block / thread

CUDA 把计算组织成三级：

- **thread**：最小执行单元，每个线程有自己的寄存器。
- **block**（线程块）：一组线程，规模一般 32 的倍数（32 个线程组成一个 warp，是 GPU 的实际执行单位）。同一 block 内线程共享一块**共享内存**（shared memory，几十 KB），可通过 `__syncthreads()` 同步。
- **grid**（网格）：多个 block 组成 grid，是一次 kernel 启动的全部线程。

启动 kernel 时写 `kernel<<<gridDim, blockDim>>>(args...)`，GPU 上就生成 `gridDim.x * blockDim.x` 个线程。

### 2. kernel 启动与显存/内存分离

CPU 端的变量默认在主机内存，GPU 端的变量在显存。两者物理分离：
- CPU 不能直接读 GPU 显存，反之亦然。
- 数据需要 `cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)` 拷过去；算完再 `DeviceToHost` 拷回来。
- kernel 启动 `<<<...>>>` 是异步的：CPU 调用立即返回，GPU 在自己的 stream 上排队执行。要等结果就 `cudaStreamSynchronize`。

这就是为什么 `ggml-cuda.cu` 里到处是 `dev[id].src0_dd`、`src1_ddf` 这种 `_dd`（device data）后缀变量——它们是主机张量对应的显存拷贝。

### 3. 合并访存（coalesced）为什么重要

GPU 显存带宽虽大，但要求"同一 warp 内 32 个线程访问的地址连续"才能跑满。这叫**合并访存**。如果 32 个线程各跳 1KB 读一个字节，带宽利用率会暴跌到几十分之一。

### 4. naive matmul 例子（每个线程算一个输出元素）

理解上面三点后，看一个最朴素的 matmul：`C[M,N] = A[M,K] x B[K,N]`：

```cuda
// 每个 thread 算 C 的一个元素
__global__ void naive_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // C 的行
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // C 的列
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];  // A 是合并的，B 不是
    }
    C[row * N + col] = acc;
}
```

启动：`naive_matmul<<<dim3((N+15)/16, (M+15)/16), dim3(16,16)>>>(...)`。

问题：每个线程都把 B 的一整列重新读一遍。1000 个输出元素就读 1000 遍同一列 B。真实 GPU 代码要做**分块（tiling）**：把 A、B 的一块先 `load` 到共享内存，block 内所有线程复用，再算部分和累加。mmq/mmf 的核心就是这个分块逻辑的极致优化版。

## 推理的两类 matmul 形态：prefill vs decode

理解 matmul 形态，才能理解为什么 CUDA 后端要准备**四种**策略，而不是一种。

LLM 推理分两阶段：

| 阶段 | n_tokens | matmul 形态 | 算术强度 | 瓶颈 |
|---|---|---|---|---|
| **prefill**（提示词处理） | 大（几十~几千） | `W[M,K] x X[K, n_tokens]`（矩阵 x 矩阵） | 高（每读 1 个权重，参与 n_tokens 次乘加） | **计算受限** |
| **decode**（逐 token 生成） | 1 | `W[M,K] x x[K,1]`（矩阵 x 向量，即 matvec） | 低（每读 1 个权重，只参与 1 次乘加） | **访存受限** |

直观例子：`M=K=4096` 的权重，约 16M 个权重元素。
- prefill `n_tokens=512`：读 16M 权重做 16M * 512 次乘加，算术强度 512。
- decode `n_tokens=1`：读 16M 权重做 16M 次乘加，算术强度 1。

GPU 的算力/带宽比通常在 50~200 之间（每字节配 50~200 次乘加才不浪费算力）。所以：
- **prefill 是计算受限**：把权重读进来不亏，关键是把算力榨干——上 Tensor Core、低精度乘加。
- **decode 是访存受限**：算力根本用不满，关键是少读权重——量化（4bit 比 16bit 少读 4 倍字节）直接线性提速。

这条性质直接解释下面四种策略的存在意义。

## 布局：一个算子一个文件

`ggml-cuda/` 里每个算子一个 `.cu` + 同名 `.cuh`，按算子命名（先 Glob 核实文件名）：
- `mmq.cu`：量化权重 matmul（**主流路径**，批量推理主路径）
- `mmf.cu`：f16/bf16/f32 权重 matmul（cuBLAS 或自定义）
- `mmvq.cu` / `mmvf.cu`：matvec（向量 x 权重）变体，对应 decode 路径
- `mmid.cu`：MoE 专家 matmul（`MUL_MAT_ID`）
- `fattn*.cu`：Flash-Attention 系（`fattn-tile.cu`、`fattn-wmma-f16.cu`、`fattn-mma-f16.cuh` 等）
- `norm.cu` / `getrows.cu` / `cpy.cu` / `dequantize.cuh` / `vecdotq.cuh` / `mma.cuh` / `allreduce.cu` ...

统一由 `ggml-cuda.cu` 里的 `ggml_cuda_compute_forward`（:2791）按 `GGML_OP_*` 分发。

## 后端接口（`ggml_backend_cuda_interface`，:4796）

CUDA 实现 `ggml_backend_i`，其中：
```
.graph_compute = ggml_backend_cuda_graph_compute   (:4468)  执行整图
```
- 走 `ggml_backend_cuda_graph_compute`：先做 graph 检查/融合（`ggml_cuda_graph_check_compability`、`ggml_cuda_can_fuse`），再逐个节点调 `ggml_cuda_compute_forward`
- 支持 **CUDA Graphs**：把整张图捕获成一个可复用的 kernel 序列，显著降低启动开销（详见下文专节）

## 核心：matmul 的四种策略（`ggml_cuda_mul_mat`，:2541）

`mul_mat` 是 FFN/attention 的主算子，按「权重类型 + src1 形状 + 硬件」选策略。决策代码在 `ggml-cuda.cu` :2550-2558，先按类型给初值，再叠加硬件判断：

| 策略 | 条件 | 实现 | 什么时候用 | 为什么该场景用它 |
|---|---|---|---|---|
| `use_mul_mat_vec_f` | 权重 f32/f16/bf16，src1 行数小 | mmvf | 单 token 生成（n=1），权重未量化 | matvec 访存为主，专用向量 kernel 每个 warp 处理一行，对 B 的访问全部合并 |
| `use_mul_mat_f` | 权重非量化，src1 是 f32 | mmf / cuBLAS | 批量但权重未量化（f16/bf16 权重） | 批量时 cuBLAS GEMM 已调到极致，自定义 mmf 处理转置/特殊形状 |
| `use_mul_mat_vec_q` | 权重量化，`src1->ne[1] <= MMVQ_MAX_BATCH_SIZE` | mmvq | 单 token + 量化权重 | n=1 时 matvec 访存为主，直接读量化 block 省 4~8 倍带宽；用 `vec_dot_q*_q8_1` 算块内点积 |
| `use_mul_mat_q` | 权重量化 | **mmq**（`mmq.cu`） | **批量推理主路径** | 批量时算术强度足够高，可分块上 Tensor Core，低精度乘加 + f32 累加跑满算力 |

关键判断（`ggml-cuda.cu` :2550-2558）：
```cpp
bool use_mul_mat_vec_f = (src0->type == F32 || F16 || BF16) && src1==F32 && dst==F32;
bool use_mul_mat_f     = !ggml_is_quantized(src0->type) && src1==F32 && dst==F32;
bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && src1==F32 && dst==F32
                         && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
bool use_mul_mat_q     = ggml_is_quantized(src0->type) && src1==F32 && dst==F32;
```

随后对每张卡叠加硬件判断（:2573-2585）：`ggml_cuda_should_use_mmq` / `_mmf` / `_mmvf` / `_mmvq` 会根据 cc（compute capability）、`src1->ne[1]`、权重形状决定是否真的启用。例如旧卡没有 Tensor Core，mmq 退化到 dp4a 路径（`MMQ_DP4A_MAX_BATCH_SIZE = 64`，`mmq.cuh` :12），批量不够大就切回 mmvq。

伪代码（合并硬件判断后的最终决策）：
```
if 权重未量化:
    if n_tokens == 1 and 形状适合:           use mmvf      # 向量路径
    else:                                    use mmf/cuBLAS
else:  # 权重量化
    if n_tokens <= MMVQ_MAX_BATCH_SIZE:      use mmvq       # 小批量走向量路径
    elif 硬件支持 mmq 且 n_tokens 足够大:     use mmq        # 批量走分块
    else:                                    fallback mmvq  # 老卡或小批量
```

关键判断：`use_mul_mat_q = ggml_is_quantized(src0) && ...`，即只要权重是量化的就走 mmq--所以**绝大多数已量化模型走 mmq**。

## mmq（量化 matmul）的数据流直觉

不解剖完整内核（mmq.cu 单文件近 4000 行），只讲数据流。前置：`06` 篇讲过的 block 量化格式（Q4_0：32 个 fp16 标度 + 32 个 4bit 量化值打包成 18 字节块；Q4_K/Q6_K 等带额外 scale/min 分级）。mmq 的核心是**权重不解量化到 f32**，按 block 反量化到寄存器/共享内存，用低精度乘法 + f32 累加。

数据流（参考 `mmq.cuh` :17-20 的三个函数指针类型）：

1. **加载权重块（`load_tiles_mmq_t`）**：从显存读一段权重 block 到共享内存。权重保持打包形态（4bit 两两成字节），不展开。
2. **激活值预量化（`quantize_mmq_q8_1_cuda`）**：把 src1（f32 激活）量化成 `block_q8_1` 格式（8bit + 标度），按 mmq 友好的布局重排（`block_q8_1_mmq`，`mmq.cuh` :28-47）。
3. **分块乘加（`vec_dot_mmq_t`）**：在共享内存里，权重块按需反量化（4bit 解成 int8/f16），与 q8_1 激活做点积；反量化只在寄存器内完成，**绝不落地到显存**。累加器是 f32。
4. **写回（`mmq_write_back_t`）**：把 f32 累加结果写回 dst 对应位置。

为什么这样快：
- 权重始终以 4bit/8bit 形态过显存带宽，相比 f16 少读 2~4 倍字节。
- 反量化只发生在寄存器/共享内存里，延迟隐藏在算术流水线后面。
- 分块后整块塞进 Tensor Core，每个时钟周期完成 `m x n x k` 的乘加（下节展开）。

## Tensor Core / WMMA 一句话直觉

Tensor Core 是 GPU 里的专用矩阵乘法单元：一条指令完成一个**小矩阵块**的乘加（如 16x16x16），输出 f32 累加。`mma.cuh` :2-7 注释明确：它封装 PTX 的 `mma.sync` / `ldmatrix` 指令，对外暴露 `A @ B = C` 的 tile 接口（A: M x K，B: K x N，C: M x N）。

为什么量化格式能配合 int8 Tensor Core：4bit 两个值打包成一个字节，加载后 unpack 成 int8 进 Tensor Core；标度（d）作为单独的 f16 标量在最后乘上累加结果。整个流程：`int8 x int8 -> int32 累加 -> 乘标度 -> f32 输出`。这就是 mmq 在 Turing+ 卡上跑满算力的关键。AMD 走 `amd_wmma_available` / `__builtin_amdgcn_wmma_*`（`mma.cuh` :1232+），逻辑对称。

旧卡没有 Tensor Core 时，mmq 退化到 **dp4a** 指令（`mma.cuh` 内 `__dp4a`）：一条指令把两个 int32 当 4 个 int8 做点积，int32 累加。性能不如 Tensor Core，但仍比 f32 matmul 快。

## CUDA Graphs：把几百次 kernel 启动压成一次

**为什么需要**：decode 时每生成一个 token，要跑完整张计算图。一张 LLM 图通常有几百个节点（norm + matmul + rope + attention + ffn + ...），每节点一个 kernel。CPU 端 kernel 启动开销约 5~10 微秒/次，几百个就是几毫秒——decode 总耗时可能才十几毫秒，启动开销占比惊人。

**CUDA Graphs 的做法**：第一次跑图时把"启动哪些 kernel、参数是什么"录制成一个 `cudaGraph`；之后每次 decode 直接 `cudaGraphLaunch`，由 GPU 自己按录好的顺序执行，CPU 只发一次启动。

llama.cpp 的实现（`ggml-cuda.cu`）：
- `ggml_cuda_graph_get_key`（:3297）：用 `cgraph->nodes[0]` 作 key，区分不同图。
- `ggml_cuda_graph_check_compability`（:3255）：检查能否捕获。**split buffer 不支持**（:3267-3272，因为多卡 split 涉及跨设备同步）；`MUL_MAT_ID` 在非量化的某些情况也禁用（:3275-3287）。
- `ggml_cuda_graph_update_required`（:3301）：检查张量属性（地址、形状）是否变化。
- `ggml_backend_cuda_graph_compute`（:4468）：主入口。流程：
  1. 检查图兼容性。
  2. **Warmup**：连续两次调用属性都没变才允许捕获（:4488-4496）。这避免捕获到一次性初始化开销。
  3. `cudaStreamBeginCapture` 进入录制模式，正常跑一遍图，所有 kernel 调用都被录进 graph。
  4. 之后每次调用直接 `cudaGraphLaunch`（在 `ggml_cuda_graph_evaluate_and_capture`，:4237 内）。

**复用条件**：形状不变才能复用。一旦 `properties_changed` 为 true（如 batch 变了、张量地址变了），warmup 计数器归零，下次重新捕获（:4499-4503）。这就解释了：decode 阶段 n_tokens=1 恒定，图能稳定复用；prefill 阶段每次 batch 可能不同，往往捕获失败退回普通执行。

## 多 GPU 切分（按行切权重 + 事件同步）

llama.cpp 的多卡切分实现在 `ggml_cuda_op_mul_mat`（:1805），不是 allreduce 风格的"各卡算全图再求和"，而是**按行切权重**。

切分逻辑（:1898-1914）：
```cpp
// tensor_split[id] 是 0~1 的累计比例，如 {0.5, 1.0} 表示 GPU0 拿前 50%
if (split) {
    const int64_t rounding = get_row_rounding(tensor_split);  // 对齐到 mmq tile
    dev[id].row_low  = (id == 0) ? 0 : ne01 * tensor_split[id];
    dev[id].row_high = (id == last) ? ne01 : ne01 * tensor_split[id+1];
    // 减去余数对齐到 tile 边界，避免切坏 block
    dev[id].row_low  -= dev[id].row_low  % rounding;
    dev[id].row_high -= dev[id].row_high % rounding;
}
```

执行（:2074-2076）：每张卡用自己的 `row_low..row_high` 行段算 `dst` 对应行，通过 `Memcpy2DPeerAsync`（:2090）直接写到主卡 `dst` 张量的正确偏移上。

同步（:2101-2122）：用 CUDA event 而非 allreduce。每张卡算完自己的分片后 `cudaEventRecord(events[id][is], stream)`；主卡在收尾时 `cudaStreamWaitEvent(main_stream, events[id][is])` 等所有分片就绪。没有显式求和——因为各卡写的是 `dst` 的不同行，本来就是拼接关系。

ASCII 图：
```
                weight W [M, K]  (M 行按 tensor_split 切)
        +------------------------+
   GPU0 |   row 0 .. row_low1-1  |  ->  dst[0 .. row_low1-1,    :]
        +------------------------+
   GPU1 | row_low1 .. row_high1  |  ->  dst[row_low1 .. row_high1, :]
        +------------------------+
   GPU2 | row_high1 .. M-1       |  ->  dst[row_high1 .. M-1,    :]
        +------------------------+
              ^ 各卡算各的
              | cudaEventRecord(events[id])
              v
        主卡 stream: cudaStreamWaitEvent(events[0..N-1])
              |
              v
        dst 张量已完整（按行拼接，无 allreduce）
```

注意：`allreduce.cu` 在仓库里存在，但用于另一条路径（tensor parallel 的跨卡规约，例如训练或不同并行方案）。普通多卡推理走的是上面的"行切 + event 同步 + 直接写回"。

## 图融合（`ggml_cuda_can_fuse`，:3617）

CUDA 后端会做**跨算子融合**，把多个节点合成一个自定义 kernel：
- `ggml_cuda_topk_moe_fusion`（:3406）：MoE 的 top-k 选专家 + 后续操作融合
- `ggml_cuda_should_fuse_rope_set_rows`（:3372）：RoPE 与 KV 写入融合
- `ggml_cuda_should_fuse_mul_mat_vec_f` / `_q`（:2468 / :2503）：把 matvec 后续的 add/act 融到一起
- 融合减少 kernel 启动次数与中间显存，是 GPU 性能优化的主要手段

## 与其他后端对比（放在学习框架里）

- **CPU**（`ggml-cpu`）：同样的算子，用 SIMD（AVX/NEON）；量化走 `ggml-quants.c`
- **CUDA**：WMMA/Tensor Core + cuBLAS，量化感知 matmul
- **Metal/Vulkan/SYCL**：另套内核，同样赌在「量化 + 融合」上
- 调度器（`07`）负责把图分给这些后端

## 追问方向
- `mmq.cu` 里 Q4_0 的 WMMA 内核具体怎么写
- `fattn-tile.cu` / `fattn-mma-f16.cuh` 的 Flash-Attention 实现
- CUDA Graphs 捕获的触发条件（split buffer / MUL_MAT_ID 禁用细节）
