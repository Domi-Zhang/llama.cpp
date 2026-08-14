# 精读：CUDA 后端内核（ggml-cuda）

> 日期：2026-08-13 | 核心文件：`ggml/src/ggml-cuda/`
> 前置：已读 `07`（后端调度）。CUDA 是 llama.cpp 最重要的 GPU 后端。

## 布局：一个算子一个文件

`ggml-cuda/` 里每个算子一个 `.cu` + 同名 `.cuh`，按算子命名：
- `mmq.cu`：量化权重 matmul（**主流路径**）
- `mmf.cu`：f16 权重 matmul（cuBLAS 或自定义）
- `mmvq.cu` / `mmvf.cu`：matvec（向量 x 权重）变体
- `mmid.cu`：MoE 专家 matmul（`MUL_MAT_ID`）
- `fattn*.cu`：Flash-Attention 系
- `norm.cu` / `getrows.cu` / `cpy.cu` / `dequantize.cuh` ...

统一由 `ggml-cuda.cu` 里的 `ggml_cuda_compute_forward`（:2791）按 `GGML_OP_*` 分发。

## 后端接口（`ggml_backend_cuda_interface`，:4796）

CUDA 实现 `ggml_backend_i`，其中：
```
.graph_compute = ggml_backend_cuda_graph_compute   (:4468)  执行整图
```
- 走 `ggml_backend_cuda_graph_compute`：先做 graph 检查/融合（`ggml_cuda_graph_check_compability`、`ggml_cuda_can_fuse`），再逐个节点调 `ggml_cuda_compute_forward`
- 支持 **CUDA Graphs**：把整张图捕获成一个可复用的 kernel 序列，显著降低启动开销

## 核心：matmul 的四种策略（`ggml_cuda_mul_mat`，:2541）

`mul_mat` 是 FFN/attention 的主算子，按「权重类型 + src1 形状 + 硬件」选策略：

| 策略 | 条件 | 实现 | 什么时候用 |
|---|---|---|---|
| `use_mul_mat_vec_f` | 权重 f32/f16/bf16，一次一行 | mmvf | 单 token 生成（n=1）|
| `use_mul_mat_f` | 权重非量化，src1 是 f32 | mmf / cuBLAS | 批量但权重未量化 |
| `use_mul_mat_vec_q` | 权重量化，src1 行数<上限 | mmvq | 单 token + 量化权重 |
| `use_mul_mat_q` | 权重量化 | **mmq**（`mmq.cu`） | **批量推理主路径** |

关键判断：`use_mul_mat_q = ggml_is_quantized(src0) && ...`，即只要权重是量化的就走 mmq——所以**绝大多数已量化模型走 mmq**。

### mmq（量化 matmul）为什么快
- 权重以 block 形式（`06` 的 Q4_0 等）直接驻留显存，**不解量化到 f32**
- 在 CUDA 里用 WMMA/Tensor Core 做「量化感知」的乘加：低精度乘法 + f32 累加
- 省下解量化带宽 + 利用硬件低精度算力，是 GPU 上量化推理的性能核心

### 多 GPU 切分（`split`）
`src0` 若在 split buffer 上，按 `tensor_split` 比例把权重行分到多卡，每卡算自己的分片再合并——对应 `07` 的图切分在 GPU 内的落地。

## 图融合（`ggml_cuda_can_fuse`，:3617）

CUDA 后端会做**跨算子融合**，把多个节点合成一个自定义 kernel：
- `ggml_cuda_topk_moe_fusion`（:3406）：MoE 的 top-k 选专家 + 后续操作融合
- `ggml_cuda_should_fuse_rope_set_rows`（:3372）：RoPE 与 KV 写入融合
- 融合减少 kernel 启动次数与中间显存，是 GPU 性能优化的主要手段

## 与其他后端对比（放在学习框架里）

- **CPU**（`ggml-cpu`）：同样的算子，用 SIMD（AVX/NEON）；量化走 `ggml-quants.c`
- **CUDA**：WMMA/Tensor Core + cuBLAS，量化感知 matmul
- **Metal/Vulkan/SYCL**：另套内核，同样赌在「量化 + 融合」上
- 调度器（`07`）负责把图分给这些后端

## 追问方向
- `mmq.cu` 里 Q4_0 的 WMMA 内核具体怎么写
- `fattn-tile.cu` / `fattn-mma-f16.cuh` 的 Flash-Attention 实现
- CUDA Graphs 捕获的触发条件