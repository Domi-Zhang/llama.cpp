# 精读：GGML 后端调度（ggml-backend.cpp）

> 日期：2026-08-13 | 核心文件：`ggml/src/ggml-backend.cpp`（调度器）+ `ggml-backend.h`（接口）
> 前置：已读 `01`，知道 `decode -> graph_compute -> ggml_backend_sched_graph_compute_async`。

## 作用

把一张大的计算图（可能横跨 CPU/GPU/Metal 等多个后端）**切分成若干子图（split）**，每个子图交给能处理它的后端执行，并在 split 之间搬运中间张量。多后端混合推理的大脑。

## 调度器三步（`ggml_backend_sched_graph_compute_async`，:1889）

```
1. ggml_backend_sched_reset()         若未重置
2. ggml_backend_sched_alloc_graph()   切分图 + 分配显存/内存
      └─ split_graph()   按后端把图切成 splits
      └─ gallocr_alloc_graph()  计算每个 split 的 buffer 布局
3. ggml_backend_sched_compute_splits()   逐个 split 执行
```

## 一、切图：`ggml_backend_sched_split_graph`（:1014）

**两趟扫描**决定每个节点跑在哪个后端：

- **pass 1**：给已有输入归属的节点分配后端；用户可 `set_tensor_backend` 显式钉住（如 `02` 里 attention 中间节点钉 CPU）
- **pass 2**：**扩展**——把 GPU 后端标记向相邻节点上下扩展，忽略最低优先级 CPU。效果：CPU 几乎不会被用到，除非权重在 CPU 上、或两 GPU 操作之间夹了 CPU 操作

划分后，相邻、同后端的节点并成一个 **split**，`sched->splits[]` 记录每个 split 的节点、输入、后端。

## 二、执行：`ggml_backend_sched_compute_splits`（:1541）

对每个 split 依序：
1. **拷贝输入**到该 split 的后端（`tensor_copy` -> 目标后端 buffer）
   - 用户输入（`GGML_TENSOR_FLAG_INPUT`）立即同步拷贝，防止用户覆盖
   - 中间张量用 event 做异步 `wait`，避免覆盖上一个 split 还在用的数据
2. **执行**：`ggml_backend_graph_compute_async(split_backend, &split->graph)`
3. 多份拷贝（`cur_copy` 轮转）实现「上一个 split 还在跑，下一个 split 的输入已就位」的流水线并行

## 三、关键工程点

### 显存分配（gallocr）
- 每个 split 用 `ggml_gallocr_alloc_graph` 做**图级复用**：同一时刻不重叠生命周期的中间张量共享一块 buffer
- 先跑一次「测量图」确定各 tensor 大小，再真正分配

### MoE 专家的按需搬运（:1576-1618）
- 当 split 的输入是权重、且首个节点是 `MUL_MAT_ID`（专家 matmul）时：**只拷贝本次用到的专家**，而不是整个专家矩阵
- 通过读 `ids` 张量，用 `ggml_bitset` 标记哪些专家被选中，省下大量跨后端搬运——大 MoE 模型混合推理的关键优化

### 后端抽象（`ggml_backend_i`，`ggml-backend-impl.h:130`）
- 每个后端实现 `graph_compute` 等接口；`ggml_backend_*` 是分发层
- 后端注册到调度器时按优先级排序，GPU 优先

## 图视角小结（串联 01/02/03）

- `01` 搭的图 → `07` 把它切成 GPU/CPU 各段 → 每段由对应后端的内核（`ggml-cpu`/`ggml-cuda`...）执行
- `02` 的 Flash-Attention 融合算子、`03` 的 MoE 稀疏，都会影响 split 的划分与搬运量
- 「图即拓扑」的精髓在此兑现：调度器靠图的依赖关系决定切分、流水线、显存复用

## 追问方向
- `ggml_gallocr` 的 buffer 复用算法（`ggml-allocator`）
- 各后端注册与优先级（`ggml_backend_register`，`ggml-backend.c`）
- CUDA graph 捕获（`ggml_backend_cuda_graph`）如何把整图变成单一 kernel