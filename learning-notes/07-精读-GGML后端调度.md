# 精读：GGML 后端调度（ggml-backend.cpp）

> 日期：2026-08-13 | 核心文件：`ggml/src/ggml-backend.cpp`（调度器）+ `ggml-backend.h`（接口）+ `ggml/src/ggml-alloc.c`（gallocr）
> 前置：已读 `00a`（张量/计算图/后端）、`01`（构图）、`02`（Attention）、`03`（MoE），知道 `decode -> graph_compute -> ggml_backend_sched_graph_compute_async`。
> 2026-08-17 修订：扩充为初学者详细版（split/显存复用图解）。

## 0. 为什么需要调度器

`01`~`03` 构出来的计算图可能有几百个节点，每个节点都有自己希望的执行设备：

- attention/FFN 的 `mul_mat` 想跑在 GPU 上（算力高、带宽高）。
- embed/lm_head 的权重表可能太大，GPU 显存装不下，只能留在 CPU 内存。
- `02` 末尾提到的 KV-cache 写回路径，非 offload 时主动钉在 CPU。
- `03` 的 MoE 在 `n_expert` 很大时，专家权重也会溢出到 CPU/SSD。

如果只有"单后端"，要么全 GPU（装不下就 OOM），要么全 CPU（慢得不可用）。调度器解决的就是**让一张图同时跨多个后端跑**，把"放不下的部分"挪到 CPU，把"算得动的部分"留在 GPU。

但跨后端是有代价的。PCIe 4.0 x16 实测单向约 25~30 GB/s，PCIe 5.0 x16 约 50~60 GB/s；而 HBM3 显存带宽 A100 是 2.0 TB/s、H100 是 3.35 TB/s，**差 50~100 倍**。一次"GPU 算 -> 拷回 CPU -> CPU 算 -> 拷回 GPU"如果中间张量是 `[4096, 1024]` 的 F16（8 MB），单次拷贝 ~0.3 ms，看似不大；但如果是 `[4096, 32768]` 的 F32（512 MB），单次 ~20 ms，已经是一个 token 几十 ms 量级的大头了。所以**搬运必须省着用**。

调度器要回答的问题清单：

1. **切在哪**：相邻节点跑在不同后端时，从切换点切开。
2. **谁执行**：每个节点选一个后端（用户钉住的优先；否则按权重位置 + 后端能力选）。
3. **中间数据怎么搬**：split 之间需要跨后端拷贝时，哪些张量要拷、用什么时机（同步/异步/event）。
4. **显存怎么省**：同一时刻不同时活跃的中间张量，能否共用一块显存。

这篇笔记就围绕这四件事展开。

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

`ggml_backend_sched_graph_compute_async`（:1889）本身极简：先 `reset`（:1892，若未重置），再 `alloc_graph`（:1896，若未分配），最后 `compute_splits`（:1901）。三步分别对应"清状态、画蓝图、动手干"。

## 一个具体例子：6 节点图切成 3 个 split

为了把"切图"讲透，构造一个最小线性图（实际 LLM 的图比这复杂得多，但切分逻辑一样）：

```
节点序号:    #0        #1       #2       #3       #4        #5
          embed  -> attn1  -> ffn1  -> attn2  -> ffn2  -> lm_head
```

假设：

- `embed` 权重 `W_emb` 在 CPU 内存（太大，GPU 装不下）。
- `attn1/ffn1/attn2/ffn2` 的权重都在 GPU，且这些算子 GPU 都支持。
- `lm_head` 权重 `W_lm` 也在 CPU（与 `W_emb` 共享 embedding 是常见做法，见 `01`）。

切分结果（3 个 split）：

```
      Split 0 (CPU)               Split 1 (GPU)              Split 2 (CPU)
  ┌─────────────────┐         ┌──────────────────────┐    ┌─────────────────┐
  │                 │         │                      │    │                 │
  │  W_emb  (CPU)   │         │  W_attn1 (GPU)       │    │  W_lm   (CPU)   │
  │     |           │         │     |                │    │     |           │
  │  embed          │         │  attn1 -> ffn1       │    │  lm_head        │
  │     |           │         │     -> attn2 -> ffn2 │    │     |           │
  │  emb_out        │ copy ──>│     |                │    │  logits         │
  │  (CPU buffer)   │ CPU->GPU│  ffn2_out (GPU buf)  │<── copy GPU->CPU    │
  │                 │         │                      │    │                 │
  └─────────────────┘         └──────────────────────┘    └─────────────────┘
   nodes: [embed]              nodes: [attn1,ffn1,         nodes: [lm_head]
                                 attn2,ffn2]
   inputs: []                   inputs: [emb_out]          inputs: [ffn2_out]
   backend: CPU                 backend: GPU               backend: CPU
```

**每个 split 的输入/输出**：

- Split 0：输入无（`W_emb` 是权重不是图输入），输出 `emb_out`。
- Split 1：输入 `emb_out`（CPU 来），输出 `ffn2_out`。
- Split 2：输入 `ffn2_out`（GPU 来），输出 `logits`。

**需要跨后端拷贝的张量**：

- `emb_out`：CPU buffer -> GPU buffer（在 Split 1 开始前拷）。
- `ffn2_out`：GPU buffer -> CPU buffer（在 Split 2 开始前拷）。

`split->inputs[]` 数组（`ggml-backend.cpp:768`，容量 `GGML_SCHED_MAX_SPLIT_INPUTS = 30`，:757）就记录这些跨后端张量。调度器还会为每个 input 在目标后端**创建一个同 layout 的拷贝张量** `input_cpy`（:1354-1369），后续 split 的节点 `src[j]` 会被改写成指向 `input_cpy`（:1370），这样 split 内部所有节点看到的都是同一后端的张量。

## 一、切图：`ggml_backend_sched_split_graph`（:1014）

**两趟扫描**决定每个节点跑在哪个后端。

### pass 1（:1035 "assign backends to ops with pre-allocated inputs"）

遍历所有 leaf 和 node，给"已经有归属线索"的节点分配后端：

- **权重决定后端**（:908-930 `1.wgt`）：如果某个 src 是 `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` 的权重，且它所在的 buffer 属于某个后端，那么这个 node 就跟着权重走。这是最常见的路径：`attn1.wq_k` 在 GPU buffer -> `attn1` 的 `mul_mat` 也归 GPU。
- **图输入**（:902-906 `1.inp`）：`GGML_TENSOR_FLAG_INPUT` 的张量（如 token ids）默认归最后一个后端（CPU，:903）。
- **view_src 跟随**（:887-893 `1.vsrc`）：view 张量永远和它的源张量同一个后端（零拷贝 view 的前提）。
- **用户钉住**：用户通过 `ggml_backend_sched_set_tensor_backend`（:1965 附近的 `usr` cause）显式指定，pass 1 不会覆盖。

### pass 2（:1072 "expand current backend assignments"）

pass 1 之后还有大量节点没分配（比如两个权重节点之间的 add/norm 算子，本身没权重）。pass 2 把已分配的后端标记**向相邻节点扩展**，分四小步：

- **expand GPU down**（:1077-1097）：从前向后扫，遇到 GPU 后端标记就向后续未分配节点扩展。
- **expand GPU up**（:1098-1118）：从后向前扫，同样扩展 GPU 标记。
- **expand rest down**（:1119-1134）：对剩下未分配的，从前向后用任意已分配后端扩展。
- **expand rest up**（:1135-1150）：从后向前再扫一遍。

**关键设计**：扩展 GPU 时会**跳过 CPU**（:1087-1092、:1108-1113 检查 `*node_backend_id == sched->n_backends - 1` 就把 `cur_backend_id` 置 -1）。也就是说，**CPU 标记不会向相邻节点扩展**。

为什么 CPU 优先级最低、不扩展？因为：

1. **CPU 永远可用**：所有算子 CPU 都能跑（`ggml-backend.cpp:1736` 在 `sched_new` 里 assert 最后一个后端必须是 CPU，作为兜底）。
2. **CPU 慢**：把能在 GPU 跑的算子误分到 CPU 是性能损失。
3. **避免无谓 split**：如果 CPU 标记也扩展，两个 GPU 段之间夹一个 CPU 节点就会把 GPU 段切成两半，多一次跨后端拷贝。

所以 CPU 几乎只在两种情况下被用到：权重在 CPU 上（pass 1 的 `1.wgt` 路径），或某算子 GPU 不支持只能靠 CPU 兜底。

后续 pass 3（:1152 升级到更高优先级后端）、pass 4（:1213 给 view_src 和 src 补全）、pass 5（:1245 真正切段、登记 inputs/copies）属于收尾。最终 `sched->splits[]` 数组里每个元素记录 `i_start`/`i_end`（节点区间）、`backend_id`、`inputs[]`、`n_inputs`。

## 二、执行：`ggml_backend_sched_compute_splits`（:1541）

对每个 split 依序：

1. **拷贝输入**到该 split 的后端（`tensor_copy` -> 目标后端 buffer，:1558）
   - 用户输入（`GGML_TENSOR_FLAG_INPUT`）立即**同步**拷贝（:1560-1567），防止用户在拷贝完成前覆盖源数据。
   - 中间张量用 **event 异步** `wait`（:1570-1574）：`ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy])`，等的是"目标后端上同一 slot 的事件"，而不是源后端。
2. **执行**（:1678）：`ggml_backend_graph_compute_async(split_backend, &split->graph)`。
3. **记录 event**（:1717-1721）：`ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend)`，标记这个 slot 上的拷贝+执行完成，下次同 slot 复用时要等它。

### 流水线：cur_copy 轮转

`sched->cur_copy` 在每次 `alloc_graph`（:1869-1870）时自增取模 `n_copies`。`n_copies` 在 `sched_new`（:1751）里设：`parallel=false` 时是 1（串行），`parallel=true` 时是 `GGML_SCHED_MAX_COPIES = 4`（:761）。events 数组 `events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES]`（:807）只在 `n_copies > 1` 时创建（:1781-1785）。

**轮转的好处**：split N 用 slot 0，split N+1 用 slot 1，split N+1 的输入拷贝**不需要等 split N 在 slot 0 上执行完**就能开始（只要源数据就绪），从而把"输入拷贝"和"上一个 split 的执行"在时间上重叠。时间线示意：

```
n_copies = 1（串行，无流水线）:
  Split 0 (CPU): [copy][====exec====][sync]
  Split 1 (GPU):                                    [copy][====exec====][sync]
  Split 2 (CPU):                                                              [copy][====exec====]
                  ^ split N+1 必须等 split N 整个 sync 完成才能开始拷贝

n_copies = 2（slot 轮转，事件同步）:
  Split 0 (CPU, slot 0): [copy][====exec====][record ev0]
  Split 1 (GPU, slot 1):            [copy][====exec====][record ev1]
                ↑ 不等 ev0（不同 slot），只等源张量就绪
  Split 2 (CPU, slot 0):                                  [wait ev0][copy][====exec====][record ev0]
                                                          ↑ 这里才等 split 0 的 ev0
```

第二条时间线里，Split 1 的 `copy`（emb_out: CPU->GPU）和 Split 0 的 `exec` 尾部可以重叠，因为 Split 1 写的是 slot 1，与 Split 0 用的 slot 0 不冲突。这是多后端流水线并行的来源。

> 注意：实际重叠程度取决于后端是否支持 `cpy_tensor_async`（:1664）。不支持时退化为同步拷贝，但 event 机制仍能避免多余的 `synchronize`。

## 三、关键工程点

### 显存分配（gallocr，`ggml-alloc.c`）

每个 split 用 `ggml_gallocr_alloc_graph`（`ggml-alloc.c:1051`）做**图级复用**：同一时刻不重叠生命周期的中间张量共享一块 buffer。核心算法在 `ggml_gallocr_alloc_graph_impl`（:717）：

- 先扫一遍图，给每个张量算 `n_children`（被几个 node 当 src）和 `n_views`（被几个 view 引用）（:731-760）。
- 顺序遍历 node：分配 src 和 node 本身（`ggml_gallocr_allocate_node`，:622），node 处理完后给每个 src 的 `n_children` 减 1，**减到 0 且没有 view 时**就 `ggml_gallocr_free_node`（:690），把它占的内存还回 free list。
- `ggml_dyn_tallocr_alloc`（:201）用 **best-fit** 在 free blocks 里找最小能放下的块；`ggml_dyn_tallocr_free_bytes`（:311）尝试与相邻空闲块合并。

**生命周期不重叠 -> 共享偏移**的小例子。设 3 个中间张量 a/b/c，大小都是 100 字节，依赖关系是 `a -> b -> c`（b 用完 a 才能释放，c 用完 b 才能释放）：

```
时刻 t1: 分配 a，offset = 0,    size = 100
时刻 t2: 分配 b，offset = 100,  size = 100   (a 还活着，b 不能占 a 的位置)
时刻 t3: b 开始用，a 的 n_children 归 0，释放 a -> free block [0, 100)
时刻 t4: 分配 c，offset = 0,    size = 100   (复用 a 释放的位置！b 在 [100, 200))
时刻 t5: c 用完，释放 c -> free block [0, 100)
时刻 t6: 释放 b -> free block [0, 200) (合并)

最终 buffer 最大占用 = 200 字节（而非 a+b+c = 300 字节）
```

偏移表：

| 张量 | 生命周期 | offset | size |
|------|---------|--------|------|
| a    | t1 ~ t3 | 0      | 100  |
| b    | t2 ~ t6 | 100    | 100  |
| c    | t4 ~ t5 | 0      | 100  |  ← 与 a 共享偏移 0

**为什么先跑"测量图"再分配**：实际推理中，第一帧（prompt 解码）的图比后续逐 token 生成的图大得多（KV cache 还在填充，节点更多）。`ggml_backend_sched_reserve`（:1847）会用 `measure_graph` 跑一遍 `split_graph` + `ggml_gallocr_reserve_n`（`ggml-alloc.c:961`），让 `ggml_dyn_tallocr` 把每个 chunk 的 `max_size` 推到最大值，再真正分配 backend buffer（:938 `ggml_vbuffer_alloc`）。这样后续每一帧都不会因为 buffer 不够而触发昂贵的重新分配（`ggml-backend.cpp:1509` 那条 fallback 路径只在节点数/形状变化时才走）。

### MoE 专家的按需搬运（:1576-1620）

当 split 的输入是权重、且首个节点是 `MUL_MAT_ID`（专家 matmul，见 `03`）时：**只拷贝本次用到的专家**，而不是整个专家矩阵。

触发条件（:1578-1583）：`input->buffer` 的 usage 是 `WEIGHTS`、且 `ggml_backend_buffer_is_host(input->buffer)`（权重在 CPU 内存，可被 GPU 直接读），split 第一个 node 的 op 是 `GGML_OP_MUL_MAT_ID` 且 `node->src[0] == input_cpy`（input 就是专家权重）。

实现（:1585-1620）：

1. 读 `node->src[2]`（即 `03` 里的 `ids` 张量，记录每个 token 选了哪些专家）。
2. 用 `ggml_bitset`（:1611-1617）标记本次 batch 用到的所有专家 id。
3. 把连续命中的专家合并成一段（:1624-1660 的 `copy_experts` lambda），用 `ggml_backend_tensor_set_async` 一次性拷过去。

**省多少**（数字例子）：64 专家、top-8、batch=32 tokens。整张专家权重 `[n_ff, n_embd, 64]`，假设 `n_ff=14336, n_embd=4096, F16`，单专家 `14336*4096*2 = 117 MB`，全部 64 个 = 7.5 GB。

- 不优化：每次 split 都要拷 7.5 GB（哪怕只用到几个专家）。
- 按需搬运：32 个 token 各选 8 个专家，去重后最多 64 个、最少 8 个；典型情况命中 ~30 个，拷 `30 * 117 MB ≈ 3.5 GB`，省 ~一半。
- 极端情况（top-1、batch=1）：只拷 1 个专家 = 117 MB，**省 64 倍**。

这是大 MoE 模型（DeepSeek-V3、LLaMA4 等）能在 GPU 显存不够装全部专家时仍可推理的关键优化。

### 后端抽象（`ggml_backend_i`，`ggml-backend-impl.h:105`，`graph_compute` 在 :130）

- 每个后端实现 `graph_compute` 等接口；`ggml_backend_*` 是分发层。
- 后端注册到调度器时按优先级排序，GPU 优先，CPU 必须在最后（`ggml-backend.cpp:1736` 的 assert）。

## 图视角小结（串联 00a/01/02/03）

- `00a` 讲的"张量驻留在 backend buffer"概念，在这里兑现：调度器靠 `tensor->buffer` 判断权重在哪，进而决定 node 在哪跑。
- `01` 搭的图 -> `07` 把它切成 GPU/CPU 各段 -> 每段由对应后端的内核（`ggml-cpu`/`ggml-cuda`...）执行。
- `02` 的 Flash-Attention 融合算子（钉 CPU 的非 offload 路径）、`03` 的 MoE 稀疏（`MUL_MAT_ID` 按需搬运），都会影响 split 的划分与搬运量。
- 「图即拓扑」的精髓在此兑现：调度器靠图的依赖关系决定切分、流水线、显存复用--无需用户手写"哪段在 GPU、哪段在 CPU"。

## 追问方向

- `ggml_gallocr` 的 buffer 复用算法细节：`ggml_dyn_tallocr` 的 best-fit + chunk 扩展（`ggml-alloc.c:201`、`:311`），多 chunk 场景（`vbuffer`，:397）何时触发。
- 各后端注册与优先级（`ggml_backend_register`，`ggml-backend-reg.cpp`）。
- CUDA graph 捕获（`ggml_backend_cuda_graph`）如何把整图变成单一 kernel launch，绕过调度器逐 split 启动开销。
- `cpy_tensor_async`（:1664）各后端实现差异：CUDA 用 dedicated copy stream，CPU 后端无此能力时退化为同步。
