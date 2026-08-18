# 精读：KV cache 管理（llama-kv-cache.cpp）

> 日期：2026-08-13 | 核心文件：`src/llama-kv-cache.cpp`（2632 行）、`src/llama-kv-cells.h`（535 行）
> 2026-08-17 修订：扩充为初学者详细版（显存计算例子等）
> 前置：已读 `01/02`，知道 attention 里 `cpy_k/cpy_v`(写) 与 `get_k/get_v`(读) 的图节点。

## 0. 为什么需要 KV cache：没有它会怎样？

Transformer 自回归生成时，每生成一个新 token，注意力都要看回**所有历史 token** 的 K 和 V。如果不缓存，就得把整段 prompt + 已生成部分再过一遍 transformer 前向，重新算一遍每层的 K、V。

用一个最小例子感受差距：已经生成了 `[t0, t1, t2]`，现在要生成 `t3`，再生成 `t4`。

```
没有 cache（每步全量重算）                有 cache（写新读旧）

step 1: 喂 [t0 t1 t2] -> t3              cache 状态: [K0 K1 K2 | V0 V1 V2]
        每层对 3 个 token 算 K/V           step 1: 只喂 t3 -> 算 K3 V3
        前向开销 O(n)                                 ↓ 写入 cache
        cache: [K0 K1 K2 K3 | V0 V1 V2 V3]
                                          attention: Q3 · [K0 K1 K2 K3] -> O(n)

step 2: 喂 [t0 t1 t2 t3] -> t4            step 2: 只喂 t4 -> 算 K4 V4
        每层对 4 个 token 算 K/V                     ↓ 写入 cache
        前向开销 O(n+1)                              cache: [K0..K4 | V0..V4]
        ...                                    attention: Q4 · [K0..K4] -> O(n+1)

生成 N 个 token 总开销 ≈ O(N^2)          生成 N 个 token 总开销 ≈ O(N)
```

差距本质：没有 cache 时**算过的 K/V 被扔掉**，下一步又得算；有 cache 时**算过的 K/V 留下**，下一步直接拿来用。核心思路就一句话：**「写新读旧」**。后面所有 cell / slot / stream / 回收的设计，都是围绕"怎么把这份 cache 在显存里管理好、复用好、回收好"展开的。

## 1. 作用

KV cache 是自回归推理的「记忆」：缓存每个历史 token 的 K/V，使生成新 token 时无需重算整段历史。
`llama_kv_cache` 类负责：**分配内存、按 slot/sequence 管理、读写张量、上下文回收**。

## 2. 显存占用：先算一笔账

KV cache 是推理时显存大头之一（另一个是权重），先弄清它的体积公式。

**每层每 token 的 K/V 字节数**（K 与 V 体积相同，合起来记）：

```
bytes_per_token_per_layer = 2 * n_head_kv * n_embd_head * sizeof(type)
                            ^    ^            ^            ^
                           K+V  KV 头数       每头维度      每元素字节
```

**总占用** = 上面这个值 × `n_layer` × `n_ctx` × `n_seq`（`n_seq` 是同时推理的 sequence 数，unified cache 下通常 1）。

实例：Qwen2.5-0.5B（`n_head_kv=2, n_embd_head=64, n_layer=24`），用 f16（2 字节）跑 `-c 4096`：

```
单层单 token：2 * 2 * 64 * 2  = 512 B
全部层单 token：512 * 24       = 12288 B ≈ 12 KB
4096 token 总共：12288 * 4096  = 50,331,648 B ≈ 48 MB
```

**为什么 GQA 省显存？** Qwen2.5-0.5B 的 `n_head=14` 但 `n_head_kv=2`（分组因子 7）。如果不用 GQA（即假设 `n_head_kv=n_head=14`），同一份计算变成：

```
单层单 token：2 * 14 * 64 * 2 = 3584 B
4096 token 总共：3584 * 24 * 4096 ≈ 336 MB
```

也就是 7 倍差距（336 MB vs 48 MB）。**KV cache 只存 `n_head_kv` 份**，Q 在 attention 时通过 repeat 拉成 `n_head` 份再算 -- 权重和 cache 都省了 7×。这就是现代 LLM 几乎都上 GQA 的原因。

> 源码里这个大小由 `size_k_bytes()` / `size_v_bytes()`（`llama-kv-cache.cpp:1806` / `:1816`）实际统计；构造函数末尾 `LLAMA_LOG_INFO(... KV buffer size ...)`（`:303`）和总览行（`:313`）会打印实际 MiB，可以对着 `llama-server` 启动日志验证上面的手算。

## 3. 核心概念：cell / slot / sequence / stream

| 概念 | 含义 | 源码对应 |
|---|---|---|
| **cell** | cache 的最小单元，一个位置放一组 K/V | `llama_kv_cells`（`llama-kv-cells.h:32`）里的一个下标 `i` |
| **slot** | 一个 sequence 一次 ubatch 占用的 cell 集合（通常连续） | `slot_info.idxs[s]`（`llama-kv-cache.h:34`） |
| **sequence** | 一条独立对话/推理流，由 `llama_seq_id` 标识 | `seq[i]`（`bitset<LLAMA_MAX_SEQ>`，`llama-kv-cells.h:489`） |
| **stream** | 可并行处理的独立缓存流（一份独立 K/V 张量） | `n_stream`，构造函数 `:98` |
| **head** | 每个 stream 当前「写入游标」位置，找空 cell 的起点 | `v_heads[s]`（`llama-kv-cache.h:267`） |

### 3.1 数据结构：`llama_kv_cells`

`llama_kv_cells` 是 cell 的**元数据层**（K/V 张量本体在 `layers[il].k / .v`，元数据只记"谁占了哪里"）。关键字段（`llama-kv-cells.h:459-499`）：

```cpp
bool                          has_shift;   // :459  pos 是否被 seq_add/seq_div 改过，待 K-shift 重算
std::set<uint32_t>            used;        // :462  哪些下标非空（O(log n) 查最小/最大）
std::vector<llama_pos>        pos;         // :464  每个下标对应的 token 位置（-1 表示空）
std::vector<llama_kv_cell_ext> ext;        // :467  2D 位置（M-RoPE 用，如 Qwen2-VL）
std::vector<llama_pos>        shift;       // :484  累积 position 平移量（滑窗/缩放用）
std::bitset<LLAMA_MAX_SEQ>    seq[];       // :489  每个下标被哪些 sequence 占用（一 cell 可多 seq 共享）
std::map<llama_pos, int>      seq_pos[LLAMA_MAX_SEQ]; // :499 反向索引：seq -> 它在 cache 里的所有 pos
```

要点：
- `pos[i] == -1` 表示空 cell；`used` 集合与 `pos` 保持同步。
- **一个 cell 可以被多个 sequence 同时共享**（`seq[i]` 是 bitset）。典型场景：两条 sequence 共享同一段 prefix prompt，那段 prefix 只存一份 K/V。
- `seq_pos[s]`（`:499`）是反向索引，给 `seq_pos_min/max`（`:334` / `:349`）用，O(log n) 查某 sequence 在 cache 里的最小/最大位置 -- `find_slot` 判 SWA 淘汰时要用。

### 3.2 n_stream 为什么 unified 时是 1

构造函数初始化列表里这一行（`llama-kv-cache.cpp:98`）：

```cpp
n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max),
```

加上紧跟的断言（`:149`）：

```cpp
GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);
```

`unified = true` 时把所有 sequence 塞进**同一份** K/V 张量（`n_stream = 1`），靠 `seq` bitset 区分谁占了哪个 cell -- 这是常规推理路径（`llama-cli`/`llama-server` 默认）。`unified = false` 时每个 sequence 一份独立张量（`n_stream = n_seq_max`），用于真正并行的多序列推理（如 beam search 的并行分支）。两种模式二选一，所以有上面的断言。K/V 张量按 `[n_embd_gqa, kv_size, n_stream]` 三维分配（`:245`/`:246`），unified 时第三维就是 1。

### 3.3 串起来：两条序列，其中一条被回收

假设 `n_stream = 1, kv_size = 8`，两条 sequence `s0, s1` 同时推理。一开始 s0 写满 0~4，s1 复用前 2 个 prefix cell（共享 prompt）后写到 5~7：

```
下标:    0       1       2      3      4      5      6      7
pos:     0       1       2      3      4      5      6      7
seq[i]: {s0,s1} {s0,s1}  {s0}   {s0}   {s0}   {s1}   {s1}   {s1}
         └──共享 prefix──┘     └─s0 独占─┘    └──s1 独占──┘
```

现在 s0 上下文超限要滑窗，删掉 `s0` 的 pos [0, 2)：

- `seq_rm(s0, p0=0, p1=2)`（`:392`）：对 cell 0、1 调 `cells.seq_rm(i, s0)`（`llama-kv-cells.h:238`）。
- cell 0、1 同时被 s0 和 s1 占用，`seq_rm` 只清掉 s0 那一位，bitset 仍非空 -> cell **不**变空，K/V 张量里那两行**保留**（s1 还要用）。
- 如果之后 s1 也删了同一区间，bitset 才清零、cell 才真正回收进 `used` 集合外。

这就是「共享 prefix 省 cache」的实现本质：靠 bitset 多 sequence 标记 + 引用计数式的回收。

## 4. 一次推理的 cache 生命周期

```
初始化   llama_kv_cache(...) :80   按 layer 创建 K/V 张量，分配 buffer
  │                                 (per-buft 建 ggml ctx，支持跨后端放置)
分配     find_slot(ubatch) :907    找个可用的 slot（空闲/可复用 cell）
  │
写入     apply_ubatch :1106         本次 batch 的 token 落到对应 cells
  │         └─ 构图里 cpy_k/cpy_v :1291/1326  -> ggml_set_rows 写进 cache 张量
  │
读取     (attention) get_k/get_v :1239/1259  -> ggml_view_4d 取整段历史 K/V 视图
  │
回收     update() :826 / seq_* 系列   上下文超限时回收最旧的 cell
```

## 5. find_slot：贪心找连续空闲 cell

`find_slot`（`:907`）的核心目标：为本次 ubatch 的 `n_tokens` 个 token，在每个用到的 stream 里找 **连续的** `n_tokens` 个可用 cell。简化伪代码：

```
head_cur = v_heads[stream]                       # 上次写完后的游标
if head_cur > used + 2*n_tokens:                 # 前面空太多 -> 从头开始找（:1014）
    head_cur = 0

n_test = cont ? n_tokens : 1                     # cont=true 需要连续 n_tokens 个
while n_tested < cells.size():
    for i in [0, n_test):
        idx = head_cur + i
        can_use = cells.is_empty(idx)            # 空的最优先
        if not can_use and cells.seq_count(idx) == 1:
            # 只被一个 seq 占 + 该 cell 已被 SWA 屏蔽 -> 可覆盖
            if is_masked_swa(pos_cell, seq_pos_max(seq_id_cell) + 1):
                can_use = true
        if can_use: idxs.push(idx) else break    # 不连续就跳出重来
    if idxs.size() == n_tokens: break
    idxs.clear(); head_cur += n_test             # 往后挪再试
```

**为什么必须连续？** 后面 `cpy_k` 用 `ggml_set_rows` 一次性把 `n_tokens` 行写进 cache 张量，`k_idxs` 是长度为 `n_tokens` 的一维数组（`:1385`），写入位置由数组下标顺序决定 -- 如果 cell 不连续，k_idxs 就得是不规则数组，而下游 attention 取 K 时也是按 `[0..n_kv)` 连续取（见下节 view_4d）。连续布局让「写」和「读」都能用最简单的 stride 算地址，避免 scatter/gather。`slot_info::is_contiguous()`（`llama-kv-cache.h:77`）就是用来快速判断是否连续的。

**SWA 复用**：上面 `is_masked_swa` 那条分支是滑窗注意力的优化 -- 某 cell 的 pos 已超出当前序列的 SWA 窗口，对 attention 不可见，那它的物理位置可以直接被新 token 覆盖，不必先 `seq_rm`。

## 6. 关键读/写路径（对齐 `02` 的 attention 备忘录）

### 6.1 写：`cpy_k` / `cpy_v`（`:1291` / `:1326`）

```cpp
// cpy_k 简化版（:1291）
k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, ...)  // 当前 token 的 K
return ggml_set_rows(ctx, k, k_cur, k_idxs);                  // 把 k_cur 每行写到 k 的 k_idxs[i] 行
```

- **为什么写 cache 也是图节点？** `ggml_set_rows` 是 ggml op，和 `ggml_mul_mat` 一样是计算图里的一个节点。这意味着它由后端 scheduler 调度执行（可在 GPU 上跑）、可参与复用图、可与前后 op 融合。把"写 cache"建模成图节点而不是 host 端 memcpy，是为了让 cache 写入和 attention 计算在同一张图里、由同一调度器安排，避免 host-device 来回拷。
- **`k_idxs` 怎么来？** 由 `build_input_k_idxs`（`:1382`）在构图阶段创建为一个 `ggml_set_input` 的 I64 一维张量（占位），真正填值是在 `set_input_k_idxs`（`:1449`）里 -- 那里把 `sinfo.idxs[s]`（`find_slot` 找到的 cell 下标）加上 stream 偏移 `s*kv_size` 写进去。也就是说，**下标是 host 端填的，但写入动作是 device 端做的**。

### 6.2 读：`get_k` / `get_v`（`:1239` / `:1259`）

```cpp
return ggml_view_4d(ctx, k,
    /* ne0 */ n_embd_head_k,                         // 每头维度
    /* ne1 */ n_head_kv,                             // KV 头数
    /* ne2 */ n_kv,                                  // 取多少个 token 位置
    /* ne3 */ ns,                                    // 多少个 stream (s1-s0+1)
    /* nb1 */ row_size(type, n_embd_head_k),         // 一个 head 的 stride
    /* nb2 */ row_size(type, n_embd_k_gqa),          // 一个 token 的 stride (跨所有 KV head)
    /* nb3 */ row_size(type, n_embd_k_gqa * kv_size),// 一个 stream 的 stride
    /* off  */ ... * sinfo.s0);                      // 起始 stream 偏移
```

四维含义：

| 维 | 大小 | 含义 |
|---|---|---|
| ne0 | `n_embd_head_k` | 单个 head 的特征维（Qwen2.5-0.5B 是 64） |
| ne1 | `n_head_kv` | KV 头数（GQA 下远小于 `n_head`，如 2） |
| ne2 | `n_kv` | 取的 token 数（padded 到 `max(n_pad, 256)` 的倍数，`get_n_kv` `:1223`） |
| ne3 | `ns = s1 - s0 + 1` | 涉及的 stream 数（unified 下通常 1） |

**为什么零拷贝可行？** `ggml_view_4d` 只构造一个新张量对象，`data` 指针指向原 `k` 张量的某段偏移，`nb[1..3]` 给出跨 head / 跨 token / 跨 stream 的 stride -- 不需要拷数据。之所以能这样，是因为 K/V 张量从构造（`:245` / `:246`）时就按 `[n_embd_gqa, kv_size, n_stream]` 三维分配，head / token / stream 在内存里是规则 stride 的，任何子集都能用 stride 描述。

V 的逻辑分两套（`:1259`）：`v_trans == false`（FA 路径）走和 K 类似的 view_4d；`v_trans == true`（非 FA，V 在内存里转置存放）维度顺序变成 `[n_kv, n_head_kv, n_embd_head_v, ns]`，对应 `v_idxs` 也要按转置布局算偏移（`set_input_v_idxs` `:1480` 起的双重循环就是干这个的）。

## 7. 上下文回收：`seq_*` 系列

当上下文超限或要管理多 sequence，`update()`（`:826`）协调下列操作腾出空间：

| 函数 | 行号 | 作用 | 典型使用场景 |
|---|---|---|---|
| `seq_rm`   | `:392` | 删除某 sequence 在 `[p0, p1)` 位置的 cell | 上下文滑窗：删最旧一段再 `seq_add` 平移 |
| `seq_cp`   | `:460` | 把 src sequence 的 cell 标记也给 dst（同 stream 仅改 bitset；跨 stream 才真拷数据，`:515` 起排队到 `update` 里执行） | beam search 分叉、并行采样复制 prefix |
| `seq_keep` | `:552` | 只保留指定 sequence，其它全清 | beam search 收敛后只留最优那条 |
| `seq_add`  | `:579` | 把某 sequence 所有 cell 的 pos 加一个 shift | 滑窗后把剩余 token 平移回 0 附近 |
| `seq_div`  | `:629` | 把 pos 除以 d | 某些长上下文外推策略（如 POS 缩放） |

注意 `seq_add` / `seq_div` 只改 `pos[]` 元数据，**K/V 张量本体不动** -- 但 RoPE 是位置相关的，pos 变了 K 也得跟着重算，所以会置 `has_shift = true`（`llama-kv-cells.h:422` / `:455`），`update()` 检测到后调用 `build_graph_shift`（`:881`）跑一遍 K-shift 重算所有受影响 cell 的 K。

> 注意：这里管理的是「缓存**布局**」，真正的「token 是否要重算」还取决于 `llama-kv-cache-dsa.cpp` / cparams 的上下文策略。

## 8. LLaMA 解码器与 cache 的衔接

回顾 `01/02`：每层 attention 里
- `cpy_k/cpy_v` 把**新** token 的 K/V 写入
- `get_k/get_v` 取**全部**历史

这个「写新读旧」就是 KV cache 让自回归每 token 生成开销近似 O(1)（而非 O(n) 重算）的根本原因。

## 9. 追问方向
- `find_slot` 的复用启发式（如何选最合适被顶掉的 cell）-- SWA 复用之外还有没有更细的策略？
- 上下文回收策略如何与 server 的 `SLOT_STATE` 状态机联动
- DSA cache（`llama-kv-cache-dsa.cpp`）与普通 cache 的差异
- 跨 stream `seq_cp` 排队到 `update` 执行的设计：为什么不直接在 `seq_cp` 里同步拷？
