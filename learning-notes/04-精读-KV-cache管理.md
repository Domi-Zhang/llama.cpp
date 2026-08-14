# 精读：KV cache 管理（llama-kv-cache.cpp）

> 日期：2026-08-13 | 核心文件：`src/llama-kv-cache.cpp`（2632 行）
> 前置：已读 `01/02`，知道 attention 里 `cpy_k/cpy_v`(写) 与 `get_k/get_v`(读) 的图节点。

## 作用

KV cache 是自回归推理的「记忆」：缓存每个历史 token 的 K/V，使生成新 token 时无需重算整段历史。
`llama_kv_cache` 类负责：**分配内存、按 slot/sequence 管理、读写张量、上下文回收**。

## 核心概念

| 概念 | 含义 |
|---|---|
| **cell** | cache 的最小单元，一个位置放一组 K/V |
| **slot** | 一个 sequence 占用的连续 cell 区间 |
| **stream** | 可并行处理 sequence 的独立缓存流（`n_stream = unified ? 1 : n_seq_max`） |
| **head** | 每个 stream 当前「写入游标」位置 |

## 一次推理的 cache 生命周期

```
初始化   llama_kv_cache(...) :80   按 layer 创建 K/V 张量，分配 buffer
  │                                 (per-buft 建 ggml ctx，支持跨后端放置)
分配     find_slot(ubatch) :907    找个可用的 slot（空闲/可复用 cell）
  │
写入     apply_ubatch :1106         本次 batch 的 token 落到对应 cells
  │         └─ 构图里 cpy_k/cpy_v :1291/1326  → ggml_set_rows 写进 cache 张量
  │
读取     (attention) get_k/get_v :1239/1259  → ggml_view_4d 取整段历史 K/V 视图
  │
回收     update() :826 / seq_* 系列   上下文超限时回收最旧的 cell
```

## 关键读/写路径（对齐 `02` 的 attention 备忘录）

### 读：`get_k/get_v`（:1239/:1259）
- 返回 `ggml_view_4d`，**零拷贝视图**，不复制数据
- 视图形状：`[embd_head_k, n_head_kv, n_kv, ns]`，即「取该层该 stream 从 s0 起的 n_kv 个位置的全部头」
- `v_trans` 时 V 的布局翻转（针对量化/FA 优化），`get_v` 分两套 view 逻辑

### 写：`cpy_k/cpy_v`（:1291/:1326）
- `ggml_set_rows(cache_k, k_cur, k_idxs)`：把当前 token 的 K 按 `k_idxs` 位置写进 cache 张量
- `k_idxs` 由 `build_input_k_idxs`（:1382）预构建，构图层面指定「写到哪个 cell」
- 多 stream 时先 reshape 合并 buffer 再写（idx 全局）

## 上下文回收（关键工程点）

当上下文超限，`update()` 协调 `seq_*` 操作腾出空间：
- `seq_rm`（:392）：删除某 sequence 一段位置
- `seq_cp`（:460）：复制 sequence（如 beam/并行分支）
- `seq_keep`/`seq_add`/`seq_div`：保留/平移/缩放 position
- `find_slot`（:907）+ `update`（:826）配合决定复用哪个 slot

> 注意：这里管理的是「缓存**布局**」，真正的「token 是否要重算」还取决于 `llama-kv-cache-dsa.cpp` / cparams 的上下文策略。

## LLaMA 解码器与 cache 的衔接

回顾 `01/02`：每层 attention 里
- `cpy_k/cpy_v` 把**新** token 的 K/V 写入
- `get_k/get_v` 取**全部**历史

这个「写新读旧」就是 KV cache 让自回归 O(1) 每 token 生成（而非 O(n) 重算）的根本原因。

## 追问方向
- `find_slot` 的复用启发式（如何选最合适被顶掉的 cell）
- 上下文回收策略如何与 server 的 `SLOT_STATE` 状态机联动
- DSA cache（`llama-kv-cache-dsa.cpp`）与普通 cache 的差异