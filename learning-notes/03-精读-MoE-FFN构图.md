# 精读：MoE-FFN 构图（build_moe_ffn，路由器 + top-k）

> 日期：2026-08-13 | 核心文件：`src/llama-graph.cpp:1454` `build_moe_ffn`
> 前置：已读 `01`，知道 MoE 分支在 `build_ffn` 的 else 分支触发。
> 触发条件：`layer.ffn_gate_inp != nullptr`（存在路由器权重）。
> 2026-08-17 修订：扩充为初学者详细版（路由数值例子等）。

## 为什么需要 MoE：稀疏激活 vs 稠密

普通 Dense FFN 里，每个 token 都要过同一套 `gate/up/down` 权重，参数量与每 token 计算量是绑在一起的：模型想变「聪明」就得加宽 `n_ff`，计算量立刻同比例上升。

MoE（Mixture of Experts，混合专家）把这件事拆开：
- **权重显存**：由 `n_expert`（专家总数）决定，可以很大。
- **每 token 计算量**：只由 `n_expert_used`（每 token 实际激活的专家数）决定，可以远小于 `n_expert`。

这就实现了「参数量与计算量解耦」。

**数字对比（n_embd=4096, n_ff=14336, SwiGLU 三矩阵）**：

| 配置 | 权重参数量（单层 FFN） | 每 token 浮点乘加（约） |
| --- | --- | --- |
| Dense FFN | 3 × 4096 × 14336 ≈ 176M | 2 × 3 × 4096 × 14336 ≈ 352M |
| MoE: 8 专家 top-2 | 3 × 8 × 4096 × 14336 ≈ 1.41G | 2 × 3 × 4096 × 14336 × 2 ≈ 704M |

也就是说：相比把 Dense FFN 加宽到 ~57k（参数量同样 1.41G），MoE 用 8 个 14k 的「窄」专家替换，但每 token 只算 2 份 FFN，计算量约为加宽方案的 1/4。这是 DeepSeek/Mixtral/LLaMA4 这类模型能在单卡显存装得下、又跑得动的关键。

代价是：所有专家权重都得常驻显存（或按需搬运，见 07 篇），且引入路由选择这一额外步骤。

## 作用

稀疏混合专家前馈：每个 token 只激活少量「专家」，大幅降低 FFN 计算量。
MoE FFN 所有权重形状多一维 `[.., n_expert]`，即「每个专家一套 FFN 权重」。

## 一个 token 走完 MoE 的全过程：手算例子

为了把抽象的「路由 -> top-k -> 加权合并」讲透，用一个 4 专家、top-2 的小例子手算。
假设某个 token 在路由器打分后得到 logits = `[0.1, 2.0, 1.5, -0.5]`（4 个专家）。

**第 1 步：softmax 把 logits 变概率**

`exp([0.1, 2.0, 1.5, -0.5]) = [1.1052, 7.3891, 4.4817, 0.6065]`，求和 = 13.5825。
`probs = [0.0814, 0.5441, 0.3300, 0.0447]`。

**第 2 步：top-2 选专家**

按概率从大到小排序，选最大两个：专家 1（0.5441）和专家 2（0.3300）。
这就是 `ggml_argsort_top_k` 在做的事：返回索引 `[1, 2]`。

**第 3 步：取权重**

`weights = [0.5441, 0.3300]`（直接从 probs 用 `ggml_get_rows` 抠出来）。

**第 4 步（可选）：归一化**

如果该模型开启了 `norm_w`（如 DeepSeek 部分配置），把 top-2 权重除以它们的和：
`sum = 0.8741`，归一后 `weights = [0.6225, 0.3775]`，让两个专家权重加起来 = 1。
Mixtral 等模型不归一，直接用原 softmax 概率。

**第 5 步：专家前向 + 加权合并**

设专家 1 的 FFN 输出向量 `e1 = [1.0, 0.0]`，专家 2 的输出 `e2 = [0.0, 1.0]`（示意）。
若不归一：`out = 0.5441·e1 + 0.3300·e2 = [0.5441, 0.3300]`。
若归一：`out = 0.6225·e1 + 0.3775·e2 = [0.6225, 0.3775]`。

实际代码里，第 5 步在图上是由 `ggml_mul_mat_id`（按行索引批量矩阵乘）+ `ggml_view_2d`+`ggml_add` 完成的，下面单独讲。

## 数据流（`ffn_moe_*` 命名对应图节点）

```
cur (归一化后的输入, [n_embd, n_tokens])
  │
  ├─ ① 路由器打分  logits = gate_inp·cur        -> [n_expert, n_tokens]
  ├─ ② 门控激活    probs = softmax/sigmoid(logits)
  ├─ ③ 选专家      selected = argsort_top_k(probs, n_expert_used)
  │                   -> [n_expert_used, n_tokens]
  ├─ ④ 取权重      weights = get_rows(probs, selected)
  │                   -> [1, n_expert_used, n_tokens]
  ├─ ⑤ 专家前向    (每个选中专家算 up·SiLU(gate·x)·down)
  └─ ⑥ 加权合并    out = Σ weights[i] * expert_out[i]
```

## 逐段说明

### ① 路由器打分（`llama-graph.cpp:1527`）

`logits = build_lora_mm(gate_inp, cur)`，输出 `[n_expert, n_tokens]`，即每个 token 对每个专家的偏好分。
`gate_inp` 是个 `[n_embd, n_expert]` 的小矩阵，跟普通线性层一样，只是输出维度 = 专家数。
支持 `probs_in` 直接注入（跨层/预计算时复用），以及 `gate_inp_b` 给 logits 加偏置（`llama-graph.cpp:1533`）。

### ② 门控激活（`switch (gating_op)`，`llama-graph.cpp:1539-1554`）

- `SOFTMAX`（`LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`）：对全部专家做 softmax，所有概率和为 1。Mixtral/DeepSeek-MoE/V3 默认。
- `SIGMOID`（`LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`）：逐专家 sigmoid，互不影响，不再归一。LLaMA4 用。
- `SOFTMAX_WEIGHT`（`LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT`）：保留 logits 当 probs，把 softmax 推迟到选出 top-k 之后再做（在 `llama-graph.cpp:1619-1624`）。GrooveMoE 等用。

注意 `selection_probs` 与 `probs` 是两个不同的量（`llama-graph.cpp:1559`）：
- `probs` 用于**取权重**（line 1615 `get_rows`）。
- `selection_probs` 用于**选 top-k**（line 1602 `argsort_top_k`）。
默认它们相等；DeepSeek-V3 用 `exp_probs_b` 给 selection 加偏置但不污染权重（`llama-graph.cpp:1560-1563`）；LLaMA4 把 selection 直接换成 raw logits（`llama-graph.cpp:1567-1569`）；GrooveMoE 把 selection 换成 sigmoid（`llama-graph.cpp:1571-1574`）。

### ③ 选取 top-k 专家（`llama-graph.cpp:1602`）

`ggml_argsort_top_k(selection_probs, n_expert_used)` 选出每个 token 得分最高的 `n_expert_used` 个专家。
返回 `[n_expert_used, n_tokens]` 的 I32 索引张量，每列就是某 token 选中的专家编号。

### ③' 专家分组路由（DeepSeek-V3 风格，`llama-graph.cpp:1578-1599`）

当 `hparams.n_expert_groups > 1` 时，先按组粗筛再组内细选：
1. `n_exp_per_group = n_expert / n_expert_groups`，把 `selection_probs` reshape 成 `[n_exp_per_group, n_expert_groups, n_tokens]`。
2. 对每个组取组内 top-2（`ggml_argsort_top_k(..., 2)`），加起来当「组得分」。这一步是「组内最强专家的代表分」。
3. 再对组得分取 top `n_group_used` 个组（`ggml_argsort_top_k`）。
4. 把没选中的组用 `-INFINITY` 掩掉（`ggml_set_rows` + `ggml_fill(..., -INFINITY)`），reshape 回 `[n_expert, n_tokens]`，再交给第 ③ 步的 `argsort_top_k` 选最终 top-k。

**小例子**：8 专家、2 组每组 4 个、选 top-2 组、组内 top-2 专家。一个 token 对 8 个专家的得分为
`[0.5, 0.3, 0.1, 0.0 | 0.9, 0.8, 0.2, 0.1]`（竖线分两组）。
- 组 0 top-2 = `[0.5, 0.3]`，代表分 = 0.8。
- 组 1 top-2 = `[0.9, 0.8]`，代表分 = 1.7。
- 选 top-1 组（`n_group_used=1`） -> 选组 1。
- 组 0 全部掩成 `-INFINITY`，最终 top-2 专家只能在组 1 内选 -> 专家 4 和 5。

**动机**：纯 top-k 容易让大量 token 挤进少数「热门」专家，其他专家长期闲置，既浪费参数也破坏负载均衡。先分组、再限制每组最多贡献 `n_group_used` 个，强制专家使用更均匀。DeepSeek-V3 论文管这叫 `grouped-limited routing`。

### ④ 专家权重（`llama-graph.cpp:1615-1644`）

`ggml_get_rows(probs, selected_experts)` 取出选中专家的概率当权重，shape `[1, n_expert_used, n_tokens]`。

可选的后处理（按模型开关）：
- `SOFTMAX_WEIGHT`（`llama-graph.cpp:1619-1624`）：把 top-k 权重在 `n_expert_used` 维上再做一次 softmax。
- `norm_w`（`llama-graph.cpp:1626-1640`）：对权重按 token 归一（除以 sum，`clamp` 下限 `6.103515625e-5` 防 F16 下溢除零）。
- `w_scale`（`llama-graph.cpp:1641-1644`）：整体缩放，例如 Qwen3-MoE 等模型在权重上乘一个 scale。

`weight_before_ffn`（仅 LLaMA4，`llama-graph.cpp:1522, 1651-1656`）：把上面算出的权重在 FFN **之前**乘到输入 `cur` 上（`ggml_repeat_4d` 把 `[n_embd, 1, n_tokens]` 广播成 `[n_embd, n_expert_used, n_tokens]` 再 `mul`），而不是按常规在 FFN 之后乘到 `experts` 输出上。这样每个专家看到的是已经按权重缩放过的输入；好处是下游 `mul_mat_id` 一次乘就能把权重折进结果里，省一次 elementwise mul。

### ⑤ 专家前向 + ⑥ 加权合并（`llama-graph.cpp:1658-1829`）

每个选中专家跑一版「up -> SiLU(gate) -> down」，再按权重加总。关键在于这步是用 `ggml_mul_mat_id` 一个图节点完成「每 token 过自己那 k 个专家」的批量矩阵乘，而不是写 k 个 `for` 循环。

- 若 `gate_up_exps` 非空（`llama-graph.cpp:1661-1679`）：gate 和 up 权重合并存储，先做一次 `mul_mat_id` 得到 `[n_ff*2, n_expert_used, n_tokens]`，再用两个 `ggml_view_3d` 切成 gate 与 up 两半。这样可以省一次 `mul_mat_id` 调度。
- 否则（`llama-graph.cpp:1680-1709`）：gate、up 各一次 `mul_mat_id`，输出都是 `[n_ff, n_expert_used, n_tokens]`。
- 激活（`llama-graph.cpp:1713-1778`）：`LLM_FFN_SILU` 走 `ggml_swiglu_split`，`LLM_FFN_GELU` 走 `ggml_geglu_split`，等等；Step35 模型还会对 routed expert 做 per-layer clamp。
- down（`llama-graph.cpp:1780`）：再一次 `mul_mat_id` 把 `[n_ff, n_expert_used, n_tokens]` 投影回 `[n_embd, n_expert_used, n_tokens]`。
- 加权（`llama-graph.cpp:1792-1795`）：非 LLaMA4 情况下 `experts = mul(experts, weights)`，把每个专家的输出按权重缩放。
- 合并（`llama-graph.cpp:1799-1820`）：对 `n_expert_used` 维切 `n_expert_used` 个 `ggml_view_2d`，依次 `ggml_add` 累加成 `[n_embd, n_tokens]`。这里循环上限用 `hparams.n_expert_used` 而非入参 `n_expert_used`，是为了 warmup 时不至于生成过多 add 节点（注释里提到 PR #14753）。

## 关键算子深入：`ggml_argsort_top_k` 与 `ggml_mul_mat_id`

### `ggml_argsort_top_k`（`ggml/src/ggml.c:5307-5321`）

签名：

```c
struct ggml_tensor * ggml_argsort_top_k(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   k);
```

语义：先对 `a` 沿 dim0 做降序 argsort（调用 `ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC)`，`ggml.c:5289-5303`），返回与 `a` 同形状的 I32 索引张量；再用 `ggml_view_4d` 取前 k 个索引，结果 shape 为 `[k, a->ne[1], a->ne[2], a->ne[3]]`，dtype `I32`。注意返回的是**索引**（专家编号），不是值。断言 `a->ne[0] >= k`。

简化伪代码：

```
argsort_top_k(a, k):
    idx = argsort_desc(a)        # I32, shape == a.shape, 每列是该维排序后的下标
    return idx[:k, ...]          # view 取前 k 行
```

### `ggml_mul_mat_id`（`ggml/src/ggml.c:3290-3314`）

签名和形状约定（直接抄源码注释）：

```c
/*
    c = ggml_mul_mat_id(ctx, as, b, ids);

    as  -> [cols, rows, n_expert]        # 一组专家权重，每片是一个 [cols, rows] 矩阵
    b   -> [cols, n_expert_used, n_tokens]  # 输入：每个 token 在 k 个槽位上的向量
    ids -> [n_expert_used, n_tokens] (i32)  # 每个 token 的 k 个槽位分别用哪个专家
    c   -> [rows, n_expert_used, n_tokens]

    c ~= as[:,:,i] @ b[:,i%r,t], i = ids[e,t] for all e,t in ids
*/
```

关键点：
- `as` 是「专家权重栈」，第三维 `n_expert` 是全部专家，未选中的专家权重不参与计算。
- `b` 是被广播过的输入：每个 token 复制 k 份（`n_expert_used` 维），准备让 k 个专家各算一次。
- `ids` 是上一步 `argsort_top_k` 的输出：`ids[e, t]` = token t 选中的第 e 个专家编号。
- 输出 `c[r, e, t] = Σ_c as[c, r, ids[e,t]] * b[c, e, t]`，即 token t 在槽位 e 上由 `ids[e,t]` 号专家做了一次普通 matmul。

简化伪代码：

```
mul_mat_id(as, b, ids):           # as:[cols,rows,n_expert] b:[cols,k,n_tok] ids:[k,n_tok]
    c = zeros([rows, k, n_tok])
    for t in range(n_tok):
        for e in range(k):
            i = ids[e, t]                  # token t 的第 e 个槽用专家 i
            c[:, e, t] = as[:, :, i] @ b[:, e, t]   # 一次普通矩阵-向量乘
    return c
```

这把「每个 token 只过自己的专家」用一个图节点表达，避免在图里写 `for e, for t` 的两层 Python/C++ 循环；后端（CPU/CUDA/Metal）各自用一次 grouped GEMM 高效实现。

## 设计动机澄清

**为什么 sigmoid 门控不归一？**
Softmax 把所有专家概率挤成「和为 1」，意味着专家之间是零和博弈：A 强 B 就弱。Sigmoid 让每个专家独立开闭，A 强 B 也可以强，权重之和不是 1 也没关系。模型可以学到「这两个专家对当前 token 都重要，都给大权重」。代价是不同 token 的总权重量级不一致，所以下游常配合 `norm_w` 或 `w_scale` 重新校准。

**SOFTMAX vs SOFTMAX_WEIGHT 的区别？**
- `SOFTMAX`：在全部 `n_expert` 上做 softmax，未选中的专家也参与了归一化分母，权重天然偏小（因为分母大）。
- `SOFTMAX_WEIGHT`：先用 raw logits 选 top-k，再只在 `n_expert_used` 个被选专家上做 softmax。归一化分母小，权重相对更大；且不会因为存在某个超大 logit 的「网红专家」把其他权重压成 0。LLaMA4 选 top-k 用 raw logits（`selection_probs = logits`），权重用 sigmoid，本质上是「不归一」的极端版本，再靠 `weight_before_ffn` 把权重折进输入。

**`weight_before_ffn`（LLaMA4）为什么要在 FFN 前乘权重？**
常规流程是「FFN 算完 -> 输出乘权重 -> 相加」，需要：1 次 `mul_mat_id`（up）、1 次 `mul_mat_id`（down）、1 次 `mul`（输出乘权重）、`n_expert_used-1` 次 `add`。
LLaMA4 改成「输入乘权重 -> FFN -> 直接相加」。由于 FFN 是线性的（up/down 是 matmul，SiLU 非线性但作用在 gate 路径），把权重提前乘到输入上等价于把权重折进 down 的输出。这样可以省一次 elementwise `mul`，且让 `mul_mat_id` 直接吃带权重的输入，对硬件更友好。注意 SiLU 不是线性的，所以严格等价要求权重在 SiLU 之前就作用完——这正是 LLaMA4 把权重放在最前面的原因。

## 显存与计算权衡

- **显存**：MoE 层权重 ≈ `3 × n_embd × n_ff × n_expert`。`n_expert` 越大，显存压力越大。DeepSeek-V3 把 `n_expert` 拉到 256，单层 FFN 权重达数十 GB，整个模型权重 600GB+。
- **计算**：每 token 只算 `n_expert_used` 份 FFN，与 `n_expert` 无关。Mixtral `8×7B` 实际上每 token 只算 2 份 7B 模型的 FFN，因此推理速度接近一个 14B 稠密模型。
- **按需搬运**：当 `n_expert` 大到单卡装不下全部专家权重时，可以让专家权重驻留 CPU/SSD，只把当前 batch 选中的那几个专家搬到 GPU（07 篇会展开）。`ggml_mul_mat_id` 的 grouped GEMM 形态让这种「按需」调度变得自然——每个 batch 只需准备 `n_expert_used × n_tokens` 个专家权重切片。
- **batch 维度**：`n_tokens` 越大，多个 token 选中的专家并集越大，单步需要加载的专家权重越多。极端情况 batch 里每个专家都被某个 token 选中，等价于稠密计算。这就是为什么 MoE 在小 batch 单 token 生成时省算力最明显，大 batch 训练时省得没那么夸张。

## 关键实现细节

1. **`argsort_top_k` 是核心算子**：一次拿到 top-k 的索引，图里反复用于选专家/选组（`ggml.c:5307`）。它复用 `ggml_argsort` + `ggml_view_4d`，本身不引入新 op。
2. **稀疏性=省算力的来源**：`n_expert` 通常 8~256，但 `n_expert_used` 常只有 1~8，未被选中的专家权重不参与计算（`ggml_mul_mat_id` 只索引被选中的片）。
3. **分组路由**：DeepSeek-V3 的 `n_expert_groups`/`n_group_used` 参数（`llama-hparams.h:93-95`），先按组粗筛再组内细选，保证不同 token 的专家分布不扎堆。
4. **门控函数可配**：`llama_expert_gating_func_type` 枚举让同一图代码支持 Mixtral/DeepSeek/LLaMA4/GrooveMoE 的差异（`llama-graph.cpp:1539-1554`）。
5. **bias 与 scaling**：`exp_probs_b`（DeepSeek-V3 专家选择偏置，`llama-graph.cpp:1560-1563`）、`w_scale` 权重缩放（`llama-graph.cpp:1641-1644`）、`up_exps_s`/`gate_exps_s`/`down_exps_s` 的 per-expert scale（在 `build_lora_mm_id` 内叠加，`llama-graph.cpp:1123-1130`）都可在图里叠加。
6. **LoRA 透传**：`build_lora_mm_id`（`llama-graph.cpp:1116`）在 `ggml_mul_mat_id` 之上再叠加 LoRA 的 `ggml_mul_mat_id(lw->b, mul_mat_id(lw->a, cur, ids), ids)`，让 MoE 层也能挂 LoRA。
7. **合并循环上限用 hparams**：`llama-graph.cpp:1816` 用 `hparams.n_expert_used` 而非入参，是为了 warmup 时图节点数稳定（见 PR #14753 注释）。

## 追问方向

- `ggml_argsort_top_k` 内核实现（CPU/CUDA 后端如何并行做 top-k + argsort，`ggml/src/ggml.c:5307`）。
- `ggml_mul_mat_id` 后端实现：CUDA 上是用 grouped GEMM 还是循环调用 cublas？
- MoE 的显存/计算权衡：`n_expert_used` 与 `n_batch` 的关系、专家按需搬运（见 07 篇）。
- DeepSeek-V3 的 `n_group_used` 与 `expert_group_scale` 怎么在训练时被辅助 loss 驱动到均衡。
