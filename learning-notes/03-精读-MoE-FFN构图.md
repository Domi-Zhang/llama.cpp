# 精读：MoE-FFN 构图（build_moe_ffn，路由器 + top-k）

> 日期：2026-08-13 | 核心文件：`src/llama-graph.cpp:1454` `build_moe_ffn`
> 前置：已读 `01`，知道 MoE 分支在 `build_ffn` 的 else 分支触发。
> 触发条件：`layer.ffn_gate_inp != nullptr`（存在路由器权重）。

## 作用

稀疏混合专家前馈：每个 token 只激活少量「专家」，大幅降低 FFN 计算量。
MoE FFN 所有权重形状多一维 `[.., n_expert]`，即「每个专家一套 FFN 权重」。

## 数据流（`ffn_moe_*` 命名对应图节点）

```
cur (归一化后的输入, [n_embd, n_tokens])
  │
  ├─ ① 路由器打分  logits = gate_inp·cur        → [n_expert, n_tokens]
  ├─ ② 门控激活    probs = softmax/sigmoid(logits)
  ├─ ③ 选专家      selected = argsort_top_k(probs, n_expert_used)
  │                   → [n_expert_used, n_tokens]
  ├─ ④ 取权重      weights = get_rows(probs, selected)
  │                   → [1, n_expert_used, n_tokens]
  ├─ ⑤ 专家前向    (每个选中专家算 up·SiLU(gate·x)·down)
  └─ ⑥ 加权合并    out = Σ weights[i] * expert_out[i]
```

## 逐段说明

### ① 路由器打分（`llama-graph.cpp:1527`）
`logits = build_lora_mm(gate_inp, cur)`，输出 `[n_expert, n_tokens]`，即每个 token 对每个专家的偏好分。
支持 `probs_in` 直接注入（跨层/预计算时复用）。

### ② 门控激活（`switch (gating_op)`）
- `SOFTMAX`：对全部专家做 softmax（最常用，DeepSeek/Mixtral 系）
- `SIGMOID`：逐专家 sigmoid（不再归一，专家可独立开闭）
- `SOFTMAX_WEIGHT`：保留 logits，权重在后面对 top-k 再做 softmax

### ③ 选取 top-k 专家
`ggml_argsort_top_k(selection_probs, n_expert_used)` 选出每个 token 得分最高的 `n_expert_used` 个专家。
**专家分组（DeepSeek-V3 风格）**：当 `n_expert_groups > 1` 时，先按组选 top group，再在选中组内选专家，组内非选中专家用 `-INFINITY` 掩掉（`llama-graph.cpp:1578-1599`）。

### ④ 专家权重
`ggml_get_rows(probs, selected_experts)` 取出选中专家的概率当权重。
可选 `norm_w`：对权重按 token 归一（除以 sum，防除零有 clamp）。分支特例见源码 `1619-1633`。

### ⑤ 专家前向 + ⑥ 加权合并
每个选中专家跑一版「up → SiLU(gate) → down」，再按权重加总。
`weight_before_ffn`（LLaMA4）特例：sigmoid 后的权重在 FFN **之前**应用。

## 关键实现细节

1. **`argsort_top_k` 是核心算子**：一次拿到 top-k 的索引，图里反复用于选专家/选组。
2. **稀疏性=省算力的来源**：`n_expert` 通常 8~256，但 `n_expert_used` 常只有 1~8，未被选中的专家权重不参与计算。
3. **分组路由**：DeepSeek-V3 的 `n_expert_groups`/`n_group_used` 参数，先按组粗筛再组内细选，保证不同 token 的专家分布不扎堆。
4. **门控函数可配**：`llama_expert_gating_func_type` 枚举让同一图代码支持 Mixtral/DeepSeek/LLaMA4/GrooveMoE 的差异。
5. **bias 与 scaling**：`exp_probs_b`（DeepSeek-V3 专家选择偏置）、`w_scale` 权重缩放都可在图里叠加。

## 追问方向
- `ggml_argsort_top_k` 内核实现（`ggml/src/ggml.c`）
- MoE 的显存/计算权衡：`n_expert_used` 与 `n_batch` 的关系