# 精读：LLaMA 解码器构图（build_graph 内部）

> 学习目标：理解一次前向的 transformer 计算图如何逐层构建。
> 日期：2026-08-13
> 核心文件：`src/models/llama.cpp`（架构构图）+ `src/llama-graph.cpp`（构件辅助）
> 前置：已读 `00-主流程`，知道 `decode()` -> `model.build_graph()`。

---

## 一图总览：一个 LLaMA 层的计算图

```
inp_embd (token embedded)  inp_pos (位置)  inp_attn (KV cache 相关 mask/pos)
   │
   ▼
 ┌─────────────────────────────── 第 il 层循环 (n_layer 次) ───────────────────────────────┐
 │  attn_norm (RMS) ──▶ QKV 投影 ──▶ RoPE ──▶ Attention(写读 KV cache) ──▶ +residual ──▶ ffn_inp │
 │  ─────────────────────────────────────────────────────────────────────────────────── │
 │  ffn_norm (RMS) ──▶ FFN (SiLU gated) ──▶ +residual ──▶ inpL(下一层输入)                    │
 └──────────────────────────────────────────────────────────────────────────────────────┘
   │
   ▼ (最后一层输出 inpL)
 output_norm (RMS) ──▶ lm_head(linear) ──▶ logits  [shape: n_vocab x n_tokens]
```

---

## 构图源码（`src/models/llama.cpp:98` 的 `graph<embed>::graph`）

### 0) 三个图输入（先建输入张量）
对应 `build_graph` 里最前面几行：

| 输入 | 构件 | 是什么 | 形状 |
|---|---|---|---|
| `inpL` | `build_inp_embd(model.tok_embd)` | token 查表嵌入；支持 token 或向量两条路径二选一（`ggml_build_forward_select`） | `[n_embd, n_tokens]` |
| `inp_pos` | `build_inp_pos()` | 每个 token 的位置，喂给 RoPE | `[n_pos_per_embd * n_tokens]` |
| `inp_attn` | `build_attn_inp_kv()` | 预计算好 KV cache 相关的 mask / 位置 / 索引，一个「容器」供 attention 复用 | 结构体（内含多个张量） |

> 设计要点：KV mask、位置偏移等**在层循环前算一次**，每层 attention 直接引用，避免重复构图。

### 1) 逐层循环（`llama.cpp:126` `for il`）

每层内部按顺序做以下 8 步，产物 `inpL` 传给下一层：

**① Attn 归一化** `build_norm(..., LLM_NORM_RMS)`
- 输入：`inpL`；输出：归一化后的 `cur`
- RMSNorm（无均值，只除方差），权重 `attn_norm`

**② QKV 投影** `build_qkv(layer, cur, ...)` (`llama-graph.cpp:1190`)
- 输入：归一化 `cur`；输出：`Qcur/Kcur/Vcur`
- 两条路径：fused `wqkv`（一次 matmul 出全部，再 view 切分）或分列 `wq/wk/wv`
- 每个都过 `build_lora_mm`（支持 LoRA 适配）

**③ RoPE** `ggml_rope_ext(Qcur/Kcur, inp_pos, rope_factors, ...)`
- 输入：Q/K + 位置；输出：旋转后的 Q/K
- 位置编码注入；`rope_factors` 对 Llama3 系列（long/short/freqs）可为空

**④ Attention（含 KV cache 读写）** `build_attn(inp_attn, ...)` (`llama-graph.cpp:2311`)
- 输入：Q/K/V + `inp_attn`；输出：`attn_out`
- 关键动作：
  1. `cpy_k/cpy_v` 把新 token 的 K/V **写入** KV cache（`llama-graph.cpp:2349`）
  2. `mctx_cur->get_k/get_v` 从 cache 取**全部**历史 K/V
  3. `build_attn_mha` 做真实注意力（Q·K^T 加权 V，`kq_mask` 做因果掩码）
- 这是「K/V 来自缓存、Q 来自新 token」的核心，也是 KV cache 让自回归能重复利用历史的原因

**⑤ 残差 + 注意力输出** `ggml_add(attn_out, inpSA)` -> `ffn_inp`
- 残差连接，稳定深层训练/推理

**⑥ FFN 归一化** `build_norm(ffn_inp, ffn_norm, LLM_NORM_RMS)`

**⑦ FFN（前馈网络）** `build_ffn(...)` (`llama-graph.cpp:1267`)
- 输入：归一化 `ffn_inp`；输出：`ffn_out`
- 结构：`up` 线性 -> SiLU 门控(`gate`) -> 逐元素乘 -> `down` 线性（SwiGLU）
- 参数 `LLM_FFN_SILU, LLM_FFN_PAR` 指定激活与并行结构
- **MoE 分支**：若 `ffn_gate_inp` 存在，改走 `build_moe_ffn`（路由器 softmax 选 top-k 专家）

**⑧ 残差** `ggml_add(ffn_out, ffn_inp)` -> `inpL`
- 层循环结束，`inpL` 进入下一层

### 2) 输出头（`llama.cpp:229-244`）
```
cur = inpL
    -> build_norm(..., output_norm, LLM_NORM_RMS)   // result_norm
    -> build_lora_mm(model.output, cur)             // lm_head 线性投影
    -> res->t_logits                                 // 最终 logits
```

> 技巧：`output` 权重若缺失，会复用 `tok_embd`（权重共享，`load_arch_tensors` 里 `TENSOR_DUPLICATED`）。

---

## 关键构件（`llama-graph.cpp` 里的辅助函数）

| 构件 | 行号 | 作用 |
|---|---|---|
| `build_inp_embd` | 1833 | token 查表得嵌入；token/向量双路径 |
| `build_inp_pos` | 1922 | 位置张量 |
| `build_qkv` | 1190 | QKV 投影（fused 或分列） |
| `build_norm` | 1154 | 归一化（RMS / LayerNorm 等） |
| `build_ffn` | 1267 | 标准 FFN（SwiGLU） |
| `build_moe_ffn` | 1454 | MoE FFN（专家路由） |
| `build_attn` | 2311 | 注意力 + KV cache 读写 |
| `build_attn_mha` | 2066 | 核心多头注意力算子组合 |

---

## 读图小结（为什么构图是「理解性能的钥匙」）

1. **图即拓扑**：`ggml_cgraph` 的节点=算子、边=张量依赖，后端据此决定执行顺序与显存分配。
2. **KV cache 在构图层面可见**：`cpy_k/cpy_v` 与 `get_k/get_v` 是图节点，所以「缓存写读」也是可并行/可 fused 的计算。
3. **复用预构张量**：mask/位置在层循环外算一次，所有层共享，减少图体积。
4. **构图与执行解耦**：架构代码只负责「搭图」，真正算力在 `ggml_backend_sched_graph_compute_async`。

---

## 追问方向（可选下一步）
- 深入 `build_attn_mha`：flash-attention 融合、`kq_mask` 因果掩码如何实现
- 深入 `build_moe_ffn`：路由器 + top-k 选择 + 专家权重组合
- 深入 KV cache：`src/llama-kv-cache.cpp` 的分配/复用/上下文回收（对应 `04-KV cache` 专题）