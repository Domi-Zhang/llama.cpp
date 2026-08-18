# 精读：LLaMA 解码器构图（build_graph 内部）

> 2026-08-17 修订：扩充为初学者详细版（公式推导+数值例子）。
> 学习目标：理解一次前向的 transformer 计算图如何逐层构建，读完能对照源码讲清一层 transformer 的每一步。
> 日期：2026-08-13（初稿）/ 2026-08-17（扩充版）
> 核心文件：`src/models/llama.cpp`（架构构图）+ `src/llama-graph.cpp`（构件辅助）
> 前置：已读 `00-主流程`，知道 `decode()` -> `model.build_graph()`；了解 ggml 张量是列主序、`ne[0]` 是连续维（ne/nb 概念详见 `00a-张量与计算图基础`，未读可先记住"`ne` 是各维长度、`nb` 是各维步长，`ne[0]` 对应最内层连续维"）。

---

## 0. 读者约定与符号

本文假设你「会基本 C++ 和线性代数，知道注意力有 Q/K/V，但第一次读 llama.cpp 构图代码」。因此下文先约定符号，再上图，再逐环节讲。

| 符号 | 含义 |
|---|---|
| `n_tokens` (`T`) | 本批要算的 token 数（一个 ubatch） |
| `n_embd` | 隐藏维度（embedding 维） |
| `n_head` (`H`) | Q 的注意力头数 |
| `n_head_kv` (`H_kv`) | K/V 的注意力头数；GQA 时 `H_kv < H` |
| `n_embd_head` (`d_h`) | 每头维度 = `n_embd / n_head` |
| `n_embd_q` | `d_h * H`，Q 投影输出维 |
| `n_embd_kv` | `d_h * H_kv`，K/V 投影输出维（单条 K 或 V） |
| `n_embd_gqa` | GQA 下 K/V 在「Q 头视角」的等价维度 = `d_h * H`；但在权重里 K/V 只存 `n_embd_kv = d_h * H_kv` 份 |
| `n_ff` | FFN 中间层维度 |
| `n_vocab` | 词表大小 |
| `n_layer` | transformer 层数 |
| `eps` | RMSNorm 的数值稳定小量（`f_norm_rms_eps`，如 1e-5 / 1e-6） |

形状写法：`[d0, d1, d2, ...]` 表示 ggml 张量 `ne[0]=d0, ne[1]=d1, ...`，`d0` 是连续维。`A·B` 表示矩阵乘；`⊙` 表示逐元素相乘（Hadamard 积）。

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

数学上，一个 LLaMA 层可以写成：

```
# 自注意力子层
h_attn = x + W_o * Attn(RoPE(W_q * RMSNorm(x; γ_1)),
                        RoPE(W_k * RMSNorm(x; γ_1)),
                        W_v * RMSNorm(x; γ_1))
# FFN 子层
h_out  = h_attn + W_down * ( SiLU(W_gate * RMSNorm(h_attn; γ_2)) ⊙ W_up * RMSNorm(h_attn; γ_2) )
```

其中 `RMSNorm(x; γ) = γ * x / sqrt(mean(x_i^2) + eps)`，`Attn` 是带因果 mask 的多头注意力，`⊙` 是逐元素乘。下面三件事（RMSNorm / SwiGLU / 残差）是初学者最容易卡住的点，先单独讲透，再回到代码。

---

## 公式预备：RMSNorm、SwiGLU、残差（先讲数学，再讲代码）

### A. RMSNorm

#### 定义与推导

给定向量 `x = (x_1, ..., x_d)`，权重 `γ = (γ_1, ..., γ_d)`，稳定常量 `eps`：

```
rms(x) = sqrt( (1/d) * Σ_{i=1..d} x_i^2 + eps )

RMSNorm(x; γ) = γ * x / rms(x)
            = (γ_1 * x_1 / rms, ..., γ_d * x_d / rms)
```

分量形式：

```
y_i = γ_i * x_i / sqrt( (1/d) * Σ_j x_j^2 + eps )
```

关键观察：`rms(x)` 是个**标量**（一个 token 一个值），所有维度共享，所以归一化只需一次「求平方和 + 一次 sqrt + 一次除法」。这与 LayerNorm 不同。

#### 手算例子（d=4）

取 `x = (1, 2, 2, 4)`，`γ = (2, 1, 1, 0.5)`，`eps = 1e-6`。

1. 平方和：`1 + 4 + 4 + 16 = 25`
2. 均方：`25 / 4 = 6.25`
3. `rms = sqrt(6.25 + 1e-6) ≈ 2.5000002`
4. 归一化后：`x / rms ≈ (0.4000, 0.8000, 0.8000, 1.6000)`
5. 乘权重：`y = (2*0.4, 1*0.8, 1*0.8, 0.5*1.6) = (0.8, 0.8, 0.8, 0.8)`

输出每维都是 `0.8`，这就是「逐元素乘以学习到的缩放」的效果。

#### RMSNorm vs LayerNorm

LayerNorm 公式（标准版）：

```
μ = (1/d) Σ x_i
σ^2 = (1/d) Σ (x_i - μ)^2
LayerNorm(x; γ, β) = γ * (x - μ) / sqrt(σ^2 + eps) + β
```

差异：
- LayerNorm **先减均值再除标准差**，RMSNorm **不减均值、只除均方根**。
- LayerNorm 有两个可学习参数 `γ`（缩放）和 `β`（偏置），RMSNorm 只有 `γ`。
- LayerNorm 要算 `μ`、再算 `σ^2`（两次规约）；RMSNorm 只算一次平方和（一次规约）。

#### 为什么 LLM 普遍用 RMSNorm

1. **省算力**：少一次「减均值」的规约，对 GPU 来说少一遍全向量扫描。
2. **省内存带宽**：归一化是个 memory-bound 算子，少读少写一次向量能显著提速。
3. **效果相当**：经验上把 LayerNorm 的「减均值」去掉，模型质量几乎不下降（参考 RMSNorm 原论文，与 LayerNorm 持平）。
4. **数值更稳**：当某些维度被激活函数推到很大的正值时，均方根永远为正、不为 0（加 eps 后），不容易出现极端缩放。

llama.cpp 的对应代码（`build_norm` 走 `LLM_NORM_RMS` 分支）：

```cpp
// src/llama-graph.cpp:1154
ggml_tensor * llm_graph_context::build_norm(...) {
    switch (type) {
        case LLM_NORM_RMS:
            cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps);  // 第 1162 行
            break;
        ...
    }
    if (mw) cur = ggml_mul(ctx0, cur, mw);   // 乘 γ，第 1176 行
    if (mb) cur = ggml_add(ctx0, cur, mb);   // 加 β（RMSNorm 时 mb=NULL）
    return cur;
}
```

`ggml_rms_norm` 内部一次性算出 `1/rms` 并逐元素乘 `x`，输出再点乘权重 `mw`。RMSNorm 路径里 `mb` 为 NULL（看 `llama.cpp:132-134`），所以不会加偏置。

---

### B. SwiGLU（SiLU-Gated Linear Unit）

#### SiLU 定义

```
sigmoid(z) = 1 / (1 + exp(-z))
SiLU(z)    = z * sigmoid(z)            # 也叫 Swish
```

#### SiLU 数值例子

- `SiLU(0)   = 0 * 0.5      = 0`
- `SiLU(1)   = 1 * 0.7311   = 0.7311`    （sigmoid(1)≈0.7311）
- `SiLU(2)   = 2 * 0.8808   = 1.7616`    （sigmoid(2)≈0.8808）
- `SiLU(-1)  = -1 * 0.2689  = -0.2689`   （sigmoid(-1)≈0.2689）
- `SiLU(-5)  ≈ -5 * 0.0067  ≈ -0.0335`   （大负数趋近 0，但有微弱负值）

注意 SiLU 在大正数处趋近 `z`（线性），在大负数处趋近 `0`，但在中段负数处有**轻微负值**，这是它与 ReLU 最大的区别，给激活带来平滑的非线性。

#### SwiGLU 完整式

LLaMA 的 FFN 用「门控 + SiLU」结构：

```
h_up   = W_up   * x          # 升维到 n_ff
h_gate = W_gate * x          # 升维到 n_ff（另一条投影，称为「门」）
h_mid  = SiLU(h_gate) ⊙ h_up   # 逐元素：门控的 SiLU 与 up 的结果相乘
h_out  = W_down * h_mid        # 降维回 n_embd
```

#### SwiGLU 数值例子（n_ff=2，单个 token）

设 `x = [0.5, -0.5, 1.0]`（n_embd=3 为例），投影后：
- `h_up   = [2.0, 3.0]`
- `h_gate = [1.0, -1.0]`

则：
- `SiLU(h_gate) = [SiLU(1.0), SiLU(-1.0)] = [0.7311, -0.2689]`
- `SiLU(h_gate) ⊙ h_up = [0.7311 * 2.0, -0.2689 * 3.0] = [1.4622, -0.8067]`

最后 `W_down` 把 `[1.4622, -0.8067]` 投影回 `n_embd` 维。

#### 门控直觉

把 `h_up` 想成「**我有什么信息**」，把 `SiLU(h_gate)` 想成「**我把多少比例放出来**」。`SiLU` 输出在 `(0, 1)` 附近时是「门半开」，输出大于 1 时是「放大通过」，负值时「反向通过」，整体像一个**带负向调控、平滑可导的门**。相比 ReLU 门控（ReLUGLU）只开/关两态，SwiGLU 的门更平滑，梯度更稳，所以被 LLaMA/PaLM/Qwen 等普遍采用。

llama.cpp 对应代码（`build_ffn`，`llama-graph.cpp:1267`，SwiGLU 分支 `:1369`）：

```cpp
// src/llama-graph.cpp:1305-1369（节选）
ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;   // h_up   = W_up * x
...
cur = build_lora_mm(gate, cur);                           // h_gate = W_gate * x  (PAR 分支)
...
case LLM_FFN_SILU:
    if (gate && type_gate == LLM_FFN_PAR) {
        ...
        cur = ggml_swiglu_split(ctx0, cur, tmp);          // SiLU(h_gate) ⊙ h_up （融合算子）
        cb(cur, "ffn_swiglu", il);
        ...
    } else {
        cur = ggml_silu(ctx0, cur);
    }
```

`ggml_swiglu_split(a, b)` 把 `SiLU(a) ⊙ b` 融合成单个算子，减少中间张量读写——这在 memory-bound 的 FFN 上能省可观的带宽。融合前后语义等价于「先 SiLU 再逐元素乘」。

---

### C. 残差连接

公式：`y = x + F(x)`，其中 `F` 是某个子层（注意力或 FFN）。

#### 训练视角（直觉版，详细推导见 `15-Transformer论文与llama.cpp实现对照`）

深层网络最大的训练难题是**梯度消失**：反向传播时，每经过一层，梯度要乘以该层的雅可比；层数堆叠后梯度指数级衰减，浅层几乎学不动。

残差把 `y = F(x)` 改成 `y = x + F(x)`，反向传播时：

```
∂y/∂x = I + ∂F/∂x
```

也就是「直连通道 `I`」让梯度可以**不走 `F` 的雅可比**直接传回 `x`。即使 `∂F/∂x` 很小，`I` 仍能保证至少一份梯度传到上一层。这就是为什么 transformer 能堆到几十上百层。

#### 推理视角

推理时残差同样重要：它让每一层学到的「修正量」`F(x)` 加到原 `x` 上，相当于每层只学「补丁」，不必从头重算全部信息。注意力子层的 `inpSA` 副本、FFN 子层的 `ffn_inp` 都是为此保留的「加法支点」。

代码对应：`llama.cpp:178`（注意力后残差）和 `llama.cpp:220`（FFN 后残差），都是一句 `ggml_add`。

---

## 一个 LLaMA 层：逐环节精讲（输入 -> 算子 -> 输出）

> 以下每个环节都对应 `src/models/llama.cpp` 里层循环 `for (il...)`（`:126`）中的一段代码，算子实现都在 `src/llama-graph.cpp`。
> 形状约定（ggml 张量是列主序，`[d0, d1, d2]` 中 d0 是连续维）。

### 第 0 步：三个输入张量（层循环之外，先建一次）

| 张量 | 代码 | 是什么 | 形状 |
|---|---|---|---|
| `inpL` | `build_inp_embd(model.tok_embd)`（`llama.cpp:108`；构件在 `llama-graph.cpp:1833`） | 本层输入 = token 嵌入（或上一层输出）。`build_inp_embd` 内部用 `ggml_get_rows(tok_embd, tokens)` 查表取行（`:1858`），或用 `inp->embd` 走向量路径（`:1887`），再用 `ggml_build_forward_select` 二选一（`:1893`） | `[n_embd, n_tokens]` |
| `inp_pos` | `build_inp_pos()`（`llama.cpp:111`；构件 `:1922`） | 每个 token 的**绝对位置**，喂给 RoPE 生成旋转角。类型 I32 | `[n_tokens]`（I32；当 `n_pos_per_embd>1` 时是 `[n_tokens*n_pos_per_embd]`） |
| `inp_attn` | `build_attn_inp_kv()`（`llama.cpp:119`；构件 `:2303`） | 预构好的「KV cache 容器」：内含 mask、cache 位置、索引等，供每层 attention 复用，避免重复构图 | 结构体（内含多个张量） |

还有 `kq_scale`（`llama.cpp:122`）：`1/sqrt(n_embd_head)`（或超参 `f_attention_scale`），注意力缩放系数，层循环外算一次。

### 层循环内：8 步逐一说清楚

**① Attn 归一化**（`llama.cpp:132` -> `build_norm :1154`）
- 输入：`inpL [n_embd, n_tokens]`
- 算子：`ggml_rms_norm`（RMSNorm，只除均方根、不减均值），再逐元素乘权重 `layer.attn_norm [n_embd]`
- 输出：`cur [n_embd, n_tokens]`（归一化后的激活）
- 作用：把每 token 的向量拉到稳定尺度，让后续 QKV 投影数值可控。详见前文「公式预备 A」。

**② QKV 投影**（`llama.cpp:143` -> `build_qkv :1190`）
- 输入：归一化 `cur [n_embd, n_tokens]`
- 算子：一组 `build_lora_mm` 线性投影（= `ggml_mul_mat`，支持 LoRA 适配）
- 两条路径（决定走哪条的是 `layer.wqkv` 是否存在，见 `llama-model.cpp:2694`）：
  - **fused**（`:1202-1221`）：`wqkv` 一次 matmul 出 `[n_embd_q + 2*n_embd_kv, n_tokens]`，再用 `ggml_view_3d` **零拷贝切分**成 Q/K/V（`llama-graph.cpp:1214-1221`）
  - **分列**（`:1222-1256`）：`wq/wk/wv` 三个分别 matmul，再 `ggml_reshape_3d`（`:1254-1256`）
- 输出（3D，已按头排好）：
  - `Qcur [n_embd_head, n_head, n_tokens]`
  - `Kcur [n_embd_head, n_head_kv, n_tokens]`
  - `Vcur [n_embd_head, n_head_kv, n_tokens]`
- 作用：把每个 token 的 hidden 向量投影到 Q/K/V 三个语义空间，并按注意力头切分。设计细节见下文「代码设计讲解」。

**③ RoPE**（`llama.cpp:146-156` -> `ggml_rope_ext`）
- 输入：`Qcur`、`Kcur` + `inp_pos`；可选 `rope_factors`（Llama3 系列长文本缩放因子；Llama2 为 nullptr）
- 算子：`ggml_rope_ext`（对 Q/K 的每个维度做分组旋转，角度由位置决定）
- 输出：旋转后的 `Qcur`、`Kcur`（形状不变）
- 作用：把**位置信息**注入 Q/K，让注意力能区分 token 先后顺序。（见下方「术语：RoPE」）

**④ Attention（含 KV cache 读写）**（`llama.cpp:169` -> `build_attn :2311`）
- 输入：旋转后 `Qcur/Kcur/Vcur` + `inp_attn`（mask/位置/索引）
- 算子序列（`build_attn` `:2311` 的 KV cache 分支）：
  1. `cpy_k` / `cpy_v`（`ggml_set_rows`）：把**本批新 token** 的 K/V 写入 KV cache（`:2349-2350`）
  2. `k = mctx_cur->get_k(...)` / `v = mctx_cur->get_v(...)`：从 cache 取**全部历史** K/V（`ggml_view_4d` 零拷贝视图，`:2356-2357`）
  3. `build_attn_mha(cur, v, k, ...)`（构件 `:2066`）：真实多头注意力 -- `Q·K^T`（`ggml_mul_mat`，`:2132`），`kq_mask` 加因果掩码（`ggml_soft_max_ext` 把 mask 与 scale 一起做，`:2166`），对 V 加权求和（`ggml_mul_mat`，`:2176`）
  4. `wo` 线性投影（`build_lora_mm`，`:2375`）拼回 `n_embd`
- 输出：`cur = attn_out [n_embd, n_tokens]`
- 作用：**Q 来自新 token、K/V 来自缓存** -- 这就是 KV cache 让自回归能复用全部历史、且每步只算新 token 的原因。也是构图里最核心、最费算力的一段。

**⑤ 残差连接**（`llama.cpp:178` -> `ggml_add`）
- 输入：`attn_out` + `inpSA`（= 本层最开始的 `inpL`，即「Self-Attention 的输入」预留副本，`llama.cpp:129`）
- 算子：`ggml_add`（逐元素相加）
- 输出：`ffn_inp [n_embd, n_tokens]`
- 作用：把注意力输出加回原始输入，形成残差，稳定深层网络训练（详见前文「公式预备 C」）。

**⑥ FFN 归一化**（`llama.cpp:184` -> `build_norm` RMSNorm）
- 输入：`ffn_inp`；权重 `layer.ffn_norm [n_embd]`
- 输出：归一化 `cur`

**⑦ FFN（前馈网络）**（`llama.cpp:189` -> `build_ffn :1267`）
- 输入：归一化 `cur`；权重 `ffn_up / ffn_gate / ffn_down`
- 算子（SwiGLU，`LLM_FFN_SILU + LLM_FFN_PAR`，关键行 `llama-graph.cpp:1305-1375`）：
  1. `up` 线性投影：`tmp = up·x`（维度扩到 FFN 中间层 `n_ff`，`:1305`）
  2. `gate` 线性投影：`cur = gate·x`（并行分支，PAR 模式，`:1327`）
  3. `ggml_swiglu_split(cur, tmp)`：即 `SiLU(gate(x)) ⊙ up(x)`，逐元素门控相乘（融合算子，`:1369`）
  4. `down` 线性投影：缩回 `n_embd`
- 输出：`ffn_out [n_embd, n_tokens]`
- 作用：逐 token 的非线性变换，是模型「记忆/计算知识」的主要容量。详见前文「公式预备 B」。**MoE 分支**：若该层有 `ffn_gate_inp`，改走 `build_moe_ffn`（`llama.cpp:203`，专家路由，见 `03` 专题）。

**⑧ 残差连接**（`llama.cpp:220` -> `ggml_add`）
- 输入：`ffn_out` + `ffn_inp`
- 输出：`inpL [n_embd, n_tokens]` -> 进入下一层（`for il` 下一次迭代，`llama.cpp:227` 更新 `inpL`）

### 数据流速览（一层内张量名字）

```
inpL ──RMS──▶ cur ──QKV──▶ {Qcur,Kcur,Vcur} ──RoPE──▶ {Q,K,V} ──Attn(KV cache)──▶ attn_out
  │                          (3D 按头)                                             │
  └─────────────── inpSA(预留副本) ←───────────────────────────────────────────────┘
                                │ + (⑤残差)
                                ▼
                             ffn_inp ──RMS──▶ cur ──SwiGLU FFN──▶ ffn_out ──+──▶ inpL(下一层)
```

### 输出头（所有层之后，`llama.cpp:229-244`）
```
cur = inpL(最后一层输出)
   -> build_norm(cur, output_norm, RMS)      // result_norm [n_embd, n_tokens]（:231）
   -> build_lora_mm(model.output, cur)       // lm_head 线性投影 [n_vocab, n_tokens]（:240）
   -> res->t_logits                          // 最终 logits（:243）
```
> 若 `output` 权重缺失，会复用 `tok_embd`（权重共享，`TENSOR_DUPLICATED`，见 `llama.cpp:44-46`）。

---

## 代码设计讲解：为什么这么写

这一节专门回答初学者最容易问的三个「为什么」。

### 设计点 1：build_qkv 的 fused 与分列两条路径

llama.cpp 同一段 `build_qkv`（`llama-graph.cpp:1190`）支持两种权重布局：

```cpp
// 路径 A：fused wqkv（一次大 matmul 出全部，再零拷贝切分）
if (layer.wqkv) {
    ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);   // :1204
    // 一次 matmul，输出 [n_embd_q + 2*n_embd_kv, n_tokens]
    Qcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head,    n_tokens, ...);  // :1214
    Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens, ...);  // :1216
    Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens, ...);  // :1219
} else {
    // 路径 B：分列 wq/wk/wv（三次小 matmul，再 reshape）
    Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);  // :1224
    Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);  // :1234
    Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);  // :1244
    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);    // :1254
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);    // :1255
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);    // :1256
}
```

为什么 fused 路径快？三个理由：

1. **一次大 matmul 比 三次小 matmul 对 GPU/缓存友好**：
   - GPU 的 matmul 是 compute-bound，但其「算力利用率」与矩阵面积有关。三次 `[n_embd, n_embd_q] * [n_embd, T]` 的小 matmul，每次都要重新加载输入 `cur [n_embd, T]`，并启动一次 kernel。把三次合成一次 `[n_embd, n_embd_q+2*n_embd_kv] * [n_embd, T]` 的大 matmul，`cur` 只读一遍、kernel 只启动一次，单位访存的算术强度更高，CUDA 上一般能多榨 10%-30% 吞吐。
2. **kernel 启动开销**：每次 `ggml_mul_mat` 对应一个 kernel launch，3 次比 1 次多 2 个 launch。在短 batch（推理时的典型情况）下这个固定开销占比明显。
3. **调度器更省心**：1 个图节点比 3 个图节点更容易让 `ggml_backend_sched` 找到好分片。

但 fused 路径只在 GGUF 里**确实存了 `wqkv` 张量时**才走（`llama-model.cpp:2694` 的 `if (layer.wqkv)` 判断）。Qwen2.5 / Llama2 的官方 HF 权重是分开的 `q_proj/k_proj/v_proj`，所以默认走分列路径；llama.cpp 自己转换时若 fused（部分转换工具会自动合并），就走 fused。

#### `ggml_view_3d` 零拷贝切分为什么可行

`ggml_view_3d` 不复制数据，它只是**新建一个张量描述符**，共享底层 buffer，只改 `ne`（每维长度）和 `nb`（每维步长）。原理如下（ne/nb 的完整定义见 `00a-张量与计算图基础`）：

- `ne[0]` = 连续维长度 = `n_embd_head`（一个头的元素数）
- `nb[0]` = 一个元素的字节数（如 f16 是 2）
- `nb[1]` = 跨一个头要跳多少字节
- `nb[2]` = 跨一个 token 要跳多少字节
- 偏移 `offset` = 该子张量起始地址相对父张量首地址的字节差

fused qkv 的输出按 `n_embd_q | n_embd_kv | n_embd_kv` 顺序排布（即先 Q 区、再 K 区、再 V 区），每个区内按 `[n_embd_head, n_head(*), n_tokens]` 排。所以：

- `Qcur`：从 offset=0 开始，`ne=[n_embd_head, n_head, n_tokens]`
- `Kcur`：从 offset=`n_embd_q * sizeof(elem)` 开始，`ne=[n_embd_head, n_head_kv, n_tokens]`
- `Vcur`：从 offset=`(n_embd_q + n_embd_kv) * sizeof(elem)` 开始，`ne=[n_embd_head, n_head_kv, n_tokens]`

每个子张量通过设置 `nb[1] = n_embd_head * sizeof(elem)`（同一 token 内相邻头之间紧挨着）和 `nb[2] = qkv->nb[1]`（同一头内相邻 token 之间用父张量原始 token 步长），就能让「按子张量的索引」自然落到「父张量对应位置的字节」上。整件事没有任何 `memcpy`。

对比分列路径里的 `ggml_reshape_3d`（`:1254-1256`）：reshape 也是零拷贝的（只改 `ne`，`nb` 保持继承），但它**要求张量已经是按头紧挨着的连续布局**——分列路径三次独立 matmul 的输出天然连续，所以 reshape 即可，不需要 view 的 offset 偏移。

### 设计点 2：build_lora_mm 为什么包一层

`build_lora_mm`（`llama-graph.cpp:1085`）是 `ggml_mul_mat` 的薄包装：

```cpp
// src/llama-graph.cpp:1085
ggml_tensor * llm_graph_context::build_lora_mm(ggml_tensor * w, ggml_tensor * cur, ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);     // 基础 matmul
    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);                  // NVFP4 等量化需要的逐元素 scale
    }
    // LoRA 适配：对每个加载的 LoRA 适配器，查这个权重 w 有没有对应的 (a, b) 低秩矩阵
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) continue;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);
        // 计算 ΔW·x = b·(a·x)·scale，加到主结果上
        ggml_tensor * ab_cur = ggml_mul_mat(ctx0, lw->b,
                                ggml_mul_mat(ctx0, lw->a, cur));
        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }
    return res;
}
```

包一层的原因：

1. **LoRA 适配点统一**：所有线性层（QKV、wo、FFN 的 up/gate/down、lm_head）都可能挂 LoRA。如果每个调用点都手写「先 matmul 再叠加 LoRA」的 4 行代码，会有几十处重复。包一层后，业务代码只看到 `build_lora_mm(w, x)`，LoRA 透明叠加。
2. **量化 scale 统一**：NVFP4 等量化格式需要附加 `w_s` scale 张量，这一步也统一进 `build_lora_mm`。
3. **未来扩展**：要加新的适配器类型（如 IA3、DoRA），只改这一处即可。

对没有加载 LoRA 的常见情况，循环为空，`build_lora_mm` 退化为单次 `ggml_mul_mat`，零额外开销。

### 设计点 3：build_attn 把 KV cache 读写暴露为图节点

`build_attn`（`llama-graph.cpp:2311`）把 `cpy_k/cpy_v`（写 cache）和 `get_k/get_v`（读 cache）都作为 `ggml_build_forward_expand` 加进图：

```cpp
// src/llama-graph.cpp:2335-2357（节选）
ggml_build_forward_expand(gf, q_cur);
ggml_build_forward_expand(gf, v_cur);
ggml_build_forward_expand(gf, k_cur);
...
ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));   // :2349
ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));   // :2350
...
ggml_tensor * k = mctx_cur->get_k(ctx0, il);   // :2356
ggml_tensor * v = mctx_cur->get_v(ctx0, il);   // :2357
```

这样 KV cache 的读写**和注意力计算在同一张图里**，好处：

- 后端调度器可以整体规划（哪些层并行、哪些换设备），不必把 cache 读写当特殊操作。
- flash-attn 等 fused kernel 可以直接读 cache 视图（`ggml_view_4d` 零拷贝），不需要在 Python 层面拼装。
- 对设备分离（CPU+GPU 混合）的部署，cache 的位置和读写顺序由调度器自动决定。

---

## 完整形状演算：以 Qwen2.5-0.5B 为例

> Qwen2.5-0.5B 官方 `config.json` 关键值：`hidden_size=896`、`num_attention_heads=14`、`num_key_value_heads=2`、`num_hidden_layers=24`、`intermediate_size=4864`、`vocab_size=151936`、`head_dim=64`、`rms_norm_eps=1e-6`、`rope_theta=1000000`（以官方 config 为准；`src/models/qwen2.cpp` 也是按这套参数加载的）。
>
> 推导出：
> - `n_embd = 896`、`n_head = 14`、`n_head_kv = 2`、`n_layer = 24`
> - `n_embd_head = 896/14 = 64`
> - `n_embd_q  = 64 * 14 = 896`
> - `n_embd_kv = 64 * 2  = 128`
> - `n_ff = 4864`、`n_vocab = 151936`
> - `kq_scale = 1/sqrt(64) = 0.125`
> - GQA 分组：`n_head / n_head_kv = 7`，即每 7 个 Q 头共享 1 组 K/V

下面表格以「单层 + 输出头」为单位，列出从 `inpL` 到 `logits` 每一步张量的形状。设本批 `n_tokens = T`（推理时 T 通常是 1，prefill 时是几十到几千）。

| 步 | 代码位置 | 张量 | 形状 `[ne0, ne1, ne2, ne3]` | 元素数 |
|---|---|---|---|---|
| 输入 | `llama.cpp:108` | `inpL`（token 嵌入） | `[896, T]` | 896·T |
| 层循环（每层 24 次重复，下面只写一层） | | | | |
| ① Attn 归一化 | `llama.cpp:132` | `cur = attn_norm(inpL)` | `[896, T]` | 896·T |
| ②a QKV 投影（分列，Qwen2.5 走这条） | `llama-graph.cpp:1224` | `wq·cur -> Qcur` | `[896, T]` | 896·T |
| ②b | `:1234` | `wk·cur -> Kcur` | `[128, T]` | 128·T |
| ②c | `:1244` | `wv·cur -> Vcur` | `[128, T]` | 128·T |
| ②d reshape | `:1254-1256` | `Qcur` | `[64, 14, T]` | 896·T |
| ②e reshape | `:1255` | `Kcur` | `[64, 2, T]` | 128·T |
| ②f reshape | `:1256` | `Vcur` | `[64, 2, T]` | 128·T |
| ②' 假如走 fused 路径（参考） | `:1204` | `wqkv·cur -> qkv` | `[896+128+128=1152, T]` | 1152·T |
| ②'' fused 切 Q/K/V | `:1214-1221` | `Qcur/Kcur/Vcur` | 同上 `[64, 14, T] / [64, 2, T] / [64, 2, T]` | 同上 |
| ③ RoPE | `llama.cpp:146/152` | `Qcur, Kcur`（旋转后，形状不变） | `[64, 14, T]`、`[64, 2, T]` | 896·T、128·T |
| ④a 写 KV cache | `llama-graph.cpp:2349-2350` | `cpy_k(Kcur)`, `cpy_v(Vcur)` | 写入每层 cache `[64, 2, n_ctx]` | 128·n_ctx |
| ④b 取历史 K/V | `:2356-2357` | `k = get_k(il)`, `v = get_v(il)` | `[64, 2, T_past+T]` | 128·(T_past+T) |
| ④c Q·K^T | `build_attn_mha :2132` | `kq = K·Q^T` | `[T_past+T, 14, T]` | 14·T·(T_past+T) |
| ④d softmax + mask + scale | `:2166` | `kq_soft_max` | `[T_past+T, 14, T]` | 14·T·(T_past+T) |
| ④e 加权 V | `:2176` | `kqv = V·kq` | `[64, 14, T]` | 896·T |
| ④f wo 投影 | `:2375` | `attn_out = wo·kqv` | `[896, T]` | 896·T |
| ⑤ 残差 | `llama.cpp:178` | `ffn_inp = attn_out + inpSA` | `[896, T]` | 896·T |
| ⑥ FFN 归一化 | `llama.cpp:184` | `cur = ffn_norm(ffn_inp)` | `[896, T]` | 896·T |
| ⑦a FFN up | `llama-graph.cpp:1305` | `tmp = up·cur` | `[4864, T]` | 4864·T |
| ⑦b FFN gate | `:1327` | `cur = gate·cur`（PAR） | `[4864, T]` | 4864·T |
| ⑦c SwiGLU | `:1369` | `cur = SiLU(gate) ⊙ up` | `[4864, T]` | 4864·T |
| ⑦d FFN down | `:1410` 附近 | `ffn_out = down·cur` | `[896, T]` | 896·T |
| ⑧ 残差 | `llama.cpp:220` | `inpL = ffn_out + ffn_inp` | `[896, T]` | 896·T |
| 重复 24 层 → 最后一层 `inpL` | | | `[896, T]` | 896·T |
| 输出归一化 | `llama.cpp:231` | `result_norm = output_norm(inpL)` | `[896, T]` | 896·T |
| lm_head | `llama.cpp:240` | `logits = output·result_norm` | `[151936, T]` | 151936·T |

**几个值得记住的数字**（Qwen2.5-0.5B，T=1，单层）：

- 注意力 `Q·K^T` 元素数 = `14 * 1 * (T_past+1)`，T_past 较大时主导是 cache 长度。
- FFN 中间层 4864 比 hidden 896 大 5.4 倍，这是 SwiGLU FFN 的「升维-非线性-降维」容量来源。
- 单层 forward 一共约 `896*T + 1152*T (QKV) + 896*T (wo) + 4864*T (up) + 4864*T (gate) + 4864*T (down) ≈ 17536*T` 个矩阵乘法参与计算的元素（不含 attention 的 score），FFN 占了 83% —— 这也是为什么大模型推理时 FFN 是访存瓶颈之一。
- KV cache 每层每 token 占 `128*2 = 256` 个元素；24 层全装满 32K 上下文需要 `24 * 256 * 32768 * 2 字节(f16) ≈ 384 MB` 显存——这就是 GQA 把 `n_head_kv` 从 14 砍到 2 带来的 7 倍节省。

---

## 构图源码（`src/models/llama.cpp:98` 的 `graph<embed>::graph`）

### 0) 三个图输入（先建输入张量）
对应 `build_graph` 里最前面几行：

| 输入 | 构件 | 是什么 | 形状 |
|---|---|---|---|
| `inpL` | `build_inp_embd(model.tok_embd)`（`llama.cpp:108`） | token 查表嵌入；支持 token 或向量两条路径二选一（`ggml_build_forward_select`，`:1893`） | `[n_embd, n_tokens]` |
| `inp_pos` | `build_inp_pos()`（`llama.cpp:111`） | 每个 token 的位置，喂给 RoPE | `[n_pos_per_embd * n_tokens]` |
| `inp_attn` | `build_attn_inp_kv()`（`llama.cpp:119`） | 预计算好 KV cache 相关的 mask / 位置 / 索引，一个「容器」供 attention 复用 | 结构体（内含多个张量） |

> 设计要点：KV mask、位置偏移等**在层循环前算一次**，每层 attention 直接引用，避免重复构图。

### 1) 逐层循环（`llama.cpp:126` `for il`）

每层内部按顺序做以下 8 步，产物 `inpL` 传给下一层：

**① Attn 归一化** `build_norm(..., LLM_NORM_RMS)`（`llama.cpp:132`）
- 输入：`inpL`；输出：归一化后的 `cur`
- RMSNorm（无均值，只除方差），权重 `attn_norm`

**② QKV 投影** `build_qkv(layer, cur, ...)`（`llama.cpp:143` -> 构件 `llama-graph.cpp:1190`）
- 输入：归一化 `cur`；输出：`Qcur/Kcur/Vcur`
- 两条路径：fused `wqkv`（一次 matmul 出全部，再 view 切分）或分列 `wq/wk/wv`
- 每个都过 `build_lora_mm`（支持 LoRA 适配）

**③ RoPE** `ggml_rope_ext(Qcur/Kcur, inp_pos, rope_factors, ...)`（`llama.cpp:146/152`）
- 输入：Q/K + 位置；输出：旋转后的 Q/K
- 位置编码注入；`rope_factors` 对 Llama3 系列（long/short/freqs）可为空

**④ Attention（含 KV cache 读写）** `build_attn(inp_attn, ...)`（`llama.cpp:169` -> 构件 `llama-graph.cpp:2311`）
- 输入：Q/K/V + `inp_attn`；输出：`attn_out`
- 关键动作：
  1. `cpy_k/cpy_v` 把新 token 的 K/V **写入** KV cache（`llama-graph.cpp:2349-2350`）
  2. `mctx_cur->get_k/get_v` 从 cache 取**全部**历史 K/V（`:2356-2357`）
  3. `build_attn_mha` 做真实注意力（Q·K^T 加权 V，`kq_mask` 做因果掩码，构件 `:2066`）
- 这是「K/V 来自缓存、Q 来自新 token」的核心，也是 KV cache 让自回归能重复利用历史的原因

**⑤ 残差 + 注意力输出** `ggml_add(attn_out, inpSA)` -> `ffn_inp`（`llama.cpp:178`）
- 残差连接，稳定深层训练/推理

**⑥ FFN 归一化** `build_norm(ffn_inp, ffn_norm, LLM_NORM_RMS)`（`llama.cpp:184`）

**⑦ FFN（前馈网络）** `build_ffn(...)`（`llama.cpp:189` -> 构件 `llama-graph.cpp:1267`）
- 输入：归一化 `ffn_inp`；输出：`ffn_out`
- 结构：`up` 线性 -> SiLU 门控(`gate`) -> 逐元素乘 -> `down` 线性（SwiGLU）
- 参数 `LLM_FFN_SILU, LLM_FFN_PAR` 指定激活与并行结构
- **MoE 分支**：若 `ffn_gate_inp` 存在，改走 `build_moe_ffn`（`llama.cpp:203`，路由器 softmax 选 top-k 专家，构件 `:1454`/`:1496`）

**⑧ 残差** `ggml_add(ffn_out, ffn_inp)` -> `inpL`（`llama.cpp:220`）
- 层循环结束，`inpL` 进入下一层（`:227`）

### 2) 输出头（`llama.cpp:229-244`）
```
cur = inpL
    -> build_norm(..., output_norm, LLM_NORM_RMS)   // result_norm（:231）
    -> build_lora_mm(model.output, cur)             // lm_head 线性投影（:240）
    -> res->t_logits                                 // 最终 logits（:243）
```

> 技巧：`output` 权重若缺失，会复用 `tok_embd`（权重共享，`load_arch_tensors` 里 `TENSOR_DUPLICATED`，见 `llama.cpp:44-46`）。

---

## 关键构件（`llama-graph.cpp` 里的辅助函数）

| 构件 | 行号 | 作用 |
|---|---|---|
| `build_lora_mm` | 1085 | 基础 matmul + NVFP4 scale + LoRA 适配叠加（所有线性层共用） |
| `build_norm` | 1154 | 归一化（RMS / LayerNorm / GroupNorm） |
| `build_qkv` | 1190 | QKV 投影（fused 或分列） |
| `build_ffn` | 1267 | 标准 FFN（SwiGLU/GeGLU/ReGLU 等） |
| `build_moe_ffn` | 1454 / 1496 | MoE FFN（专家路由），两个重载 |
| `build_attn_mha` | 2066 | 核心多头注意力算子组合（flash 或拆分实现） |
| `build_attn`（KV cache 版） | 2311 | 注意力 + KV cache 读写编排 |
| `build_attn`（no-cache 版） | 2226 | 无 cache 的注意力（嵌入模型用） |
| `build_attn_inp_kv` | 2303 | 预构 KV cache 输入容器（mask/索引） |
| `build_inp_embd` | 1833 | token 查表得嵌入；token/向量双路径 |
| `build_inp_pos` | 1922 | 位置张量 |

---

## 读图小结（为什么构图是「理解性能的钥匙」）

1. **图即拓扑**：`ggml_cgraph` 的节点=算子、边=张量依赖，后端据此决定执行顺序与显存分配。
2. **KV cache 在构图层面可见**：`cpy_k/cpy_v` 与 `get_k/get_v` 是图节点，所以「缓存写读」也是可并行/可 fused 的计算。
3. **复用预构张量**：mask/位置在层循环外算一次，所有层共享，减少图体积。
4. **构图与执行解耦**：架构代码只负责「搭图」，真正算力在 `ggml_backend_sched_graph_compute_async`。
5. **形状与算子选择决定性能**：`build_qkv` 选 fused 还是分列、`build_attn_mha` 选 flash 还是拆分、`ggml_swiglu_split` 是否融合，都直接影响 kernel 数量和访存量。

---

## 术语解释（inp、RoPE 等）

| 术语 | 全称/含义 | 在本文件里的位置 |
|---|---|---|
| **inpL** | **input of Layer**，每一层的输入张量。第一层=token 嵌入，之后每层循环末尾被更新为本层输出 | 层顶创建、`:227` 更新 |
| **inp_pos** | **input position**，位置输入，token 的绝对位置号，RoPE 用它生成旋转角 | `build_inp_pos` |
| **inp_attn** | **input attention**，预计算的注意力容器（mask/位置/索引），各层共享复用 | `build_attn_inp_kv` |
| **inpSA / inp_embd** | **inp**ut **S**elf-**A**ttention：残差里被加的那个「原始输入副本」；同族命名 `inp_embd` 是起点的嵌入输入 | `:129` `:178` |
| **cur** | **current**，当前流水线里正在计算的那个张量（临时变量，随环节不断被覆盖） | 全程 |
| **Q/K/V** | **Query/Key/Value**，注意力三件套：Q=「我想找什么」，K=「我有什么特征」，V=「我携带的内容」 | ②③④ |
| **RoPE** | **Rotary Position Embedding**，旋转位置编码。把 Q/K 的相邻两维当复平面，按「位置×基频」旋转；位置差越大、因果方向越明确，让注意力感知 token 顺序。**只对 Q/K 做、对 V 不做** | ③ |
| **RMSNorm** | **Root Mean Square Norm**，只除以均方根、不居中（不减均值）的归一化，比 LayerNorm 省内存 | ①⑥；详见「公式预备 A」 |
| **SiLU / Swish** | `x * sigmoid(x)`，平滑可导的激活函数，SwiGLU 的门控核心 | 「公式预备 B」 |
| **SwiGLU** | **SiLU × Gated Linear Unit**，`SiLU(gate(x)) ⊙ up(x)` 再 `down` 投影的 FFN 变体，LLaMA 标配 | ⑦；详见「公式预备 B」 |
| **KV cache** | **Key-Value 缓存**，把历史 token 的 K/V 存起来，下次自回归只算新 token 的 Q、复用历史 K/V | ④ |
| **GQA** | **Grouped-Query Attention**，多个 Q 头共享一组 K/V 头（`n_head > n_head_kv`），省缓存省算力 | ②④；详见下方补注 |
| **MHA / MQA** | Multi-Head Attention（`n_head_kv == n_head`）/ Multi-Query Attention（`n_head_kv == 1`，所有 Q 头共享一组 K/V），GQA 是介于两者之间的折中 | ②④ |
| **residual** | 残差连接，把子层输出加回输入，`x + F(x)` | ⑤⑧；详见「公式预备 C」 |
| **kq_scale** | K·Q^T 的缩放系数 `1/sqrt(n_embd_head)`，防点积过大导致 softmax 饱和 | 层外预算 `llama.cpp:122` |
| **logits** | 输出层未归一化的原始分数 `[n_vocab, n_tokens]`，采样器拿它选下一个 token | 输出头 |
| **ubatch** | 一个 micro-batch，`n_tokens` 个 token 一起算的单位 | 形状约定 |
| **fused / 分列** | fused = 一次大 matmul 出 QKV 再 view 切分；分列 = 三次小 matmul | ②；详见「代码设计讲解」 |
| **ne / nb** | ggml 张量的「每维长度 / 每维步长」描述符；`ne[0]` 是连续维。详见 `00a-张量与计算图基础` | 设计点 1 |
| **LoRA** | **Low-Rank Adaptation**，把权重增量分解为 `b·a` 两个小矩阵，推理时叠加到主权重上 | `build_lora_mm` |

### 补注：GQA 为什么 `n_head_kv` 可以小于 `n_head`，共享怎么实现

设 `n_head=14, n_head_kv=2`（Qwen2.5-0.5B 的情况）。**K/V 只存 2 组**，每组维度 `d_h=64`，总 K 维度 `2*64=128`。**Q 有 14 组**，每组 `d_h=64`。

在 `Q·K^T` 时，需要每个 Q 头都对应一组 K。GQA 的做法是**把 2 组 K 复制/广播到 14 组**：每 `14/2 = 7` 个 Q 头共享同一组 K。实现上有两种：

1. **显式 repeat**：在 `build_attn_mha` 前用 `ggml_repeat` 把 K/V 从 `[d_h, 2, T]` 扩成 `[d_h, 14, T]`，然后按普通 MHA 算。简单但费显存。
2. **隐式 broadcast / 在 kernel 内处理**：后端 kernel 在算 `Q·K^T` 时，根据 `n_head / n_head_kv` 的比例直接定位到对应的 K 头，不显式复制。llama.cpp 在多数后端用这种更省内存的方式（具体见 `02-精读-Attention构图`）。

无论哪种实现，**权重和 KV cache 都只存 `n_head_kv` 份**，因此 KV cache 体积直接缩小 `n_head / n_head_kv` 倍。Qwen2.5-0.5B 这个比例是 7，所以同样上下文长度下显存占用是 MHA 的 1/7——这对长上下文推理是巨大优势，是 GQA 在现代 LLM 里几乎成为默认选项的原因。

---

## 追问方向（可选下一步）
- 深入 `build_attn_mha`：flash-attention 融合、`kq_mask` 因果掩码如何实现（`02-精读-Attention构图`）
- 深入 `build_moe_ffn`：路由器 + top-k 选择 + 专家权重组合（`03-精读-MoE-FFN构图`）
- 深入 KV cache：`src/llama-kv-cache.cpp` 的分配/复用/上下文回收（对应 `04-KV cache` 专题）
- 深入 RoPE：旋转矩阵的复平面推导、`rope_factors` 与 YaRN 长文本外推（`15-Transformer论文与llama.cpp实现对照`）
