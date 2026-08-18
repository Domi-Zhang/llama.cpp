# 精读：Attention 构图（build_attn_mha，含 Flash-Attention）

> 日期：2026-08-13 | 核心文件：`src/llama-graph.cpp:2066` `build_attn_mha`
> 前置：已读 `01-精读-LLaMA解码器构图`，知道每层 attention 的入参。
> 2026-08-17 修订：扩充为初学者详细版（在线 softmax 手算例子等）。

## 作用

把某层的 Q/K/V 做完整自注意力，输出 `attn_out`。这是 transformer 每层里最重的计算。
入口签名（`src/llama-graph.cpp:2066`）：

```cpp
ggml_tensor * build_attn_mha(
    ggml_tensor * q,        // 当前 token 的 query（新算）
    ggml_tensor * k,        // 全部历史 K（来自 KV cache，get_k 取出）
    ggml_tensor * v,        // 全部历史 V（来自 KV cache，get_v 取出）
    ggml_tensor * kq_b,     // KQ bias（部分老模型用）
    ggml_tensor * kq_mask,  // 因果掩码 + 跨序列隔离
    ggml_tensor * sinks,    // attention sink token（可选）
    ggml_tensor * v_mla,    // MLA 压缩投影（可选）
    float   kq_scale,       // 1/sqrt(d_k)
    int     il) const;
```

要理解这个函数，得先搞清三件事：(1) 注意力在数学上算什么、(2) 软最大值为什么容易溢出、(3) 掩码怎么把不该看见的位置屏蔽掉。下面分别铺开，然后再回到两条路径的对比。

## 前置：标准注意力公式（单头）

设某头：`d = n_embd_head`（头维），`L = n_tokens`（序列长）。Q/K/V 都是 `[L, d]`，掩码 `M` 是 `[L, L]`（值待定）。

```
s_ij   = (Q·K^T)_ij / sqrt(d)        // 分数，缩放 1/sqrt(d_k)
p_ij   = s_ij + M_ij                  // 加掩码：允许的位置 M_ij=0，禁止的位置 M_ij=-inf
alpha  = softmax(p, 沿 j 轴)          // 每行(对 j)做 softmax，得到注意力权重
out_i  = Sum_j alpha_ij * V_j         // 用权重对 V 加权求和
```

softmax 的定义是 `softmax(x)_i = exp(x_i) / Sum_j exp(x_j)`。整个注意力就是「打分 -> 归一化 -> 加权」三步。下面先看一个朴素实现的麻烦。

## 软最大值的数值稳定性（动机）

### 朴素 softmax 会溢出

直接按定义 `softmax(x)_i = exp(x_i) / Sum_j exp(x_j)` 写代码，立刻撞上一个问题：`exp` 长得极快。

举例：`x = [1000, 1001, 1002]`
- `exp(1000)` 在 float32 里直接是 `+inf`（float32 最大约 `3.4e38`，`exp(88.7)` 就到上限了）。
- 三个分子全 `inf`，分母 `Sum = inf`，于是 `inf / inf = NaN`。
- 整行输出 `[NaN, NaN, NaN]`，后面的 V 加权也全废。

注意力里的 `s_ij = Q·K^T / sqrt(d)` 完全可能很大：Q、K 各分量是 F16/F32 任意实数，点积几十几百很常见，乘以几十维之后上千并不稀奇。所以朴素 softmax 根本跑不起来。

### max-shift softmax（标准解法）

把每个 `x_i` 都减去本行最大值 `m = max_j x_j` 再做 exp：

```
m       = max_j x_j
softmax(x)_i = exp(x_i - m) / Sum_j exp(x_j - m)
```

数学上等价（分子分母同乘 `exp(-m)`），但数值上：
- 最大的那项变成 `exp(0) = 1`，其余都是 `exp(负数) <= 1`。
- 永远不会溢出，最小也只会下溢成 0（不影响归一化）。

再算 `x = [1000, 1001, 1002]`：
- `m = 1002`
- `exp(-2) = 0.1353`，`exp(-1) = 0.3679`，`exp(0) = 1.0`
- `Sum = 1.5032`
- 输出 `[0.0900, 0.2447, 0.6652]`，干净。

### 这解释了两件事

1. **经典路径为什么把 `kq` 强制 F32**（`src/llama-graph.cpp:2137`：`ggml_mul_mat_set_prec(kq, GGML_PREC_F32)`）。
   F16 的最大值约 `65504`，`exp(11.1)` 就溢出了，连 max-shift 都救不回来（因为 max-shift 只让被减后的值不溢出，但中间的 `s_ij` 本身先得能存下）。F32 的最大值约 `3.4e38`，给 `s_ij` 留了充足的范围。然后 `ggml_soft_max_ext` 内部再做 max-shift（见 `ggml/src/ggml-cpu/ops.cpp:5400-5408`：先 `ggml_vec_max_f32` 找行最大，再 `ggml_vec_soft_max_f32(..., max)` 减去它）。

2. **Flash 路径为什么用「在线重缩放」**。
   Flash 不物化整张 `kq`，所以没法先扫一遍找 max 再算 exp。它的做法是：**边扫边维护一个「运行最大值 m」**，每遇到更大的分数就把之前累计的输出按 `exp(m_old - m_new)` 缩回去。这样永远只用 `exp(s - m)`，`s - m <= 0`，同样不会溢出。详见下面在线 softmax 一节。

## 因果掩码与跨序列隔离

掩码张量 `kq_mask` 由 `llama_kv_cache::set_input_kq_mask`（`src/llama-kv-cache.cpp:1715`）在每次推理前填好。核心实现是模板函数 `set_input_kq_mask_impl`（`src/llama-kv-cache.cpp:1528`），里面只有两个魔法值（`:1542-1543`）：

```cpp
const T mask_keep = llama_cast<T>(0.0f);        // 允许 attend
const T mask_drop = llama_cast<T>(-INFINITY);   // 禁止 attend
```

填表的规则在 `:1603-1671` 的大循环里，对每个 query token `i` 和每个 KV cell `j`：
- 若 `cells.is_empty(j)`：`j` 是空槽，drop（`:1617-1619`）。
- **若 `!cells.seq_has(j, seq_id)`：`j` 不属于当前 query 所在序列，drop（`:1622-1624`）。这就是跨序列隔离。**
- 若 `causal && p0 > p1`：KV cell 的位置 `p0` 比 query 位置 `p1` 更靠后（未来），drop（`:1637-1641`）。这就是因果掩码。
- 若启用 SWA（sliding window）且超出窗口：drop（`:1656-1660`）。
- 否则 keep。

下面分别看因果掩码和跨序列隔离。

### 3x3 因果掩码例子

设序列长 3，token 位置 0、1、2。纯因果（无 SWA）下，掩码 `M` 是下三角：

```
        key:  0     1     2
query 0      [ 0,   -inf, -inf ]
query 1      [ 0,    0,   -inf ]
query 2      [ 0,    0,    0   ]
```

`M[i][j] = 0` 当 `i >= j`（query i 可以看到 key j，j 不在 i 的未来），否则 `-inf`。

假设原始分数 `s = Q·K^T / sqrt(d)`（已经缩放）是：

```
S = [ [1.0,  5.0,  2.0],
      [0.5,  3.0,  4.0],
      [2.0,  1.0,  6.0] ]
```

加掩码 `p = S + M`：

```
P = [ [1.0,  -inf, -inf],
      [0.5,   3.0, -inf],
      [2.0,   1.0,  6.0] ]
```

每行做 max-shift softmax：

- 第 0 行：max=1.0，`exp([0, -inf, -inf]) = [1, 0, 0]`，sum=1，alpha_0 = `[1, 0, 0]`。query 0 只看 key 0。
- 第 1 行：max=3.0，`exp([-2.5, 0, -inf]) = [0.0821, 1, 0]`，sum=1.0821，alpha_1 = `[0.0759, 0.9241, 0]`。query 1 主要看 key 1，少量看 key 0，不看 key 2。
- 第 2 行：max=6.0，`exp([-4, -5, 0]) = [0.0183, 0.0067, 1]`，sum=1.025，alpha_2 = `[0.0179, 0.0066, 0.9756]`。query 2 主要看 key 2。

注意每一行 `Sum_j alpha_ij = 1`，因为 `-inf` 的 `exp` 是 0，不参与归一化。这就是「未来位置被屏蔽」的本质：在 softmax 分母里根本不计入。

### 跨序列隔离

batch 推理时，一个 batch 里可能同时有多个独立序列（A 和 B）。它们彼此绝对不能看见，否则相当于串话。

`set_input_kq_mask_impl` 用 `seq_has(j, seq_id)`（`src/llama-kv-cells.h:301`）检查 KV cell `j` 是否属于当前 query 的序列 `seq_id`。每个 KV cell 内部存了一个 bitset `seq[i]`，记录它被哪些序列占用（多序列共享一个 cell 在某些 KV 复用场景下会出现）。

设想一个极简 batch：序列 A 有 `a0, a1`，序列 B 有 `b0, b1`，KV cache 里 4 个 cell 按位置 `[a0, a1, b0, b1]` 排列。那么 query `a1` 看到的 mask 行是：

```
[a0,  a1,  b0,  b1]
[ 0,   0, -inf, -inf]   <- a1 可以看到 a0、a1（因果允许），但 b0、b1 属于别的序列，drop
```

而 query `b1` 看到的 mask 行是：

```
[a0,  a1,  b0,  b1]
[-inf,-inf, 0,   0 ]    <- b1 只能看到 b0、b1
```

注意这里因果掩码和序列隔离叠加生效：`b1` 不能看 `b2`（因果，假设 b2 在未来 cell），也不能看任何 `a*`（序列隔离）。代码里这两步是顺序判断的：先 `seq_has`（`:1622`），后 `p0 > p1`（`:1639`），任一不满足就 `goto skip` 写 `-inf`。

`set_input_kq_mask` 还支持一个加速：同一个序列的多个 query token 共享大部分 mask（只有「靠近 batch 内其他 token 位置」的 cell 会变），用 `seq_srct`/`seq_idxs` 跳过重复计算（`:1556-1601`，注释引用 PR #18842）。这一段是性能优化，不影响 mask 的语义。

## 两条路径

以 `cparams.flash_attn` 开关分叉（`src/llama-graph.cpp:2089`：`use_flash_attn = cparams.flash_attn && kq_b == nullptr`），这是性能分水岭。

### 路径 A：Flash-Attention（`ggml_flash_attn_ext`）
```
q/k/v 先 permute 成 [embd_head, n_head_kv, n_tokens, n_stream]
  -> ggml_flash_attn_ext(q, k, v, kq_mask, kq_scale, alibi, soft_cap)
  -> reshape 回 [n_embd_head*n_head, n_tokens]
```
- **融合**：Q·K^T、掩码、softmax、加权 V 在一个内核里完成，不落地中间大矩阵。
- 要求：`kq_b == nullptr`（FATTN 不支持 KQ bias）；K/V 需 F16（`:2098-2104` 把 F32 cast 成 F16）。
- 显存省：不物化 `[n_tokens, n_tokens]` 的注意力矩阵。
- 内核里实际跑的是「分块 + 在线 softmax」算法，下一节细讲。

### 路径 B：经典三算子（`ggml_mul_mat` 直算）
```
kq  = ggml_mul_mat(k, q)                              // Q·K^T      [n_tokens, n_tokens]
kq  = ggml_soft_max_ext(kq, kq_mask, kq_scale, alibi) // 掩码+softmax
kqv = ggml_mul_mat(v, kq)                             // (softmax@QK^T)·V
cur = ggml_permute(kqv, ...)                          // 调回 [embd_head*n_head, n_tokens]
```
- 中间会物化 `[n_tokens, n_tokens]` 的 `kq`，长上下文时是这个路径的显存瓶颈。
- `kq` 默认强制 F32 精度：`ggml_mul_mat_set_prec(kq, GGML_PREC_F32)`（`:2137`，注释说 "this op tends to require high floating point range"）。
- `ggml_soft_max_ext` 把 scale、mask、ALiBi 偏置一步到位（`:2166`），内部用 max-shift。

## 在线 softmax：Flash 的核心（完整推导）

这一节是本篇最关键的新增内容。Flash-Attention 的「省显存」和「数值稳定」都靠它。我们一步一步推。

### 问题：分块算 softmax 的难处

经典 softmax 是「整行算完再归一」：要先看到整行的所有 `s_ij`，求出 `max`，再算所有 `exp(s - max)`，再求 `sum`，最后归一。这要求整行（`[L, L]` 的某一行）必须同时在内存里。

Flash 想分块：一次只读一小块 K、V 进 SRAM，算完一块丢掉再读下一块。但 softmax 的 `max` 和 `sum` 都依赖整行，分块后每块只看到部分 `s_ij`，怎么保证最后结果和整行算一致？

### 关键观察：max 和 sum 可以「增量合并」

设整行分数分成两块：`x = [x_1, x_2]`（`x_1` 是第一块的分数向量，`x_2` 是第二块）。整行 softmax 是：

```
m   = max(max(x_1), max(x_2))   = max(m_1, m_2)
l   = Sum exp(x - m)
    = Sum exp(x_1 - m) + Sum exp(x_2 - m)
    = exp(m_1 - m) * Sum exp(x_1 - m_1)  +  exp(m_2 - m) * Sum exp(x_2 - m_2)
    = exp(m_1 - m) * l_1  +  exp(m_2 - m) * l_2
```

其中 `m_1, l_1` 是只看第一块时的 max 和 sum，`m_2, l_2` 同理。也就是说：**已知两块各自的 `(m, l)`，可以用 `exp(m_old - m_new)` 把旧的 `l` 重缩放后再相加，得到合并后的 `l`**。`m` 就是两者的 max。

### 加权 V 也能用同样的方式合并

注意输出 `out = Sum_j alpha_j * V_j = (1/l) * Sum_j exp(s_j - m) * V_j`。把 `O = Sum_j exp(s_j - m) * V_j`（未归一的累计输出）一起维护：

```
O = exp(m_1 - m) * O_1 + exp(m_2 - m) * O_2
```

每块的 `O_k = Sum_{j in block k} exp(s_j - m_k) * V_j`，只用块内信息算。合并时按 `exp(m_k - m)` 缩放。

### 在线算法

把上面的合并公式串起来，逐块流入，维护三个量：运行最大值 `m`、运行和 `l`、运行未归一输出 `O`。

```
初始化：m = -inf, l = 0, O = 0

for 每块 K[Bk], V[Bk]:
    s   = Q · K[Bk]^T / sqrt(d)          // 本块分数，形状 [Bq, Bk]
    m_b = max(s, 沿 Bk 轴)                // 本块行最大
    m_new = max(m, m_b)                   // 合并后的新运行 max

    # 关键一步：重缩放旧累计
    alpha = exp(m - m_new)                # <= 1，把旧 O、l 缩到新基准下
    O     = O * alpha
    l     = l * alpha

    # 把本块贡献加进去
    p     = exp(s - m_new[:, None])       # 本块的 exp(s - m_new)，形状 [Bq, Bk]
    O    += p @ V[Bk]                     # 累加本块的 V 贡献
    l    += sum(p, 沿 Bk 轴)

    m     = m_new                         # 更新运行 max

# 全部块处理完，归一化
out = O / l
```

每一步只用当前块的信息 + 三个累计量，所以整张 `[L, L]` 分数矩阵从头到尾不需要物化。SRAM 里只需要放 `[Bq, Bk]` 的小块。

### 手算例子：1 个 query 对 4 个 key，分两块

为了把上面抽象公式落实，用具体小数字走一遍。设 `d` 任意（只关心分数），4 个 key 的分数（已经做完 `Q·K^T/sqrt(d)`）是：

```
s = [2, 6, 8, 3]
```

V 行随便取（2 维方便手算）：

```
V_1 = [10, 0],  V_2 = [0, 10],  V_3 = [1, 1],  V_4 = [2, 2]
```

分两块：块 1 = `(s=2, V_1)` 和 `(s=6, V_2)`；块 2 = `(s=8, V_3)` 和 `(s=3, V_4)`。

**先算「真值」做参照**（整行一次性 softmax）：
- `m = max(s) = 8`
- `exp(s - m) = [exp(-6), exp(-2), exp(0), exp(-5)] = [0.00248, 0.1353, 1.0, 0.00674]`
- `l = 0.00248 + 0.1353 + 1.0 + 0.00674 = 1.14452`
- `alpha = [0.00217, 0.1182, 0.8737, 0.00589]`
- `out = 0.00217*[10,0] + 0.1182*[0,10] + 0.8737*[1,1] + 0.00589*[2,2]`
- `    = [0.0217, 0] + [0, 1.182] + [0.8737, 0.8737] + [0.01178, 0.01178]`
- `    = [0.90718, 2.06748]`

**现在用在线 softmax 分块算**。

初始化：`m = -inf, l = 0, O = [0, 0]`

#### 块 1：s_b = [2, 6]
- 块内 max：`m_b = max(2, 6) = 6`
- 合并 max：`m_new = max(-inf, 6) = 6`
- 重缩放因子：`alpha = exp(m - m_new) = exp(-inf - 6) = 0`
  - `O = O * 0 = [0, 0]`（一开始就是 0，没影响）
  - `l = l * 0 = 0`
- 更新 `m = 6`
- 块内 `p = exp(s_b - m_new) = [exp(2-6), exp(6-6)] = [exp(-4), exp(0)] = [0.0183, 1.0]`
- 累加 V 贡献：`O += p @ [V_1; V_2] = 0.0183*[10,0] + 1.0*[0,10] = [0.183, 0] + [0, 10] = [0.183, 10]`
- 累加和：`l += 0.0183 + 1.0 = 1.0183`

块 1 处理完状态：`m = 6, l = 1.0183, O = [0.183, 10]`

#### 块 2：s_b = [8, 3]
- 块内 max：`m_b = max(8, 3) = 8`
- 合并 max：`m_new = max(6, 8) = 8`  **(注意：m_new > m，需要重缩放旧累计)**
- 重缩放因子：`alpha = exp(m - m_new) = exp(6 - 8) = exp(-2) = 0.1353`
  - `O = O * 0.1353 = [0.183*0.1353, 10*0.1353] = [0.02476, 1.353]`  **(旧的 O 被缩到新基准 m=8 下)**
  - `l = l * 0.1353 = 1.0183 * 0.1353 = 0.13778`
- 更新 `m = 8`
- 块内 `p = exp(s_b - m_new) = [exp(8-8), exp(3-8)] = [1.0, exp(-5)] = [1.0, 0.00674]`
- 累加 V 贡献：`O += p @ [V_3; V_4] = 1.0*[1,1] + 0.00674*[2,2] = [1, 1] + [0.01348, 0.01348] = [1.01348, 1.01348]`
  - `O = [0.02476 + 1.01348, 1.353 + 1.01348] = [1.03824, 2.36648]`
- 累加和：`l += 1.0 + 0.00674 = 1.00674`
  - `l = 0.13778 + 1.00674 = 1.14452`

块 2 处理完状态：`m = 8, l = 1.14452, O = [1.03824, 2.36648]`

#### 归一化
- `out = O / l = [1.03824 / 1.14452, 2.36648 / 1.14452] = [0.90716, 2.06748]`

对比真值 `[0.90718, 2.06748]`：完全一致（差最后一位是手算 `exp` 取 4 位小数的舍入）。

#### 关键看「重缩放」这一步

块 2 进来时 `m_new = 8 > m = 6`，旧 `O = [0.183, 10]` 是基于 `exp(s - 6)` 算的，现在基准变成 8，必须乘 `exp(6 - 8) = 0.1353` 把它「平移」到新基准下，否则两块的贡献就不在同一把尺子上，加起来会错。

代码里对应 `ggml/src/ggml-cpu/ops.cpp:8471-8481`：

```cpp
if (s > M) {
    // s 是新最大，ms < 1.0，把旧的 VKQ 和 S 缩到新基准下
    M = s;
    ms = expf(Mold - M);
    ggml_vec_scale_f32(DV, VKQ32, ms);   // O *= ms
} else {
    vs = expf(s - M);                     // 本块贡献直接算
}
// ...
S = S*ms + vs;                            // l 也按同样因子缩，再加本块和
```

`M`/`S`/`VKQ` 就是上面手算里的 `m`/`l`/`O`。这就是「在线 softmax」的 C++ 落地。

### 为什么数学等价

把循环展开后会发现：每一步的重缩放因子乘进 `O` 和 `l`，最终 `O` 的每一项都变成了 `exp(s_j - m_final)` 乘 `V_j`，`l` 也是 `Sum exp(s_j - m_final)`，与整行算的结果一字不差。所以 Flash 和经典路径算的是**同一个数学函数**，只是求和顺序（以及随之的浮点舍入）不同。

## 两条路径对比（含显存/带宽量级）

### 数学等价性表

| 维度 | 经典路径（B） | Flash 路径（A） |
|---|---|---|
| 是否物化 `[L,L]` 分数矩阵 | 是（O(L^2) 显存） | 否（分块，O(L) 额外） |
| softmax 实现 | 整张算完再归一（max-shift） | 在线 + 运行 max 重缩放 |
| 精度策略 | `kq` 强制 F32 防分数溢出（`:2137`） | F16 输入 + 在线重缩放防溢出（`:2111` 设 F32 累加） |
| 中间张量 | `kq` `[L,L]`、`kq_soft_max` `[L,L]`、`kqv` `[L,d]` | 仅 `cur` `[L,d]` 输出 |
| 数学结果 | `alpha * V` | 同一 `alpha * V` |

### 显存/带宽量级分析

设序列长 `L`，头维 `d`，块大小 `B`（典型 64-128）。单头单层看：

**经典路径**（HBM 流量）：
1. `kq = mul_mat(k, q)`：写 `[L, L]` -> `4*L^2` 字节。
2. `soft_max_ext(kq, ...)`：读 `[L, L]`、写 `[L, L]` -> `8*L^2` 字节。
3. `kqv = mul_mat(v, kq)`：读 `[L, L]`、读 `[L, d]`、写 `[L, d]` -> `4*L^2 + 8*L*d` 字节。
4. 峰值显存：`[L, L]` 这张中间矩阵常驻，约 `4*L^2` 字节（F32）。

`L = 4096` 时，单头单层就要 `4 * 4096^2 = 64 MB` 的 `kq` 中间张量。32 头 32 层 = 1024 个这样的张量（虽然是分批跑的，但峰值仍可观）。

**Flash 路径**（HBM 流量）：
1. 每个 query 块（`Bq` 行）扫一遍所有 K/V 块，K/V 每元素被读 `L/Bq` 次。
2. 总 K/V 读：`(L/Bq) * 2 * L * d` 字节（K 和 V 各 `L*d`）。
3. 写出：`L * d` 字节（仅最终输出）。
4. 峰值显存：`O(L*d + Bq*Bk + Bq*d)`，与 `L^2` 无关。

数值对比（`L=4096, d=128, Bq=64`）：
- 经典中间张量：`4 * 4096^2 = 64 MB`。
- Flash 中间量：`4 * 4096 * 128 + 4 * 64 * 128 = 2.1 MB`，省 30 倍。
- 经典 HBM 流量：约 `16 * L^2 = 256 MB`。
- Flash HBM 流量：约 `(L/Bq) * 2 * L * d = (4096/64) * 2 * 4096 * 128 = 64 MB`，省 4 倍。

实际加速比取决于硬件 SRAM 大小和后端实现，但量级差异是结构性的：**经典是 O(L^2) 显存 + O(L^2) 带宽，Flash 是 O(L) 显存 + O(L^2 * d / Bq) 带宽**。这就是长上下文推理（`L` 上万）必须切到 Flash 的原因。

## GQA 下的 K/V 头共享

Grouped Query Attention（GQA）和 Multi-Query Attention（MQA）让 K/V 的头数 `n_head_kv` 少于 Q 的头数 `n_head`（如 Llama-3-70B：`n_head=64, n_head_kv=8`，每 8 个 query 头共享 1 组 K/V 头）。好处是 KV cache 显存直接砍到 `n_head_kv / n_head`。

### 图里没有显式的 repeat 操作

很多教程会画一个「把 K/V 复制 `n_head/n_head_kv` 份再和 Q 对齐」的步骤。在 llama.cpp 的图里**找不到这样的节点**——没有 `ggml_repeat` 之类的 op 用来扩 K/V 头。原因是：广播是 **`mul_mat` 内部隐式做的**。

证据链：

1. **Q/K/V 的形状**。`get_k`（`src/llama-kv-cache.cpp:1239`，调用 `:1251-1252`）返回的 K 视图形状是 `[n_embd_head, n_head_kv, n_kv, n_stream]`——头维度是 `n_head_kv`，不是 `n_head`。Q 则是 `[n_embd_head, n_head, n_tokens_q, n_stream]`。两者头维度不同。

2. **`build_attn_mha` 里的 permute**（`:2081-2085`）把三者都按 `permute(0, 2, 1, 3)` 重排，得到：
   - Q：`[n_embd_head, n_tokens_q, n_head, n_stream]`
   - K：`[n_embd_head, n_kv, n_head_kv, n_stream]`
   - V：同 K。

3. **`ggml_can_mul_mat` 的契约**（`ggml/src/ggml.c:3231-3237`）：
   ```cpp
   return (t0->ne[0] == t1->ne[0])  &&
          (t1->ne[2] % t0->ne[2] == 0)  &&  // t0(K) 的头维度必须能整除 t1(Q) 的头维度
          (t1->ne[3] % t0->ne[3] == 0);
   ```
   也就是说，只要 `n_head % n_head_kv == 0`，mul_mat 就允许 K 和 Q 头维度不同。结果张量的形状（`ggml/src/ggml.c:3246`）是 `[a->ne[1], b->ne[1], b->ne[2], b->ne[3]]`——**头维度取 Q 的 `n_head`**，K 的 `n_head_kv` 被「广播掉」。

4. **内核里的索引映射**（`ggml/src/ggml-cpu/ggml-cpu.c:1176-1177, 1210-1211`）：
   ```cpp
   const int64_t r2 = ne12 / ne02;   // = n_head / n_head_kv，广播倍数
   // ...
   const int64_t i02 = i12 / r2;     // query 头 i12 对应的 K 头索引
   ```
   对 query 头 `i12`，取 K 头 `i12 / r2`。`r2 = 8` 时，query 头 0..7 都取 K 头 0，query 头 8..15 取 K 头 1，依此类推。这就是「共享」。

5. **Flash 路径完全相同**（`ggml/src/ggml-cpu/ops.cpp:8368, 8431`）：
   ```cpp
   const int64_t rk2 = neq2/nek2;    // 广播倍数
   // ...
   const int ik2 = iq2 / rk2;        // query 头 iq2 对应的 K 头
   ```
   `ggml_flash_attn_ext` 的结果形状也用 Q 的头维度（`ggml/src/ggml.c:5392`：`ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] }`）。

### 为什么这样设计

显式 repeat 会真的分配 `n_head * d * L` 大小的张量，把 GQA 省下来的显存又还回去。隐式广播只在内核里多算一个除法 `i12 / r2`，没有任何额外存储。代价是每个后端（CPU、CUDA、Metal、Vulkan...）的 mul_mat / flash_attn 内核都得自己实现这个 `i02 = i12 / r2` 的索引。代码里管这个叫 "broadcast factors"（`ggml-cpu.c:1175`），不叫 "repeat_kv"——grep 整个仓库搜不到 `repeat_kv` 这个函数名。

## 关键实现细节

1. **permute 布局**：`ggml_permute(ctx, x, 0, 2, 1, 3)` 把 `[embd_head, n_head, n_tokens]` 重排，让 matmul 沿正确维度做，实际是零拷贝的 view 技巧（不改元素值，只改 stride 解释）。
2. **n_stream 分流**：`n_stream = k->ne[3]`（`:2079`），KV cache 可按 sequence 分多条 stream，`ggml_view_4d`（`:2081`）把 q 切成 `n_stream` 份并行。
3. **softmax 的扩展参数**：除 `kq_scale` 外还支持 ALiBi 偏置（`hparams.f_max_alibi_bias`）和 logit soft-capping（tanh 缩放到 `attn_soft_cap`）。
4. **Grok 特例**（`:2139-2150`）：`30*tanh(kq/30)` 的注意力缩放，其他架构走统一 softmax。
5. **后端归属**（`:2190-2193`）：非 offload 时，KV 存储到 attention 输出之间的节点钉在 CPU（`ggml_backend_sched_set_tensor_backend`），避免反复搬运。
6. **MLA 解压**（`:2113-2128` / `:2180-2183`）：`v_mla` 非空时再乘一次投影，把 MQA 的压缩表示解回 MHA 权重。
7. **Flash 的 sink 处理**（`:2110` `ggml_flash_attn_ext_add_sinks`）：sink token 作为虚拟「全局最大」并入在线 softmax 的 `m`（见 `ops.cpp:8518-8533`）。
8. **Flash 精度**（`:2111` `ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32)`）：累加 `O`、`m`、`l` 强制 F32，即使 K/V 是 F16。

## 图视角小结

- Flash 路径把「QK-softmax-V」3 个算子折叠成 1 个融合算子节点，图更小、后端融合机会更多。
- 经典路径则显式展开 3 个节点，便于观察中间 `kq` 张量（也便于调试）。
- 切换到 Flash 往往是长上下文推理提速/省显存的第一切入点。

## 追问方向

- `ggml_flash_attn_ext` 各后端实现（`ggml/src/ggml-cuda/fattn-*.cu`、`ggml/src/ggml-metal/llama.cpp` 等），重点看分块大小 `Bq/Bk` 如何依 SRAM 容量选取。
- 因果掩码 `kq_mask` 在 `set_input_kq_mask`（`src/llama-kv-cache.cpp:1715`）里如何逐 token 生成，特别是 `seq_has` 跨序列隔离（`:1622`）和 SWA 窗口（`:1656`）。
- 多 stream（`n_stream > 1`）时 mask 如何按 stream 切片，避免不同 stream 串话。
