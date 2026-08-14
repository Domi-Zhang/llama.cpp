# 精读：Attention 构图（build_attn_mha，含 Flash-Attention）

> 日期：2026-08-13 | 核心文件：`src/llama-graph.cpp:2066` `build_attn_mha`
> 前置：已读 `01-精读-LLaMA解码器构图`，知道每层 attention 的入参。

## 作用

把某层的 Q/K/V 做完整自注意力，输出 `attn_out`。这是 transformer 每层里最重的计算。
入口签名：`build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il)`

- `q`：当前 token 的 query（新算）
- `k`/`v`：**全部历史**（来自 KV cache，经 `get_k/get_v` 取出）
- `kq_mask`：因果掩码（attend 到的位置）
- `kq_b`/`kq_scale`：bias 与缩放
- `sinks`：attention sink token（可选）
- `v_mla`：MLA 的压缩投影（可选，仅切 MLA 模型）

## 两条路径

以 `cparams.flash_attn` 开关分叉，这是性能分水岭：

### 路径 A：Flash-Attention（`ggml_flash_attn_ext`）
```
q/k/v 先 permute 成 [embd_head, n_head, n_tokens, n_stream]
  -> ggml_flash_attn_ext(q, k, v, kq_mask, kq_scale, alibi, soft_cap)
  -> reshape 回 [n_embd_head*n_head, n_tokens]
```
- **融合**：Q·K^T、掩码、softmax、加权 V 在一个内核里完成，不落地中间大矩阵
- 要求：`kq_b == nullptr`（FATTN 不支持 KQ bias）；K/V 需 F16
- 显存省：不物化 `[n_tokens, n_tokens]` 的注意力矩阵

### 路径 B：经典三算子（`ggml_mul_mat` 直算）
```
kq  = ggml_mul_mat(k, q)          // Q·K^T      [n_tokens, n_tokens]
kq  = ggml_soft_max_ext(kq, kq_mask, kq_scale, alibi)   // 掩码+softmax
kqv = ggml_mul_mat(v, kq)         // (softmax@QK^T)·V
cur = ggml_permute(kqv, ...)       // 调回 [embd_head*n_head, n_tokens]
```
- 中间会物化 `[n_tokens, n_tokens]` 的 `kq`，长上下文时是这个路径的显存瓶颈
- `kq` 默认强制 F32 精度：`ggml_mul_mat_set_prec(kq, GGML_PREC_F32)`（数值稳定性）

## 关键实现细节

1. **permute 布局**：`ggml_permute(ctx, x, 0, 2, 1, 3)` 把 `[embd_head, n_head, n_tokens]` 重排，让 matmul 沿正确维度做，实际是零拷贝的 view 技巧。
2. **n_stream 分流**：`n_stream = k->ne[3]`，KV cache 可按 sequence 分多条 stream，`ggml_view_4d` 把 q 切成 `n_stream` 份并行。
3. **softmax 的扩展参数**：除 `kq_scale` 外还支持 ALiBi 偏置（`f_max_alibi_bias`）和 logit soft-capping（tanh 缩放到 `attn_soft_cap`）。
4. **Grok 特例**：`30*tanh(kq/30)` 的注意力缩放，其他架构走统一 softmax。
5. **后端归属**：非 offload 时，KV 存储到 attention 输出之间的节点钉在 CPU（`ggml_backend_sched_set_tensor_backend`），避免反复搬运。
6. **MLA 解压**：`v_mla` 非空时再乘一次投影，把 MQA 的压缩表示解回 MHA 权重。

## 图视角小结

- Flash 路径把「QK-softmax-V」3 个算子折叠成 1 个融合算子节点，图更小、后端融合机会更多。
- 经典路径则显式展开 3 个节点，便于观察中间 `kq` 张量（也便于调试）。
- 切换到 Flash 往往是长上下文推理提速/省显存的第一切入点。

## 追问方向
- `ggml_flash_attn_ext` 内核实现（`ggml/src/ggml-*.c` 各后端）
- 因果掩码 `kq_mask` 在 `set_input_kq_mask`（`llama-kv-cache.cpp:1715`）里如何逐 token 生成