# 精读：词表与 Tokenizer（llama-vocab.cpp）

> 日期：2026-08-13 | 核心文件：`src/llama-vocab.cpp`（+ `.h`）
> 前置：读过 `00`（采样拿到的 token id 要回显成文本）。llama.cpp **内置** tokenizer，不依赖 HF tokenizers。

## 作用

把「文本 <-> token id」互转。词表（vocab）是 id <-> 词元的映射表；tokenizer 是切分算法。llama.cpp 把两者都实现在 `llama-vocab.cpp`，加载时从 GGUF 的 KV 元数据重建。

## 词表类型（`llama_vocab_type`，:1765 impl）

| 类型 | 来源 | 说明 |
|---|---|---|
| `SPM` | SentencePiece | 最常用（LLaMA1/2 等），`▁` 表示空格 |
| `BPE` | GPT-2 风格 | 带 pre-tokenizers，LLaMA3 等 |
| `WPM` | WordPiece | BERT 系 |
| `UGM` | unigram | Gemma 等 |
| `RWKV` / `PLAMO2` | 特殊 | 特定架构 |

`get_type()`（:3004）在 GGUF 加载时按 `tokenizer.ggml.model` KV 键决定。

## tokenize 流程（`llama_tokenize_internal`，按类型分发）

`llama_vocab::impl::tokenize`（:2093 附近）按 `type` 走不同分支：

1. **BPE**：先按 **pre-tokenizer** 切分（如 `LLAMA_VOCAB_PRE_TYPE_LLAMA3`），再对每个片段做 BPE merge（`ignore_merges` 可跳过 merge，直接查表）
2. **SPM**：SentencePiece 风格，`▁` 预切 + 合并
3. **WPM**：WordPiece，`##` 前缀子词

**关键参数**（不同模型差异就在这）：
- `add_bos` / `add_eos`：是否加 BOS/EOS 特殊 token
- `add_space_prefix`：是否在开头加空格
- `clean_spaces` / `escape_whitespaces`：空格处理
- `tokenizer_pre`：pre-tokenizer 类型（模板正则/规则）

> 这些由 GGUF 里的 `tokenizer.ggml.pre` KV 键驱动（:2087 的一长串 if-else），所以「同是 BPE，LLaMA3 和 GPT-2 不一样」。

## 加载与重建（`init_tokenizer`，:3083）

`llama_vocab::impl::init_tokenizer(type)` 按词表类型初始化内部的 tokenizer 结构（SPM 用 `llama_vocab::impl` 自己的 SentencePiece 实现，不依赖外部库）。

## 与采样/回显的衔接

- 采样器（`05`）产出 token **id**
- 回显文本：`llama_token_to_piece`（id -> 文本片段）；`llama_byte_to_token`（单字节 -> token）
- 服务端 `common_token_to_piece`（`common/` 层）包装，处理特殊 token 与 UTF-8 边界

## 为什么 llama.cpp 自带 tokenizer

- 零外部依赖（符合项目「纯 C/C++ 无依赖」目标）
- GGUF 里存好 vocab + tokenizer 配置，加载即得，无需调 HF tokenizers 库
- 代价：需为每种 tokenizer 风格单独实现/维护切分逻辑（所以 `tokenizer_pre` 分支那么多）

## 追问方向
- `llama_vocab::impl::tokenize` 的 SPM 具体实现（unigram/Viterbi）
- BPE pre-tokenizer 的正则模板（GPT-2 的 `gpt2` 正则）
- `llama_token_to_piece` 的 UTF-8 边界处理