# 精读：词表与 Tokenizer（llama-vocab.cpp）

> 2026-08-17 修订：扩充为初学者详细版（BPE 手算例子等）
> 日期：2026-08-13 | 核心文件：`src/llama-vocab.cpp`（+ `src/llama-vocab.h`）
> 前置：读过 `00`（采样拿到的 token id 要回显成文本）。llama.cpp **内置** tokenizer，不依赖 HF tokenizers。

## 1. 为什么需要 tokenization

神经网络只会吃浮点张量、吐浮点张量。要让模型理解「文本」，必须先有一座桥把「字符串」翻译成「整数 id 序列」，再通过 embedding 表把 id 映射成向量。这座桥就是 tokenizer，桥两端的对照表就是 vocab（词表）。

- **文本 -> id**：`tokenize`，输入是一段 `std::string`，输出是 `std::vector<llama_token>`（就是 `int32_t`）。
- **id -> 文本**：`detokenize` / `llama_token_to_piece`，反向把 id 序列拼回字符串，用于回显。

词表大小 `n_tokens` 决定了 id 的取值范围：`0 .. n_tokens-1`。LLaMA-2 是 32000，LLaMA-3 是 128256，Qwen2 还要更大。每个 id 在 `id_to_token` 数组里对应一条 `token_data{text, score, attr}`（见 `src/llama-vocab.cpp:1812` 附近的 `id_to_token`）。

**为什么不能直接按字符切？** 中文有数万汉字，按字符切词表会爆炸（动辄十万级），且每个汉字的语义信息太碎，模型要学很多步才能拼出「词语」的概念。**为什么不能按整词切？** 英文有几十万词、加上变形和专有名词根本列不完，遇到训练时没见过的词就只能输出 `<UNK>`，泛化能力差；序列长度也短不下来。

折中方案就是「子词」（subword）：把词切成比词短、比字符长的片段。常用词整体进词表（如 `low`、`er`），罕见词拆成更小的子词拼出来（如 `lower` = `low` + `er`）。这样既控制了词表大小，又能编码任意 UTF-8 文本，几乎不会出现 `<UNK>`。

主流子词算法有三种：BPE（GPT 系列、LLaMA3）、SentencePiece/Unigram（LLaMA1/2、Gemma）、WordPiece（BERT）。下面前两节详讲 BPE 和 SentencePiece。

## 2. BPE 算法详解

### 2.1 训练直觉（理解算法怎么来的）

BPE = Byte-Pair Encoding，原本是一种压缩算法。训练流程：

1. 把训练语料里每个「词」拆成单字符序列，统计每个字符对（pair）的出现次数。
2. 选出现次数最多的 pair，合并成一个新的「子词」，写进词表和合并规则表（merges），并给一个 rank（合并优先级，越小越先合）。
3. 用新的子词替换语料里的对应 pair，重新统计 pair 频次。
4. 重复 2-3，直到词表达到目标大小或频次低于阈值。

例：语料里 `"low"` 出现 5 次、`"lower"` 出现 2 次、`"newest"` 出现 6 次... 第一次合并最高频的 pair `e, s -> es`，下次再找最高频... 最终词表既有单字符 `l, o, w, e, r`，也有合并出来的 `lo, low, er, lower, es, est, newest` 等等。

训练结果保存为两张表：**词表**（vocab，token 字符串 -> id）和**合并规则表**（merges，pair -> rank）。GGUF 把这两张表都存进 metadata。

### 2.2 推理编码过程

推理时拿到一段文本，对一个**片段**（pre-tokenize 切出来的，见第 4 节）做 BPE 编码：

1. 把片段按 UTF-8 字符拆成初始符号链表：`['l', 'o', 'w', 'e', 'r']`。
2. 找出当前所有相邻 pair 里 **rank 最小**（即合并优先级最高）的那个，合并成一个新符号。
3. 重复 2 直到没有可合并的 pair 为止。
4. 把最终符号链表里的每个符号去词表查 id。查不到的（理论上是 byte-level BPE 的字节 token），按单字节编码成 `<0xXX>` 形式的 byte token。

llama.cpp 的 BPE session 实现见 `llm_tokenizer_bpe_session::tokenize`（`src/llama-vocab.cpp:597`）。它用一个**优先队列**（`llm_bigram_bpe::queue`，:271）维护所有可合并的 pair，按 `rank` 升序弹出，每次合并后只更新受影响的左右两个新 pair（:665-666），不需要每次重扫整条链。

### 2.3 完整手算例子

设小词表 `V = {"l", "o", "w", "e", "r", "lo", "low", "er", "lower"}`，对应的合并规则表（按 rank 升序）：

| rank | merge  |
|------|--------|
| 0    | l + o -> lo     |
| 1    | lo + w -> low   |
| 2    | e + r -> er     |
| 3    | low + er -> lower |

编码 `"lower"`：

**第 0 步**：拆字符
```
['l', 'o', 'w', 'e', 'r']
相邻 pair: (l,o) rank 0, (o,w) -, (w,e) -, (e,r) rank 2
最小 rank = 0 -> 合并 (l,o)
```

**第 1 步**：合并后
```
['lo', 'w', 'e', 'r']
相邻 pair: (lo,w) rank 1, (w,e) -, (e,r) rank 2
最小 rank = 1 -> 合并 (lo,w)
```

**第 2 步**：合并后
```
['low', 'e', 'r']
相邻 pair: (low,e) -, (e,r) rank 2
最小 rank = 2 -> 合并 (e,r)
```

**第 3 步**：合并后
```
['low', 'er']
相邻 pair: (low,er) rank 3
最小 rank = 3 -> 合并 (low,er)
```

**第 4 步**：合并后
```
['lower']
无相邻 pair 可合 -> 结束
```

最终序列 `['lower']`，去词表查得 `lower` 的 id，输出 `[id("lower")]`。如果输入是 `"lowering"`（`ing` 不在词表），前面步骤相同，最后会剩 `['lower', 'i', 'n', 'g']`，再各自查表（`i/n/g` 都按单字符或 byte token 编码）。

### 2.4 `ignore_merges` 是什么

`ignore_merges`（`llama_vocab::impl::ignore_merges`，声明 :1802）是一个开关：**在跑 BPE merge 之前，先查整个片段是不是已经在词表里**。如果是，直接当成一个 token，跳过 merge。

代码见 `llm_tokenizer_bpe_session::tokenize`（:612）：

```cpp
if (vocab.get_ignore_merges() && vocab.text_to_token(word) != LLAMA_TOKEN_NULL) {
    symbols.emplace_back(llm_symbol{-1, -1, word.c_str(), word.size()});
    offset = word.size();
}
```

什么时候需要？LLaMA3、MiniCPM5 等模型把整个常用词都加进了词表，希望优先按整词编码，而不是按 BPE 规则切。所以加载这类模型时 `ignore_merges = true`（设置点 :2110、:2122）。结果就是「如果整个片段就是词表里的 token，就直接用，否则才走 BPE merge」。

## 3. SPM（SentencePiece）详解

SentencePiece 是 Google 出的另一套 tokenizer 框架。算法本身有两种模式：**BPE 模式**（同上）和 **Unigram LM 模式**（更常用，下面讲）。

### 3.1 Unigram 语言模型 + Viterbi 的直觉

Unigram 模型训练时假设：每段文本的最佳切分，是「使所有子词概率乘积最大」的那种切分。每个子词在词表里有一个 `score`（对数概率，越小越罕见）。比如 `"lower"` 可能的切分有：

- `['lower']`：score = log P(lower)
- `['low', 'er']`：score = log P(low) + log P(er)
- `['l', 'o', 'w', 'e', 'r']`：5 个 log P 之和

每种切分的总 score 不同，Viterbi 算法动态规划地找出文本上「总 score 最大」的那条切分路径。直觉就是：能用一个高概率的长子词覆盖就别拆，实在不行才拆成更短的。

llama.cpp 的 UGM（Unigram）实现在 `llm_tokenizer_ugm_session::tokenize`（`src/llama-vocab.cpp:964`）。注释 :951-963 完整描述了流程：从左到右逐个 UTF-8 字符扫，每到一个位置枚举所有可能的子词（用 trie 前缀匹配），把「到当前位置为止的最佳切分 score」存在 `tokenization_results[offset]` 里，最后从末尾回溯（:1035）拿到 token 序列。

> 注：`LLAMA_VOCAB_TYPE_SPM`（LLaMA1/2 风格）在 llama.cpp 里走的是另一条路 —— 不是 Viterbi，而是 **priority-queue 按 token.score 合并**（见 `llm_tokenizer_spm_session::tokenize`，:117）。优先队列的 comparator 是 `l.score < r.score`（:99），即 score 大的先合并。这等价于「贪心合并最大 score 子词」，对 LLaMA1/2 的词表能复现原始 SentencePiece 的结果，但实现更轻量。真正的 Viterbi 留给 `LLAMA_VOCAB_TYPE_UGM`（Gemma 等）。

### 3.2 `▁` 表示空格的约定

SentencePiece 不用普通空格 `' '`，而用占位字符 `▁`（U+2581）来表示「词边界/空格」。`"hello world"` 在 SPM 里被编码成 `["▁hello", "▁world"]`（开头也加 `▁`）。这样词与空格的边界信息被保留进 token 字符串本身，detokenize 时再把 `▁` 还原成空格即可。

llama.cpp 里：
- encode 侧：`llama_escape_whitespace`（:3251）把字符串里的空格替换成 `▁`。
- decode 侧：`llama_unescape_whitespace` 反向替换（在 `impl::token_to_piece` 里调用，见 :3534、:3553）。

`add_space_prefix`（:1798）控制是否在文本开头补一个 `▁`（即把 `"hello"` 编码成 `["▁hello"]`），LLaMA1/2 默认开启。

### 3.3 Byte fallback 为什么必要

如果某个字符完全不在词表里（比如训练时没见过的生僻汉字），没有 token 对应它。SentencePiece 的兜底是 **byte fallback**：把这个字节当成单独的 token 输出。词表里预先放好 256 个形如 `<0xXX>` 的 byte token（见 `llama_vocab::byte_to_token`，:3818，对 SPM/UGM 走 `<0xXX>` 形式，:3824；对 BPE/WPM 走 GPT-2 风格的 `unicode_byte_to_utf8(ch)`，:3835）。

SPM session 的 `resegment`（:176）就体现了这个回退：找不到 token 时，把符号的每个字节调 `vocab.byte_to_token(...)` 输出（:191-194）。UGM 侧用「unknown token + 单字节 penalty」处理（:1018-1026）。

意义：**任何 UTF-8 文本都能编码成 id 序列，永远不会因为「词表里没这个字符」而失败**。这对多语言、emoji、错别字非常重要。

## 4. pre-tokenizer 是什么、为什么需要

### 4.1 为什么 BPE 不能直接跑整段文本

如果直接对 `"Hello, world!"` 跑 BPE merge：

- 空格会和后面的词粘在一起，可能合并成 `" world"` 这种带空格的子词，也可能漏掉。
- 不同模型对「空格是否保留」「数字几位一组」「标点是否单独成 token」有不同要求，规则不统一。
- 整段文本上做 BPE merge 也会很慢。

所以在跑 BPE 之前，要先用一组**正则规则**把文本切成「片段」（fragment），每个片段独立做 BPE merge，最后把 id 序列拼起来。这就是 pre-tokenizer。

### 4.2 GPT-2 风格正则的例子

GPT-2 的 pre-tokenizer 正则大致是：

```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
```

切 `"Hello, world!"` 会得到：

```
["Hello", ",", " world", "!"]
```

（`"Hello"` 是 ` ?\p{L}+` 匹配；`","` 是 ` ?[^\s\p{L}\p{N}]+`；`" world"` 是 ` ?\p{L}+` 连带前导空格；`"!"` 同 `,`。）每个片段再各自跑 BPE merge，最后拼成 id 序列。

### 4.3 llama.cpp 里那一长串 `tokenizer_pre` 分支

不同模型用的正则不同 —— LLaMA3 一套、DeepSeek 一套、Qwen 又一套。llama.cpp 加载时从 GGUF 读 `tokenizer.ggml.pre` 这个 KV 字符串，再 if-else 把它翻译成 `enum llama_vocab_pre_type`（声明在 `src/llama-vocab.h:10`），并构造对应的 `regex_exprs` 向量。

if-else 链从 `src/llama-vocab.cpp:2097` 开始（`if (tokenizer_pre.empty())`），一路 `else if (tokenizer_pre == "llama3")`、`"deepseek-llm"`、`"qwen2"`... 几十个分支，每个分支设 `pre_type` 和 `ignore_merges` 等标志。BPE 构造函数 `llm_tokenizer_bpe::llm_tokenizer_bpe`（:280 起）再按 `pre_type` 选 `regex_exprs`（switch 从 :282 开始，覆盖 LLaMA3、GPT-2、Qwen2、Tekken、GPT-4o 等）。

所以「同是 BPE，LLaMA3 和 GPT-2 不一样」的根本原因就是这：pre-tokenizer 正则不同 + `ignore_merges` 不同。第 6 节的表格保留了 `tokenizer_pre` 的意义注释。

## 5. llama.cpp 实现：tokenize 流程图 + 缓存 + detokenize

### 5.1 tokenize 流程

```
llama_tokenize(model, text, ...)
  -> llama_vocab::tokenize(text, add_special, parse_special)
        -> impl::tokenize(text, add_special, parse_special)    // src/llama-vocab.cpp:3279
              1. 把 text 包成一个 fragment_buffer
              2. tokenizer_st_partition(buffer, parse_special)  // :3116
                   按 cache_special_tokens 里的特殊 token 把文本切成
                   [RAW_TEXT | TOKEN | RAW_TEXT | TOKEN | ...]
              3. switch (get_type())                            // :3293
                   SPM  -> llm_tokenizer_spm_session::tokenize   // :117
                   BPE  -> llm_tokenizer_bpe_session::tokenize   // :597
                   WPM  -> llm_tokenizer_wpm_session::tokenize    // :765
                   UGM  -> llm_tokenizer_ugm_session::tokenize   // :964
              4. 在头部加 BOS、尾部加 EOS（按 add_bos/add_eos）
```

- **第 2 步** `tokenizer_st_partition`（:3116）按特殊 token（如 `<|im_start|>`）把文本切成片段，避免把特殊 token 当普通文本编码。
- **第 3 步 BPE 分支**（:3345）里，`llm_tokenizer_bpe_session::tokenize`（:597）先 `unicode_regex_split` 用 `regex_exprs` 把片段再切成子片段（pre-tokenize），然后对每个子片段跑 BPE merge（用 `add_new_bigram` + 优先队列，:720-744），最后查表（:692-714）输出 token。

### 5.2 缓存的作用

`impl::cache_token_to_piece`（声明 :1815）是一个长度为 `n_tokens` 的字符串数组，加载时（:2918 起）一次性把每个 token 的 `token_to_piece(id, special=true)` 结果算好存进去。运行时 `impl::token_to_piece(llama_token)`（:3602-3604）只是 `return cache_token_to_piece.at(token)`，O(1) 查表，避免每次 detokenize 都重新算 whitespace 转义之类。

`impl::cache_special_tokens`（:1814）按 token 字符串长度降序排好（:2908-2912），给 `tokenizer_st_partition` 用 —— 长的特殊 token 先匹配，避免短的特殊 token 截断长的（例如 `<|im_start|>` 不能被 `<|` 之类的截掉）。

### 5.3 detokenize 侧：`llama_token_to_piece` 与 UTF-8 边界

`llama_token_to_piece`（`src/llama-vocab.cpp:4314`）把单个 token 转成字符串片段，写入调用者提供的 `buf`。它委托给 `llama_vocab::token_to_piece`（:4058），最终走到 `impl::token_to_piece` 的 switch（:3521-3597）：

- `LLAMA_TOKEN_ATTR_NORMAL`：把 token text 里的 `▁` 还原成空格再写回（:3534、:3553）。
- `LLAMA_TOKEN_ATTR_BYTE`：取 `token_to_byte(token)`，写一个字节（:3537-3539）。
- 特殊 token：直接拷文本。

**UTF-8 边界问题**：一个中文字（如 `"中"`）在 UTF-8 里占 3 字节。BPE 训练时如果某个 token 恰好是 `"中"` 的前 2 字节，detokenize 出来就是一段**不完整的 UTF-8 字节序列**，单独打印会乱码。流式输出时尤其严重：服务器把 token 一个个回显，如果当前 token 切断了多字节字符，下个 token 才能补齐。

llama.cpp 的处理：

1. **`common_token_to_piece`**（`common/common.cpp:1692`）包装 `llama_token_to_piece`，先按默认容量试取，若返回负数（buffer 不够）则按 `-n_chars` 重新分配再取一次，保证拿完整 piece。
2. **`common_utf8_is_complete`**（`common/unicode.cpp:72`）判断一个 piece 是不是「完整的 UTF-8 字符串」。`common/reasoning-budget.cpp:87-91` 就在流式场景里用它决定是否要等下一个 token 到了再处理。
3. **服务端流式输出**（`tools/server/server-context.cpp:1851`）用 `validate_utf8(slot.generated_text) < slot.generated_text.size()` 检测末尾是否切断了多字节字符；若切断就暂时不发出，等下个 token 补齐再发。
4. **`tokens_to_output_formatted_string`**（`tools/server/server-common.cpp:1331`）处理「size==1 且首字节高位是 1」的孤立字节 token，打印成 `byte: \xXX` 而不是乱码。

### 5.4 词表类型（`llama_vocab_type`，声明 `include/llama.h:72`，`impl` 结构在 `src/llama-vocab.cpp:1765`）

| 类型 | 来源 | 说明 |
|---|---|---|
| `SPM` | SentencePiece | LLaMA1/2 等；`▁` 表示空格；llama.cpp 实现按 token score 用优先队列合并（非 Viterbi） |
| `BPE` | GPT-2 风格 | 带 pre-tokenizer，LLaMA3、Qwen、DeepSeek 等 |
| `WPM` | WordPiece | BERT 系；`##` 前缀表子词 |
| `UGM` | Unigram + Viterbi | Gemma 等；llama.cpp 真正实现 Viterbi |
| `RWKV` / `PLAMO2` | 特殊 | RWKV 走贪婪；PLaMo-2 用 Aho-Corasick + DP |

`impl::get_type()`（`src/llama-vocab.cpp:3004`）直接返回 `type` 字段（:1771）。`type` 在加载时由 `tokenizer.ggml.model` KV 决定（设置点：SPM 在 :1940、WPM 在 :1950、BPE 在 :1962、UGM 在 :2003、RWKV 在 :2034、PLAMO2 在 :2043）。

## 6. 加载与重建（`init_tokenizer`，:3083）

`llama_vocab::impl::init_tokenizer(enum llama_vocab_type type)`（`src/llama-vocab.cpp:3083`）按 type new 出对应的 tokenizer 对象（SPM -> `llm_tokenizer_spm`，BPE -> `llm_tokenizer_bpe`，等等），存进 `impl::tokenizer`（`std::unique_ptr`）。BPE 构造函数（:280）里同步把 `regex_exprs` 配好（switch 从 :282 开始），UGM 构造时会加载 `precompiled_charsmap`（用于 normalize 阶段查表）。

整个加载流程不依赖任何外部库（没有 HF tokenizers、没有 SentencePiece C++ 库），全在 `llama-vocab.cpp` 里实现，符合项目「纯 C/C++ 无依赖」目标。代价是要为每种 tokenizer 风格单独维护切分逻辑，所以 `tokenizer_pre` 分支那么多。

## 7. 与采样/回显的衔接

- 采样器（`05-精读-采样器体系.md`）产出 token **id**。
- 回显文本：`llama_token_to_piece`（:4314，id -> 文本片段写入 buf）；`common_token_to_piece`（`common/common.cpp:1692`）是它的 C++ 包装，返回 `std::string`。
- 反向：`llama_byte_to_token`（`:3818`）把单字节映射成词表里的 byte token；`text_to_token`（:3848）查整串是否在词表里。
- 服务端 `server-context.cpp` 在流式输出时用 `validate_utf8` 守护 UTF-8 边界（见 5.3）。

## 8. 为什么 llama.cpp 自带 tokenizer

- **零外部依赖**（符合项目「纯 C/C++ 无依赖」目标）。
- **GGUF 里存好 vocab + tokenizer 配置**，加载即得，无需调 HF tokenizers / SentencePiece 库。
- **代价**：需为每种 tokenizer 风格单独实现/维护切分逻辑（所以 `tokenizer_pre` 分支那么多、`enum llama_vocab_pre_type` 一直加新枚举）。

## 9. 追问方向

- `llm_tokenizer_spm_session` 的 `rev_merge` 表（:228）作用：记录「这个子词是由哪两个 symbol 合出来的」，给 `resegment` 回溯用。
- BPE pre-tokenizer 的正则模板（GPT-2 的 `gpt2` 正则 vs LLaMA3 的 `llama3` 正则）切同一段文本的差异，可以拿 `unicode_regex_split` 的输出对比。
- `llama_token_to_piece` 的 `lstrip` 参数：流式场景里把前导空格去掉对齐。
- `cache_token_to_piece` 占内存大小（加载日志 `token to piece cache size = %.4f MB`，:2931）和命中率。
