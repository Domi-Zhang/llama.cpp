# 精读：GGUF 文件格式（gguf-py）

> 日期：2026-08-13 | 核心文件：`gguf-py/gguf/`（gguf_writer.py / gguf_reader.py / constants.py）
> 2026-08-17 修订：扩充为初学者详细版（迷你 GGUF 字节布局手推）
> 前置：已读 `00`/`00a`/`08`。GGUF 是 llama.cpp 的模型容器，加载器（`src/llama-model-loader.cpp`）按同样布局读。
> 注：仓库当前没有 `docs/gguf.md`，规格以 `gguf-py/gguf/constants.py` 与 `gguf_writer.py` 为准。

## 作用

GGUF（GPT-Generated Unified Format）是 llama.cpp 自定义的**单文件二进制容器**，把「元数据（KV）+ 张量数据」塞进一个 `.gguf` 文件。llama.cpp 推理前先由 `llama_model_load_from_file` 解析它，再决定怎么构图、加载权重。

## 为什么不直接用 safetensors / ONNX / PyTorch checkpoint

会基本 Python 的同学都见过这几种格式，先对比下取舍：

- **PyTorch checkpoint（`.bin`/`.pt`）**：本质是 pickle 序列化。优点是能存任意 Python 对象；缺点是加载时要 `torch.load` 反序列化，**有代码执行风险**（pickle exploit），且依赖 PyTorch 版本。大模型用它在生产里不合适。
- **safetensors**：单文件，按"头 JSON + 张量数据连续区"布局，安全（不执行代码）、能 mmap。但元数据只能是扁平 JSON，词表（几万个字符串数组）、tokenizer 配置、RoPE 缩放参数等结构化信息要么硬塞进 JSON 字符串、要么外置成 `tokenizer.json` 等多文件。多文件分发对用户不友好。
- **ONNX**：是个标准化的计算图交换格式，但目标是跨框架推理图，自带 schema 演进包袱（每个 opset 版本都要兼容），且不擅长存"只有权重、构图在运行时做"的场景。llama.cpp 的图是自己用 ggml 搭的，不需要 ONNX 的图语义。
- **HuggingFace `pytorch_model.bin` + `config.json` + `tokenizer.json`**：多文件组合，下载/校验/分片要逐个对，传播时容易缺件。

GGUF 的设计目标就是"一个文件搞定一切"：

1. **单文件**：模型超参、tokenizer 词表、张量权重全在一个 `.gguf` 里，下载一个链接就完事。
2. **可 mmap**：张量数据区连续、对齐，操作系统按需把文件页映射进内存，**冷启动只读头**，权重按页 lazy 加载（`06`/`08` 提过）。
3. **无 schema 演进**：每个 KV 自带类型标记（`vtype`），读取端见到不认识的 key 直接按 type 长度跳过，新加字段不会让旧版本崩溃。这是和 ONNX 最大的区别——ONNX 是"图结构变更要走 opset"，GGUF 是"加个 KV 不影响老读取器"。

代价是：GGUF 是 llama.cpp 专属，工具链窄；不像 ONNX 有跨框架生态。但在 llama.cpp 的场景里这个取舍很值。

## 三段二进制布局（`gguf_writer.py`）

### 1. Header（`write_header_to_file` :214）
依次 4 个字段（`gguf_writer.py:230-233`）：
```
uint32  GGUF_MAGIC   "GGUF"（0x46554747，constants.py:10）
uint32  GGUF_VERSION （当前 = 3，constants.py:11）
uint64  张量数量 n_tensors
uint64  KV 数量     n_kv
```
共 4+4+8+8 = **20 字节**。magic 不是字符串而是 4 字节整数，按小端写出就是 `47 47 55 46`（ASCII 正好 "GGUF"），这样 `cat` 文件能看到文件头是 "GGUF"。

### 2. KV Metadata（`write_kv_data_to_file` :237）
若干「键-值」对，每个键值对（`_pack_val` :1345）：
```
uint64  key_len
bytes   key        （utf8 字节）
uint32  value_type （GGUFValueType 枚举，constants.py:4480）
bytes   value      （按 vtype 编码）
```
注意 key 本身就是个"长度 + 字节"的字符串，没有尾终止符；这和 C 字符串不同，读取端按 key_len 读完即止。值类型前缀让读取端不需预设 schema，也便于向前兼容——见到不认识的 key，按 vtype 跳过固定长度即可。

### 3. Tensor 信息 + 数据（`write_ti_data_to_file` :254 + `write_tensors_to_file` :438）
先写**所有张量的信息区**（不写数据），再写**数据区**。这样布局是为了让读取端一次性把所有 tensor info 读完后，再决定数据区怎么映射。每个张量信息（`gguf_writer.py:264-270`）：
```
uint64  name_len
bytes   name           如 "blk.0.attn_q.weight"
uint32  n_dims
uint64  shape[n_dims]  （倒序存，见下文）
uint32  type           GGMLQuantizationType（如 F32=0、Q4_0=2）
uint64  offset         本张量数据在数据区的偏移（相对数据区起点，不是文件头！）
```
数据区按 `data_alignment`（默认 32，`constants.py:12`）对齐。每个张量写完后，下一个张量的 offset 累加 `ggml_pad(nbytes, alignment)`（`gguf_writer.py:327`），保证每个张量数据起始都是 32 字节对齐。

## 为什么 shape 倒序存

GGML 张量用 `ne[0]` 作为**最快变化维度**（column-major / Fortran order，详见 `00a`）。而 PyTorch / numpy 默认 row-major，最后一维最快。所以一个 numpy shape `(2, 3)` 的张量，写到 GGUF 时要把 shape 倒过来存成 `[3, 2]`，字节流本身不变——numpy 的 row-major 字节序正好对应 GGML 的 column-major 字节序（`gguf_writer.py:268`：`ti.shape[n_dims - 1 - j]`）。读取端再 `tuple(reversed(dims))` 还原（`gguf_reader.py:330`）。

## 为什么 offset 相对数据区而非文件头

KV 区和 tensor info 区都是变长的（KV 数量、字符串长度、张量数都影响总长），如果 offset 相对文件头，每写一个新 KV 都得回去改所有张量的 offset。改成相对数据区起点（`gguf_writer.py:270`，写入时累加 `offset_tensor`），写入端只关心"我在数据区里第几个、多大"，和头长解耦。

读取端先扫完 KV 和 tensor info 算出数据区起点 `data_offset`（`gguf_reader.py:181-184`：`padding = offs % alignment; offs += alignment - padding; self.data_offset = offs`），再 `data_offs = start_offs + offset_tensor[0]`（`gguf_reader.py:333`）拿到绝对偏移。`llama_model_loader` mmap 时把整个文件映射进来，`cur->data = (uint8_t *)mapping->addr() + w.offs`（`llama-model-loader.cpp:1391`）——`w.offs` 是绝对文件偏移，所以数据区一次映射就能按 offset 直接拿指针。

## 为什么 `data_alignment = 32`

SIMD 指令（AVX2/AVX-512/NEON）和 GPU 拷贝通常要求 16/32/64 字节对齐才不走慢路径。32 是个稳妥默认值：满足 AVX2（32 字节）、NEON（16 字节向下兼容）、绝大多数 GPU host→device 拷贝的对齐要求。用户也能用 `general.alignment` KV 改（`gguf_writer.py:505` `add_custom_alignment`，要求非零 2 的幂），读取端在 `gguf_reader.py:173-180` 读到这个 KV 后覆盖默认值。

## KV 类型系统：`GGUFValueType`（`constants.py:4480`）

| 枚举值 | 名称 | 字节数 | 说明 |
|--------|------|--------|------|
| 0 | UINT8 | 1 | |
| 1 | INT8 | 1 | |
| 2 | UINT16 | 2 | |
| 3 | INT16 | 2 | |
| 4 | UINT32 | 4 | 模型超参最常用 |
| 5 | INT32 | 4 | |
| 6 | FLOAT32 | 4 | RoPE freq_base、eps 等 |
| 7 | BOOL | 1 | `use_parallel_residual` 等 |
| 8 | STRING | 变长 | `uint64 len + utf8 bytes` |
| 9 | ARRAY | 变长 | 嵌套，见下 |
| 10 | UINT64 | 8 | shape 值用 |
| 11 | INT64 | 8 | |
| 12 | FLOAT64 | 8 | |

标量类型在 `_simple_value_packing`（`gguf_writer.py:72`）里用 `struct` 格式符映射；STRING 是 `uint64 len + bytes`。

**ARRAY 类型的嵌套**（`gguf_writer.py:1358-1377`）：
```
uint32  ARRAY      (vtype 本身)
uint32  ltype      (子元素类型，必须是标量/STRING，不能再嵌套 ARRAY)
uint64  len        (元素个数)
bytes   items[len] (每个元素按 ltype 编码，不再写 vtype 前缀)
```
典型例子：tokenizer 词表 `tokenizer.ggml.tokens`（`Keys.Tokenizer.LIST`，`constants.py:252`）就是 **ARRAY of STRING**——文件里写下 `ltype=8(STRING), len=32000`，接着 32000 个 `uint64+bytes` 字符串。`gguf_reader.py:239-255` 的 ARRAY 分支递归调用 `_get_field_parts` 把每个元素读出来，最终 `ReaderField.contents()` 返回字符串列表。

## 量化类型（`constants.py:4382` GGMLQuantizationType）

GGUF 里张量类型直接对应 `ggml` 的量化类型（`06` 的 block 格式）。`GGML_QUANT_SIZES` 表（`constants.py:4561`）给出每个类型每 block 的 `(block_size, type_size)`，用于计算张量字节数与偏移。例如 `F32: (1, 4)`、`Q4_0: (32, 18)`——32 个元素压成 18 字节（1 个 f16 scale + 16 个 4-bit pack）。

## 手推迷你 GGUF 字节布局

构造一个最小可解析的 GGUF：1 个 KV（`"llama.context_length" = 4096`，UINT32）+ 1 个张量（2×3 f32，名字 `"blk.0.attn_q.weight"`）。默认 `data_alignment = 32`，GGUF v3，小端。

设张量数据为 `[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]`（numpy C-order，shape (2,3)）。

### 段 1: Header（20 B，偏移 0–19）
```
偏移  字节                         含义
0x00  47 47 55 46                  magic "GGUF" (0x46554747 LE)
0x04  03 00 00 00                  version = 3
0x08  01 00 00 00 00 00 00 00      n_tensors = 1
0x10  01 00 00 00 00 00 00 00      n_kv = 1
```

### 段 2: KV 区（36 B，偏移 20–55）
key `"llama.context_length"` 共 20 个 ASCII 字符；vtype=4（UINT32）；value=4096=0x1000。
```
偏移  字节                         含义
0x14  14 00 00 00 00 00 00 00      key_len = 20
0x1C  6C 6C 61 6D 61 2E 63 6F     "llama.co"
      6E 74 65 78 74 5F 6C 65     "ntext_le"
      6E 67 74 68                  "ngth"     (共 20 B)
0x30  04 00 00 00                  vtype = 4 (UINT32)
0x34  00 10 00 00                  value = 4096 (0x1000 LE)
```
小计：8 + 20 + 4 + 4 = 36 B。

### 段 3: Tensor 信息区（60 B，偏移 56–115）
name `"blk.0.attn_q.weight"` 20 字符；n_dims=2；shape 倒序存为 `[3, 2]`（因 numpy shape (2,3) 倒过来）；type=0（F32）；offset=0（数据区里第一个）。
```
偏移  字节                         含义
0x38  14 00 00 00 00 00 00 00      name_len = 20
0x40  62 6C 6B 2E 30 2E 61 74     "blk.0.at"
      74 6E 5F 71 2E 77 65 69     "tn_q.wei"
      67 68 74                      "ght"      (共 20 B)
0x54  02 00 00 00                  n_dims = 2
0x58  03 00 00 00 00 00 00 00      shape[0] = 3  (numpy 最后一维)
0x60  02 00 00 00 00 00 00 00      shape[1] = 2  (numpy 第一维)
0x68  00 00 00 00                  type = 0 (F32)
0x6C  00 00 00 00 00 00 00 00      offset = 0  (相对数据区)
```
小计：8 + 20 + 4 + 8 + 8 + 4 + 8 = 60 B。

### 段 4: 对齐 padding（12 B，偏移 116–127）
当前偏移 = 20 + 36 + 60 = **116**。116 % 32 = 20，需补 32 - 20 = **12 字节 0**，让数据区起点对齐到 32 的倍数（128）。
```
0x74  00 00 00 00 00 00 00 00 00 00 00 00   (12 B padding)
```
**data_offset = 128（0x80）**。

### 段 5: 数据区（24 B，偏移 128–151）
2×3 f32 = 6 个 float = 24 字节。numpy C-order 字节序正好对应 GGML column-major：`1.0, 2.0, 3.0, 4.0, 5.0, 6.0`。
```
0x80  00 00 80 3F   1.0f (0x3F800000 LE)
0x84  00 00 00 40   2.0f (0x40000000 LE)
0x88  00 00 40 40   3.0f (0x40400000 LE)
0x8C  00 00 80 40   4.0f (0x40800000 LE)
0x90  00 00 A0 40   5.0f (0x40A00000 LE)
0x94  00 00 C0 40   6.0f (0x40C00000 LE)
```

### 总览
```
[0     .. 19  ] Header        20 B
[20    .. 55  ] KV 区          36 B  (1 个 KV)
[56    .. 115 ] Tensor info    60 B  (1 个张量)
[116   .. 127 ] padding        12 B  (对齐到 32)
[128   .. 151 ] 数据区         24 B  (6 个 f32)
总文件大小 = 152 字节
```
读取端拿到的就是这个布局；`data_offset=128`，第一个张量 `data_offs = 128 + 0 = 128`，读 24 字节解释成 shape `(2,3)` 的 f32。

## 读取端（`gguf_reader.py` + `llama-model-loader.cpp`）

### `gguf_reader.py`：惰性读取
`GGUFReader.__init__`（`gguf_reader.py:132`）第一步 `np.memmap(path, mode='r')`（:133），把整个文件 mmap 进 numpy 数组，**不真正读盘**。之后所有 `_get(offset, dtype, count)`（:197）只是返回 memmap 的 view，操作系统按页 demand-load。

读取流程（`__init__` :132-185）：
1. 校验 magic（:137）、版本（:142，支持 v2 和 v3，`READER_SUPPORTED_VERSIONS = [2, GGUF_VERSION]`）；
2. 读 `tensor_count` / `kv_count`（:165）；
3. `_build_fields`（:289）逐个解析 KV——key 长度、key 字节、vtype、value。ARRAY 用 `_get_field_parts`（:221）递归把每个元素加进 `parts` 列表；
4. `_build_tensor_info`（:310）解析每个张量的 name/n_dims/shape/type/offset；
5. 计算 padding 与 `data_offset`（:181-184）；
6. `_build_tensors`（:318）构造 `ReaderTensor`，`data = self._get(data_offs, item_type, item_count).reshape(np_dims)`（:367）——仍是 memmap view，**没拷贝**。

`ReaderField`（:39）记录每个字段的偏移和 parts，`ReaderTensor`（:100）记录张量元信息加 data view。这俩 namedtuple 就是惰性读取的载体。

### `llama-model-loader.cpp`：按名字找张量 + mmap 取指针
C++ 端流程对应：
1. `gguf_init_from_file`（`llama-model-loader.cpp:547`）读 header + KV + tensor info，构造 `ggml_context` 里一堆 `ggml_tensor`（**只填元信息，不分配数据**）；
2. 遍历所有 tensor，把 `{name → (file_idx, offs, ggml_tensor*)}` 塞进 `weights_map`（:584，:650，:694）；
3. `init_mappings()` 里 `std::make_unique<llama_mmap>(file.get(), ...)`（:1351）把整个文件 mmap 进虚拟地址空间；
4. `load_data_for(cur)`（:1385）按 tensor name 找到 `w`，如果 `use_mmap`：`cur->data = (uint8_t *)mapping->addr() + w.offs`（:1391）——**直接把指针指向 mmap 区域，零拷贝**；不用 mmap 时才 `file->seek + read_raw` 拷贝（:1399-1400）。

所以"加载模型"实际上是：读头→建索引→mmap→给每个 ggml_tensor.data 赋指针。真正的页故障发生在第一次访问张量数据时，由 OS 按需读盘。`ml.get_key(LLM_KV_*)`（`llama-model-loader.cpp:400`）就是按标准 KV 键名（如 `llama.context_length`）从 `metadata` 里取元数据。

## 版本兼容策略

- **version 字段**（`constants.py:11`，当前 3）：读取端 `READER_SUPPORTED_VERSIONS = [2, 3]`（`gguf_reader.py:36`），不认识的版本直接报错（:149-150）。新版本要走不兼容改动才升。
- **未知 KV 跳过**：读取端见到不认识的 key 怎么知道这个 KV 占多长？答案是 vtype 是定长可跳过的——标量类型有固定字节数（UINT32=4、FLOAT32=4 等），STRING 是 `uint64 len + len 字节`，ARRAY 是 `uint32 ltype + uint64 len + len 个 ltype 元素`。`_get_field_parts`（`gguf_reader.py:221`）按 vtype 分支计算出本 KV 总长，offset 推进到下一个 KV 起点即可，不认识的 key 也能精确跳过。
- **加新 KV 不破坏老读取器**：这是 GGUF 设计的核心。新加字段如 `llama.attention.key_length_mla`（`constants.py:185`），老版本读取器不认识就跳过，按默认值或缺失处理；新版本读取器认识就用。
- **量化类型扩展**：`GGMLQuantizationType` 不断加新值（如 NVFP4=40、Q1_0=41），老读取器见到不认识的 type 会报错，但不会破坏文件其他部分——这是有限的前向兼容。

## 为什么 GGUF 是单文件友好的（小结）

- 头 + KV 是**可变长的键值**，模型超参/词表/自定义元数据都塞在 KV 里，无需固定 schema；
- 张量信息区集中、数据区连续且对齐，天然支持 mmap 懒加载（`06`/`08` 提过）；
- 版本号 + 值类型前缀让读取端优雅处理新旧版本，加字段不破坏老读取器；
- 单文件分发：一个 `.gguf` 就是全部，校验一个哈希即可。

## 追问方向

- `gguf_reader.py` 的 `ReaderField.contents`（:57）如何处理 ARRAY of STRING 的切片访问；
- `quants.py` 的 `quant_shape_to_byte_shape` / `quant_shape_from_byte_shape`：量化形状↔字节形状换算（block_size 折叠进最后一维）；
- 分片（multi-shard）GGUF：`SHARD_NAME_FORMAT`（`gguf_writer.py:38`）、`Split` KV（`add_shard_kv_data` :201）与 `llama-model-loader.cpp:590-665` 的多文件加载、跨文件 offset 如何统一进 `weights_map`。
