# 精读：GGUF 文件格式（gguf-py）

> 日期：2026-08-13 | 核心文件：`gguf-py/gguf/`（gguf_writer.py / gguf_reader.py / constants.py）
> 前置：已读 `08`（转换链）。GGUF 是 llama.cpp 的模型容器，加载器（`src/llama-model-loader.cpp`）按同样布局读。
> 规格文档：`docs/gguf.md`

## 作用

单文件二进制容器，存「元数据（KV）+ 张量数据」。llama.cpp 推理前先由 `llama_model_load_from_file` 解析它。

## 三段二进制布局（`gguf_writer.py`）

### 1. Header（`write_header_to_file` :214）
依次 4 个字段：
```
uint32  GGUF_MAGIC   "GGUF"（0x46554747）
uint32  GGUF_VERSION
uint64  张量数量 n_tensors
uint64  KV 数量     n_kv
```

### 2. KV Metadata（`write_kv_data_to_file` :237）
若干「键-值」对，每个键值对：
```
string  key        （长度 + utf8 字节）
vtype   value_type （值的类型标记）
value              （按 vtype 编码）
```
`GGUFValueType` 枚举：`UINT8/INT8/.../STRING/ARRAY` 等。值类型前缀让读取端不需预设 schema，也便于向前兼容。

### 3. Tensor 信息 + 数据（`write_ti_data_to_file` :254 + `write_tensors_to_file` :438）
先写**所有张量的信息区**（不写数据），再写**数据区**：
```
张量信息（每个张量）：
  string name           如 "blk.0.attn_q.weight"
  uint32 n_dims
  uint64 shape[n_dims]   （倒序，column-major）
  uint32 type            GGMLQuantizationType（如 Q4_0）
  uint64 offset          本张量数据在数据区的偏移
数据区：
  按 offset 依次写每个张量的原始字节（f16/bf16/量化块）
```
张量数据按 `data_alignment` 对齐（默认 32 字节），便于后端对齐读取。

## 量化类型（`constants.py` GGMLQuantizationType）

GGUF 里张量类型直接对应 `ggml` 的量化类型（`06` 的 block 格式）。`GGML_QUANT_SIZES` 表给出每个类型每 block 的字节数，用于计算张量大小与偏移。

## 读取端（`gguf_reader.py`）

- `ReaderField` 描述每个字段的偏移/数据，支持**惰性读取**
- 与 `src/llama-model-loader.cpp` 的 `llama_model_loader` 对应：加载器按 name 逐个找张量，读类型/形状/偏移，再 mmap 映射数据
- `llama-model-loader.cpp` 里 `ml.get_key(LLM_KV_*)` 就是按标准 KV 键名取元数据

## 为什么 GGUF 是单文件友好的

- 头 + KV 是**可变长的键值**，模型超参/词表/自定义元数据都塞在 KV 里，无需固定 schema
- 张量信息区集中、数据区连续，天然支持 mmap 懒加载（`06/08` 提过）
- 版本号 + 值类型前缀让读取端优雅处理新旧版本

## 追问方向
- `gguf_reader.py` 的惰性读取 `LazyTensor` 实现
- `quants.py` 的 `quant_shape_to_byte_shape`（量化形状换算）
- 分片（multi-shard）GGUF 的 offset 如何跨文件排布