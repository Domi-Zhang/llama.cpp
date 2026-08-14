# llama.cpp 学习备忘录索引

> 本文件夹存放学习 llama.cpp 的走读/精读备忘录，按数字前缀排序。
> 00 = 主流程（第一遍必读），01+ = 精读专题。

## 文档列表

- [00-主流程-推理链路.md](00-%E4%B8%BB%E6%B5%81%E7%A8%8B-%E6%8E%A8%E7%90%86%E9%93%BE%E8%B7%AF.md) - 第一遍：完整推理链路 + 调用关系图（含每步输入/产出）
- [01-精读-LLaMA解码器构图.md](01-%E7%B2%BE%E8%AF%BB-LLaMA%E8%A7%A3%E7%A0%81%E5%99%A8%E6%9E%84%E5%9B%BE.md) - 精读：build_graph 内部，一层 transformer 的逐层构图
- [02-精读-Attention构图.md](02-%E7%B2%BE%E8%AF%BB-Attention%E6%9E%84%E5%9B%BE.md) - 精读：build_attn_mha，Flash-Attention vs 经典路径
- [03-精读-MoE-FFN构图.md](03-%E7%B2%BE%E8%AF%BB-MoE-FFN%E6%9E%84%E5%9B%BE.md) - 精读：build_moe_ffn，路由器 + top-k 专家选择
- [04-精读-KV-cache管理.md](04-%E7%B2%BE%E8%AF%BB-KV-cache%E7%AE%A1%E7%90%86.md) - 精读：llama-kv-cache.cpp，分配/读写/回收
- [05-精读-采样器体系.md](05-%E7%B2%BE%E8%AF%BB-%E9%87%87%E6%A0%B7%E5%99%A8%E4%BD%93%E7%B3%BB.md) - 精读：llama_sampler_i 接口 + 链式组合
- [06-精读-量化系统.md](06-%E7%B2%BE%E8%AF%BB-%E9%87%8F%E5%8C%96%E7%B3%BB%E7%BB%9F.md) - 精读：block 格式 + 量化算法 + 类型编排
- [07-精读-GGML后端调度.md](07-%E7%B2%BE%E8%AF%BB-GGML%E5%90%8E%E7%AB%AF%E8%B0%83%E5%BA%A6.md) - 精读：图切 split + 后端执行 + 显存复用
- [08-精读-模型转换链.md](08-%E7%B2%BE%E8%AF%BB-%E6%A8%A1%E5%9E%8B%E8%BD%AC%E6%8D%A2%E9%93%BE.md) - 精读：HF -> GGUF 转换流程与三段结构
- [09-精读-CUDA后端内核.md](09-%E7%B2%BE%E8%AF%BB-CUDA%E5%90%8E%E7%AB%AF%E5%86%85%E6%A0%B8.md) - 精读：ggml-cuda，matmul 四策略 + mmq + 图融合
- [10-精读-GGUF文件格式.md](10-%E7%B2%BE%E8%AF%BB-GGUF%E6%96%87%E4%BB%B6%E6%A0%BC%E5%BC%8F.md) - 精读：gguf-py，三段二进制布局 + 读取
- [11-精读-词表与Tokenizer.md](11-%E7%B2%BE%E8%AF%BB-%E8%AF%8D%E8%A1%A8%E4%B8%8ETokenizer.md) - 精读：llama-vocab.cpp，SPM/BPE/WPM 与 tokenize
- [12-精读-llama-server架构.md](12-%E7%B2%BE%E8%AF%BB-llama-server%E6%9E%B6%E6%9E%84.md) - 精读：HTTP/任务队列/槽位，并发推理引擎
- [13-精读-对话模板与Jinja引擎.md](13-%E7%B2%BE%E8%AF%BB-%E5%AF%B9%E8%AF%9D%E6%A8%A1%E6%9D%BF%E4%B8%8EJinja%E5%BC%95%E6%93%8E.md) - 精读：common/jinja，自研 Jinja 渲染对话
- [14-精读-grammar与PEG解析.md](14-%E7%B2%BE%E8%AF%BB-grammar%E4%B8%8EPEG%E8%A7%A3%E6%9E%90.md) - 精读：GBNF 约束采样 + PEG 解析
- [开发手册-构建与运行.md](%E5%BC%80%E5%8F%91%E6%89%8B%E5%86%8C-%E6%9E%84%E5%BB%BA%E4%B8%8E%E8%BF%90%E8%A1%8C.md) - 实操：构建/运行 CLI 与 server/常见调试

## 待写专题（计划）
- [ ] 15-多模态 (src/models/mtmd，vision/audio 输入)
- [ ] 16-投机解码 speculative (common/speculative.cpp)
- [ ] 17-嵌入与重排序模型 (embedding/rerank 推理路径)

## 阅读约定
- 每个文件顶部写：学习目标 / 日期 / 状态
- 优先看 .h 接口，再进 .cpp 实现
- 读不懂的算子先标记，不阻塞主线