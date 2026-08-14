# 精读：Grammar 采样约束与 PEG 解析（llama-grammar.cpp + common/peg-parser）

> 日期：2026-08-13 | 核心文件：`src/llama-grammar.cpp`（GBNF 约束采样）+ `common/peg-parser.cpp/h`（PEG 解析）
> 前置：读过 `05`（grammar 是采样链的一节）。文档：`docs/development/parsing.md`、`docs/autoparser.md`。
> 作用：让模型输出强制符合某语法（JSON schema、代码、固定格式），是 llama.cpp 结构化输出的基础。

## 两种机制

| 机制 | 位置 | 用途 |
|---|---|---|
| **GBNF grammar** | `src/llama-grammar.cpp` | 训练时**采样**强制约束输出 token |
| **PEG 解析** | `common/peg-parser.cpp` | 解析已生成文本，构建 AST（higher-level，可转成 grammar） |

关系：PEG 是「解析器」，grammar 是「约束采样器」。llama.cpp 用 PEG 描述更复杂的格式，再转成 GBNF 去约束采样。

## 一、GBNF grammar 约束采样（src/llama-grammar.cpp）

### 语法解析（`llama_grammar_parser`)
- `parse(src)`（:687）把 GBNF 文本解析成 `llama_grammar_rules`
- 规则由 `llama_grammar_rule`（一系列 `llama_gretoken`）组成，token 类型：`CHAR`/`CHAR_NOT`/`CHAR_RANGE`/`RULE_REF`/`END` 等
- `parse_alternates`/`parse_sequence`/`parse_rule`（:434/:451/:663）递归下降

### 运行时状态：栈式 VM
`llama_grammar` 持有一组 `stacks`（`:stacks`），每个 stack 是「当前可匹配的规则位置」的活动集合。生成时维护栈，表示语法当前可能所处的状态。

### 采样约束（核心）
`llama_grammar_sample` 对**每个候选 token**：
1. 把 token 的文本片段虚拟地喂进当前所有栈
2. `llama_grammar_reject_candidates`（:936）对每个 stack 调 `reject_candidates_for_stack`，**剔除**会让语法卡死的 token
3. 产出「被允许的候选子集」，采样器在其间选

> 关键：**按 token 过滤而非字符**。候选 token 的字节序列逐一试探，只有「能延续当前语法状态」的 token 才保留。这就是为什么 `grammar` 采样器能强制 JSON 合法——非法 token 直接被拒。

### 接受（`llama_grammar_accept`，:1042 / `accept_token` :1456）
选中 token 后，把它的文本真正推进语法栈，更新状态（供下一个 token 约束）。

### 左递归检测（:955）
预处理阶段检测左递归规则，避免无限递归（`llama_grammar_detect_left_recursion`）。

## 二、PEG 解析（common/peg-parser）

**Builder 模式**：用 C++ 运算符组合语法，`operator+`=序列、`operator|`=选择、`<<`=空格分隔序列：
```cpp
common_peg_parser integer = peg_parser_builder.build_integer();
common_peg_parser json = object | array | string | number | ...;
```
- 生成 AST（`common_peg_ast_id`），供上层（如 `chat-auto-parser`）分析
- 可把 PEG 语法**编译成 grammar**（`common_grammar_builder`），再走 GBNF 采样约束

### 与 GBNF 的分工
- **GBNF**：离线定义好、直接约束采样的格式
- **PEG**：定义更丰富/可组合的语法，运行时解析文本，必要时转成 GBNF 约束后续生成

## 三、在采样链中的位置（对照 `05`）

`llama_sampler_grammar`（`05` 里表格）封装了这套逻辑：
- apply：调 `llama_grammar_sample` 过滤候选
- accept：调 `llama_grammar_accept` 推进状态
- reset：新序列时清空栈

## 追问方向
- `reject_candidates_for_stack` 的具体栈推进算法（`llama-grammar.cpp`）
- `common_grammar_builder` 如何把 PEG 转成 GBNF（`common/`）
- JSON schema -> grammar 的转换（`common/json-schema-to-grammar.cpp`，`examples/json_schema_to_grammar.py`）