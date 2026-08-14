# 精读：对话模板与 Jinja 引擎（common/chat.cpp + common/jinja/）

> 日期：2026-08-13 | 核心文件：`common/chat.cpp`（模板封装）+ `common/jinja/`（Jinja 引擎实现）
> 前置：读过 `00`（CLI 组装 messages 后要转成模型需要的 prompt 文本）。

## 作用

把「对话消息列表（role + content）」渲染成模型期望的**单段 prompt 文本**。每个模型有自己对话格式（ChatML、Llama3、Gemma...），用 **Jinja 模板**描述。llama.cpp 反问：**自己实现了一个轻量 Jinja 引擎**，无需 Python。

## 为什么需要对话模板

模型训练时对话被格式化成特定样子（如：
```
<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>...
```
）。推理时若不以同样格式喂回，模型表现会崩。模板就是把「结构性消息」→「这个模型的格式文本」的规则。

## 两层设计

```
common/chat.cpp        高层：管理模板集合、BOS/EOS、tool 模板、fallback
  └─ common/jinja/     底层：真正的 Jinja 语言引擎（lexer/parser/runtime/value）
```

## 一、模板集合 `common_chat_templates`（chat.cpp:286）

- `template_default`：默认对话模板（无显式模板时按架构 fallback 到 chatml 等）
- `template_tool_use`：工具调用专用模板（可选）
- `add_bos`/`add_eos`：是否加特殊 token
- `has_explicit_template`：模型是否自带模板

### 初始化 `common_chat_templates_init`（:733 附近）
1. 取模板源码（GGUF 的 `tokenizer.chat_template` KV，或 `--jinja` 显式指定）
2. 从 vocab 取 BOS/EOS token 文本（若无则模板里对应变量为空）
3. `new common_chat_template(src, bos, eos)` 编译模板（失败可 `--no-jinja` 回退）

### 应用 `common_chat_templates_apply`（:2621）
```cpp
return inputs.use_jinja ? apply_jinja(...)     // 走内置 Jinja 引擎
                        : apply_legacy(...);   // 旧式硬编码模板
```

## 二、Jinja 引擎（common/jinja/）

一个自包含的 Jinja2 子集实现，四件套：

| 文件 | 作用 | 对应 Python Jinja |
|---|---|---|
| `lexer.cpp` | 把模板源码切成 token（`{{ }}`/`{% %}`/文本） | tokenize |
| `parser.cpp` | 生成 AST | parse |
| `runtime.cpp` | 解释执行 AST，输出文本 | render |
| `value.cpp` | 模板里的值类型（str/int/list/dict/bool...） | - |
| `caps.cpp` | 模板能力声明（哪些函数/过滤器可用） | - |

支持 `{{ var }}` 表达式、`{% if %}`/`{% for %}` 控制流、过滤器（`|length`）、属性访问（`message['content']`）等常用子集。

## 三、渲染输入（`chat.cpp:419` render_message_to_json）

把 `std::vector<common_chat_msg>` 转成符合 Jinja 预期的结构（`messages` 数组，每项含 `role/content/tool_calls` 等），再交给模板渲染。这保证「模型自己的模板」能直接吃 llama.cpp 的消息模型。

## 关键点

1. **零依赖**：C++ 实现 Jinja，不依赖 Python/PyJinja，符合项目无依赖目标
2. **模型自带模板优先**：`has_explicit_template` 时用 GGUF 里的模板，否则 fallback 到内置（chatml 兜底）
3. **工具模板分离**：`template_tool_use` 让工具调用可走不同格式
4. **回退路径**：Jinja 失败可 `--no-jinja` 用 legacy 硬编码模板，保证可用性

## 追问方向
- `parser.cpp` 的 AST 节点类型（如何表达 for/if/过滤器）
- `runtime.cpp` 如何求值 `message['tool_calls']` 这类嵌套访问
- `caps.cpp` 的能力检测（判断某模板能否用某项功能）