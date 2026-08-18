# 精读：对话模板与 Jinja 引擎（common/chat.cpp + common/jinja/）

> 日期：2026-08-13 | 核心文件：`common/chat.cpp`（模板封装）+ `common/jinja/`（Jinja 引擎实现）
> 2026-08-17 修订：扩充为初学者详细版（模板渲染走查、Jinja 语法入门、引擎分层等）
> 前置：读过 `00`（CLI 组装 messages 后要转成模型需要的 prompt 文本）。

## 作用

把「对话消息列表（role + content）」渲染成模型期望的**单段 prompt 文本**。每个模型有自己对话格式（ChatML、Llama3、Gemma...），用 **Jinja 模板**描述。llama.cpp 反问：**自己实现了一个轻量 Jinja 引擎**，无需 Python。

## 为什么训练格式必须推理复现

模型不是天生会聊天，是在「特定对话格式语料」上微调出来的。训练时每条样本长这样（以 ChatML 风格为例）：

```
<|im_start|>system
你是一个助手<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么我可以帮你的吗？<|im_end|>
```

模型学会的其实是「看到 `<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n` 这种上下文，就开始说人话」。如果推理时换种喂法，模型就懵了：

- **正确格式**喂 `...<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n`，模型续写「你好！有什么我可以帮你的吗？」，自然停在外部 EOS。
- **错误格式**（比如直接把所有 content 拼起来）喂 `你是一个助手你好`，模型没见过这种"光秃秃文本"的训练分布，输出常常是：复读输入、胡言乱语、或者根本不知道何时停（一直生成到 ctx 满）。

直觉：模型是「条件概率机器」，它的条件是「上文长这个样子」。模板就是把推理时的上文塑形回训练时的样子。这就是为什么 HF 上每个 instruct 模型都附带一份 `chat_template.jinja`——它定义了"这个模型的对话长什么样"。

## 为什么需要对话模板

模型训练时对话被格式化成特定样子（如：
```
<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>...
```
）。推理时若不以同样格式喂回，模型表现会崩。模板就是把「结构性消息」->「这个模型的格式文本」的规则。

## Jinja 语法 5 分钟入门

Jinja2 是 Python 生态的模板语言，常见于 Flask/Django 网页渲染。HF 模型借它来描述对话格式。核心只有 4 个语法点：

| 语法 | 含义 | 例子 |
|---|---|---|
| `{{ 表达式 }}` | 把表达式的值转成字符串拼到输出 | `{{ message.role }}` 输出 `user` |
| `{% 语句 %}` | 控制流（if/for/set/macro），不直接产出文本 | `{% if add_generation_prompt %}...{% endif %}` |
| `|` 过滤器 | 把左侧值传给右侧函数处理 | `message.content | trim` 去首尾空白 |
| `.` / `[ ]` 访问 | 取对象属性或数组下标 | `message.role` / `messages[0]` |

另外有两个常用约定：
- `{{- ... -}}` 中的 `-` 表示吃掉一侧的空白（`{{-` 吃左侧空白，`-}}` 吃右侧）。模板里到处都是换行和缩进，不加 `-` 输出会塞满空行。
- `{# 注释 #}` 是注释，不产出任何输出。

### 一段真实风格的 chatml 模板（简化版）

下面是 llama.cpp 内置的 CHATML 兜底模板（`common/chat.cpp:606` `CHATML_TEMPLATE_SRC` 宏）的近似版，逐行注释：

```jinja
{%- for message in messages -%}              遍历 messages 数组，{%- -%} 吃掉本行两端空白
  {{- '<|im_start|>' + message.role + '\n'   拼出 "<|im_start|>system\n" 这样的开头
      + message.content + '<|im_end|>\n' -}} 再拼上正文和结尾标记
{%- endfor -%}                               for 循环结束
{%- if add_generation_prompt -%}             若调用方要求"开始助手回复"
  {{- '<|im_start|>assistant\n' -}}          拼出 assistant 段的开头，等模型续写
{%- endif -%}                                if 结束
```

要点：
- `messages`、`add_generation_prompt` 是模板的**输入变量**，由 llama.cpp 在渲染时通过 `jinja::context` 注入（见后文 `common_chat_template_direct_apply_impl`，`common/chat.cpp:839`）。
- `+` 在 Jinja 里既做数字加也做字符串拼，这里全是字符串。
- `{%- ... -%}` 严格控制空白，保证输出里没有多余空行/缩进——这点对 token 化很关键，多一个空格可能改变 token 边界。

## 渲染实例走查

输入：

```json
{
  "messages": [
    {"role": "system",    "content": "你是一个助手"},
    {"role": "user",      "content": "你好"}
  ],
  "add_generation_prompt": true
}
```

模板就用上面那段 chatml。一步步手推：

**第 1 步：进入 `for message in messages` 循环**

迭代 1，`message = {"role": "system", "content": "你是一个助手"}`：
- `{{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}`
- = `'<|im_start|>' + 'system' + '\n' + '你是一个助手' + '<|im_end|>\n'`
- = `<|im_start|>system\n你是一个助手<|im_end|>\n`

迭代 2，`message = {"role": "user", "content": "你好"}`：
- = `'<|im_start|>' + 'user' + '\n' + '你好' + '<|im_end|>\n'`
- = `<|im_start|>user\n你好<|im_end|>\n`

`for` 循环结束后，到目前为止的输出：
```
<|im_start|>system
你是一个助手<|im_end|>
<|im_start|>user
你好<|im_end|>
```

**第 2 步：`if add_generation_prompt`**

`add_generation_prompt = true`，进入 if 体：
- `{{ '<|im_start|>assistant\n' }}` -> `<|im_start|>assistant\n`

**最终输出**（拼起来）：
```
<|im_start|>system
你是一个助手<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
```

最后那个 `<|im_start|>assistant\n` 就是"留个头"让模型从这里开始续写——它续出来的内容会被外层 PEG 解析器按 assistant 段格式拆回 `common_chat_msg`（见笔记 14）。

> 注意：每行末尾的换行符都是模板里**显式**写的 `\n`，不是 Jinja 自己加的。模板里 `{%- -%}` 把行间空白全吃掉了，所以输出里看不到模板自身的换行/缩进。这是写 chat template 的常见手法。

## 两层设计

```
common/chat.cpp        高层：管理模板集合、BOS/EOS、tool 模板、fallback
  └─ common/jinja/     底层：真正的 Jinja 语言引擎（lexer/parser/runtime/value）
```

## 一、模板集合 `common_chat_templates`（chat.cpp:282）

`common/chat_templates` 结构（`common/chat.cpp:282`）：

- `template_default`：默认对话模板（无显式模板时按架构 fallback 到 chatml 等）
- `template_tool_use`：工具调用专用模板（可选）
- `add_bos`/`add_eos`：是否加特殊 token
- `has_explicit_template`：模型是否自带模板

### 初始化 `common_chat_templates_init`（chat.cpp:655）

`common_chat_templates_init`（`common/chat.cpp:655`）流程：

1. 取模板源码（GGUF 的 `tokenizer.chat_template` KV，或 `--jinja` 显式指定）
2. 从 vocab 取 BOS/EOS token 文本（若无则模板里对应变量为空）
3. `new common_chat_template(src, bos, eos)` 编译模板（失败可 `--no-jinja` 回退）

`common_chat_template` 构造函数（`common/chat.h:59`）做三件事：
```cpp
jinja::lexer lexer;
auto lexer_res = lexer.tokenize(src);        // 1. 词法分析
this->prog = jinja::parse_from_tokens(lexer_res);  // 2. 语法分析，得到 AST
// ...
this->caps = jinja::caps_get(prog);          // 3. 能力探测
```

注意：**模板编译只做一次**，编译出的 AST（`jinja::program prog`）存在 `common_chat_template` 对象里，之后每次渲染都复用这份 AST。这是引擎分层的关键收益（见下文）。

### 应用 `common_chat_templates_apply`（chat.cpp:2621）

```cpp
return inputs.use_jinja ? apply_jinja(...)     // 走内置 Jinja 引擎
                        : apply_legacy(...);   // 旧式硬编码模板
```

入口在 `common/chat.cpp:2621`，二选一分支在 2624-2625 行。`use_jinja` 默认 `true`（`common/chat.h:195`），用 `--no-jinja` 关掉。

## 二、Jinja 引擎（common/jinja/）

一个自包含的 Jinja2 子集实现，五件套：

| 文件 | 作用 | 对应 Python Jinja |
|---|---|---|
| `lexer.cpp` | 把模板源码切成 token（`{{ }}`/`{% %}`/文本） | tokenize |
| `parser.cpp` | 生成 AST | parse |
| `runtime.cpp` | 解释执行 AST，输出文本 | render |
| `value.cpp` | 模板里的值类型（str/int/list/dict/bool...） | - |
| `caps.cpp` | 模板能力声明（哪些函数/过滤器可用） | - |

支持 `{{ var }}` 表达式、`{% if %}`/`{% for %}` 控制流、过滤器（`|length`）、属性访问（`message['content']`）等常用子集。

### 2.1 为什么 lexer -> parser -> runtime 三段分层？

经典编译器三段式，对应三个阶段：

**lexer（`common/jinja/lexer.cpp:32` `tokenize`）**：把字符流变成 token 流。比如 `<|im_start|>{{ message.role }}` 被切成：
```
text("<|im_start|>")
open_expression("{{")
identifier("message")
dot(".")
identifier("role")
close_expression("}}")
```
token 类型定义在 `common/jinja/lexer.h:13`，覆盖 `text`/`identifier`/`open_statement`/`pipe` 等约 20 种。lexer 还处理空白控制（`{%- -%}`），见 `lexer.cpp:115-118` 默认开启 `lstrip_blocks` 和 `trim_blocks`。

**parser（`common/jinja/parser.cpp` `parse_from_tokens`）**：吃 token 流，按文法生成 AST。AST 节点定义在 `common/jinja/runtime.h`：
- 语句节点：`program`（146 行）、`if_statement`（157 行）、`for_statement`（178 行）、`set_statement`、`macro_statement`
- 表达式节点：`identifier`（310 行）、`member_expression`（280 行，对应 `a.b` / `a[b]`）、`filter_expression`（414 行，对应 `x | filter`）、`binary_expression`、`call_expression`

`for` 的解析见 `parser.cpp:295`，`if` 的解析见 `parser.cpp:241`，递归下降，运算符优先级从低到高：`or` -> `and` -> `not` -> 比较 -> `+/-` -> `*///%` -> `is` -> `|` -> `.`/`[]`。

**runtime（`common/jinja/runtime.cpp`）**：拿 AST + 上下文变量，递归执行每个节点的 `execute_impl(ctx)`，最终把每个表达式的值拼接成字符串。`if_statement::execute_impl` 在 `runtime.cpp:449`，`for_statement::execute_impl` 在 `runtime.cpp:470`，`member_expression::execute_impl` 在 `runtime.cpp:798`。

**为什么这样分层？** 关键收益是**编译一次，多次渲染**：
- 模板源码在 `common_chat_template` 构造时编译成 AST（`common/chat.h:62`），存进 `prog` 字段。
- 每次用户对话调用 `common_chat_template_direct_apply_impl`（`common/chat.cpp:831`），只走第 868 行的 `runtime.execute(tmpl.prog)`，不重新 lex/parse。
- 这跟 Python 里 Jinja2 一样：模板字符串进 `Environment.from_string` 编译一次，之后 `template.render(**ctx)` 只走 AST。聊天场景每条消息都要渲染一次，省掉重复解析很值。

附带好处：错误信息可以指回源码位置。每个 AST 节点都存了 `pos`（`runtime.h:106`），运行时出错会带"line X, column Y"（`runtime.cpp:33-45` 的 `get_line_col`），方便定位是模板哪一行炸了。

### 2.2 value.cpp 的动态类型为什么需要？

Jinja 模板里的值类型在**写模板时不确定**，得运行时检查。比如 `message.content`：
- 有时是字符串（`"你好"`）
- 有时是数组（多模态消息：`[{"type":"text","text":"看这张图"}, {"type":"image",...}]`）
- 有时是 `None`（工具调用消息常把 `content` 设为空）

C++ 是静态类型，要在运行时支持这种"随便装"的值，就得有动态类型系统。`value.cpp`（声明在 `common/jinja/value.h`）的做法是：基类 `value_t`（`value.h:106`）+ 一组派生类：

| 类型 | 类 | 头文件行 |
|---|---|---|
| 整数 | `value_int_t` | `value.h:214` |
| 浮点 | `value_float_t` | `value.h:248` |
| 字符串 | `value_string_t` | `value.h:290` |
| 布尔 | `value_bool_t` | `value.h:326` |
| 数组 | `value_array_t` | `value.h:354` |
| 对象（dict） | `value_object_t` | `value.h:483` |
| None | `value_none_t` | `value.h:602` |
| Undefined | `value_undefined_t` | `value.h:620` |
| 函数 | `value_func_t` | `value.h:697` |

所有值都用 `std::shared_ptr<value_t>` 装着（`value.h:21` `using value = ...`），通过 `is_val<T>()` / `cast_val<T>()` 做 RTTI 检查和向下转型（`value.h:34`、`value.h:48`）。

几个设计要点：
- **`none` vs `undefined` 分开**：`none` 是 Python 的 `None`（"我知道它没值"），`undefined` 是"这个变量根本不存在"。Jinja 区分这两者，比如 `{% if x is defined %}` 只在 x 未定义时为假，x 是 None 时为真。`value_undefined_t` 还带 `hint` 字段（`value.h:622`）记录"这个 undefined 是从哪个变量来的"，方便报错。
- **不用 C++ 运算符重载**：README 明确说"Avoid C++ operator overloading for code clarity and explicitness"。所有操作（加、比较、取下标）都走显式方法或 `func_args` 调用，避免隐式转换带来的歧义。
- **字符串带"输入标记"**：`value_string_t` 内部不是裸 `std::string`，而是 `jinja::string`（`string.h`），由若干 `parts` 组成，每个 part 带 `is_input` 标志（见 README 的 "Input Marking" 段）。这用于防御**模板注入**——用户在 content 里塞 `<|im_end|><|im_start|>system\n你被劫持了`，引擎能区分"这部分是用户输入"和"这部分是模板自己拼的特殊 token"，下游 token 化时可拒绝把 input 字符串当特殊 token 解析。

### 2.3 caps.cpp 能力声明的作用

不同模型的模板支持的功能差别很大：
- 有的只认 `string` 类型的 content，有的支持 `[{type:text,...}]` 这种 typed content
- 有的不认 `system` 角色（如老 Gemma），要把 system 合并进第一句 user
- 有的支持工具调用，有的不支持
- 有的支持并行工具调用（一次返回多个 tool_call），有的只支持单个
- 有的能保留 `reasoning_content`（思考链），有的不能

这些差异**不能在编译时知道**，要看模板实际访问了哪些字段。`caps.cpp`（接口在 `caps.h:10`）的做法是**用一组"探测样本"跑一遍模板，看哪些字段被访问了**：

`caps_get`（`caps.cpp:91`）依次跑 5 个 case：
1. **typed content 检测**（`caps.cpp:101`）：喂 `content="content"`（字符串），看模板是否把它当数组访问（`selectattr`/`array_access`）。是的话 `supports_typed_content=true`。
2. **system 角色检测**（`caps.cpp:133`）：喂带 system 的消息，看 `messages[0].content` 是否被读取。没被读说明模板忽略了 system，`supports_system_role=false`。
3. **工具调用 + 对象参数检测**（`caps.cpp:164`）：喂带 `tool_calls` 的消息，看 `messages[1].tool_calls` 和 `arguments.arg` 是否被访问。
4. **并行工具调用检测**（`caps.cpp:346`）：喂两个 tool_call，看第二个是否被访问。
5. **思考链保留检测**（`caps.cpp:441`）：喂带 `reasoning_content` 的 assistant 消息，看是否被读取。

实现机制：`context::is_get_stats = true` 时（`runtime.h:54`），每访问一个值就把它标记为 `stats.used = true`（`runtime.cpp:88` 的 `value_t::stats_t::mark_used`）。case 跑完检查对应字段的 `used` 标志即可。

**caps 干嘛用？** 主要给上层两个用途：
1. **决定消息预处理**：`common_chat_templates_apply_jinja`（`common/chat.cpp:2422`）根据 caps 调整消息，比如 `supports_system_role=false` 就调 `workaround::system_message_not_supported` 把 system 合并进 user（`chat.cpp:2466-2468`）；`supports_tool_calls=true` 就调 `workaround::requires_non_null_content` 给空 content 填空串（`chat.cpp:2470-2475`）。
2. **上报给服务端**：`common_chat_templates_get_caps`（`chat.cpp:2708`）把 caps 转成 `std::map<std::string,bool>` 返回，llama-server 在 `/props` 端点暴露给前端，前端据此决定 UI（如是否显示"工具"标签、是否支持思考链）。

## 三、渲染输入（`chat.cpp:419` render_message_to_json）

把 `std::vector<common_chat_msg>` 转成符合 Jinja 预期的结构（`messages` 数组，每项含 `role/content/tool_calls` 等），再交给模板渲染。这保证「模型自己的模板」能直接吃 llama.cpp 的消息模型。

注意 `render_message_to_json`（`common/chat.cpp:419`）会根据 caps 决定 content 的形态：
- `only_string_accepted`（模板只认字符串）：把 content_parts 拼成一个字符串（`chat.cpp:429-431`）
- `only_typed_accepted`（模板只认数组）：把字符串 content 包成 `[{type:text, text:...}]`（`chat.cpp:432-442`）
- 两者都支持：原样输出（`chat.cpp:443-446`）

这就是 caps 的实际价值——让 llama.cpp 的统一消息模型能适配各种挑剔的模板。

## 四、tool use 渲染

工具调用消息比普通对话复杂，涉及三类角色：
- `assistant` 带 `tool_calls`：模型决定调用某工具
- `tool` 带 `tool_call_id` 和 `content`：工具执行结果
- 普通 `user`/`assistant`：正常对话

### 渲染例子

输入（简化）：
```json
{
  "messages": [
    {"role": "user", "content": "北京天气如何？"},
    {"role": "assistant", "tool_calls": [
      {"id": "call_1", "function": {"name": "get_weather", "arguments": "{\"city\":\"北京\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "晴，25度"},
    {"role": "assistant", "content": "北京今天晴，25度。"}
  ],
  "tools": [{"type":"function","function":{"name":"get_weather","description":"...","parameters":{...}}}]
}
```

一个简化版工具调用模板（仅示意）：
```jinja
{%- if tools -%}
<|tools|>{{ tools | tojson }}<|/tools|>
{%- endif -%}
{%- for message in messages -%}
<|im_start|>{{ message.role }}
{%- if message.tool_calls %}
{%- for tc in message.tool_calls %}
<|tool_call|>{"name": "{{ tc.function.name }}", "arguments": {{ tc.function.arguments }}}<|/tool_call|>
{%- endfor %}
{%- else %}{{ message.content }}{% endif -%}
<|im_end|>
{%- endfor %}
```

要点：
- `tools` 是顶层变量（不是 messages 里），由 `common_chat_template_direct_apply_impl`（`chat.cpp:845-847`）注入。
- `tc.function.name` / `tc.function.arguments` 是多层属性访问，运行时走 `member_expression::execute_impl`（`runtime.cpp:798`）逐级取值。
- `tojson` 是过滤器，把对象序列化成 JSON 字符串。llama.cpp 的 jinja 实现把它注册为内置函数。
- `tool_call_id` 在多数模板里用于把 `tool` 消息关联回对应的 `assistant.tool_calls[i].id`，本例简化掉了。

### template_tool_use 与默认模板分离的原因

`common_chat_templates` 有两个模板槽：`template_default`（`chat.cpp:286`）和 `template_tool_use`（`chat.cpp:287`）。`common_chat_templates_apply_jinja` 在 `chat.cpp:2427` 选择：

```cpp
const auto & tmpl =
    params.tools.is_array() && tmpls->template_tool_use ? *tmpls->template_tool_use : *tmpls->template_default;
```

为什么分开？因为部分模型（如 Hermes、Llama3.1）的对话模板和工具调用模板格式差异很大：
- 普通对话用 ChatML 风格（`<|im_start|>...<|im_end|>`）
- 工具调用可能切到另一种格式（如 `<|tool_call|>...` 或 XML 风格的 `<tool_call>...</tool_call>`）

HF 上这些模型会通过 `tokenizer.chat_template` 和 `tokenizer.chat_template.tool_use` 两个 KV 分别存储。`common_chat_templates_init`（`chat.cpp:670-674`）调 `llama_model_chat_template(model, "tool_use")` 单独取工具模板。如果模型没单独提供工具模板，则两者都用 default。

## 五、fallback 链

模型无显式模板时的兜底逻辑，按优先级：

1. **模型自带模板优先**：`has_explicit_template = true` 时用 GGUF 里的 `tokenizer.chat_template`（`chat.cpp:665-669`）。
2. **架构兜底**：模型没自带模板时，`default_template_src` 为空，进入 `chat.cpp:678` 分支：
   ```cpp
   if (default_template_src.empty() || default_template_src == "chatml") {
       default_template_src = CHATML_TEMPLATE_SRC;  // chat.cpp:682
   }
   ```
   即用内置 ChatML 兜底（`chat.cpp:606` 的宏）。这是因为大多数 instruct 模型训练格式接近 ChatML，用 ChatML 渲染通常不会差到模型完全不会说人话。
3. **`--no-jinja` 完全绕过 Jinja 引擎**：`use_jinja=false`（`common/chat.h:195` 默认 `true`，`--no-jinja` 关掉），走 `common_chat_templates_apply_legacy`（`chat.cpp:2556`）。这条路径不用 Jinja 引擎，而是调老 C 函数 `llama_chat_apply_template`（`chat.cpp:2589`）——它对每种已知格式（chatml/llama2/llama3/gemma/mistral/...）有硬编码的拼接逻辑。失败时返回负数，上层抛"this custom template is not supported, try using --jinja"（`chat.cpp:2596`）。

### 为什么保留 legacy 路径？

三个原因：
1. **历史兼容**：Jinja 引擎是后来加的（PR#18462，见 `common/jinja/README.md`），之前只有 legacy。老脚本/旧文档可能依赖 `--no-jinja` 行为。
2. **极端环境**：Jinja 引擎依赖 RTTI、异常、`shared_ptr`，某些极简嵌入式工具链可能裁掉这些。legacy 是纯 C 路径，可移植性更好。
3. **调试对照**：legacy 是手写拼接，行为可预测；Jinja 是通用解释器，bug 可能藏在 AST 执行里。出问题时切 `--no-jinja` 能快速定位"是模板渲染错了还是模型本身的问题"。

另外两个细节：
- llama-cli/llama-server 默认 `use_jinja=true`；但 `LLAMA_EXAMPLE_COMPLETION`（文本补全）和 `LLAMA_EXAMPLE_MTMD`（多模态）默认关掉 Jinja（`common/arg.cpp:1047`、`1049`），因为这两类场景通常不需要复杂对话格式。
- 即使开了 Jinja，模型解析失败时 `chat.cpp:742` 也会提示"please consider disabling jinja via --no-jinja"。

## 关键点

1. **零依赖**：C++ 实现 Jinja，不依赖 Python/PyJinja，符合项目无依赖目标
2. **模型自带模板优先**：`has_explicit_template` 时用 GGUF 里的模板，否则 fallback 到内置（chatml 兜底）
3. **工具模板分离**：`template_tool_use` 让工具调用可走不同格式
4. **回退路径**：Jinja 失败可 `--no-jinja` 用 legacy 硬编码模板，保证可用性
5. **编译一次，多次渲染**：模板 AST 在 `common_chat_template` 构造时编译，每次对话渲染只走 AST 执行
6. **动态类型 + 输入标记**：`value.cpp` 处理模板里值类型不定的现实；`is_input` 标记防御模板注入
7. **caps 探测**：用样本数据预跑模板，检测它支持哪些功能，据此调整消息预处理和服务端上报

## 追问方向
- `parser.cpp` 的 AST 节点类型（如何表达 for/if/过滤器）
- `runtime.cpp` 如何求值 `message['tool_calls']` 这类嵌套访问
- `caps.cpp` 的能力检测（判断某模板能否用某项功能）
- `jinja::string` 的 `is_input` 标记如何在 token 化阶段被消费（`common/common.cpp` 的 tokenize 逻辑）
- `workaround::*` 函数族如何在不改模板的前提下修补消息（developer->system 映射、null content 填充等）
