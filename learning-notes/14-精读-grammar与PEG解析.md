# 精读：Grammar 采样约束与 PEG 解析（llama-grammar.cpp + common/peg-parser）

> 日期：2026-08-13 | 核心文件：`src/llama-grammar.cpp`（1510 行）+ `common/peg-parser.cpp`（2256 行）/`common/peg-parser.h`（536 行）
> 2026-08-17 修订：扩充为初学者详细版（文法入门+栈演进手算）
> 前置：读过 `05`（grammar 是采样链的一节）。文档：`docs/development/parsing.md`（288 行）、`docs/autoparser.md`。
> 作用：让模型输出强制符合某语法（JSON schema、代码、固定格式），是 llama.cpp 结构化输出的基础。

## 0. 问题：LLM 会输出"不合法 JSON"

故事从一句抱怨开始。你让模型返回 JSON，它可能给你：

```
{"name": "Alice", "age": 30,}      <- 末尾多了逗号
{"name": "Alice", "age": 30         <- 缺右花括号
我帮你查了一下：{"name": "Alice"}   <- JSON 前面插了散文
```

`json.loads` 一Parse就抛错。这类问题**不能**靠"提示词写得更严格"根治--LLM 是概率采样，总有非零概率走偏，长输出里几乎必然出现一次。

**约束采样（constrained sampling）**换个思路：每一步采样时，把"会让输出违反语法"的候选 token 直接屏蔽（`logit = -INFINITY`），模型只能在合法分支里挑。这样无论模型多想乱说，输出注定合法。这就是 llama.cpp 的 `grammar` 采样器（`llama-sampler.cpp:2519`）。

llama.cpp 用 **GBNF**（Grammar BNF）描述语法，用 `src/llama-grammar.cpp` 里的"栈式 VM"做约束。本篇先讲文法基础，再讲 GBNF 怎么写，然后手算栈怎么随生成推进，最后讲 PEG 解析器（`common/peg-parser.*`）如何把更复杂的语法编译成 GBNF。

## 1. 文法基础 5 分钟入门

### 1.1 终结符 / 非终结符 / 规则 / 推导

**文法（grammar）**是一组重写规则。以经典例子：

```
S -> "a" S "b" | "ab"
```

读法：
- `S` 是**非终结符**（nonterminal，可继续展开的名字）
- `"a"`、`"b"` 是**终结符**（terminal，字面字符）
- `|` 是"或"，分两个分支
- `->`（或 GBNF 里的 `::=`）读作"定义为"

**推导（derivation）**：从起始符号开始，反复把非终结符替换成它的某个分支，直到全是终结符。推导出 `aabb`：

```
S
=> "a" S "b"        # 用第一分支展开 S
=> "a" "ab" "b"     # 把内层 S 用第二分支展开
=> "aabb"           # 拼接所有终结符
```

而 `aabbb`、`abab`、`aab` 都无法从此文法推出，故非法。

### 1.2 PEG 与 CFG 的差别

上面这种文法叫 **CFG（上下文无关文法）**。CFG 的 `|` 是无序选择：两条分支都"可能"匹配，解析器可以挑任意一条，必要时回溯。

**PEG（解析表达式文法）**把 `|` 改成**有序选择 `/`**：左侧分支优先，左侧能匹配就用左侧，绝不试右侧。这看似小改动，实则消除歧义--同一文法只有一种解析结果。代价是：`A / B` 与 `B / A` 不等价，写规则要小心顺序（例如 `number / integer` 里 `number` 已涵盖 `integer`，`integer` 永远不被尝试）。

PEG 还有 `&A`（正向预查，不消耗输入）、`!A`（负向预查）等运算符，能表达 CFG 难以表达的东西（如"匹配到分隔符为止"）。

llama.cpp 的 GBNF 走 CFG 路线（支持歧义文法，靠多栈并行处理），而 `common/peg-parser` 是 PEG 路线（用 builder 组合语法，靠有序选择避免歧义）。两者关系后文展开。

## 2. 两种机制（总览）

| 机制 | 位置 | 用途 |
|---|---|---|
| **GBNF grammar** | `src/llama-grammar.cpp` | **采样时**强制约束输出 token |
| **PEG 解析** | `common/peg-parser.cpp` | 解析已生成文本，构建 AST；可编译成 GBNF |

关系：PEG 是「解析器」，grammar 是「约束采样器」。llama.cpp 用 PEG 描述更复杂的格式（如带 JSON schema 的工具调用），再编译成 GBNF 去约束采样。

## 3. GBNF 语法教学

### 3.1 一个 JSON 子集的 GBNF

```gbnf
root   ::= object                                  # 起始规则：必须是对象
object ::= "{" ws "}"                              # 空对象 {}
         | "{" ws member (ws "," ws member)* ws "}"  # 至少一个成员
member ::= json_string ws ":" ws value             # "key": value
value  ::= json_string | number | object | "true" | "false" | "null"
json_string ::= "\"" ( [^"\\] | "\\" . )* "\""     # 字符串，支持转义
number ::= [0-9]+ ("." [0-9]+)?                    # 简化版数字
ws     ::= [ \t\n]*                                # 可选空白
```

逐行注解：
- `root ::= object`：根规则只能展开成 `object`，所以输出必然以 `{` 开头。
- `::=` 是"定义为"，`|` 是分支（CFG 风格，无序）。
- `"{"` 是字面量，匹配字符 `{`。
- `[0-9]+`：字符类 `[0-9]` 加 `+`（一次或多次）。
- `[^"\\]`：字符类取反，匹配除 `"` 和 `\` 外的任意字符。
- `(...)*`：分组加 `*`（零次或多次）。
- `ws` 是辅助规则，被其他规则**引用**（写规则名即引用，类似函数调用）。
- `\` 在 GBNF 字符串里需要转义，所以文本里写 `"\\"` 表示一个反斜杠字符。

### 3.2 GBNF 写法速查

| 写法 | 含义 | 例 |
|---|---|---|
| `"abc"` | 字面字符串 | `"{"` |
| `[abc]` | 字符类（a/b/c 任一） | `[0-9]` |
| `[^abc]` | 取反字符类 | `[^"]` |
| `[a-z]` | 范围 | `[0-9a-fA-F]` |
| `.` | 任意字符 | `.*` |
| `*` `+` `?` | 重复 0+ / 1+ / 0或1 | `[0-9]+` |
| `{m,n}` | 重复 m~n 次 | `[0-9]{2,4}` |
| `A B C` | 序列（空格分隔） | `"{" ws "}"` |
| `A | B` | 选择 | `"true" | "false"` |
| `( ... )` | 分组 | `("a"|"b")+` |
| `rule_name` | 规则引用 | `value` |
| `<[123]>` | 按 token id 匹配（少用） | `<[42]>` |

### 3.3 解析过程（`llama_grammar_parser`）

GBNF 文本由 `llama_grammar_parser::parse(src)`（`llama-grammar.cpp:687`）解析。它是递归下降：
- `parse_rule`（`:663`）切出 `name ::= ...` 一行
- `parse_alternates`（`:434`）按 `|` 分支
- `parse_sequence`（`:451`）解析一条分支内的字符类、字面量、引用、`*+?{}` 等

结果是一张 `llama_grammar_rules`（`vector<vector<llama_grammar_element>>`），每条规则是一串 `llama_grammar_element`。元素类型见 `llama-grammar.h:13` 的 `enum llama_gretype`：
- `CHAR`/`CHAR_NOT`/`CHAR_RNG_UPPER`/`CHAR_ALT`/`CHAR_ANY`：字符相关
- `RULE_REF`：规则引用
- `ALT`/`END`：分支分隔 / 规则结束
- `TOKEN`/`TOKEN_NOT`：按 token id 匹配（用于 `<[id]>` 语法）

例如 `[0-9]+` 会被 `handle_repetitions`（`:462`）改写成等价的右递归规则（源码注释 :470-483 给出 `S+ -> S S'`，`S' ::= S S' | <空>`）。

## 4. 核心新增：语法栈演进手算

这一节是关键。理解了栈怎么演进，就理解了"约束采样"的全部魔法。

### 4.1 栈是什么

`llama_grammar`（`llama-grammar.h:126`）持有 `stacks`（`vector<vector<const llama_grammar_element *>>`）。每个 **stack** 是一个指针列表，记录"当前正在匹配的规则位置"。多个 stack 表示"语法当前可能处于多个状态"（CFG 允许歧义，所有可能的状态都要保留）。

为什么用指针？因为规则存在 `rules` 这个大 vector 里，栈元素是指向 `rules[rule_id][i]` 的指针，省得拷贝。

### 4.2 极小文法

为了手算，用一个极小文法：

```gbnf
root   ::= number
number ::= digit+ "." digit+
digit  ::= [0-9]
```

合法输出如 `3.14`、`00.99`；非法如 `3.`、`.5`、`3.14.15`。

解析后 `rules` 大致是（`+` 被 `handle_repetitions` 改写后）：
- `rules[0]` (root): `[RULE_REF number, END]`
- `rules[1]` (number): `[RULE_REF digit, RULE_REF number_rec1, CHAR '.', RULE_REF digit, RULE_REF number_rec2, END]`
- `number_rec1`/`number_rec2`：`digit number_recN | <空>` 的右递归辅助规则
- `rules[2]` (digit): `[CHAR '0'-'9', END]`

### 4.3 初始栈

`llama_grammar_init_impl`（`:1126` / `:1195`）建初始栈。过程（`:1156`-`:1176` 或 `:1249`-`:1269`）：
1. 取起始规则 `root` 的第一个分支
2. 调 `llama_grammar_advance_stack`（`:853`）把栈"展开"--遇到 `RULE_REF` 就把引用替换成被引用规则的内容（每个分支生成一条新栈），直到栈顶是**终结符**（`CHAR`/`CHAR_NOT`/`CHAR_ANY`/`TOKEN`/`TOKEN_NOT`）

初始栈展开后（简化表示）：
```
栈1: [CHAR '0'-'9' (来自 digit), <number 剩余: number_rec1, ".", digit, number_rec2, END>]
```
栈顶是 digit 的 `[0-9]`，后面挂着 number 规则剩余的部分。

**候选 token 集合怎么算**：栈顶是 `[0-9]`，所以**首字符**只允许 `0`-`9`。词表里凡是首字节不在 `0`-`9` 的 token 都被屏蔽。例如 token `"3"` 被允许，token `"."` 被屏蔽（因为没有以 `.` 开头的合法 number），token `"\n"` 被屏蔽。

### 4.4 接受 token `"3"` 后栈怎么推进

假设采样器选中了 token `"3"`（字符 `'3'`）。`llama_grammar_accept_token`（`:1456`）把它喂进栈：

1. 对栈顶 `[0-9]` 调 `llama_grammar_match_char`（`:758`）：`'3'` 在 `0`-`9` 范围内，匹配成功。
2. `llama_grammar_accept_chr`（`:1016`）弹出栈顶，把"匹配后剩下的部分"压回去。`[0-9]` 后面没有 `CHAR_ALT`/`CHAR_RNG_UPPER`，所以 `match_char` 返回的 `pos` 指向 digit 规则的 `END`，再弹掉。
3. 栈变成 `[<number 剩余: number_rec1, ".", digit, number_rec2, END>]`。
4. 调 `advance_stack` 展开：`number_rec1` 有两个分支（再吃一个 digit / 空分支），所以**栈分裂成两条**：
   ```
   栈A: [CHAR '0'-'9' (number_rec1 第1分支的 digit), <number_rec1, ".", digit, number_rec2, END>]
   栈B: [CHAR '.' (number_rec1 第2分支为空，跳到 number 里的 "."), <digit, number_rec2, END>]
   ```
   两条栈的栈顶分别是"还能再吃 digit"（栈A）和"已经吃完 digit+，准备吃 `.`"（栈B）。

这就是"多栈=语法可能处于多个状态"的含义：模型既可以继续吃数字（栈A），也可以转去吃小数点（栈B），两条路都合法。

### 4.5 候选集合的更新

现在 `apply` 时：
- 栈A 栈顶是 `[0-9]`，允许 `0`-`9`
- 栈B 栈顶是 `CHAR '.'`，允许 `.`
- 取并集：候选首字符 = `{0-9, .}`

所以下一步既可能采样到 `"."`，也可能采样到另一个数字 token（如 `"1"`）。如果选了 `"."`，栈A 全部死掉（吃不下 `.`），只剩栈B；如果选了 `"1"`，栈B 全部死掉，栈A 继续分裂。

### 4.6 关键：约束采样不回溯

注意：**采样是单线的**。每步选一个 token，就**永久落定**，不会"等下不对我退回去重选"。所以约束采样必须保守地保留**所有**可能继续的栈。任何一条栈死了就丢，但只要还有一条活着，输出就还能合法。

这就是为什么 `advance_stack` 要枚举所有分支并去重（`:868` 的 `seen` 集合防循环展开）：不能漏掉任何一条活路。也是为什么 `init_impl` 要检测左递归（见第 8 节）--左递归会让 `advance_stack` 无限分裂。

### 4.7 接受结束符

当 number 走到最后，栈会变成空。`llama_grammar_apply_impl`（`:1339`）检查（`:1347`-`:1352`）：只要有任一栈为空，就允许 EOG（end-of-generation）token（如 `<eos>`）。如果所有栈都非空却来了 EOG，apply 会把 EOG 屏蔽（`:1364`-`:1367`），强制继续生成。

## 5. reject_candidates 机制：按 token 而非字符过滤

### 5.1 为什么不能按字符过滤

词表里的 token 是多字节串。比如 `"\n}"` 是一个 token，`".5"` 也是一个 token。如果只看 token 首字符是否合法，会误判：
- 栈期待 `[0-9]`，token `"3.14"` 首字符 `'3'` 合法，但 token 内部有 `.`，而栈此刻不期待 `.`--按字符过滤会**误放行**。
- 栈期待 `.`，token `".5"` 首字符 `.` 合法，但后面 `5` 期待 `digit`--按字符过滤不知道怎么处理非首字符。

所以必须**逐字节**试探 token 是否能延续当前语法。

### 5.2 算法（`llama_grammar_reject_candidates_for_stack`，`:1053`）

对每个候选 token（已 decode 成 code_points 数组，末尾带 0 终止符）：
1. 取栈顶 `stack_pos`（`:1070`）。
2. 若是 `TOKEN`/`TOKEN_NOT`（按 token id 匹配）：直接比对 token id（`:1073`-`:1086`）。
3. 若是字符类：
   - 若 token 字节已用完（`*tok.code_points == 0`，`:1092`）：检查有没有半截 UTF-8（`tok.partial_utf8.n_remain != 0`）。有且补不全成合法字符就 reject（`:1095`-`:1097`）；否则**不加入 rejects**（部分接受，见 5.3）。
   - 若第一个 code point 匹配字符类（`:1099`）：把 token 的 code_points 指针后移一位（`:1100`），栈也推进到 `match_char` 返回的下一位置，**递归**调 `reject_candidates` 处理剩余字节（`:1116`）。
   - 否则不匹配 -> reject（`:1102`）。

外层 `llama_grammar_reject_candidates`（`:936`）对**每个**栈都跑一遍 `reject_candidates_for_stack`，对 rejects 取**累积**--只有"所有栈都拒绝"的 token 才最终被屏蔽。换句话说，只要有一条栈能接受这个 token，它就保留。

### 5.3 "半匹配"举例

**例 1**：栈期待字符串 `"}"`，候选 token 是 `"\n}"`（两个字符）：
1. 第一个 code point `'\n'`，`match_char` 比对 `'}'`：不匹配 -> reject。
2. 整个 token 被屏蔽。

**例 2**：栈期待 `[^"]`（任意非引号字符），候选 token 是 `"ab`（BPE 切分到一半，缺右引号）：
1. `'a'` 匹配 `[^"]`，token 指针后移，栈推进。
2. `'b'` 匹配 `[^"]`，token 指针后移，栈推进。
3. token 字节用完（`*tok.code_points == 0`），没有 partial UTF-8 -> 这个 token 是**部分接受（partial accept）**：它能"暂时"被语法吃下，但语法还没结束（栈非空，期待下一个字符是 `[^"]` 或 `"`）。这个 token 被保留，下一步采样可以选它，选完栈推进，下一步候选要继续匹配 `[^"]` 或 `"`。

这就是"一个 token 可能半匹配"的处理：token 走完了语法没走完，没关系，把它当作"推进了一部分"接受，下一步继续约束。

### 5.4 partial UTF-8 的细节

多字节 UTF-8 字符（如中文，3 字节）可能被词表切到一个 token 边界上：token 只含前 2 字节。`llama_partial_utf8`（`llama-grammar.h:52`）记录"已累积的位值 + 还差几字节"。`llama_grammar_match_partial_char`（`:788`）算出"这个半截字符可能补全成的码点范围 [low, high]"（`:803`-`:804`），再判断范围是否与栈顶字符类相交（`:815`-`:831`）。相交就保留（有补全成合法字符的可能），不相交就 reject。

`apply_impl`（`:1371`）decode token 时带上 `grammar.partial_utf8`（前一个 token 留下的半截），保证跨 token 边界的 UTF-8 字符也能正确约束。

## 一、GBNF grammar 约束采样（src/llama-grammar.cpp）

### 语法解析（`llama_grammar_parser`）
- `parse(src)`（:687）把 GBNF 文本解析成 `llama_grammar_rules`
- 规则由 `llama_grammar_rule`（一系列 `llama_grammar_element`）组成，token 类型：`CHAR`/`CHAR_NOT`/`CHAR_RANGE`/`RULE_REF`/`END` 等
- `parse_alternates`/`parse_sequence`/`parse_rule`（:434/:451/:663）递归下降

### 运行时状态：栈式 VM
`llama_grammar` 持有一组 `stacks`（`llama-grammar.h:134`），每个 stack 是「当前可匹配的规则位置」的活动集合。生成时维护栈，表示语法当前可能所处的状态（详见第 4 节手算）。

### 采样约束（核心）
`llama_grammar_apply_impl`（:1339）对**每个候选 token**：
1. 把 token 的文本片段 decode 成 code_points（`:1371`）
2. `llama_grammar_reject_candidates`（:936）对每个 stack 调 `reject_candidates_for_stack`（:1053），**剔除**会让语法卡死的 token
3. 把被拒 token 的 `logit` 设为 `-INFINITY`（`:1378`），产出「被允许的候选子集」，采样器在其间选

> 关键：**按 token 过滤而非字符**。候选 token 的字节序列逐一试探，只有「能延续当前语法状态」的 token 才保留。这就是为什么 `grammar` 采样器能强制 JSON 合法--非法 token 直接被拒。

### 接受（`llama_grammar_accept`，:1042 / `accept_token` :1456）
选中 token 后，把它的文本真正推进语法栈，更新状态（供下一个 token 约束）。

### 左递归检测（:955）
预处理阶段检测左递归规则，避免无限递归（`llama_grammar_detect_left_recursion`）。详见第 8 节。

## 二、PEG 解析（common/peg-parser）

**Builder 模式**：用 C++ 运算符组合语法，`operator+`=序列、`operator|`=选择、`operator<<`=空格分隔序列（`peg-parser.h:43`-`:56`）：

```cpp
common_peg_parser_builder p;
auto integer = p.rule("integer", [&]() {
    auto digit = p.chars("[0-9]", 1, 1);              // 一个数字
    auto first = p.chars("[1-9]", 1, 1);               // 首位 1-9
    return p.choice({
        p.literal("0"),                                // 单独 0
        p.sequence({ first, p.zero_or_more(digit) })   // 1-9 后跟任意数字
    });
});
```

对应 PEG 文法（与 `peg-parser.cpp:1327` 的 `json_number` 思路一致）：

```
integer <- "0" / [1-9][0-9]*
```

builder 产生 `common_peg_parser_variant`（`peg-parser.h:284`-`:306`，一个 `std::variant`，覆盖所有解析器类型），存进 `common_peg_arena`。`arena.parse(ctx)` 跑 PEG 解析，产出 AST（`common_peg_ast_node`，`peg-parser.h:74`）。

### PEG 转编译成 GBNF 再约束采样

`common_peg_arena::build_grammar`（`peg-parser.cpp:1713`）把 PEG 树翻译成 GBNF 文本。翻译规则在 `to_gbnf` lambda（`:1761`）里：
- `literal("abc")` -> `"abc"`
- `sequence(A, B)` -> `A B`
- `choice(A, B)` -> `A | B`
- `zero_or_more(p)` -> `p*`
- `chars("[0-9]")` -> `[0-9]*`
- `negate(p)` -> **无法翻译**（PEG 的 `!` 在 CFG 里表达不了，`docs/development/parsing.md:131` 警告）

完整流水线（`common/chat.cpp` 等用）：

```
JSON schema ──┐
              ├─► common_peg_parser_builder ──► common_peg_arena
              │       (PEG 树, 可执行解析)            │
              │                                       │
              │                                       ├─► parse()  -> AST (运行时解析模型输出)
              │                                       │
              │                                       └─► build_grammar() ──► GBNF 文本
              │                                                                    │
              │                                                                    ▼
              └────────────────────────────────────────────────► llama_sampler_init_grammar
                                                                                  │
                                                                                  ▼
                                                              运行时约束采样（本篇第 4-6 节）
```

### 与 GBNF 的分工
- **GBNF**：离线定义好、直接约束采样的格式（用户传 `--grammar-file`，server 直接用）
- **PEG**：定义更丰富/可组合的语法，运行时解析文本（如解析工具调用的 AST），必要时编译成 GBNF 约束**后续**生成。`docs/development/parsing.md:117` 起有完整说明

## 三、在采样链中的位置（对照 `05`）

`llama_sampler_grammar`（`llama-sampler.cpp:2519` 的 `llama_sampler_grammar_i`）封装了这套逻辑：

| 接口 | 函数 | 干什么 |
|---|---|---|
| `apply` | `llama_sampler_grammar_apply`（`llama-sampler.cpp:2448`） | 调 `llama_grammar_apply_impl`（`:1339`）：decode 候选 -> `reject_candidates` 算屏蔽集 -> 设 `-INFINITY` |
| `accept` | `llama_sampler_grammar_accept_impl`（`llama-sampler.cpp:2441`） | 调 `llama_grammar_accept_impl`（`:1382`）-> `llama_grammar_accept_token`（`:1456`）：推进所有栈 |
| `reset` | `llama_sampler_grammar_reset` | 新序列时清空栈、重建初始栈 |

呼应 `05`：grammar 在链尾 apply 阶段过滤候选，dist 在被允许集合里采样，accept 推进栈。三步循环，直到 EOG。

### lazy grammar（按需触发）

`llama-grammar.h:142` 起的 `lazy`/`awaiting_trigger` 字段：lazy 模式下，grammar 先不约束，等模型输出"触发词/触发 token/触发正则"后才开始约束。常用于"模型先输出散文，遇到 `<tool_call>` 后强制 JSON"的场景。触发前 `apply` 直接 return（`:1342`-`:1344`），触发后回放 buffer 里的 token 重建栈（`:1404`-`:1422`）。

## 四、左递归检测（预处理）

`llama_grammar_detect_left_recursion`（`:955`）在 `init_impl`（`:1150` / `:1243`）里跑。CFG 允许 `A -> A B` 这种左递归，但本实现用栈展开，遇到左递归会无限分裂（`advance_stack` 的 `seen` 集合虽能防死循环，但会导致初始栈为空，行为不对）。所以预处理检测到就拒绝：

```cpp
if (llama_grammar_detect_left_recursion(vec_rules, i, ...)) {
    LLAMA_LOG_ERROR("unsupported grammar, left recursion detected ...");
    return nullptr;
}
```

写 GBNF 时把左递归改写成右递归即可（`A -> B A'`，`A' -> B A' | <空>`），就像 `handle_repetitions` 对 `+` 做的那样。

## 追问方向
- `reject_candidates_for_stack` 的具体栈推进算法（`llama-grammar.cpp:1053`-`:1122`，本篇第 5 节已展开）
- `common_grammar_builder` 如何把 PEG 转成 GBNF（`common/peg-parser.cpp:1713`，本篇第 7 节已展开）
- JSON schema -> grammar 的转换（`common/json-schema-to-grammar.cpp`，`examples/json_schema_to_grammar.py`）
- lazy grammar 的触发回放细节（`:1382`-`:1427`）
