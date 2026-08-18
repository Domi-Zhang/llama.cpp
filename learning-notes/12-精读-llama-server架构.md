# 精读：llama-server 架构（tools/server）

> 日期：2026-08-13 | 核心文件：`tools/server/server.cpp`（HTTP 路由，407 行）+ `server-context.cpp`（引擎，5405 行）+ `server-queue.cpp`（任务队列，460 行）+ `server-http.cpp`（HTTP 封装，782 行）+ `server-common.h`（372 行）
> 2026-08-17 修订：扩充为初学者详细版（并发时间线走查等）
> 前置：读过 `00`（CLI 复用 server 引擎）、`04`（KV cache 与 `seq_rm/seq_add`）。server 是 llama.cpp 的现代服务入口，也是 `llama-cli` 的底层。

## 0. 为什么要这么复杂：单请求串行推理的浪费

先问一个朴素问题：**为什么不直接"HTTP 收到一个请求 -> 跑一次推理 -> 返回"？**

直觉上这是最简单的实现，但 GPU 推理有一个非常关键的成本结构：**算力便宜、带宽贵**。每跑一次 `llama_decode`，无论 batch 里只有 1 个 token 还是有 512 个 token，都要先把模型权重从显存读一遍到计算单元。假设读一次 7B 模型权重花 8 ms，纯计算 1 个 token 的 attention+FFN 只花 1 ms：

```
单请求串行（两个用户 A、B 各生成 1 个 token）
时间轴：0ms          9ms        10ms       18ms       19ms
        |------------|----------|-----------|----------|
        A: 读权重8ms  算1ms      B:读权重8ms 算1ms
        ^^^^^^^^^^^^            ^^^^^^^^^^^^
        权重读了 2 遍（GPU 利用率低）
```

问题是：**第二次读权重完全是浪费** -- 如果 B 的 token 也在同一个 batch 里，它就能搭 A 的便车，权重的 8 ms 摊给 2 个 token，等效每 token 4.5 ms 而不是 9 ms。这就是 **batching** 的收益直觉：一次 decode 把权重读一遍，可以同时服务 N 个请求的 token，每个请求摊到的成本随 N 增大而降低。

但 batching 又带来一个新问题：**A 和 B 的进度不一样**。A 可能还在 prompt 阶段（要处理 1000 个 prompt token），B 已经生成到第 50 个 token 了。怎么把它们塞到同一个 batch 里？这就是 **continuous batching**（动态批处理）：每个 step 重新决定"这一批里要带哪几个 slot 的 token"，新请求随时插进来，老请求随时结束走人。`llama.cpp` 的 `cont_batching` 默认就是 `true`（`common/common.h:553`：`insert new sequences for decoding on-the-fly`）。

这一节是后面所有架构的动机。三层架构、slot 状态机、任务队列，本质都是为了把"多个并发请求的 token 塞进同一个 `llama_batch`"这件事做好。还有另一个硬约束：**`llama_context` 本身不支持并发 `llama_decode`**（KV cache 状态、内存管理都是单线程假设），所以推理必须串行 -- 这决定了"HTTP 线程不能直接调推理，必须丢队列给单线程消费"的解耦设计。

## 1. 作用

提供 OpenAI 兼容的 HTTP API（`/v1/chat/completions` 等），把多用户请求调度到推理引擎并流式返回。核心三件套：**HTTP 层 -> 任务队列 -> 槽位(slot)推理**。

## 2. 层次模型

```
HTTP 层   server.cpp + server-http.cpp    解析请求、写响应、SSE 流式
   |
   |  (post task -> queue_results.recv)
   v
任务队列   server_queue (server-queue.cpp)  派发任务、多请求并发
   |
   |  (callback_new_task / callback_update_slots)
   v
引擎       server_context (server-context.cpp)
   ├── 槽位 slot                              每个并发请求占一个 slot
   ├── 批处理 llama_batch                     多个 slot 的 token 合批喂 llama_decode
   └── 任务处理 update_slots -> pre_decode/decode/post_decode
```

**逐层职责与解耦原因**：

- **HTTP 层**：只负责 HTTP 协议本身（解析 JSON、组装 SSE 分片、超时、SSL）。它不关心 slot、不关心 KV cache，更不能阻塞 -- 一个 HTTP 线程卡住不能影响其他 HTTP 线程。HTTP 线程把请求转成 `server_task` 丢进队列后就守在 `queue_results.recv` 上等结果，期间可以处理 SSE ping、客户端断连（`req.should_stop`）。
- **任务队列**：是 HTTP 线程和推理线程之间的"信箱"。HTTP 线程往里塞 task，推理线程单线程消费。队列本身只做"派发"和"结果路由"，不做推理 -- 这一层很薄，但解耦了"什么时候来请求"和"什么时候做推理"两件事。
- **引擎**：单线程串行处理。所有 slot 的 token 都进同一个 `llama_batch`，一次 `llama_decode` 并行处理 -- 这是 batching 落地的地方。`llama_context` 不能被多线程并发 `decode`（KV cache 是单线程假设），所以推理必须单线程串行 -- 这是设计约束，不是性能瓶颈，因为 batching 已经把单线程的算力榨干了。

三层解耦的关键：**HTTP 线程不阻塞推理线程，推理线程不阻塞 HTTP 线程**。靠队列 + 结果路由解耦，每个 HTTP 线程只看自己的 `id_tasks` 集合里的结果，互不干扰。

## 3. HTTP 层（`server.cpp` + `server-http.cpp`）

`ctx_http.post("/v1/chat/completions", ...)` 注册各路由（`server.cpp:188-228`）。关键路由：

- `/v1/chat/completions`（OpenAI 兼容对话，最常用）
- `/completion` / `/v1/completions`（补全）
- `/v1/embeddings` / `/rerank` / `/tokenize` / `/detokenize` / `/infill` / `/v1/responses` / `/v1/messages`（Anthropic 兼容）
- `/slots`（槽位状态查询与 save/restore）、`/metrics`、`/health`

**路由模式**：每个 handler 是 lambda，解析 body -> 调 `handle_completions_impl`（`server-context.cpp:4110`）-> 返回 `server_res_generator`。`ex_wrapper`（`server.cpp:40`）包一层异常处理，把 C++ 异常转成 500 / 400 错误 JSON，确保 handler_t 永不抛异常（避免 httplib 进异常处理兜底）。

```cpp
// server.cpp:40 简化版
static handler_t ex_wrapper(handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        try { return func(req); }
        catch (const std::invalid_argument & e) { /* 400 */ }
        catch (const std::exception & e)        { /* 500 */ }
        catch (...)                              { /* 500 unknown */ }
    };
}
```

**流式响应**：`task_response_type`（`server-task.h:33`）决定响应格式；流式用 **SSE**（`text/event-stream`），每生成一个 token 发一个 `data:` 分片。`handle_completions_impl` 在流式分支（`server-context.cpp:4233` 之后）把第一块结果作为 HTTP 头发出去，然后挂一个 `res->next` 回调 -- httplib 每次调这个回调，server 就从 `response_reader` 拉一个新结果，格式化为 `data: {json}\n\n` 推回客户端。客户端断连时 `req.should_stop()` 返回 true，回调返回 false 终止流。

## 4. 任务队列（`server-queue.cpp` + `server-task.h`）

### 4.1 server_task 的类型表

`enum server_task_type`（`server-task.h:16`）：

| 类型 | 用途 | 触发方 |
|---|---|---|
| `SERVER_TASK_TYPE_COMPLETION` | 文本补全（`/completion`、`/v1/chat/completions`） | HTTP |
| `SERVER_TASK_TYPE_EMBEDDING` | 取 embedding（`/v1/embeddings`） | HTTP |
| `SERVER_TASK_TYPE_RERANK` | 重排打分（`/rerank`） | HTTP |
| `SERVER_TASK_TYPE_INFILL` | 代码补全（FIM，`/infill`） | HTTP |
| `SERVER_TASK_TYPE_CANCEL` | 取消某个 task（客户端断连） | response_reader 析构 |
| `SERVER_TASK_TYPE_CONTROL` | 实时控制（如 `reasoning_end` 提前结束思考） | HTTP |
| `SERVER_TASK_TYPE_NEXT_RESPONSE` | 让 `update_slots` 再跑一轮 | 引擎自身（`server-context.cpp:2773`） |
| `SERVER_TASK_TYPE_METRICS` | 取指标 | HTTP |
| `SERVER_TASK_TYPE_SLOT_SAVE/RESTORE/ERASE` | slot 状态持久化 | HTTP `/slots/:id` |
| `SERVER_TASK_TYPE_GET_LORA/SET_LORA` | LoRA 热切换 | HTTP `/lora-adapters` |

### 4.2 父子任务：parallel sampling（**不是 speculative decoding**）

注意：原版笔记写"父子任务（speculative）"是个常见误解，实际看 `server-task.h:143-148`：

```cpp
// used by parallel sampling (multiple completions from same prompt)
int id_parent  = -1;
// temporary store of child tasks for scheduling
std::vector<server_task> child_tasks;
```

**父子任务是为 parallel sampling 服务**的 -- 即 OpenAI 的 `n` 参数：同一个 prompt 要生成 N 个不同采样结果。`server-context.cpp:4185-4190` 里如果 `n_cmpl > 1`，就给主 task 挂上 N-1 个 child task：

```cpp
if (task.params.n_cmpl > 1) {
    int n_children = task.params.n_cmpl - 1;
    for (int j = 0; j < n_children; j++) {
        task.add_child(task.id, rd.get_new_id());  // 每个 child 用不同 seed
    }
}
```

**为什么要父子结构而不是 N 个独立 task？** 因为子 task 共享主 task 的 prompt 处理结果：parent 先跑完 prompt 阶段（写完 KV cache），然后通过 `slot.copy_state_to(*child)`（`server-context.cpp:3693`）把 KV cache 直接拷给每个 child，child 跳过 prompt 阶段直接进入 GENERATING。这样 N 个采样只跑一次 prompt，而不是 N 次。看状态机：child 在 `launch_slot_with_task` 里被置为 `SLOT_STATE_WAIT_OTHER`（`server-context.cpp:1828`），parent 跑到 `SLOT_STATE_DONE_PROMPT` 后 `decode()` 里把它改成 `SLOT_STATE_DONE_PROMPT`（`server-context.cpp:3694`）。

speculative decoding 走的是**另一套机制**：每个 slot 自己有 `spec_draft`、`spec_i_batch`、`spec_ckpt` 字段（`server-context.cpp:170-175`），draft 模型和 target 模型在同一个 slot 内协同 -- 不涉及父子 task。

### 4.3 response_reader 的 id 匹配机制

每个 HTTP 请求都会创建一个 `server_response_reader`（`server-queue.h:168`），它持有一个 `id_tasks` 集合（多 task 场景下是多个 id）。整个 id 匹配流程：

```
HTTP 线程                          引擎线程
   |                                  |
   |  rd.get_new_id() 生成 task.id     |
   |  rd.post_task(task) ------------->|  queue_tasks.post(task)
   |    -> add_waiting_task_id(id)    |
   |    -> id_tasks.insert(id)         |
   |                                   |
   |   rd.next(should_stop)             |  process_single_task -> update_slots
   |     -> queue_results.recv(        |    -> send_partial_response(slot)
   |          id_tasks, timeout)        |        res->id = slot.task->id
   |                                   |        queue_results.send(res)
   |                                   |          -> 检查 id 是否在 waiting_task_ids
   |                                   |          -> 是则 push 到 queue_results
   |   <--------- 结果回到 HTTP 线程 ---|
   |   找到 id 匹配的结果，返回        |
```

关键三步（`server-queue.cpp`）：
1. **登记**：`post_task` 调 `queue_results.add_waiting_task_id(id)`（`:360`），id 进入 `waiting_task_ids` 集合。
2. **过滤**：`queue_results.send(result)`（`:319`）扫 `waiting_task_ids`，只有 id 在集合里才 push 到 `queue_results` -- 别的 HTTP 请求的结果不会跑错线程。
3. **取回**：`recv(id_tasks)`（`:266`）阻塞等 `queue_results` 非空，然后线性扫找 id 匹配的元素。`recv_with_timeout` 加了超时，让 HTTP 线程能周期性检查 `req.should_stop()`。

这个机制保证：**多个 HTTP 线程同时等结果时，结果通过 id 路由到正确的线程，不需要锁住整个队列**。

## 5. 槽位模型（slot，核心并发单元）

`struct server_slot`（`server-context.cpp:159`）是并发的最小单元。一个 slot 持有：

- **KV cache 序列**：`id` 就是 `llama_seq_id`，每个 slot 独占一条 KV cache 序列。
- **采样器链**：`smpl`（`common_sampler_ptr`），每个 slot 一套独立的采样状态（repetition penalty 等需要历史 token）。
- **prompt 状态**：`server_tokens prompt.tokens`，已写入 KV cache 的 token 序列 + 位置。
- **task 指针**：`std::unique_ptr<const server_task> task`，正在服务的请求。
- **统计**：`n_decoded`、`t_start_process_prompt`、`t_start_generation` 等。

`n_parallel` 决定 slot 数量上限（即并发请求数上限），每个 slot 的 `n_ctx` = 总 ctx / n_parallel（`server-context.cpp:1352`：`slot.n_ctx = n_ctx_slot`）。多个 slot 的 token 合进同一个 `llama_batch`（`struct server_batch`，`server-context.cpp:68`），`server_batch::token` 里每个 token 记录 `id_slot`（`:73`），decode 完后 `post_decode` 用 `slot.i_batch` 反向找到本 slot 的 logits 索引。

### 5.1 slot 状态机详解

`enum slot_state`（`server-context.cpp:57-64`）：

```
IDLE ──(launch_slot_with_task)──> STARTED ──(pre_decode 首次看到)──> PROCESSING_PROMPT
                                                                              |
                                                                              | (整段 prompt 写入 batch)
                                                                              v
                                                                          DONE_PROMPT
                                                                              |
                                                                              | (post_decode 采样出第 1 个 token)
                                                                              v
                                                                          GENERATING
                                                                              |
                                                                              | (stop 条件命中 / 预算耗尽)
                                                                              v
                                                                            IDLE
```

每个状态的进入/退出条件（**全部已核对行号**）：

| 状态 | 进入条件 | 退出条件 |
|---|---|---|
| `SLOT_STATE_IDLE` | 初始状态；`release()` 把 state 设回 IDLE（`:490`） | `launch_slot_with_task` 接到新 task |
| `SLOT_STATE_WAIT_OTHER` | `launch_slot_with_task` 且 `task->is_child()`（`:1828-1830`） | parent 进 `DONE_PROMPT` 后，`decode()` 把 child 直接置为 `DONE_PROMPT`（`:3694`） |
| `SLOT_STATE_STARTED` | `launch_slot_with_task` 且非 child（`:1830`） | `pre_decode` 第一次扫到该 slot 时（`:3069`） |
| `SLOT_STATE_PROCESSING_PROMPT` | `pre_decode` 把 STARTED 改为此（`:3069`） | 整段 prompt 已写入 batch（`:3505` 改为 DONE_PROMPT） |
| `SLOT_STATE_DONE_PROMPT` | `slot.prompt.n_tokens() == slot.task->n_tokens()`（`:3504-3505`） | `post_decode` 第一次扫到，对 completion 改为 GENERATING（`:3755`）；对 embedding/rerank 直接 release |
| `SLOT_STATE_GENERATING` | `post_decode` 在 DONE_PROMPT 分支里改成（`:3755`） | `process_token` 返回 false（stop 条件）-> `release()` -> IDLE |

**为什么需要 WAIT_OTHER**？因为子 slot 跟 parent 共享 prompt 的 KV cache，但 parent 还没算完 prompt 时 child 不能自己跑 -- 跑了就是白算。child 必须等 parent 写完 KV，然后 `copy_state_to` 把 parent 的 KV 拷给 child。注意 `WAIT_OTHER` 的 child 不会被 `pre_decode` 加进 batch（`:3052-3054`：跳过 WAIT_OTHER slot），等 parent DONE_PROMPT 后一次性激活所有 child。

**DONE_PROMPT 与 GENERATING 的分界**：DONE_PROMPT 是"prompt 的 KV 已经写完、最后一个 prompt token 的 logits 也算出来了，但还没采样"。`post_decode` 在 DONE_PROMPT 状态下（`:3736-3755`）做第一次采样、改状态为 GENERATING、记录 `t_start_generation`。embedding/rerank 在这里直接 `send_embedding` / `send_rerank` 然后 release。这分界让"prompt 处理"和"生成"两个阶段清晰可观察 -- `t_prompt_processing = (t_start_generation - t_start_process_prompt) / 1e3` 正好用这两个时点算。

## 6. 两个并发请求的完整时间线（核心场景走查）

读者读完前面五节后，应该能讲清"两个用户同时请求时系统内部发生了什么"。下面走一个具体场景：

**设定**：
- 服务端：`n_parallel = 2`（slot 0、slot 1）、`n_batch = 4`、`n_ubatch = 4`（小到能看清 batching 边界）。
- 用户 A：prompt 1000 token，要求生成 `n_predict = 4`。
- 用户 B：prompt 500 token，要求生成 `n_predict = 4`。
- B 的请求比 A 早 1 step 到达。

**简化假设**：不考虑 prompt cache 复用、speculative、checkpoint、context shift。每个 step 处理的 token 数限制在 `n_batch` 内。

```
step │ HTTP 线程 / 队列 │ 引擎线程 (update_slots) │ slot 0 │ slot 1 │ batch 内容 (id_slot:token:pos)
─────┼──────────────────┼──────────────────────────┼────────┼────────┼──────────────────────────────────
 1   │ B 到达, post B   │ 处理 task B: launch B    │  IDLE  │ IDLE   │  (空)
     │   (id=1)         │   slot 1 = STARTED       │  IDLE  │ STARTED│
 2   │                  │ pre_decode: slot1 STARTED │  IDLE  │ PROCESS│ 1:t0:0, 1:t1:1, 1:t2:2, 1:t3:3
     │                  │   -> PROCESSING_PROMPT    │        │ ING_PR │   (slot1 占满 n_batch=4)
     │                  │ decode(4 tokens)         │        │        │
     │                  │ post_decode: 还没到 DONE │        │        │
 3   │ A 到达, post A   │ pre_decode: slot1 继续推 │  IDLE  │ PROCESS│ 1:t4:4, 1:t5:5, 1:t6:6, 1:t7:7
     │   (id=2)         │   A 还没 launch（队列里  │        │ ING_PR │
     │                  │   等 slot1 release 或   │        │        │
     │                  │   另一个 slot 空出）     │        │        │
  ...│                  │ ...（继续吃 slot1 prompt）│        │        │
 127  │                  │ slot1 prompt 写完       │  IDLE  │ DONE_  │ 1:t499:499 (最后一 token, logits 出)
     │                  │   -> DONE_PROMPT         │        │ PROMPT │
 128 │                  │ pre_decode: slot1 DONE   │  IDLE  │ DONE_  │ 1:s0:500  (slot1 第 1 个采样 token)
     │                  │   -> GENERATING (在      │        │ PROMPT │   *此时 batch 还有空位，A 已 launch*
     │                  │   post_decode 里改)       │        │ ->GEN  │
     │                  │ A 在 process_single_task │ STARTED│        │
     │                  │   里 launch 到 slot0    │        │        │
 129 │                  │ pre_decode: slot0 STARTED│ PROCESS│        │ 1:s1:501, 0:a0:0, 0:a1:1, 0:a2:2
     │                  │   -> PROCESSING_PROMPT   │ ING_PR │ GEN    │   (A、B 同 batch！A 的 prompt + B 的 gen)
     │                  │ decode 同一批             │        │        │
     │                  │ post_decode:             │        │        │
     │                  │   slot1 采到 s1           │        │        │
     │                  │   slot0 还没到 DONE      │        │        │
 130 │                  │ pre_decode: slot0 继续   │ PROCESS│ GEN    │ 0:a3:3 (slot1 这次没新 token 加进 batch
     │                  │   slot1 加新采样 token   │ ING_PR │        │   因为 slot0 还有 prompt 没写完，
     │                  │                          │        │        │   但 slot1 仍保持 GENERATING)
  ...│                  │ ...                      │        │        │
 1129│                  │ slot0 prompt 写完       │ DONE_  │ GEN    │ 0:a999:999 (A 的 logits 出来)
     │                  │   -> DONE_PROMPT         │ PROMPT │        │
 1130│                  │ pre_decode: slot0 DONE   │ GEN    │ GEN    │ 0:s0:1000, 1:s2:502
     │                  │   -> GENERATING         │        │        │   (A、B 同时在生成阶段，同 batch)
     │                  │ decode 同一批             │        │        │
 1131│                  │ post_decode:             │ GEN    │ GEN    │ 0:s1:1001, 1:s3:503
     │                  │   slot0 采到 s1          │        │        │
     │                  │   slot1 采到 s3          │        │        │
 1132│                  │ slot0 stop (n_predict=4) │ IDLE   │ GEN    │ (空)
     │                  │   send_final_response   │        │        │
     │                  │   release               │        │        │
 1133│                  │ slot1 stop (n_predict=4) │ IDLE   │ IDLE   │ (空)
     │                  │   send_final_response   │        │        │
     │                  │   release               │        │        │
```

读这张图时关注几条主线：

1. **slot 状态机演化**：每个 slot 独立走 `IDLE -> STARTED -> PROCESSING_PROMPT -> DONE_PROMPT -> GENERATING -> IDLE`。多个 slot 的状态机互不同步 -- B 可能已经 GENERATING 了，A 还在 PROCESSING_PROMPT。

2. **prompt 阶段是否合批**：上面的简化场景里，slot1 的 prompt 没和 slot0 的 prompt 同时写。但源码确实支持 prompt 合批 -- `pre_decode` 在 `iterate(slots, ...)`（`server-context.cpp:3037`）里同时把多个 PROCESSING_PROMPT 状态的 slot 的 prompt token 塞进同一个 batch，限制只是 `batch.size() < n_batch`（`:3443`）。也就是说**只要 n_batch 还没满，A 的 prompt 和 B 的 prompt token 可以同 batch decode**。这一点是 batching 的真正威力 -- prompt 阶段是 IO 密集（读权重），合批的收益最大。

3. **生成阶段的合批**：`handle_last_sampled_token`（`server-context.cpp:446-479`）把每个 GENERATING slot 刚采样的 token 加进 batch。`iterate(generating, ...)`（`:3022-3024`）对所有 GENERATING slot 调一次。所以 step 1130、1131 里 A、B 的 token 是同一个 `llama_batch` 一起 decode 的 -- 这就是文章开头讲的"一次读权重服务 N 个请求"。

4. **采样是 slot 各自独立做**：`post_decode` 里 `iterate(slots, ...)`（`:3723`）对每个 slot 用自己的 `smpl` 调 `common_sampler_sample`（`:3774`）。每个 slot 独立查自己的 logits 索引 `tok_idx = slot.i_batch - off`（`:3769`），互不影响。

5. **SSE 推回**：每个 slot 采到 token 后，`process_token`（`:1839`）调 `send_partial_response`（`:2061`），把 `server_task_result_cmpl_partial` 丢进 `queue_results`。HTTP 线程的 `rd.next` 拿到结果，按 `res_type`（OAI_CHAT / OAI_CMPL / OAI_RESP / ANTHROPIC）格式化成 SSE 分片写回客户端。A 的 HTTP 线程只看 A 的 `id_tasks`，B 的 HTTP 线程只看 B 的 `id_tasks`，通过 id 路由互不串扰。

6. **slot 释放后唤醒 deferred 任务**：假设 C 在 step 500 到达，但 slot0、slot1 都在忙，`process_single_task` 会调 `queue_tasks.defer(task)`（`:2379`）。slot1 release 时（`server-context.cpp:499`）触发 `callback_on_release`（`:1359-1361`）-> `queue_tasks.pop_deferred_task(id_slot)`（`server-queue.cpp:77`），把 C 的 task 从 deferred 队列提到主队列前面继续处理。

7. **NEXT_RESPONSE 自驱动**：`update_slots` 开头（`:2766-2776`）如果发现不是全 idle，就 post 一个 `SERVER_TASK_TYPE_NEXT_RESPONSE` task -- 这是为了让 `start_loop` 跑完一轮后能立刻接着跑下一轮，不阻塞在等新 HTTP 请求上。这就是"continuous batching"在代码层的自驱动机制。

## 7. 上下文管理：slot 的 prompt 超过 n_ctx 时怎么办

这是初学者最容易担心的点。源码里分**三个层次**处理：

### 7.1 提交 prompt 时就超长 -> 报错

`pre_decode` 在 STARTED -> PROCESSING_PROMPT 转换后立即检查（`server-context.cpp:3108-3139`）：

- **`can_split() == false`**（embedding + 无 memory 模型，必须单 ubatch）：如果 `task->n_tokens() > n_ubatch` 报 `ERROR_TYPE_SERVER`（`:3110-3118`）；如果 `> n_ctx` 报 `ERROR_TYPE_EXCEED_CONTEXT_SIZE`（`:3120-3129`）。
- **`can_split() == true`**（普通 causal 模型）：如果 `task->n_tokens() >= n_ctx` 报 `ERROR_TYPE_EXCEED_CONTEXT_SIZE`（`:3131-3139`）。

报错路径：`send_error(slot, ...)` -> `slot.release()` -> IDLE，task 拿到 error result，HTTP 线程把 error 转成 4xx/5xx JSON 响应。**注意是 `>=` 不是 `>`**：要留至少 1 个位置给生成阶段的 token。

### 7.2 生成阶段快用完 -> context shift

`pre_decode` 里检查 GENERATING 状态的 slot（`server-context.cpp:2835-2898`）：

```cpp
if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
    if (!params_base.ctx_shift) {
        send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
        slot.release(); return;
    }
    if (mctx) GGML_ABORT("not supported by multimodal");
    if (slot.task->is_parent() || slot.task->is_child()) {
        send_error(slot, "context shift cannot be used for shared prompt", ...);
        slot.release(); return;
    }
    // 算 n_keep（保留 token 数）和 n_discard（丢弃多少）
    int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;
    if (add_bos_token) n_keep += 1;
    n_keep = std::min(slot.n_ctx - 4, n_keep);
    const int n_left    = slot.prompt.n_tokens() - n_keep;
    int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);
    n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

    // 对应 04 篇的 seq_rm + seq_add，整段向左搬
    common_context_seq_rm (ctx_tgt, slot.id, n_keep,            n_keep + n_discard);
    common_context_seq_add(ctx_tgt, slot.id, n_keep + n_discard, slot.prompt.n_tokens(), -n_discard);
    // ...
    slot.truncated = true;
}
```

这就是和 `04` 篇 `seq_rm/seq_add` 呼应的地方：把 `[n_keep, n_keep+n_discard)` 这段 KV cache 删掉，然后把它后面所有 token 的 pos 减 `n_discard` 平移过来，等效"丢掉中间一段历史，但保留 system prompt 和最近上下文"。`n_keep` 默认保留到 `n_ctx - 4`（留 4 个位置给后续生成），`n_discard` 默认丢一半。

**关掉 context shift 的条件**：`--no-context-shift`（`common.h:556` 的 `ctx_shift = false`）。关掉后超限直接报错，不做 shift。多模态（有 `mctx`）和 parallel sampling 的 parent/child slot 也都不支持 shift。

### 7.3 KV cache 满了但还没到 n_ctx -> try_clear_idle_slots

`decode()` 返回非 0 错误码时（`server-context.cpp:3656-3663`），先试 `try_clear_idle_slots()`（`:1678`）把别的 idle slot 的 KV 清掉腾位置；如果还失败，就把 `n_batch` 减半重试 -- 这是 fallback 机制，正常路径下不应该走到。

## 8. 请求全流程（对照 `00` 的链路）

```
curl POST /v1/chat/completions  { messages: [...], stream: true }
  -> server.cpp 路由匹配 (server.cpp:199)
  -> routes.post_chat_completions (server-context.cpp:4749)
     -> oaicompat_chat_params_parse(body)         把 OpenAI 字段转内部 task.params
     -> handle_completions_impl (server-context.cpp:4110)
        -> tokenize_input_prompts / process_mtmd_prompt   生成 server_tokens
        -> 构造 server_task，rd.get_new_id()               分配 task.id
        -> rd.post_tasks(tasks)                            投队列 + 登记 id
  -> rd.next(should_stop) 阻塞等结果（流式则挂 res->next 回调）
  ↓ 队列侧
  -> server_queue.start_loop (server-queue.cpp:125) 消费
     -> callback_new_task = process_single_task (server-context.cpp:1448)
        -> get_available_slot (LRU/相似度选 slot)
        -> launch_slot_with_task                            slot 置 STARTED
     -> callback_update_slots = update_slots (server-context.cpp:1451)
        -> pre_decode()    slot 状态推进 + prompt token 进 batch + context shift
        -> batch.render()
        -> decode(n_batch, off, batch_view)  调 llama_decode + speculative
        -> post_decode()   采样 + send_partial_response + DONE_PROMPT->GENERATING
  -> queue_results.send(partial_result)  按 id 路由到 HTTP 线程
  -> HTTP 线程 rd.next 拿到结果，format_oai_sse 推回客户端
  -> 循环直到 rd.has_next() == false，发 "data: [DONE]\n\n"
```

## 9. 关键工程点

1. **OpenAI 兼容**：`oaicompat_chat_params_parse` 把 OpenAI 的 `messages/tools/stream` 转成内部 `task.params`，一处解析、多路复用。`/v1/responses`、`/v1/messages`（Anthropic）也走同一套 `handle_completions_impl`，只是 `res_type` 不同（`server-context.cpp:4110-4115`）。
2. **路由模式**：`server_models_routes`（`server-models.cpp`）支持多模型，`proxy_*` 把请求转发给对应模型实例。router 模式下 server.cpp 主进程不加载模型（`server.cpp:93-94`），只做 HTTP 转发。
3. **动态批处理**：多 slot 合批是吞吐关键。`server_batch` 追踪每个 token 属于哪个 slot（`id_slot`，`server-context.cpp:73`），`iterate(slots, callback)`（`:2678`）逐个处理。batch 的"哪些 slot 能合"由 `can_batch_with`（`:383-387`）判断 -- task 类型相同且 LoRA 配置相同。
4. **上下文管理**：slot 超限时优先 context shift（`seq_rm/seq_add`，呼应 `04`），关掉 shift 则报错；prompt cache（`--cache-ram`）让 idle slot 的 KV 落盘，下次相似 prompt 直接复用（`server-context.cpp:1563` 的 `get_available_slot` 会按 LCP 相似度选 slot）。
5. **取消机制**：HTTP 客户端断连时 `req.should_stop()` 返回 true，`response_reader` 析构调 `stop()`（`server-queue.cpp:441-460`）投递 `SERVER_TASK_TYPE_CANCEL` task，引擎线程在 `process_single_task` 里（`server-context.cpp:2426-2435`）找到对应 slot 调 `release()`。
6. **Sleeping 状态**：所有 slot idle 超过 `idle_sleep_ms`（`server-queue.cpp:178`）后进入 sleeping，可以释放 GPU 资源（`SERVER_STATE_SLEEPING`，`server-context.h:59`）。新请求来时 `wait_until_no_sleep`（`server-queue.cpp:102`）唤醒。

## 10. 追问方向

- `server_queue` 的 deferred 队列重排实现（`server-queue.cpp:77` 的 `pop_deferred_task`，按 `id_slot` 优先匹配）
- `oaicompat_chat_params_parse` 的字段映射细节（`server-common.cpp`，工具调用、reasoning、multimodal 字段如何转）
- SSE 流式的连接生命周期与断连处理（`server-http.cpp`、`server-context.cpp:4267-4340` 的 `res->next` 回调）
- speculative decoding 在 slot 内部的 draft/accept 时序（`server-context.cpp:3755-3942`，`common_speculative_*` API）
- prompt cache 的 LRU + 相似度选 slot 策略（`server-context.cpp:1563-1671`）
