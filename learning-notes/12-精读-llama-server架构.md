# 精读：llama-server 架构（tools/server）

> 日期：2026-08-13 | 核心文件：`tools/server/server.cpp`（HTTP 路由）+ `server-context.cpp`（引擎）+ `server-queue.cpp`（任务队列）+ `server-http.cpp`（HTTP 封装）
> 前置：读过 `00`（CLI 复用 server 引擎）。server 是 llama.cpp 的现代服务入口，也是 `llama-cli` 的底层。

## 作用

提供 OpenAI 兼容的 HTTP API（`/v1/chat/completions` 等），把多用户请求调度到推理引擎并流式返回。核心三件套：**HTTP 层 -> 任务队列 -> 槽位(slot)推理**。

## 层次模型

```
HTTP 层  server.cpp + server-http.cpp    解析请求、写响应、SSE 流式
  │
任务队列  server_queue (server-queue.cpp)  派发任务、多请求并发
  │
引擎      server_context (server-context.cpp)
  ├─ 槽位 slot                              每个并发请求占一个 slot
  ├─ 批处理 llama_batch                     多个 slot 的 token 合批喂 llama_decode
  └─ 任务处理 process_single_task -> decode -> post_decode
```

## 一、HTTP 层（`server.cpp`）

`ctx_http.post("/v1/chat/completions", ...)` 注册各路由（:195-219）。关键路由：
- `/v1/chat/completions`（OpenAI 兼容对话）
- `/completion` / `/v1/completions`（补全）
- `/v1/embeddings` / `/rerank` / `/tokenize` / `/detokenize` ...
- `/slots`（槽位状态）、`/metrics`（指标）、`/health`

**路由模式**（`server-context.cpp:4725`）：每个 handler 是 lambda，解析 body -> 调 `handle_completions_impl` -> 返回 `server_http_res_ptr`。`ex_wrapper`（:40）包一层异常处理成错误 JSON。

**流式**：`TASK_RESPONSE_TYPE_OAI_CHAT` 等决定响应格式；流式用 **SSE**（`text/event-stream`），每生成一个 token 发一个 `data:` 分片。

## 二、任务队列（`server-queue`）

- `server_task_type`：COMPLETION / EMBEDDING / CONTROL 等
- `post()` 投递，`start_loop()`/`start_loop(ms)` 消费
- 支持**父子任务**（speculative 的 draft/target 协同）、**任务重排**（高优先级先跑）
- 每个任务带 `id`，结果经 `response_reader` 配 ID 回给对应 HTTP 连接

## 三、槽位模型（slot，核心并发单元）

`enum slot_state`（:57）状态机：
```
IDLE -> WAIT_OTHER / STARTED -> PROCESSING_PROMPT -> DONE_PROMPT -> GENERATING -> IDLE
```
- **槽位** = 一条独立的 KV cache 序列 + 采样器链 + 状态。`n_slots` 决定并发上限
- 多个 slot 的 token 合进同一个 `llama_batch`（`server_batch`），一次 `llama_decode` 并行处理 → **动态批处理**
- prompt 处理与生成都走 `decode()/post_decode()`（对照 `00`）

## 四、请求全流程（对照 `00` 的链路）

```
curl POST /v1/chat/completions
  -> server.cpp 路由 -> post_chat_completions (server-context.cpp:4749)
     -> oaicompat_chat_params_parse(body)    转 OpenAI 参数为内部 task
     -> handle_completions_impl              建 server_task + response_reader
  -> rd.post_task(task)                      投队列
  -> server_queue 消费 -> process_single_task (调度到某 slot)
     -> pre_decode / decode() / post_decode()  推理（同 00）
     -> send_partial_response()              每个 token SSE 推回
  -> 客户端逐分片收流式响应
```

## 关键工程点

1. **OpenAI 兼容**：`oaicompat_chat_params_parse` 把 OpenAI 的 `messages/tools/stream` 转成内部 `task.params`，一处解析、多路复用
2. **路由模式**：`server_models_routes`（`server-models.cpp`）支持多模型，`proxy_*` 把请求转发给对应模型实例
3. **动态批处理**：多 slot 合批是吞吐关键；`server_batch` 追踪每个 token 属于哪个 slot（`id_slot`），`iterate(slots,...)` 逐个处理
4. **上下文管理**：slot 超限时 `prompt_clear`/`shift`（对照 `04`），配合 checkpoint 恢复

## 追问方向
- `server_queue` 的父子任务 & 重排实现（`server-queue.cpp`）
- `oaicompat_chat_params_parse` 的字段映射（`server-common.cpp`）
- SSE 流式的连接生命周期与断连处理（`server-http.cpp`）