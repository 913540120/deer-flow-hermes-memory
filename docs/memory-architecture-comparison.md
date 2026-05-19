# Hermes vs DeerFlow 记忆架构对比与迁移方案

## 一、设计哲学对比

| 维度 | Hermes | DeerFlow |
|------|--------|----------|
| **核心理念** | Agent 主动管理记忆（工具调用） | 系统自动提取记忆（LLM 后处理） |
| **控制权** | Agent 决定何时、存什么 | LLM 后台分析会话后自动提取 |
| **可见性** | Agent 通过工具随时读写 | Agent 完全无感知，后台自动运行 |
| **存储格式** | Markdown 文本（人机可读） | JSON 结构化数据 |
| **缓存策略** | Frozen Snapshot（会话内不可变） | 每次加载最新文件（可变） |

Hermes 的记忆是 **Agent 自己的笔记本**，Agent 主动决定记什么、改什么、删什么。
DeerFlow 的记忆是 **系统为用户建立的档案**，后台自动从对话中提取，Agent 无法干预。

这两种模式并不冲突，理想状态下应该并存：系统自动提取 + Agent 主动补充。

---

## 二、存储层对比

### 2.1 存储格式

**Hermes：Markdown 文件 + 分隔符**

```
$HERMES_HOME/memories/
├── MEMORY.md    # Agent 个人笔记（环境事实、工具特性、经验教训）
└── USER.md      # 用户画像（偏好、沟通风格、工作习惯）
```

文件内容示例：
```markdown
用户偏好使用 TypeScript 而非 JavaScript 进行项目开发
§
项目部署在阿里云 ECS 上，使用 Docker Compose 编排
§
每次提交前必须运行 pnpm check，否则 CI 会失败
```

- 分隔符：`§`（section sign）
- 限制单位：字符数（模型无关）
- MEMORY.md 默认 2200 字符，USER.md 默认 1375 字符

**DeerFlow：JSON 结构化数据**

```
{base_dir}/users/{user_id}/memory.json
```

```json
{
  "version": "1.0",
  "lastUpdated": "2026-05-13T10:00:00Z",
  "user": {
    "workContext":    {"summary": "...", "updatedAt": "..."},
    "personalContext": {"summary": "...", "updatedAt": "..."},
    "topOfMind":      {"summary": "...", "updatedAt": "..."}
  },
  "history": {
    "recentMonths":      {"summary": "...", "updatedAt": "..."},
    "earlierContext":    {"summary": "...", "updatedAt": "..."},
    "longTermBackground": {"summary": "...", "updatedAt": "..."}
  },
  "facts": [
    {
      "id": "fact_a1b2c3d4",
      "content": "用户偏好 TypeScript",
      "category": "preference",
      "confidence": 0.95,
      "createdAt": "2026-05-13T10:00:00Z",
      "source": "thread_xxx"
    }
  ]
}
```

- 限制单位：Token 数（tiktoken）
- 注入上限：默认 2000 tokens
- 最大 facts：默认 100 条

### 2.2 差距分析

| 差距点 | Hermes | DeerFlow 缺失 | 影响 |
|--------|--------|---------------|------|
| **Agent 可读写** | Agent 通过 memory 工具主动 add/replace/remove | Agent 无法主动写入记忆 | 重要偏好和纠正无法即时保存 |
| **双存储区** | memory（Agent 笔记）+ user（用户画像）分离 | 所有信息混在 facts 列表中 | 无法区分"关于环境的事实"和"关于用户的事实" |
| **字符数限制** | 模型无关的字符数限制 | 依赖 tiktoken 的 Token 限制 | 换模型时 token 计算可能不准确 |
| **Frozen Snapshot** | 会话开始时拍快照，会话内不可变 | 每次调用读最新文件 | 可能破坏 LLM Provider 的 prefix cache |
| **安全扫描** | 写入前检测 prompt 注入、角色劫持、数据外泄 | 无安全检查 | 恶意用户可能通过对话注入记忆内容 |
| **并发安全** | fcntl/msvcrt 文件锁 + 原子写入 | 原子写入（无显式锁） | 多线程/多进程写入风险较低但非零 |

---

## 三、提取/写入机制对比

### 3.1 Hermes：Agent 主动写入 + 后台回顾

**主动写入（核心）**：Agent 在对话过程中通过 `memory` 工具直接写入：

```
工具 Schema:
  action: add | replace | remove
  target: memory | user
  content: 条目内容（add/replace 时必填）
  old_text: 短子串匹配（replace/remove 时必填）
```

写入流程：
1. 安全扫描（prompt 注入、不可见字符检测）
2. 获取文件锁
3. 从磁盘重新读取（拾取其他会话的写入）
4. 去重检查
5. 字符数限制校验
6. 原子写入（tempfile + fsync + os.replace）
7. 立即持久化到磁盘，但**不更新 system prompt 快照**

**后台回顾（辅助）**：每 N 轮（默认 10 轮）在后台线程启动独立 Agent，回顾对话内容并主动决定是否保存记忆。

### 3.2 DeerFlow：系统自动提取

**完整流程**：

```
对话结束
  ↓
MemoryMiddleware（after_agent 钩子）
  ↓ 过滤消息（只保留 user + 最终 AI 回复）
  ↓ 检测纠正/强化信号
  ↓
MemoryUpdateQueue.add()（防抖 30 秒）
  ↓
MemoryUpdater.update_memory()
  ↓ 调用 LLM 分析对话，输出 JSON 更新指令
  ↓ 解析 JSON，应用更新
  ↓ 去重（case-insensitive 内容比对）
  ↓ 置信度阈值过滤（>= 0.7）
  ↓ 上传文件相关内容清除
  ↓
FileMemoryStorage.save()（原子写入）
```

### 3.3 差距分析

| 差距点 | 说明 |
|--------|------|
| **Agent 无法主动写入** | DeerFlow 的 Agent 没有任何工具可以主动保存记忆。遇到用户说"记住这个偏好"时无法即时响应 |
| **无后台回顾提醒** | DeerFlow 没有类似 Hermes 的 Nudge 机制，完全依赖会话结束后的被动提取 |
| **无安全扫描** | LLM 自动提取的记忆内容没有经过安全扫描，理论上可能被间接注入 |
| **无纠正强化信号反馈给 Agent** | DeerFlow 检测到纠正/强化信号后仅传给后台 LLM，不会在当前会话中提醒 Agent |

---

## 四、注入机制对比

### 4.1 Hermes：Frozen Snapshot 注入

```python
# 会话开始时
store.load_from_disk()           # 从磁盘加载
store._system_prompt_snapshot = { # 拍快照
    "memory": render("memory", entries),
    "user":    render("user", entries),
}

# 注入时（每次 LLM 调用都用同一份快照）
store.format_for_system_prompt("memory")  # 返回冻结快照
store.format_for_system_prompt("user")    # 返回冻结快照
```

System Prompt 中的呈现：
```
════════════════════════════════════════════════
MEMORY (your personal notes) [45% — 990/2,200 chars]
════════════════════════════════════════════════
用户偏好使用 TypeScript 而非 JavaScript 进行项目开发
§
项目部署在阿里云 ECS 上，使用 Docker Compose 编排
```

关键特性：**会话内写入不影响 system prompt**，保护 prefix cache。

### 4.2 DeerFlow：动态加载注入

```python
# 每次 LLM 调用前（prompt.py）
def get_memory_prompt_section():
    memory_data = get_memory_data(agent_name, user_id=...)  # 读文件
    content = format_memory_for_injection(memory_data, max_tokens=2000)
    return f"<memory>\n{content}\n</memory>"
```

System Prompt 中的呈现：
```xml
<memory>
User Context:
- Work: 全栈开发者，主要使用 TypeScript 和 Python
- Personal: 中文母语，偏好简洁的沟通方式
- Current Focus: 正在将 Hermes 记忆系统迁移到 DeerFlow

History:
- Recent: 最近在研究多个 AI Agent 项目的架构对比
- Background: 有丰富的 LLM 应用开发经验

Facts:
- [preference | 0.95] 偏好使用 TypeScript 进行前端开发
- [context | 0.90] 项目部署在阿里云 ECS 上
- [correction | 0.95] API 调用必须使用 v2 版本（avoid: 使用了 v1 导致 404）
</memory>
```

### 4.3 差距分析

| 差距点 | 说明 |
|--------|------|
| **无 Frozen Snapshot** | DeerFlow 每次调用 `get_memory_prompt_section()` 都重新读文件。如果后台正在写 memory.json，下次调用内容会变化，破坏 prefix cache |
| **无流式输出清洗** | Hermes 有 `StreamingContextScrubber` 从输出中剥离 `<memory-context>` 标签，防止记忆上下文泄漏给用户。DeerFlow 无此机制 |
| **无 Token 预算精细控制** | DeerFlow 使用 tiktoken 按 token 计算限制，但 tiktoken 编码器与实际模型不一定匹配 |

---

## 五、Hermes 独有功能

### 5.1 会话搜索（Session Search）

Hermes 基于 SQLite FTS5 实现全历史对话检索：

```sql
CREATE VIRTUAL TABLE messages_fts USING fts5(
    content, session_id, role,
    content=messages, content_rowid=id
);
```

搜索流程：
1. FTS5 关键词匹配，按会话分组，取 top N
2. 截取匹配位置前后约 100k 字符
3. 发送给辅助 LLM 做摘要总结
4. 返回聚焦摘要 + 元数据

DeerFlow 完全没有此功能。

### 5.2 外部 Memory Provider 架构

Hermes 实现了可插拔的记忆后端：

```python
class MemoryProvider(abc.ABC):
    def initialize(session_id, **kwargs)         # 连接、创建资源
    def system_prompt_block() -> str             # 注入到 system prompt 的静态文本
    def prefetch(query, session_id)              # 每轮对话前的上下文召回
    def sync_turn(user, asst, session_id)        # 对话完成后持久化
    def get_tool_schemas() -> list               # 暴露给 Agent 的工具 schema
    def handle_tool_call(name, args)             # 处理工具调用
    def shutdown()                               # 清理退出
```

内置支持：Honcho、Mem0、SuperMemory、Holographic、ByteRover、Hindsight、OpenViking、RetainDB 等。

DeerFlow 只有 `MemoryStorage` 抽象（纯存储层），没有 Provider 级别的编排能力。

### 5.3 后台 Nudge 机制

每 N 轮（默认 10 轮）在后台线程启动独立 Agent 实例：

```python
_MEMORY_REVIEW_PROMPT = (
    "Review the conversation above and consider saving to memory if appropriate.\n"
    "Focus on:\n"
    "1. Has the user revealed things about themselves?\n"
    "2. Has the user expressed expectations about behavior?\n"
    "If something stands out, save it using the memory tool."
)
```

使用独立客户端，不干扰主会话的 prefix cache。

---

## 六、迁移方案：从基础建设开始

按照从底层到上层的顺序，将 DeerFlow 的记忆系统对齐到 Hermes 的架构水平。

### Phase 1：存储格式改造（基础建设）

**目标**：将记忆存储从纯 JSON 改为 JSON + Markdown 双层存储，引入双存储区概念。

#### 1.1 引入 MEMORY.md 和 USER.md

**当前**：所有记忆存储在单一 JSON 文件中。

**目标**：
```
{base_dir}/users/{user_id}/memory/
├── memory.json       # 结构化数据（facts、summaries，供程序读写）
├── MEMORY.md         # Agent 笔记（自由文本，供 Agent 工具读写）
└── USER.md           # 用户画像（自由文本，供 Agent 工具读写）
```

**需要修改的文件**：

| 文件 | 修改内容 |
|------|----------|
| `packages/harness/deerflow/agents/memory/storage.py` | `FileMemoryStorage` 增加对 MEMORY.md / USER.md 的读写方法；`_get_memory_file_path()` 扩展为返回目录或具体文件路径 |
| `packages/harness/deerflow/agents/memory/updater.py` | `MemoryUpdater._finalize_update()` 同步更新 Markdown 文件（从 JSON facts 中渲染） |
| `packages/harness/deerflow/config/memory_config.py` | 新增 `memory_char_limit`（默认 2200）和 `user_char_limit`（默认 1375）配置项 |

**预估代码量**：~200 行

#### 1.2 Frozen Snapshot 机制

**当前**：每次 LLM 调用都重新读取文件。

**目标**：会话/线程开始时拍快照，整个会话生命周期内 system prompt 注入使用同一份快照。

**需要修改的文件**：

| 文件 | 修改内容 |
|------|----------|
| `packages/harness/deerflow/agents/memory/storage.py` | 在 `FileMemoryStorage` 中增加 snapshot 缓存：`_snapshot_cache: dict[tuple, dict]`，提供 `load_snapshot()` 和 `invalidate_snapshot()` 方法 |
| `packages/harness/deerflow/agents/lead_agent/prompt.py` | `get_memory_prompt_section()` 改为调用 `load_snapshot()` 而非 `load()` |
| `packages/harness/deerflow/agents/middlewares/memory_middleware.py` | 线程结束时调用 `invalidate_snapshot()`，确保下次线程创建时刷新 |

**预估代码量**：~80 行

#### 1.3 安全扫描

**当前**：无任何写入安全检查。

**目标**：写入记忆前检测 prompt 注入、角色劫持、数据外泄、不可见字符等威胁模式。

**需要新增的文件**：

| 文件 | 内容 |
|------|------|
| `packages/harness/deerflow/agents/memory/security.py` | 从 Hermes 移植 `_scan_memory_content()`、`_MEMORY_THREAT_PATTERNS`、`_INVISIBLE_CHARS` |

**需要修改的文件**：

| 文件 | 修改内容 |
|------|----------|
| `packages/harness/deerflow/agents/memory/updater.py` | 在 `_apply_updates()` 中对新 fact 内容调用安全扫描，拒绝或标记不安全内容 |
| `packages/harness/deerflow/agents/memory/storage.py` | 手动创建 fact 的 API（`create_memory_fact()`）增加安全扫描 |

**预估代码量**：~120 行（新文件 ~80 行 + 修改 ~40 行）

---

### Phase 2：Agent 主动记忆工具

**目标**：让 Agent 能通过工具主动管理记忆，实现与 Hermes 一致的控制能力。

#### 2.1 新增 memory_write 内置工具

**工具 Schema**：
```python
{
    "name": "memory_write",
    "description": "主动保存持久化记忆...",
    "parameters": {
        "action": {"enum": ["add", "replace", "remove"]},
        "target": {"enum": ["memory", "user"]},
        "content": {"type": "string", "description": "条目内容"},
        "old_text": {"type": "string", "description": "短子串匹配标识"}
    }
}
```

**需要新增/修改的文件**：

| 文件 | 修改内容 |
|------|----------|
| `packages/harness/deerflow/tools/builtins/memory_tool.py`（新增） | 实现 `memory_write_tool`，内部调用 `MemoryStore` 的 add/replace/remove；移植 Hermes 的子串匹配逻辑、去重、限制检查 |
| `packages/harness/deerflow/tools/builtins/__init__.py` | 注册新工具 |
| `packages/harness/deerflow/agents/lead_agent/agent.py` | 在 `get_available_tools()` 中条件注册（`memory.enabled` 时添加） |

**关键实现细节**：
- 写入立即持久化到磁盘（MEMORY.md / USER.md）
- 同时更新 JSON 中的 facts 列表（双向同步）
- **不更新 Frozen Snapshot**（遵循 Hermes 模式）
- 工具返回当前 live state（非快照），让 Agent 看到写入结果

**预估代码量**：~250 行

#### 2.2 与自动提取的协调

Agent 工具写入和后台自动提取可能产生冲突，需要合并策略。

**需要修改的文件**：

| 文件 | 修改内容 |
|------|----------|
| `packages/harness/deerflow/agents/memory/updater.py` | `_apply_updates()` 增加：如果新 fact 与 Agent 工具写入的 fact 内容重复（case-insensitive），保留置信度更高或更新的那个 |
| `packages/harness/deerflow/agents/memory/queue.py` | `add()` 时检查是否有 Agent 工具刚写入的条目，如果有则跳过对应内容 |

**预估代码量**：~60 行

---

### Phase 3：后台回顾 Nudge 机制

**目标**：定期提醒 Agent 主动回顾对话并保存记忆，而非仅依赖会话结束后的被动提取。

#### 3.1 Nudge 注入

**需要修改的文件**：

| 文件 | 修改内容 |
|------|----------|
| `packages/harness/deerflow/agents/middlewares/memory_middleware.py` | 在 `after_model` 钩子中增加轮次计数器；达到阈值（默认 10 轮）时，注入一条系统消息："请回顾本轮对话，如果有值得持久保存的信息，使用 memory_write 工具保存" |
| `packages/harness/deerflow/config/memory_config.py` | 新增 `nudge_interval` 配置项（默认 10） |

**预估代码量**：~60 行

#### 3.2 流式输出清洗

**目标**：防止记忆上下文从 LLM 输出中泄漏给用户。

**需要新增/修改的文件**：

| 文件 | 修改内容 |
|------|----------|
| `packages/harness/deerflow/agents/memory/scrubber.py`（新增） | 从 Hermes 移植 `StreamingContextScrubber` 状态机，处理流式 delta 中的 `<memory>` 标签剥离 |
| `app/gateway/routers/threads.py` 或流式响应管道 | SSE 流式输出经过 scrubber 过滤 |

**预估代码量**：~100 行

---

### Phase 4：会话搜索（高级功能）

**目标**：实现全历史对话检索能力。

#### 4.1 会话持久化与索引

**需要新增的文件**：

| 文件 | 内容 |
|------|------|
| `packages/harness/deerflow/memory_search/`（新目录） | 会话搜索模块 |
| `packages/harness/deerflow/memory_search/storage.py` | SQLite + FTS5 存储：sessions 表、messages 表、messages_fts 虚拟表 |
| `packages/harness/deerflow/memory_search/indexer.py` | 中间件钩子：对话完成后自动索引消息 |
| `packages/harness/deerflow/memory_search/search_tool.py` | `session_search` 工具实现：FTS5 搜索 + LLM 摘要 |

**需要修改的文件**：

| 文件 | 修改内容 |
|------|----------|
| `packages/harness/deerflow/agents/middlewares/` | 新增 `MemorySearchIndexMiddleware`，对话结束后将消息写入 FTS5 |
| `packages/harness/deerflow/agents/lead_agent/agent.py` | 注册 `session_search` 工具 |
| `packages/harness/deerflow/config/` | 新增 `memory_search` 配置节 |

**预估代码量**：~500 行

---

### Phase 5：外部 Memory Provider 架构（高级功能）

**目标**：实现可插拔的外部记忆后端（Mem0、Honcho 等）。

#### 5.1 Provider 抽象层

**需要新增的文件**：

| 文件 | 内容 |
|------|------|
| `packages/harness/deerflow/agents/memory/provider.py`（新增） | `MemoryProvider` 基类：initialize、system_prompt_block、prefetch、sync_turn、get_tool_schemas、handle_tool_call、shutdown |
| `packages/harness/deerflow/agents/memory/manager.py`（新增） | `MemoryManager` 编排器：Provider 生命周期管理、工具路由、失败隔离 |
| `packages/harness/deerflow/agents/memory/providers/`（新目录） | 内置 Provider 实现 |

**需要修改的文件**：

| 文件 | 修改内容 |
|------|----------|
| `packages/harness/deerflow/agents/memory/storage.py` | 现有 `FileMemoryStorage` 适配为 `NativeMemoryProvider` |
| `packages/harness/deerflow/config/memory_config.py` | 新增 `provider` 配置项（默认 "native"） |
| `packages/harness/deerflow/agents/lead_agent/prompt.py` | `get_memory_prompt_section()` 增加 Provider 的 `system_prompt_block()` 输出 |
| `packages/harness/deerflow/agents/lead_agent/agent.py` | 注册 Provider 暴露的工具 |

**预估代码量**：~600 行

---

## 七、修改优先级总结

| 优先级 | Phase | 功能 | 预估代码量 | 依赖 |
|--------|-------|------|-----------|------|
| **P0** | Phase 1.1 | 存储：引入 MEMORY.md / USER.md 双存储区 | ~200 行 | 无 |
| **P0** | Phase 1.2 | 存储：Frozen Snapshot 机制 | ~80 行 | Phase 1.1 |
| **P0** | Phase 1.3 | 存储：安全扫描 | ~120 行 | 无 |
| **P1** | Phase 2.1 | 工具：Agent 主动记忆工具 `memory_write` | ~250 行 | Phase 1.1 |
| **P1** | Phase 2.2 | 协调：工具写入与自动提取合并策略 | ~60 行 | Phase 2.1 |
| **P2** | Phase 3.1 | 体验：后台 Nudge 回顾提醒 | ~60 行 | Phase 2.1 |
| **P2** | Phase 3.2 | 安全：流式输出清洗 | ~100 行 | Phase 1.2 |
| **P3** | Phase 4 | 高级：会话搜索（FTS5 + LLM 摘要） | ~500 行 | 无 |
| **P4** | Phase 5 | 高级：外部 Memory Provider 插件架构 | ~600 行 | Phase 1.1 |

**总预估代码量**：~1,970 行

**建议实施顺序**：Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

Phase 1 是所有后续工作的基础，尤其是双存储区（MEMORY.md / USER.md）的设计决定了后续 Agent 工具、Provider 架构的接口形状。Frozen Snapshot 和安全扫描相对独立，可以并行开发。
