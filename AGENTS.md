# PrismRAG — Agent Instructions

> 本文件定义 Claude Code 在此项目中的行为规范，每次新 thread 启动时自动加载。

## 1. 会话恢复 — 先读 handoff

每次启动新 thread，**必须先读 `handoff.md`**，了解项目当前状态、最新评测结果、云端部署信息，再开始工作。

## 2. 本地环境限制

本机为 **macOS (M 系列, 32GB)**，耗时操作禁止在本机执行，除非用户明确指定。

禁止的操作包括但不限于：
- 下载/安装大模型（ColPali、ColQwen2、BGE、Ollama 等）
- 大规模数据索引构建（ingest 5244 页）
- 全量评测（消融 283 条 / RAGAS 全量）
- 编译 pgvector 等大型依赖

允许的本地操作：
- 代码编辑、测试（`make test`）
- 快速调试（`--max-queries 10` 以内的轻量验证）
- 文档编写、分析结果

## 3. 云端 GPU 时间保护

在云上（AutoDL / RunPod / 其他 GPU 实例）运行时，**GPU 按小时计费，下载操作浪费显存和金钱**。

必须遵守：

### 3.1 先检查缓存，再触网
执行任何需要网络下载的操作前，先检查本地是否已有缓存：
```bash
# 检查 HF 模型缓存
ls /root/autodl-tmp/huggingface/models/ 2>/dev/null
# 检查 Ollama 模型 先看看是否 ollama serve
ollama list 2>/dev/null
# 检查数据集缓存
ls /root/autodl-tmp/huggingface/datasets/ 2>/dev/null
```

### 3.2 发现网络耗时操作 → 停止并报告
一旦检测到需要联网下载（HF 模型、Ollama 模型、pip 安装、数据集），**立即停止当前操作**，向用户报告：
- 正在试图下载什么（模型名/URL/文件大小）
- 预计耗时
- 替代方案（如 Phase 1 无卡时下载 vs Phase 2 有卡时下载）

### 3.3 自动识别运行环境
```bash
# AutoDL 环境检测
source /etc/network_turbo 2>/dev/null && echo "AutoDL with proxy"
# HF 镜像设置
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/huggingface
```

### 3.4 优先使用 HF 镜像而不是代理
- 模型下载（`huggingface_hub`）→ `HF_ENDPOINT=https://hf-mirror.com`
- 数据集下载（`datasets` 库）→ 走代理 `http://172.26.1.26:12798`

## 4. 分支保护

做任何代码更改前，**必须确认当前分支不是 `main`**：

```bash
git branch --show-current
```

### 4.1 如果在 main 分支
先创建功能分支：
```bash
git checkout -b feat/<short-description>
```

### 4.2 分支命名规范
- `feat/` — 新功能（如 `feat/ragas-faithfulness`）
- `fix/` — 修复（如 `fix/skip-index-bug`）
- `docs/` — 文档更新
- `chore/` — 杂项（CI、配置、依赖）

### 4.3 分支生命周期
- 功能完成 → commit → push → PR → 合并到 main → 删除本地分支
- 合并后及时切回 main 同步

## 5. 通用规范

| 规则 | 说明 |
|------|------|
| 会话持久化 | 重要结论、评测结果、学习教训写入 `/Users/theyang/.claude/projects/-Users-theyang-Documents-ai-pdf-rag/memory/` |
| handoff 更新 | 每轮工作完成，更新 `handoff.md` |
| 运行记录 | 每次评测产生的结果归档到 `runs/YYYYMMDD-<description>/`，附带 README |
| commit message | 中英文均可，清楚描述变更内容 |
| 代码改动前 | 先读文件再改，不要凭记忆写