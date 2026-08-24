# PostgreSQL 面试向学习教程（以 PrismRAG 为实战案例）

> 目的：不是泛泛讲 PostgreSQL，而是**结合本仓库真实代码**把 PG 的核心知识点讲透，
> 每一章都给出「面试怎么答」。学完你可以自信地应对：
> 索引 / 事务 / MVCC / 锁 / 批量写入 / DDL 迁移 / pgvector 向量检索 这些高频考点，
> 并且每一项都能举出本项目里的真实例子。
>
> 代码依据：`src/store/pgvector_store.py`（存储层全部逻辑）、`src/retrieval/dense_retriever.py`、
> `docker-compose.yml`、`scripts/cloud_setup.sh`。

---

## 目录

| Part | 主题 | 面试价值 |
|------|------|----------|
| 0 | 环境复现 | 能动手验证 |
| 1 | PG 在系统中的角色 · PG vs MySQL | ⭐⭐⭐ 必考 |
| 2 | 连接与驱动（psycopg2 / autocommit） | ⭐⭐ |
| 3 | 表设计与约束（chunks 表逐列拆解） | ⭐⭐⭐ |
| 4 | DDL 与 Schema 演进（幂等迁移） | ⭐⭐ |
| 5 | 索引：B-tree + HNSW | ⭐⭐⭐ 必考 |
| 6 | 查询实战：Top-k / JOIN / 模糊匹配 / 防注入 | ⭐⭐⭐ |
| 7 | 写入与幂等（批量插入 / ON CONFLICT） | ⭐⭐ |
| 8 | 事务、MVCC 与锁（23h 事故复盘） | ⭐⭐⭐ 加分项 |
| 9 | 零停机迁移（事务内 RENAME swap） | ⭐⭐⭐ 亮点题 |
| 10 | pgvector 专项（算子 / HNSW 调优 / 选型） | ⭐⭐⭐ 稀缺点 |
| 11 | 运维：连接、VACUUM、备份、云上部署 | ⭐⭐ |
| 12 | 面试模拟题 | — |

---

## Part 0 · 环境复现（先动手）

本仓库 PostgreSQL 的使用方式：

```bash
make db              # docker compose up -d db（pgvector/pgvector:pg16，端口 5432）
psql -h localhost -U prismrag prismrag   # 连库，密码 prismrag
```

建表与索引（幂等）：`scripts/cloud_boot_r.sh` 里实际执行的就是 `PgVectorStore().create_schema()`。
你可以在 psql 里跑同样的 SQL 亲手验证（建表 SQL 见 Part 3）。

> 提示：云上（AutoDL）用的是 **PostgreSQL 14 + 从源码编译的 pgvector v0.8.4**（见 `scripts/cloud_setup.sh`），
> 本地 dev 用 **pg16 官方镜像**。二者行为一致，面试提「本地 pg16 / 云上 pg14」显得你真的部署过。

---

## Part 1 · PG 在系统中的角色 · PG vs MySQL

### 1.1 本项目为什么用 PostgreSQL？

PrismRAG 是 RAG 系统，有两路存储分工：

| 路径 | 模型 | 存储 | 职责 |
|------|------|------|------|
| 文本路（Text） | BGE（1024 维向量） | **PostgreSQL + pgvector** | chunk 全量落库、HNSW 向量检索、SQL 过滤、真相源 |
| 视觉路（Visual） | ColPali（多向量） | FAISS | MaxSim 视觉检索，GPU 加速 |
| 关键词路（BM25） | — | rank-bm25（内存） | 启动时以 pgvector 为**真相源**增量对账（`fit_from_pgvector` / `reconcile_from_pgvector`） |

面试回答要点：**PostgreSQL 在这里不只是「向量数据库」——它同时承担了关系存储、向量检索、过滤、聚合统计、幂等约束多重重任**。
文本 chunk 的元数据（doc_id、page_id、chunk_type、section_path…）天然是关系型数据，
用 PG 一份数据既能做 `WHERE doc_id = 'x'` 的过滤，又能做 `ORDER BY vector <=> query LIMIT k` 的检索，不用维护两套存储的一致性。
这就是 pgvector 的核心卖点：**向量和结构化数据共用一套事务、索引、备份体系**。

### 1.2 PostgreSQL vs MySQL（必考对比）

| 维度 | PostgreSQL | MySQL (InnoDB) |
|------|-----------|----------------|
| 并发控制 | **MVCC 多版本**，无 undo log，旧版本靠 VACUUM 清理 | MVCC + undo log，回滚段自动清理 |
| 默认隔离级别 | **Read Committed** | **Repeatable Read** |
| 索引 | B-tree、Hash、**GIN、GiST、BRIN**，扩展可加 HNSW（pgvector） | B-tree 为主，全文索引较弱 |
| DDL | **事务性，可回滚**（本教程 Part 9 的原子换表就靠这个） | 隐式提交，DDL 不可回滚 |
| 扩展性 | 强：CREATE EXTENSION（pgvector / postgis / pg_trgm…） | 弱 |
| 存储引擎 | 单一引擎，行为一致 | 可插拔（InnoDB/MyISAM），引擎间语义有差异 |
| SQL 标准 | 更贴近标准（窗口函数、CTE、FILTER、NULLS LAST） | 兼容性次之 |
| 生态适用 | 复杂查询、GIS、向量、严格 ACID | 互联网简单高并发读写、团队熟悉度 |

**面试回答模板**：选 PG 通常是因为①复杂查询与扩展能力（本项目直接 `CREATE EXTENSION vector`）；
②DDL 事务性带来的安全迁移；③ACID 严格性。选 MySQL 常见理由是互联网大流量简单场景的团队惯性。
RAG/向量场景里 pgvector 是「不加新组件就能用向量」的务实选择（对比：FAISS 只是内存索引库，MongoDB/Redis 各自要引入新集群）。

---

## Part 2 · 连接与驱动（psycopg2 / autocommit）

`src/store/pgvector_store.py` 是唯一的 PG 访问入口，值得逐段读：

```python
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

class PgVectorStore:
    def _default_conn_string(self) -> str:
        # 配置在 config/models.yaml: storage.pgvector.*
        return f"host={host} port={port} dbname={dbname} user={user} password={pass}"

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.conn_string)
            # 关键：先开 autocommit，再 register_vector
            self._conn.autocommit = True
            register_vector(self._conn)
        return self._conn
```

面试要点：

1. **`register_vector` 必须在 autocommit 下调用**：pgvector 的 Python 绑定注册类型时内部会发查询，
   如果 autocommit 还是 False，这个查询会开一个事务且永不提交（详见 Part 8 的 23h 事故）。
2. **连接是 API 单例常驻**：一个长连接复用，避免每次请求握手。缺点是单连接无并发；
   生产上应换连接池（PgBouncer 或 psycopg2.pool）。
3. **`execute_values` 批量执行**：`psycopg2.extras.execute_values` 把多行拼成一条
   `INSERT ... VALUES (...),(...),(...)`，比逐条 execute 快一个数量级（Part 7 展开）。
4. 项目里每个写方法结尾都有 `self.conn.commit()`；读方法不 commit（因为 autocommit=True，SELECT 自动结束事务）。

**高频面试题**：psycopg2 连接默认是什么模式？→ autocommit=False，即每条语句隐式开启事务，必须 commit/rollback
才结束事务；`with conn.cursor()` 只管理游标，**不管理事务**（这点和 MySQLdb 的 with conn 不同）。

---

## Part 3 · 表设计与约束（chunks 表逐列拆解）

`create_schema()` 里的真实建表语句：

```sql
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,          -- 幂等键：ON CONFLICT DO NOTHING 的依据
    page_id INTEGER NOT NULL,
    doc_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_type TEXT NOT NULL DEFAULT 'text',   -- 'text' | 'table' | ...
    text TEXT NOT NULL,
    bge_vector vector(1024) NOT NULL,          -- pgvector 向量列
    doc_ref TEXT NOT NULL DEFAULT '',
    table_summary TEXT NOT NULL DEFAULT '',
    page_hash TEXT NOT NULL DEFAULT '',        -- 页面内容哈希：page diff 判定「未变页」复用
    section_path TEXT NOT NULL DEFAULT '',
    caption TEXT NOT NULL DEFAULT '',
    prev_chunk_id TEXT NOT NULL DEFAULT '',    -- 反范式：相邻 chunk 链接
    next_chunk_id TEXT NOT NULL DEFAULT ''
);
-- B-tree 索引
CREATE INDEX idx_chunks_page_id ON chunks(page_id);
CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
-- HNSW 向量索引（Part 5 展开）
CREATE INDEX idx_chunks_bge_hnsw ON chunks USING hnsw (bge_vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
```

面试拆解：

| 设计点 | 为什么 | 面试话术 |
|--------|--------|----------|
| `chunk_id TEXT PRIMARY KEY` | chunk_id 是外部生成的稳定 ID（幂等重入库），不是自增 | 自增主键适合 OLTP 内部对象；对外/跨系统实体用业务稳定 ID + 唯一约束 |
| `vector(1024) NOT NULL` | BGE 输出 1024 维；固定维度让 pgvector 可以建 HNSW | 维度在创建时固定，索引的哈希是基于维度生成的 |
| `page_hash` 冗余列 | P2 增量更新：同一 doc 修改版只重编码**变化页**，未变页跳过昂贵的视觉编码 | 「以 hash 作缓存键」是幂等/增量系统的通用模式 |
| `prev/next_chunk_id` 反范式 | 回答生成时快速取上下文邻居，避免复杂查询 | 反范式的代价是写时维护；本项目 chunk 只增不改、整体重灌，代价可接受 |
| 全列 `NOT NULL DEFAULT ''` | 代码侧少判空；历史 9 元组/10 元组插入用 `_normalize_chunk_row` 补齐 | 空串 vs NULL 的选择：聚合/展示更省心，代价是语义上分不清「没有」和「空」 |
| `documents` 表 `content_hash UNIQUE` | 内容寻址：同一份 PDF 重复入库 → 定位到同一 doc_id，消除副本 | **UNIQUE 约束 + ON CONFLICT 是幂等写入口**（Part 7） |

`documents` 表：

```sql
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL UNIQUE,   -- 内容指纹，重复内容自动复用
    source_path TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**高频面试题**：`TIMESTAMPTZ` 和 `TIMESTAMP` 区别？→ TIMESTAMPTZ 存储 UTC 时刻，显示时按会话时区转换；
TIMESTAMP 不带时区，存什么就是什么。涉及跨时区/分布式部署一律用 TIMESTAMPTZ。

---

## Part 4 · DDL 与 Schema 演进（幂等迁移）

`create_schema()` 里对旧库平滑升级的做法：

```sql
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS table_summary TEXT NOT NULL DEFAULT '';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS page_hash TEXT NOT NULL DEFAULT '';
```

面试要点：

1. **`IF NOT EXISTS` 让迁移幂等**：新库走 CREATE TABLE，旧库走 ADD COLUMN，同一段代码跑 N 次结果一致。
   这是「启动时自愈 schema」的常用手段，省掉手工 migration 版本管理。
2. **PG 的 DDL 是事务性的**：`ALTER TABLE`、`CREATE INDEX`、`DROP` 都可以在事务里回滚。
   这是 PG 与 MySQL 的一大区别（MySQL DDL 隐式提交）。
3. 生产建议：大表 `ADD COLUMN` 在 PG 11+ 是**纯元数据操作**（默认值非常量时 PG 13+ 也很快，因为不用回填每一行）。
   `CREATE INDEX` 会锁写（默认阻塞 DML，可用 `CREATE INDEX CONCURRENTLY` 不阻塞）。

**加分回答**：`CREATE INDEX CONCURRENTLY` 不持 ACCESS EXCLUSIVE 锁、不阻塞读写，但**不能在事务块里执行**
（因为需要多个阶段）。这个知识点 Part 9 的原子换表方案其实规避了它——不原地加索引，而是直接换整张表。

---

## Part 5 · 索引：B-tree + HNSW

### 5.1 B-tree（常规列）

项目里的三个 B-tree 索引：

```sql
CREATE INDEX idx_chunks_page_id ON chunks(page_id);
CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX idx_documents_hash ON documents(content_hash);  -- UNIQUE 约束自带索引，这行是冗余的
```

面试回答要点：
- 主键和 UNIQUE 约束**自动建索引**，`content_hash` 上再建一个是冗余（可以指出「这行其实多余」，加分）。
- `WHERE page_id = ANY(%s)` / `WHERE doc_id = %s` 走 B-tree 等值查找，O(log n)。
- **什么时候用不到索引**：函数包住列（`WHERE LOWER(doc_id)=...`）、隐式类型转换、LIKE 前导通配 `%xx%`。

### 5.2 HNSW（向量索引）——面试重头

```sql
CREATE INDEX idx_chunks_bge_hnsw ON chunks USING hnsw (bge_vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
```

**原理一句话**：HNSW（Hierarchical Navigable Small World，分层可导航小世界图）把向量建成**多层图**——
上层是稀疏的「高速路」快速接近目标区域，底层是稠密的邻居精确收敛。检索时从顶层贪心走图，每层保留一个
`ef_search` 大小的候选池（beam），逐层下降直到底层。

**三个参数**（必背）：

| 参数 | 含义 | 本项目 | 影响 |
|------|------|--------|------|
| `m` | 每个节点的最大连接数 | 16 | 越大召回越好、图越大、建索引越慢 |
| `ef_construction` | 建图时的候选池大小 | 200 | 越大建图越慢、图质量越高 |
| `ef_search` | 查询时的候选池大小 | 默认 40（可 `SET hnsw.ef_search=100`） | 越大召回越好、查询越慢 |

**为什么用 HNSW 而不是暴力扫描/IVFFlat**：
- 暴力扫描 O(n·d)，52 万条 chunk（本项目量级）不可行；
- IVFFlat 先聚类再查最近簇，**召回有损且对数据分布敏感**，建好后再插入新数据需要重建；
- HNSW 召回率接近精确（99%+），支持增量插入，是 pgvector 当前默认推荐。

**面试易错点**：HNSW 索引**只在「ORDER BY 向量距离 + LIMIT」的查询里被用到**。没有 LIMIT 的
`ORDER BY <=>` 会退化成全表排序；`WHERE` 过滤掉太多行时也可能不走向量索引。可以用
`EXPLAIN` 验证是否走了 `idx_chunks_bge_hnsw`。

**快速验证命令**：

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT chunk_id, 1 - (bge_vector <=> %s::vector) AS score
FROM chunks
ORDER BY bge_vector <=> %s::vector
LIMIT 20;
```

预期看到：`Index Scan using idx_chunks_bge_hnsw`（而不是 Seq Scan）。

**扩展对比**：本项目视觉路（ColPali 多向量）用的是 FAISS。FAISS 是内存索引库（可上 GPU），
HNSW 实现成熟；但 FAISS 索引是**进程内对象**，跨进程共享、持久化、增量更新都要自己管理
（本项目用 `os.replace` 原子替换索引文件 + id map 墓碑）。而 pgvector 的 HNSW 是**真正的数据库索引**，
跟着事务走、有崩溃恢复。这就是「自管 FAISS + 真库 pgvector」混合架构的由来。

---

## Part 6 · 查询实战

### 6.1 Top-k 向量检索（`search_by_vector`）

```sql
SELECT chunk_id, page_id, ..., 1 - (bge_vector <=> %s::vector) AS score
FROM chunks
ORDER BY bge_vector <=> %s::vector
LIMIT %s
```

面试要点：
- `<=>` 是 pgvector 的**余弦距离**算子（范围 [0,2]）；`1 - 距离 = 余弦相似度`，越大越相关。
- 三个距离算子要背：`<->` L2 欧氏、`<#>` 负内积、`<=>` 余弦。对**归一化向量**，余弦和点积排序等价。
- `ORDER BY ... LIMIT k` 必须**同时出现**才会走 HNSW 索引。

### 6.2 批量 IN 查询（`get_chunks_by_page_ids`）

```sql
SELECT ... FROM chunks WHERE page_id = ANY(%s)
```

`ANY(数组)` 把 Python list 直接绑定为参数——比拼 `IN (?,?,?)` 动态占位符安全且简洁。
注意：空列表要先在代码侧判空返回，否则 `ANY('{}')` 语义不同（`= ANY('{}')` 恒为 false）。

### 6.3 聚合统计（`list_documents`）——展示 PG 的 SQL 能力

```sql
SELECT
    c.doc_id,
    COALESCE(d.content_hash, '') AS content_hash,
    COUNT(c.chunk_id)::int AS num_chunks,
    COUNT(DISTINCT c.page_id)::int AS num_pages,
    COUNT(*) FILTER (WHERE c.chunk_type = 'table')::int AS num_tables,
    MIN(c.page_number)::int AS page_from,
    MAX(c.page_number)::int AS page_to
FROM chunks c
LEFT JOIN documents d ON d.doc_id = c.doc_id
GROUP BY c.doc_id, d.content_hash, d.source_path, d.created_at
ORDER BY d.created_at DESC NULLS LAST, c.doc_id
```

这段代码一次覆盖 6 个面试考点：
1. **LEFT JOIN**：以 chunks 为底表，documents 缺失也不丢行（兼容旧数据）；
2. **`COUNT(*) FILTER (WHERE ...)`**：条件计数，比 `SUM(CASE WHEN ... THEN 1 ELSE 0 END)` 更地道；
3. **`COUNT(DISTINCT ...)`** 去重计数；
4. **`::int` 类型转换**：pg 的 COUNT 返回 bigint，转 int 对齐 Python API；
5. **`NULLS LAST`**：没有 created_at 的旧数据排最后；
6. **GROUP BY 的隐式依赖**：`d.content_hash` 出现在 GROUP BY 里——PG 对函数依赖检测严格，
   只出现在 SELECT 不出现在 GROUP BY 的列可能报错。

### 6.4 模糊匹配（`find_chunks_by_ref`）

```sql
WHERE caption ILIKE '%needle%' OR text ILIKE '%needle%' OR COALESCE(table_summary,'') ILIKE '%needle%'
```

面试点：`ILIKE` 大小写不敏感；`%..%` 前导通配**走不了普通 B-tree**，量大了要上 `pg_trgm` 扩展建 GIN 索引。
代码里还展示了动态拼 `(a OR b OR c)` 子句 + 参数化列表的手法——**SQL 是动态拼的，但值永远走参数绑定**。

### 6.5 防注入（贯穿全文件）

所有查询都用 `%s` 占位符 + 参数元组，包括动态拼的 `where` 子句——**动态拼的是 SQL 结构，数据永远参数化**。
只有标识符（表名 `{staging}`/`{live}`）用 f-string 拼接，这是安全边界：表名是代码内常量，不来自用户输入。

---

## Part 7 · 写入与幂等

### 7.1 批量插入（`insert_chunks`）

```python
psycopg2.extras.execute_values(
    cur,
    "INSERT INTO chunks (...) VALUES %s ON CONFLICT (chunk_id) DO NOTHING",
    rows,
    template="(%s, %s, ..., %s::vector, ...)",
)
```

面试要点：
1. **`execute_values` 一次发一条多 VALUES 语句**，网络往返从 N 次降到 1 次；
2. **`%s::vector` 显式类型转换**让 pgvector 识别向量文本；
3. **`ON CONFLICT (chunk_id) DO NOTHING` 实现幂等**：重复入库同一 chunk 静默跳过，
   这是 RAG 批量 re-ingest 不产生重复数据的关键。

### 7.2 upsert 语义（`documents` 表内容寻址）

```sql
INSERT INTO documents (doc_id, content_hash, source_path)
VALUES (%s, %s, %s)
ON CONFLICT (content_hash) DO NOTHING
```

`content_hash` 上的 UNIQUE 约束让「同一内容只留一条记录」，配合 `get_doc_id_by_content_hash`
实现**内容级幂等**：同一份 PDF 重新入库，自动定位到已有 doc_id，而不是产生新副本。
这是 `ON CONFLICT` 的两种典型姿势：`DO NOTHING`（跳过）和 `DO UPDATE`（覆盖），背下来。

### 7.3 TRUNCATE vs DELETE（`truncate_chunks`）

```sql
SELECT COUNT(*) FROM chunks;   -- 先记录旧行数
TRUNCATE TABLE chunks;
```

面试对比：

| | DELETE | TRUNCATE |
|--|--------|----------|
| 本质 | DML，逐行删 | 表级操作 |
| 速度 | 慢（逐行 + WAL + 索引维护） | **极快**（不逐行处理） |
| 锁 | 行级锁 | **ACCESS EXCLUSIVE 表锁** |
| MVCC | 生成死元组，需 VACUUM | 直接置空，无死元组 |
| 事务 | 可回滚 | PG 中**也可回滚**（事务性 DDL） |
| 限制 | — | 被外键引用时需 CASCADE |

**回答模板**：全量清空用 TRUNCATE（本项目 Text re-ingest 场景）；只删部分行用 DELETE。
本项目 `truncate_chunks` 特意注释「不碰 FAISS / documents」——即清空是**按路隔离**的，体现分层一致性设计。

---

## Part 8 · 事务、MVCC 与锁（23h 事故复盘）⭐

### 8.1 事故现场（来自 `pgvector_store.py` 顶部注释，真实经历）

> 本 store 是 API 单例常驻连接。每次 search 的 SELECT 都会让连接进入 **idle-in-transaction**
> 且永不提交，长期持有 chunks 的**共享锁**，挡住 DDL（ALTER TABLE 需**排他锁**），
> 表现为 API 运行数小时后拖住所有 ingest/迁移（**已验证的 23h 泄漏**）。

**事故链**（面试按这个顺序讲，逻辑闭环）：
1. `psycopg2.connect()` 默认 `autocommit=False`；
2. 常驻连接第一次 SELECT 隐式开启事务；
3. 代码没在任何地方 commit/rollback → 事务永远不结束 → 会话状态 `idle in transaction`；
4. 该事务持有 chunks 表的 **AccessShareLock**（SELECT 也要读锁）；
5. 运维/迁移要跑 `ALTER TABLE ... ADD COLUMN` → 需要 **AccessExclusiveLock**；
6. AccessShareLock 与 AccessExclusiveLock **互斥** → ALTER 无限阻塞 → 排队锁把后续所有操作拖死（23h）。

**修复**：`self._conn.autocommit = True`，并且**必须在 `register_vector` 之前**设置
（因为 register_vector 内部发查询，也会开事务）。

### 8.2 由此引出的面试知识网

**MVCC**：PostgreSQL 靠**多版本**实现读不阻塞写、写不阻塞读。UPDATE 不原地改，而是生成新行版本，
旧版本由 VACUUM 清理。MVCC 让 SELECT 不需要共享锁就能保证一致性快照——但 SELECT 依然要拿
**AccessShareLock**（表级，防止并发 DDL 把表删了）。

**锁的等级**（会排级即可）：`ACCESS SHARE`（SELECT）< `ROW SHARE` < `ROW EXCLUSIVE`（INSERT/UPDATE/DELETE）
< `SHARE` < `SHARE ROW EXCLUSIVE` < `EXCLUSIVE` < `ACCESS EXCLUSIVE`（DDL）。
其中 **DDL 的 AccessExclusive 和一切其他锁冲突**，这是事故的根因。

**idle-in-transaction 为什么普遍危险**（不止锁）：
- 持有事务快照 → 阻止 VACUUM 清理死元组 → 表膨胀；
- 占用连接池名额；
- 拖住依赖该表的 DDL 与部分运维。

**排查三板斧**：

```sql
-- 1. 找僵尸事务
SELECT pid, state, now()-xact_start AS duration, query
FROM pg_stat_activity
WHERE state = 'idle in transaction';

-- 2. 看谁在等谁（锁等待链）
SELECT a.pid, a.query, b.pid AS blocked_by, b.query
FROM pg_stat_activity a JOIN pg_locks la ON a.pid = la.pid
     JOIN pg_stat_activity b ON ... ;

-- 3. 兜底超时配置
SET idle_in_transaction_session_timeout = '5min';   -- 或 statement_timeout
```

**加分项**：`idle_in_transaction_session_timeout` 是 PG 14+ 的运行时参数，设一个值能**自动杀僵尸事务**，
是生产环境的保命配置。云上 PG14 刚好支持。

### 8.3 事务隔离级别（背表）

| 隔离级别 | 脏读 | 不可重复读 | 幻读 | PG 默认 |
|----------|------|-----------|------|---------|
| Read Committed | 防 | 可能 | 可能 | ✅ |
| Repeatable Read | 防 | 防 | 防（PG 用快照实现，能防） | |
| Serializable | 防 | 防 | 防 | |

PG 的 REPEATABLE READ 和 SERIALIZABLE 都基于**快照**，所以 PG 下 REPEATABLE READ 实际也能防幻读
（这点和 MySQL 的间隙锁思路不同，面试常被拿来对比）。隔离级别不是越高越好——快照越久，VACUUM 压力越大。

---

## Part 9 · 零停机迁移：事务内 RENAME swap ⭐

`atomic_swap_chunks` 实现**大批量全量刷新的零停机换表**：

```python
conn = self._new_connection()          # autocommit=False，专门开事务
cur = conn.cursor()
cur.execute(f"CREATE TABLE {staging} (LIKE {live} INCLUDING ALL)")   # 1. 建影子表（含索引约束）
self._insert_chunks_on(cur, insert_rows)                             # 2. 新数据全写进影子表
cur.execute(f"ALTER TABLE {live} RENAME TO {old}")                   # 3. 正表改名
cur.execute(f"ALTER TABLE {staging} RENAME TO {live}")               # 4. 影子表上位
cur.execute(f"DROP TABLE {old}")                                     # 5. 丢旧表
conn.commit()                                                        # 6. 提交：全部或全不
```

**面试回答要点**：
1. **数据在影子表里先写完**——重活（插几百万行）不占用正表任何锁；
2. RENAME 是**元数据操作，毫秒级**，只需极短的 AccessExclusive 锁，检索方几乎无感知；
3. 全程**单个事务**，中途异常 `rollback()`，不会出现「正表没了、影子表没上位」的中间态；
4. 能做到这一切的根基：**PG 的 DDL 是事务性的**（MySQL 里 ALTER/RENAME 不可回滚，这招没法安全用）。

**同类方案对比**（设计题常考）：
- 双写 + 切换开关：代码复杂、有过渡期不一致；
- 触发器同步：性能差、难维护；
- `CREATE INDEX CONCURRENTLY` 原地建索引：只解决索引，不解决「全量换数据」；
- **影子表 + RENAME**：简单、原子、可回滚（提交前随时 DROP 影子表）。这是数据库迁移的经典模式。

**跨存储的一致手法**：本项目 FAISS 用 `os.replace`（文件系统原子替换）、BM25 用临时文件 `os.replace`、
pg 用事务内 RENAME——三个存储的「零停机批量刷新」用的是同一个思想：**写副本 → 原子切换**。面试讲这个
「一以贯之的设计哲学」非常加分。

---

## Part 10 · pgvector 专项

### 10.1 类型与算子

```sql
-- 建扩展
CREATE EXTENSION IF NOT EXISTS vector;
-- 建列
bge_vector vector(1024) NOT NULL
```

| 算子 | 距离 | 用在哪 |
|------|------|--------|
| `<->` | L2 欧氏 | 归一化向量的欧氏检索 |
| `<#>` | 负内积 | 内积检索（未归一化） |
| `<=>` | 余弦 | **本项目**（BGE 向量 + 余弦相似度） |

排序时 `1 - (bge_vector <=> q)` 即相似度分数。对余弦，pgvector 内部会把向量归一化再用内积计算，效率高。

### 10.2 HNSW 参数调优（背这三个数）

- `m = 16`：连接数上限，调大到 32 提升召回但索引更大更慢；
- `ef_construction = 200`：建图 beam；越大图质量越高；
- `ef_search`（查询时）：`SET hnsw.ef_search = 100` 提升召回，代价是变慢。**建表时**的 `WITH (m=..., ef_construction=...)`
  是索引级参数，ef_search 是查询级/GUC 参数——这个区分经常被问。

### 10.3 选型：pgvector vs FAISS vs 专用向量库

| 方案 | 优点 | 缺点 | 本项目用法 |
|------|------|------|-----------|
| pgvector | 与业务数据同库、事务/备份/过滤一体化 | 纯向量性能不如专用引擎 | 文本路 1024 维 BGE |
| FAISS | 快、可 GPU、内存友好 | 非数据库：持久化/并发/恢复自理 | 视觉路 ColPali 多向量 MaxSim |
| Milvus/Qdrant | 大集群、超大规模 | 引入新组件、运维成本 | 未用 |

**面试回答模板**：数据量 < 千万级、且向量要跟结构化字段一起过滤 → pgvector 够用且省心；
纯向量规模超大、查询极高并发 → 专用向量库；**多向量（每个 patch 一个向量）**这种非标准检索
（本项目 ColPali MaxSim）→ 专用引擎（FAISS）更灵活，因为 pgvector 一张表一向量列的模型撑不住。

### 10.4 为什么 FAISS 用 flat 而 pgvector 用 HNSW？

`config/models.yaml` 里 `faiss.index_type: "flat"`，原因是 **FAISS 的 HNSW 在 macOS 上有 segfault bug**，
Linux/GPU 上可以开 hnsw。这是个很好的「踩坑」素材：**同样叫 HNSW，不同实现（FAISS / pgvector）是两套独立代码**，
调参、稳定性、存储模型都不同，别被名字迷惑。

---

## Part 11 · 运维

### 11.1 连接管理
- 本项目：API 单例长连接 + autocommit（Part 2/8 的教训）；
- 生产：连接池（PgBouncer）控制 `max_connections`；每个连接是独立进程（PG 进程模型），连接数高了内存暴涨；
- 面试点：PG 是**多进程 + 共享内存**模型（backend process per connection），MySQL 是**单进程多线程**。
  这也是 PG 连接贵、需要池化的原因。

### 11.2 VACUUM / 膨胀
- MVCC 的代价：死元组要靠 VACUUM 清；autovacuum 默认开；
- 高危信号：`UPDATE`/`DELETE` 频繁的表膨胀、`idle in transaction` 拖住清理（Part 8 关联）；
- 面试点：`VACUUM FULL` 需要 AccessExclusive 锁（会锁表），生产慎用；普通 VACUUM 不阻塞读写。

### 11.3 备份
- `pg_dump` 逻辑备份 / WAL 归档 + `pg_basebackup` 物理备份（PITR）；
- 面试点：WAL（Write-Ahead Log）先写日志再写数据页，保证崩溃恢复（ACID 的 Durability）；
  同步提交 vs 异步提交的取舍。

### 11.4 云上部署（本项目真实流程，`scripts/cloud_setup.sh`）
1. `apt install postgresql-14`（AutoDL 镜像自带，pgvector 不在 apt 里）；
2. **从源码编译 pgvector v0.8.4**（`git clone → make install`），幂等检测 `/usr/lib/postgresql/14/lib/vector.so`；
3. `pg_ctlcluster 14 main start`（或 `service postgresql start`）；
4. 建用户/建库/`CREATE EXTENSION vector`；
5. `pg_isready` 探活，Python 连库自检。

面试话术：云上每一步脚本都**幂等**（已存在则跳过）——重新执行不重复下载/编译，节省按小时计费的 GPU 时间。

---

## Part 12 · 面试模拟题

### 快问快答（30 秒每题）

1. **PG 默认隔离级别？** Read Committed。
2. **SELECT 会锁表吗？** 会拿 AccessShareLock（防 DDL），但 MVCC 下不阻塞其他读写。
3. **`<=>` 是什么？** pgvector 余弦距离，`1 - 距离 = 相似度`。
4. **HNSW 三个参数？** m、ef_construction（建图）、ef_search（查询）。
5. **TRUNCATE 能回滚吗？** PG 里能（DDL 事务性）；MySQL 不行（隐式提交）。
6. **`ON CONFLICT DO NOTHING` 前提？** 目标列必须有唯一约束/索引。
7. **为什么 `%abc%` 走不了索引？** 前导通配无法二分定位，常规 B-tree 无能为力（可上 pg_trgm GIN）。
8. **PG 连接模型？** 一连接一进程，多进程 + 共享内存。
9. **VACUUM 干什么？** 清死元组、更新统计信息、防止表膨胀。
10. **PG 里 DDL 能不能回滚？** 能，这是 pgvector 原子换表（Part 9）的根基。

### 场景设计题（3-5 分钟口述）

**题目**：给 500 万 chunk 的 RAG 库设计 PG 存储与检索方案，要求：Top-20 向量检索 < 50ms、
按 doc_id 删除、支持全量重灌不停机。

**参考答法**：
1. **建表**：chunk 行 + `vector(1024)` 列，doc/page 元数据建 B-tree 索引（Part 3 的 chunks 表就是模板）；
2. **索引**：`hnsw (m=16, ef_construction=200)`，查询端调 `ef_search` 权衡召回/延迟；
   用 `EXPLAIN (ANALYZE, BUFFERS)` 验证走 HNSW（Part 5）；
3. **删除**：`DELETE ... WHERE doc_id = %s`（配套 B-tree 索引）；量大时考虑**按 doc_id 分区**，
   `DROP PARTITION` 比 DELETE 快一个量级；
4. **不停机全量重灌**：影子表 + 事务内 RENAME swap（Part 9），几百万行写入不打正表；
   或 `CREATE INDEX CONCURRENTLY` 原地升级索引；
5. **连接与并发**：PgBouncer 池化 + `idle_in_transaction_session_timeout` 兜底（Part 8 事故经验）；
6. **监控**：`pg_stat_activity` 查长事务、`pg_stat_user_tables.n_dead_tup` 盯膨胀。

### 面试反问（向面试官）建议

- 「你们向量检索是自建索引还是用 pgvector / 专用向量库？规模多大？」——顺带展示你对选型的理解。

---

## 自测清单

- [ ] 能默写 chunks 建表 SQL，并解释每个约束的用途（Part 3）
- [ ] 能画出 HNSW 原理一句话 + 说清三个参数（Part 5/10）
- [ ] 能讲清 23h 事故的完整链路：autocommit → idle-in-transaction → 锁互斥 → DDL 阻塞（Part 8）
- [ ] 能手写 `EXPLAIN (ANALYZE, BUFFERS)` 验证 HNSW 索引（Part 5/6）
- [ ] 能说出 PG vs MySQL 在 MVCC / 隔离级别 / DDL 事务性上的差异（Part 1）
- [ ] 能口述影子表换表方案，并指出它依赖 DDL 事务性（Part 9）
- [ ] 能在 psql 里查 `pg_stat_activity` 找僵尸事务（Part 8）

---

## 延伸阅读（仓库内）

- 存储层全量代码：`src/store/pgvector_store.py`
- 检索链路：`src/retrieval/dense_retriever.py`（dense 路）、`src/retrieval/bm25_retriever.py`（真相源对账）
- 部署：`docker-compose.yml`（本地 pg16）、`scripts/cloud_setup.sh`（云上 PG14 + 编译 pgvector）
- 架构文档：`docs/architecture/ingestion.md`、`docs/industrial-pdf-rag-architecture.md`
