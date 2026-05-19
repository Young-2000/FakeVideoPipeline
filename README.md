# DeepSearch Agent — 基于检索的视频伪造检测

基于 **DeepSearch（深度检索）** 的流程：通过 **检索到原始出处视频**，再与输入视频 **逐帧对比**，定位被篡改的点。流程**不依赖**「单看视频就让 VLM 瞎猜哪里假」——伪造结论必须建立在和真实溯源素材的视觉对比之上。

> **仓库说明：** 本项目以**独立仓库**发布：`git clone` 后当前的**克隆根目录**就是包根目录（内含 `src/`、`requirements.txt`）。若你是在更大的单仓 monorepo 里把该目录解压到 `…/something/deepsearch_agent/`，同样在**该目录内**运行下面的命令即可，不必再找「再往下一层 deepsearch_agent」。

## 工作原理

```
输入视频 → [COT 分析] → [DeepSearch 循环：搜索 → 下载 → 粗筛 → 细提取] → [Judge]
```

### 阶段 A：检索与伪造点提取

1. **COT 分析**（1 次 VLM 调用）：对均匀采样的帧做链式推理，提取场地文字、球衣号码、转播角标等物质线索，以及时间连续性、比分条等可疑点；同时生成检索关键词。
2. **DeepSearch 循环**（至多 N 轮）：
   - **搜索**：用 COT 得到的关键词通过 **Serper API**（`google.serper.dev`）检索 YouTube。
   - **下载**：用 yt-dlp 下载候选视频。
   - **粗相关性**（默认 16 帧）：低成本 VLM 判断是否为同一事件 / 同源素材。
   - **细提取**（默认 64 帧）：对伪造视频与候选做详细对比，输出若干条结构化伪造点描述。
   - **充分性检查**：文本 VLM 判断证据是否够用。
   - **反思 Reflect**（失败时）：分析检索失败原因并生成新关键词。
3. **产出**：形如 `{"description": "..."}` 的伪造点列表。

**说明：** 编排器里历史上的「阶段 B」（单独再跑一层伪造分析）在当前版本中**已跳过**：伪造点在 DeepSearch 的细提取阶段已收集，Judge 直接使用这些结果。

### 阶段 C：Judge（裁判）

用大模型对比 **预测伪造点** 与 manifest 中的 **Ground Truth**，为每条 GT 打点并汇总命中率。

## 设计原则

- **禁止「只凭可疑视频空想伪造」**：不会单独问模型「这段视频哪里假？」；论断必须来自与检索到的原始视频的对比。
- **COT 贯穿全程**：物质观察、时间分析、叠加文字等上下文会带入每一轮检索与抽取。
- **单一统一循环**：一套 DeepSearch + 关键词迭代，而非对每个源并行开 N 路盲搜。
- **偏精确、偏保守**：溯源失败时召回会下降——这是刻意的取舍。

## 环境要求

- Python 3.11+
- **Serper API Key**（https://serper.dev 注册获取）：用于 YouTube 视频检索。
- **Node.js / Deno / Bun（至少其一）**：YouTube 下载解签与挑战需要 JS 运行时；未安装时下载容易 403。
- （可选）浏览器导出的 **YouTube Cookie**，用于年龄限制或人机验证场景。
- 与 OpenAI Python SDK **兼容** 的 API：`OPENAI_API_KEY`（以及可选的 `OPENAI_BASE_URL`）。仓库默认示例使用 **OpenRouter**，你可改为官方 OpenAI 或其它网关。

## 安装与配置

```bash
git clone https://github.com/Young-2000/FakeVideoPipeline.git
cd FakeVideoPipeline   # 或你自定义的克隆目录名

pip install -r requirements.txt

cp .env.example .env
# 编辑 .env，至少设置 OPENAI_API_KEY 和 SERPAPI_KEY；按需修改 OPENAI_BASE_URL / OPENAI_MODEL
```

`pip install yt-dlp[default]` 后，请确认命令行中能执行 `yt-dlp --version`。

## 使用方式

### 基本命令

```bash
python -m src.cli /path/to/video/folder
```

目录内需包含：

- 待分析的 **伪造视频**：`<id>.mp4`（`id` 与 manifest 中一致）。
- **`Edit.json`**（或通过 `--manifest` 指定文件名）：评测清单。

### Manifest（清单）格式

```json
[
  {
    "id": "视频 YouTube ID（与 forged 文件名一致）",
    "task": "2.1",
    "topic": "...",
    "video2": "预期溯源 ID 之一或留空",
    "video3": "多源任务时第二个源 ID 或留空",
    "groundtruth": [
      "第一条 GT 文本描述伪造点……",
      "第二条……"
    ]
  }
]
```

- `video2`、`video3` 及兼容的 **`videoN`** 字段会与 `id` 一起构成「oracle」期望溯源 ID（见 `oracle_eval.py`）。
- `groundtruth` 推荐 **字符串数组**（与现有数据集一致）；也支持每项为带 `description`（或 `zh`）字段的对象。
- Judge 至多采用前 **5** 条 GT（与实现对齐）。

### 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--manifest` | `Edit.json` | 清单文件名（位于数据文件夹内） |
| `--output` | `_results` | 输出子目录 |
| `--limit N` | 全部 | 只处理清单中前 N 条 |
| `--only-ids` | 全部 | 逗号分隔的 `id`，只跑指定视频 |
| `--skip-judge` | 关 | 不跑 Judge（无 GT 或仅想看检索时可开） |
| `--resume` / `--no-resume` | resume | 是否复用已有 `per_video` 结果 |
| `--parallel-videos N` | 1 | 并行处理视频数（>1 时多个线程共用同一 Agent 实例，负载高时请自行评估稳定性） |
| `--top-k K` | 10 | 每次搜索保留的候选条数上限 |
| `--max-deepsearch-rounds N` | 5 | 每个溯源组的最大 DeepSearch 轮数 |
| `--max-reflect-rounds N` | 3 | 搜索失败时 Reflect 上限 |
| `--total-sample-frames N` | 64 | COT / 细筛 / sufficiency 的输入视频采样帧数（缓存 `{id}_h480_n64`） |
| `--candidate-sample-frames N` | 64 | 细对齐 / 抽取时的候选视频采样帧数 |
| `--coarse-sample-frames N` | 16 | 粗筛：输入与候选各 16 帧（输入缓存 `{id}_h480_n16`） |
| `--reasoning-model` | `$REASONING_MODEL` 或 `$OPENAI_MODEL` | COT / Reflect 专用模型；粗筛 / 细筛仍用 `$OPENAI_MODEL` |
| `--frame-resize-workers N` | 16 | 单次 ffmpeg 扫片抽帧后，并行缩放 JPEG 到目标高度的线程数（非多路解码） |
| `--query-temperature` | 0.0 | 若干 VLM 调用的采样温度 |
| `--download-dir` | `downloads`（相对当前工作目录解析） | 候选视频下载目录 |
| `--use-cot` / `--no-cot` | 开 | 关闭 COT 时用文件名兜底关键词 |
| `--judge-model` | `$OPENAI_MODEL` | Judge 阶段模型 |
| `--search-only` | 关 | 仅评估检索 oracle：不下载 / 不做视觉验证 |
| `--quiet` | 关 | 减少 Agent 控制台输出 |
| `--log-dir` | `<output>/logs` | 自定义日志目录 |

### 输出目录

```
_results/
├── summary.json          # 机器可读汇总
├── summary.md           # Markdown 汇总
├── per_video/
│   └── <video_id>.json # 单条结果
└── logs/
    └── run_<时间戳>_<视频id>.log
```

### 帧缓存（`.frame_cache/`）

抽帧结果持久化在项目根目录 `.frame_cache/{video_id}_h{height}_n{frames}/`，输入视频与候选视频共用同一命名规则。典型目录：

- `{input_id}_h480_n64` — COT、细筛、sufficiency、Reflect
- `{input_id}_h480_n16` — 粗筛输入侧（16 帧）
- `{candidate_id}_h480_n16` — 粗筛候选侧

命中缓存时跳过 ffmpeg；未命中时单次 ffmpeg 顺序解码抽帧，再并行缩放到目标高度（默认 16 线程）。

### 仅检索模式（Search-only）

不下载、不做帧级验证，只看检索是否在 oracle 命中期望源 ID：

```bash
python -m src.cli /path/to/videos --search-only
```

## 仓库结构（克隆后的根目录）

```
.
├── src/
│   ├── cli.py                 # 批处理编排（阶段 A → C）
│   ├── agent_pipeline_v2.py   # VisualRetrievalAgentV2（DeepSearch 主逻辑）
│   ├── agent/
│   │   ├── tools.py           # VLM、Serper 搜索、yt-dlp 下载、抽帧
│   │   ├── prompts.py
│   │   ├── judge.py           # Judge
│   │   ├── session_state.py
│   │   └── oracle_eval.py     # Manifest / oracle 工具
│   └── utils/
│       ├── agent_helpers.py
│       ├── config.py          # .env 与运行时配置
│       ├── pipeline_log.py
│       └── text_utils.py
├── requirements.txt
├── .env.example
└── README.md
```



