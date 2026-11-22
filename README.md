# InferenceFactory

> 来源：factory-everything/factory-everything-项目总表.md ｜ 板块：factory-everything · L7 ｜ 落地约束：4060 8G 单机可跑

## 一、条目信息（原始字段，完整保留）

| 字段 | 内容 |
| --- | --- |
| 项目 | InferenceFactory |
| 类型 | Factory |
| 真正有意思的核心 | 模型 → Backend → Benchmark → 部署 |

---

## 二、核心命题

这个项目应该成为：

> **Model → Runtime → Backend → Deployment Artifact 的统一推理工厂。**

## 三、整体架构

```
                   Model Artifact
                         │
                         ▼
                    Model Loader
                         │
                         ▼
                   Runtime Adapter
          ┌──────────────┼───────────────┐
          ▼              ▼               ▼
       PyTorch         vLLM          llama.cpp
          │              │               │
          └──────────────┼───────────────┘
                         ▼
                      Backend
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
         GPU            CPU           Hybrid
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                    Benchmark
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Latency       Throughput       Memory
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                    Deployment
```

## 四、主流水线

```
Model
 ↓
Inspect
 ↓
Select Backend
 ↓
Convert / Load
 ↓
Warmup
 ↓
Benchmark
 ↓
Stress Test
 ↓
Profile
 ↓
Compare
 ↓
Package
 ↓
Deploy
```

## 五、核心输出不是

```
"模型能跑"
```

**而应该是：**

```
InferenceProfile
├── TTFT
├── TPOT
├── Tokens/s
├── Requests/s
├── VRAM
├── RAM
├── Batch scaling
├── Context scaling
└── Concurrency scaling
```

后三项 `scaling` 是关键——单点数字（"跑出 50 tokens/s"）几乎无信息量，真正说明问题的是**曲线**：batch 从 1 加到 32 时吞吐怎么变、context 从 1K 拉到 32K 时延迟怎么涨、并发上来后 P99 如何劣化。

## 六、真正研究

```
Performance = f(Model, Backend, Batch, Context, Hardware)
```

五个变量构成一个可搜索空间。给定部署硬件，选择哪个 backend、配什么 batch、支持多长 context，是一个可量化择优的工程决策，而不是"试试看哪个快"。

## 七、阶段性落地计划

- **M0｜加载与骨架**：定义 InferenceProfile schema，跑通"加载一个模型 → 单 backend 单次推理"（非自动）。
- **M1｜多 Backend 接入**：实现 PyTorch / vLLM / llama.cpp 三类 runtime adapter 与 GPU / CPU / Hybrid 后端选择。
- **M2｜Profile 与压测**：实现 TTFT / TPOT / Tokens/s / Requests/s / VRAM 度量与 Batch / Context / Concurrency 三维 scaling 曲线。
- **M3｜对比与部署**：多 backend 对比、打包 Deployment Artifact 与选型报告。

## 八、验收标准

能端到端完成「InferenceFactory」，并在其「核心」（模型 → Backend → Benchmark → 部署）上给出可复现的量化结果：

- 输出完整 InferenceProfile，含三条 scaling 曲线
- 给定硬件与约束，可复现地选出最优 backend 配置并产出可部署 Artifact

---

_本 README 的「条目信息」字段来自 `factory-everything-项目总表.md`，架构与流水线为 L7 深化设计稿；可按实际进展修订。_

---

# 九、交付说明：这里到底建成了什么

## 9.1 一句话

**这是一个"推理部署决策"的数字孪生**：它不加载任何真实模型（环境无 torch / triton /
numpy / GPU），而是用 **roofline + 排队论** 从一阶原理算出「某模型在某机器上、
用某后端、某量化、某 batch、某 context、面对某到达率」时的 **TTFT / TPOT / P99 / 显存 / 吞吐**，
再把 **`model × hardware × backend × quant × batch × context × arrival` 的笛卡尔积**
穷举评估、筛掉不可行、按三个目标（延迟 / 吞吐 / 显存）取 Pareto 前沿、
最后**在 SLO 约束下给出每个部署单元的一个决策**。

它不是"跑了个 benchmark"，而是**建了一个可以在秒级重跑的、口径闭合的决策模型**：
`python main.py run` 全量 1944 格，本机总墙钟 **不到 1 秒**（确切值随时间变，见附录 B）。

## 9.2 形状判定（搭骨架前先判定）

这是 **扇出-收拢型·第四形态（部署配置空间）**。

前十种形状见工作区 `SHAPES.md`。与它最近的三种：

| 对比项 | L5-quant（量化设计空间） | L7-servingoptimization | **本项目（部署配置空间）** |
| --- | --- | --- | --- |
| 候选来源 | 量化配置的笛卡尔积 | 服务/调度策略的笛卡尔积 | **模型 × 机器 × 后端 × 量化 × batch × context × 到达率** |
| 是否含随机过程 | 否 | 是（到达序列） | **是（到达率 ⇒ 排队 ⇒ P99 分位数）** |
| 硬约束来源 | 精度预算 | 显存 / P99 | **显存（配置的函数）+ SLO** |
| 收拢成什么 | 一个帕累托集 | 一个策略 | **每个部署单元一个可部署 Artifact** |

多出来的两样，正是这个项目真正的研究点：

1. **一条随机过程轴**（arrival）——它把"配置评估"从确定函数变成**排队系统的分位数**，
   于是有了 `Erlang-C（解析）` 与 `离散事件模拟（数值）` 两条独立实现必须互校。
2. **内生硬约束** —— VRAM 不是外部给定的预算，而是**配置自己的函数**
   （权重 + KV×batch×context + 框架开销）⇒ 可行性是**搜索空间的一部分**，不是过滤器。

## 9.3 三层真值（每层都有两条独立实现）

| 层 | 求的量 | 实现 A | 实现 B | 互校判据 |
| --- | --- | --- | --- | --- |
| **语义无关的成本** | FLOPs / 字节 | `cost.analytic_*`（闭式） | `cost.layerwise_*`（逐层累加） | 相对误差 ≤ 1e-12 |
| **排队论** | `P(等待>0)` / `Wq` | `queueing.analytic`（Erlang-C） | `queueing.simulate`（离散事件） | `P(等待)` 绝对误差 ≤ 0.05 |
| **决策** | 选哪个配置 | `compare.decision` | `invariants` 独立复查 SLO / 可行 / 前缀 | 逐条对抗用例 |

**为什么要两条独立实现**：一条实现错了，另一条不会跟着错——
但前提是**它们真的走了不同的路径**（闭式 vs 逐层、解析 vs 模拟），
而不是同一段代码抄了两遍。这一点由变异测试证明（见附录 B.6）。

## 9.4 定点迭代：`batch` 不是配置，是**结果**

`"配了 batch=64"` ≠ `"64 条同时在跑"`。真正的服务台数 `c` 由
**服务时间 → 系统内请求数（Little 定律）→ 实际 batch** 的不动点决定：

```
c₀ = 配置的 batch
repeat: TPOT(c) → S(服务时间) → L = λ·S（Little） → c' = clamp(⌈L⌉, 1, batch) → 阻尼
until |c' - c| 稳定（<= 8 轮，阻尼 0.5）
```

这解释了本项目**最反直觉的一个读数**：所有决策格的实际 batch 都掉到 **1~2**，
而配置写的是 8 / 64（见附录 B.5）。

## 9.5 三条 scaling 曲线

钉在一个**参考点**上（`batch=16, context=2048, arrival=1.0`），分别扫三个轴，
每条曲线都在「全可行域」上取点（不可行点带 `reason`，不是 `null`）：

- **batch scaling**：吞吐怎么随 batch 变（⇒ 翻过 roofline 拐点后**饱和**）
- **context scaling**：TPOT 怎么随 context 涨（⇒ 是**带宽**成本，不是显存成本）
- **arrival scaling**：P99 怎么随负载劣化（⇒ 稳定区几乎平，然后**掉下悬崖**）

---

# 十、快速开始

```bash
# 跑完整流水线（写 out/）
python main.py run

# 只读：从 out/ 重跑验收 + 对抗用例 + 期望（rc=1 表示有未通过）
python main.py check

# 只读：从 out/ 重建报告并与盘上的逐位比对（产物充分性）
python main.py visualize

# 只读：确定性核对（跑两遍比逐位）与产物充分性
python main.py verify

# 只读：打印摘要
python main.py show

# ⚠️ 会写盘：把当前生效的配置写回 configs/default.json
python main.py dump-config
```

**零第三方依赖**：核心路径只用标准库（`requires-python >= 3.12`）。
没有 `pip install` 步骤。

> ⚠️ **改了 `src/inference/config.py` 里的 `DEFAULTS` 之后，必须跑一次 `dump-config`。**
> 因为 `deep_merge` 里**列表整体替换、磁盘优先**，磁盘上的旧列表会**静默盖掉**新声明。
> 这是本项目最容易"改完没生效却看不出来"的一处。

## 十.1 目录契约

```
L7-inferencefactory/
├── configs/default.json     # 口径与门槛（代码里不硬编码阈值）
├── data/
│   ├── models/*.json        # 3 个模型规格（dense-1b / dense-3b / dense-7b）
│   └── hardware/*.json      # 2 个机器规格（rtx4060-8g / epyc-cpu-64g）
├── src/inference/           # 20 个模块
├── main.py                  # 入口
├── out/                     # 产物（.gitignore）
├── diagnostics/             # 诊断脚本（check_readme / prove_cases_can_fail）
└── requirements.txt         # 空的（零第三方依赖）
```

---

# 附录 A｜骨架、CLI 契约、产物

## A.1 模块骨架（`src/inference/`，20 个）

| 模块 | 职责 |
| --- | --- |
| `rng.py` | 确定性 LCG（64 位）+ `tensor_rng`——排队模拟用，可复现 |
| `emit.py` | 序列化（`newline="\n"`）/ 内容摘要 / 产物读写 / `VERSIONED_FILES` |
| `config.py` | `DEFAULTS`、`deep_merge`、`load_config`、`validate_config`、`protocol_digest` |
| `specs.py` | `ModelSpec`（frozen）/ `HardwareSpec` / 从 `data/` 加载 |
| `cost.py` | FLOPs / 字节：`analytic_*`（闭式）与 `layerwise_*`（逐层）**两条实现** |
| `memory.py` | 峰值内存（权重 + KV + 框架）；可行性；KV/分页 |
| `queueing.py` | Erlang-B / Erlang-C / `analytic` / `simulate`（离散事件）/ 静态批等待 |
| `engine.py` | `evaluate`：定点迭代求 `c`、产能 / 交付吞吐、TTFT / TPOT / P99 |
| `space.py` | `enumerate_configs` / `config_id`（无需版本号） |
| `scaling.py` | 三条 scaling 曲线（钉在参考点） |
| `compare.py` | Pareto 前沿 / 支配 / tie-break / 后端胜出统计 / 决策 |
| `pipeline.py` | 各段 `build_*` + `run_pipeline`（两趟对抗） |
| `invariants.py` | 16 门槛 + **51 对抗用例** + 报告相关用例（第二趟） |
| `expectations.py` | 15 条「设计期望 vs 实测」求值器 |
| `report.py` | `render(products) -> (md, html)`——**只吃产物字典** |
| `visualize.py` | `rebuild` / `verify`（从产物重建报告并逐位比对） |
| `cli.py` | 6 个子命令 + 路径锚定 + `--json` 单段输出 |
| `__init__.py` / `__main__.py` | 包入口 |

## A.2 产物（`out/`，23 个文件）

| 组 | 产物 | 内容 |
| --- | --- | --- |
| 规格 | `models.json` / `hardware.json` / `backends.json` | 3 模型 / 2 机器 / 4 后端 |
| 前段 | `inspect.json` / `selection.json` / `convert.json` / `warmup.json` | 检查 / 可选后端×量化 / 转换 / 预热 |
| 主矩阵 | `matrix.json` | **1944 格**（每格：可行性 + 读数 + 原因） |
| 曲线 | `scaling.json` | 60 条序列 × 3 条曲线 = 1620 点 |
| 压测 | `stress.json` | 按到达率 × 后端的可行性分布 |
| 比较 | `compare.json` | 每部署单元的 Pareto + 决策 |
| 打包 | `artifact.json` | 3 个可部署 Artifact（含期望读数 + checksum） |
| 部署 | `deploy.json` | 每个部署单元的最终状态（含 SLO 复判） |
| 核对 | `crosscheck.json` / `acceptance.json` / `adversarial.json` / `expectations.json` | 三层互校 / 门槛 / 对抗 / 期望 |
| 摘要 | `summary.json` / `config.json` / `version.json` / `timings.json` | 汇总 / 生效配置 / 版本 / 耗时（不参与版本摘要） |
| 视图 | `report.md` / `report.html` | 报告（不参与版本摘要） |

**版本摘要覆盖 19 个确定性产物**（`VERSIONED_FILES`），排除：
`version.json`（自指 + 时间戳）、`timings.json`（耗时，随机器变）、`report.*`（视图）。

## A.3 CLI 契约

| 命令 | 写盘 | 说明 |
| --- | --- | --- |
| `run` | **会写盘** | 跑完整流水线，写 `out/`（23 个文件） |
| `check` | **只读** | 从 `out/` 重跑 16 门槛 + 51 对抗 + 15 期望；rc=1 = 有未通过 |
| `visualize` | **只读** | 从产物重建报告并与盘上逐位比对（`--target` 时才另写他处） |
| `verify` | **只读** | 跑两遍比逐位（确定性）+ 产物充分性；只写临时目录 |
| `show` | **只读** | 打印摘要与 6 个部署单元的决策 |
| `dump-config` | **会写盘** | 把当前生效的配置写回 `configs/default.json` |

> 判据用**两个显式标记** `会写盘` / `只读`：每个子命令的 `--help` 里**恰好出现一个**，
> 与上表用**同一套词**。`diagnostics/check_readme.py` 逐命令机器核这两边是否一致。
> ⚠️ **哈希能证明"没变"，不能证明"没写"**（`dump-config` 写回的常是同一份内容）⇒
> 读写属性只能读 `--help` / 代码来核，不能靠"跑一遍看文件变没变"。

**所有子命令的 `--config` / `--out` / `--json` 都锚定项目根**，不锚 cwd：
从工作区根调用 `python L7-inferencefactory/main.py show` 与在项目内调用结果一致，
`out/` 仍落在项目内。

---

# 附录 B｜实测结果

> 全部数字从**最终 `out/`** 重新导出；`diagnostics/check_readme.py` 逐 token 机器核对。
> 版本 **`v1-2ebc7658daa7`**，协议指纹 `da1af0a2f9901293`。
> 耗时类数字（墙钟）随机器变，正文只给量级，精确值见 B.7 与 `out/timings.json`。

## B.0 一次性总览

| 判据 | 实测 |
| --- | --- |
| 矩阵 | **3 模型 × 2 机器 × 324 配置 = 1944 格**（缺口 **0**） |
| 可行 / 不可行 | **148 可行** / 1796 不可行 |
| 不可行原因种类 | **5 种**（`no_gpu` 567 / `unstable` 566 / `quant_unsupported` 324 / `gpu_oom` 231 / `ram_oom` 108） |
| 双实现互校 | **1458/1458 一致**（flops 相对误差 **0**；bytes 相对误差 **0**；排队 **54/54**，ΔP(等待) **0.0495**） |
| 门槛 | **16 / 16 通过** |
| 对抗用例 | **51 / 51 成立**（不可调的不变量断言） |
| 变异测试 | **51 / 51 条用例都被证明能红**（通用 39 / 配置 5 / 源码补丁 5 / 定向 2） |
| 期望 | **5 成立 / 10 被推翻 / 0 算不出来** |
| 部署决策 | **3 / 6**（另 3 个"SLO 挡住或无解"）；被 SLO 淘汰 **4** 次 |
| 后端胜出 | **2 个**（`llamacpp-cpu` / `vllm-paged`）——不存在"一个后端通吃" |
| Pareto 总点数 | **26** |

## B.1 矩阵规模与可行性

| 数量 | 值 |
| --- | --- |
| 模型 / 机器 / 后端 / 量化 / batch / context / arrival | 3 / 2 / 4 / 3 / 3 / 3 / 3 |
| 配置总数（去重后） | **324** |
| 部署单元（模型×机器） | 6 |
| 格数 | **1944** |
| 可行格 | **148** |
| 缺口（枚举遗漏） | **0** |

**可行性是真维度**：5 种不可行原因各自都非空（见上表）⇒ 可行性不是"要么全可行、
要么全不可行"的装饰。其中 `unstable`（566 格）是**排队论意义上的不稳定**（ρ ≥ 1），
不是实现里的异常分支。

## B.2 三个轴的缩放（门槛钉住的分界线）

| 轴 | 现象 | 门槛 | 实测 |
| --- | --- | --- | --- |
| batch | 吞吐随 batch 加速 | `>= 3.0` | **12.52** |
| context | 拉长后 TPOT 涨幅 | `>= 1.2` | **20.82** |
| arrival | 到达率上来后 P99 涨幅 | `>= 2.0` | **5.95** |
| roofline | 出现的 bound 种类数 | `>= 2` | **2**（`memory` / `compute` 都出现） |
| KV 占比 | 可行域内 KV 占峰值比例峰值 | `>= 0.5` | **0.981** |

三条轴都不是"平的"——这一句是门槛在管的事。**但三条轴的形状**（饱和 / 带宽 / 悬崖）
才是本项目的结论，逐条见 B.4 的期望表。

## B.3 16 条门槛（全部可调，写在 `config.acceptance`）

| 键 | 判据 | 实测 |
| --- | --- | --- |
| `matrix_gap` | `<= 0` | 0 |
| `n_cells` | `>= 1000` | 1944 |
| `n_feasible` | `>= 100` | 148 |
| `n_infeasible_reasons` | `>= 3` | 5 |
| `batch_speedup` | `>= 3` | 12.5196 |
| `context_tpot_growth` | `>= 1.2` | 20.8165 |
| `p99_growth_with_load` | `>= 2` | 5.94821 |
| `crosscheck_flops_rel` | `<= 1e-12` | 0 |
| `crosscheck_bytes_rel` | `<= 1e-12` | 0 |
| `queue_sim_abs` | `<= 0.05` | 0.0495436 |
| `kv_share_peak` | `>= 0.5` | 0.980989 |
| `roofline_bound_variety` | `>= 2` | 2 |
| `n_backend_wins` | `>= 2` | 2 |
| `n_models_with_oom` | `>= 1` | 1 |
| `n_slo_exclusions` | `>= 1` | 4 |
| `n_pareto_points` | `>= 5` | 26 |

⚠️ `queue_sim_abs` 是**顶格通过**（0.0495 vs 限 0.05）—— 见 B.4 最后一行，
这不是调参调出来的，是 warmup 比例的取舍（说明见附录 C）。

## B.4 设计期望 vs 实测（15 条；**10 条被推翻**）

> 被推翻的那几条才是这个项目的产出。**没有为了变绿而改判据。**

| # | 期望 | 判 | 实测 |
| --- | --- | --- | --- |
| 1 | 吞吐随 batch **线性**增长 | **推翻** | 最差处 batch 放大只换来线性预期的 **0.0084** 倍（`dense-1b@epyc-cpu-64g\|llamacpp-cpu\|int4`）⇒ 翻过拐点后**饱和** |
| 2 | 最优配置一定用**最大** batch | **推翻** | 1 个部署单元的最优 batch 不是 64（更大的 batch 要么 OOM、要么把 P99 推出 SLO） |
| 3 | 量化越狠越好（免费午餐） | **推翻** | 22 组里更狠的量化**反而更慢** ⇒ 量化省的是**带宽**不是算力 |
| 4 | P99 与到达率**成正比** | **推翻** | 弹性区间 **[0.000, 2.130]** ⇒ 稳定区几乎不随负载变，然后**直接掉下悬崖** |
| 5 | context 拉长只多占显存、不影响速度 | **推翻** | 12 条曲线上 TPOT 明显变慢（每步都要把 `B·C` 的 KV 读一遍，是**带宽**） |
| 6 | CPU 后端在任何部署单元上都不会最优 | **推翻** | CPU 在 **`dense-1b@epyc-cpu-64g`** 上就是最优（8G 卡装不下的模型，最优解是让 CPU 跑） |
| 7 | 存在一个 backend **通吃**所有部署单元 | **推翻** | **2 个**不同后端赢过 ⇒ "选哪个后端"**依赖模型与机器** |
| 8 | 吞吐最高的那格一定满足 SLO | **推翻** | **4 次**"吞吐最优"被 SLO 淘汰 ⇒ 吞吐最优点落在**饱和边缘** |
| 9 | 纯 eager 的 TTFT **低于** vLLM | **推翻** | vLLM 在 **18 组**上 TTFT 更低（调度器开销 < 无调度器的浪费） |
| 10 | 到达率越高、产能越高 | **推翻** | **36 段持平 / 0 段升降** ⇒ 产能是**配置的属性**，与到达率无关 |
| 11 | 分页 KV 让 context 上限与 batch 上限**解耦** | 成立 | **64 组**证据：同 `(B, C)` 下分页可行而连续预留 OOM |
| 12 | 小 batch 的 decode 是**带宽**受限 | 成立 | 最小 batch 的可行格里 **34/38** 是 bandwidth-bound |
| 13 | prefill 是**算力**受限 | 成立 | **148/148** 可行格的 prefill 是 compute-bound |
| 14 | 最长 context 下 KV 是显存主要占用 | 成立 | 最长 context（131072）下 KV 最多占 **0.981** |
| 15 | 混合 offload 是**显存换吞吐** | 成立 | **9 组**同框格子：显存更低 **9**、产能更低 **9** ⇒ 是一门**交易** |

**被推翻的 4 条里最值钱的两条**：

- **#4「P99 不随负载渐进劣化」**：负载上来时 P99 先在稳定区**几乎不动**，
  然后**直接从"可行"跳到"不可行"**——中间没有"缓慢变差"的过渡带。
  这意味着**压测里"加一点负载试试"是探不到边界的**，边界是断崖。
- **#10「产能与到达率无关」**：这条一开始被实现误报成"36 段下降"，真因是
  判据把**持平**算成了下降（见附录 C）。修好后它变成了一条**正面结论**：
  产能是**配置的属性**；随 λ 变的是 P99（排队）与**交付吞吐**，不是产能。

## B.5 部署决策（3/6）与"batch 不是配置"

| 部署单元 | 决策 | 吞吐 | P99 | TPOT | 显存占比 | 实际 batch |
| --- | --- | --- | --- | --- | --- | --- |
| `dense-1b@epyc-cpu-64g` | `llamacpp-cpu\|int8\|b8\|c2048\|a0.25` | 112.9 | 1571.1 | 5.81 | 0.0211 | **1** |
| `dense-1b@rtx4060-8g` | `vllm-paged\|int8\|b64\|c2048\|a0.25` | 1263.8 | 823.2 | 5.85 | 0.3505 | **1** |
| `dense-3b@epyc-cpu-64g` | —（SLO 挡住） | — | — | — | — | — |
| `dense-3b@rtx4060-8g` | `vllm-paged\|int4\|b64\|c2048\|a0.25` | 365.4 | 1311.1 | 8.11 | 0.4549 | **1** |
| `dense-7b@epyc-cpu-64g` | —（无解） | — | — | — | — | — |
| `dense-7b@rtx4060-8g` | —（SLO 挡住） | — | — | — | — | — |

（吞吐单位 tokens/s，P99 / TPOT 单位 ms。）

**注意最后一列**：三个决策格**配置的 batch 是 8 / 64 / 64，实际同时在跑的却是 1**。
这不是 bug，是 9.4 说的定点迭代的**结果**——
在低到达率（0.25 req/s）下，系统里平均只有约 1 条请求在服务，
**配多大的 batch 都填不满**。这正是期望 #2 被推翻的机制。

**为什么 3/6 而不是 6/6**：
- `dense-3b@epyc-cpu-64g`：6 个可行格，**0 个**满足 SLO ⇒ `slo_blocked`。
- `dense-7b@rtx4060-8g`：11 个可行格，**0 个**满足 SLO ⇒ `slo_blocked`。
- `dense-7b@epyc-cpu-64g`：**0 个可行格**（模型根本装不进也跑不动）⇒ 无解。

## B.6 变异测试：51/51 条用例都能红

对抗用例"全绿"本身没有信息量——**绿着但什么都没验的用例**才是危险。
`diagnostics/prove_cases_can_fail.py` 对每条用例施加四类变异，直到它变红：

| 变异类型 | 条数 | 说明 |
| --- | --- | --- |
| 通用变异 | **39** | 删键 / 改数 / 截列表 / 反转 / 删产物，**只作用在该用例真的读过的产物上** |
| 配置变异 | **5** | 改 `cfg`（清空轴 / 塞重复 / 改 seed / 改位宽 / 改 prompt） |
| 源码补丁 | **5** | 换纯函数（Erlang-B/C、两条 layerwise、`protocol_digest`、`weight_bytes` 反单调） |
| 定向变异 | **2** | 每条写清对应哪种真缺陷 |

**结论：51 / 51 全部被证明能红**（`n_generic: 39, n_targeted: 12, seed: 20261003`）。
它当场抓出过：

- **1 条弱的**：`protocol_digest_is_stable` 原来比的是**同一对象的两次摘要**——
  源码补丁 `lambda c: f"unstable-{id(c)}"` 照样绿 ⇒ 改成比**两份深拷贝 + 一份 dump→load 副本**。
- **探针的 3 个自身 bug**（详见附录 C）：不共享 `touched` / 靶集匹配 / 记类型不记值。

## B.7 确定性与产物充分性

- **确定性**：两次 `run` → **23 个产物中 21 个逐位相同**；
  不同的**只有** `timings.json`（耗时）与 `version.json`（时间戳）。
- **产物充分性**：从 `out/` 的产物字典重建报告（`visualize`）⇒
  `report.md` 与 `report.html` **与盘上的逐位相同** ⇒ **产物足以重造报告**
  （渲染器一行不读世界 / 模型 / 真值）。
- **只读契约**：8 个只读组合跑完，`out/` 逐位不变；`--json` 时 stdout 都是**一段**合法 JSON。
- **从工作区根调用**：6 个子命令 rc=0，`out/` 仍落项目内。
- **墙钟**：本机 `run` 总墙钟约 **0.58 s**（`timings.json` 记精确值，随机器变——
  **不写进正文**，因为每重跑一次它就成了"上一次的读数"）。

---

# 附录 C｜踩过的坑

> **编号跨项目连续**：本项目占 **C.389–403**（接上一项目 `L7-kvcachefactory` 的 388）。
> 每条都对应一次真实的红 / 一次真实的误判，不是"注意事项清单"。

### 389. `kv_share` 的口径写窄了，于是它假装是「可行域」的问题

`_obs_kv_share` 第一版取的是「**最长 context 下** KV 占峰值比例的最大值」。
这在门槛 `kv_share_peak >= 0.5` 下给出了 0.2699 —— 不达标，看起来像"可行域太薄"。

真因是**口径**：问的是「KV 在可行域内最多能占多少峰值占用」，
而"最长 context"那一档恰恰是**最难可行的**（KV 越大越容易 OOM）⇒
在"最长 context"上采样，等于在**最不可能出现高占比的地方**找高占比。
改成「**全可行域**最大」后是 0.981。

⇒ 两个**看起来一样**的口径（"最大 context 下最大" vs "全可行域最大"）
回答的是**两个不同的问题**。改窄口径不会报错，只会给出一个**稳稳偏小**的数，
而它看起来像个正常的读数。

### 390. λ 轴不跨产能两侧 ⇒ 高位整段是死的

`arrival` 轴第一版是 `[0.5, 2.0, 8.0]`，而可行产能中位数约 **5.2 req/s**。
于是 8.0 那一档的**几乎所有格都是 `unstable`**（ρ ≥ 1），
`p99_growth_with_load` 与 `n_backend_wins` 都被这个死区拖死。

改成 `[0.25, 1.0, 4.0]` 后：低位（0.25）远低于产能 ⇒ 各项裕度看得清；
中位（1.0）跨在中间；高位（4.0）逼近产能 ⇒ **P99 的悬崖出现**。

⇒ **设计轴必须跨在被测量量的两侧**。只在一侧取点，
拿到的不是"这个轴的形状"，而是"这个轴在死区里的形状"。

### 391. 把 `llamacpp-cpu` 设成单流 ⇒ 两条轴同时假死

`llamacpp-cpu` 原来配的是 `slots_mode="fixed"` / `slots_fixed=1` / `batching="none"`。
后果有两个，**都不报错**：

1. `slots_mode="fixed"` 意味着**服务台数 `c` 与 batch 无关**（恒为 1）
   ⇒ batch 轴对它**完全失效**（改 batch 读数一动不动）；
2. `c=1` 是一个 M/M/1 系统 ⇒ **P99 必然爆掉** ⇒ CPU 后端永远过不了 SLO
   ⇒ 期望 #6「CPU 从不是最优」**看起来成立**，其实是被人为设定造出来的。

改成 `slots_mode="batch"`（`c` = 配置的 batch）+ `batching="static"` 后，
batch 轴活了、CPU 在 `dense-1b@epyc-cpu-64g` 上赢了下来，
期望 #6 被**诚实地推翻**。

⇒ **`slots_mode` 决定排队模型的服务台数 `c`，比 `gpu_eff` 重要得多**。
一个"性能参数"写错，会让**两条独立的设计轴**同时失去分辨力，
而症状只是"这个后端好像总是最差"。

### 392. 时间键白名单漏了**速率**：`_s` 后缀是歧义的

`no_undeclared_time_key_in_the_versioned_products` 要求"产物里不许出现未声明的时间键"。
白名单原来只收 `_ms` / `_seconds` / `_s` 这类**时间**后缀。

但 `_s` 同时匹配 `tokens_per_s`——那是**速率**（1/秒），**不是时间**。
只收前者 ⇒ 用例去咬吞吐读数（`tokens_per_s` / `delivered_tokens_per_s`
/ `capacity_req_per_s` / `served_req_per_s` / `max_gain_ms`），
把**目标轴**当成**失真**。

⇒ **时间键白名单必须同时收时间与速率**，或者干脆按"是不是测量"来分类。
后缀匹配在 `_s` 上是**多义的**。

### 393. `exempt` 用错豁免集：口径层产物是整体豁免的

同一个用例还报了 `backends.json` 里的 `launch_overhead_us` / `scheduler_overhead_ms`。
第一版的修法是 `exempt = (name == "config.json")` —— 只豁免了配置。

真因是**口径层产物整体豁免**：`config.json` 与 `backends.json` 里的时间键是
**参数**（超时、launch 开销），不是**测量**。把豁免集写成单个文件名，
就会在下一个同类文件上再红一次。

⇒ 改成 `exempt = (name in _CATALOG_FILES)`，并把这个集合**显式命名**。
**"哪些文件的数字是参数不是测量"本身是一条口径**，不该散落在一个 `==` 里。

### 394. `hybrid_beats_pure_cpu_on_throughput` 结构上不可测

这条期望原来写的是「混合 offload 比**纯 CPU** 吞吐高」。
但 `llamacpp-hybrid` 的 `device="hybrid"` **需要 GPU**，
而纯 CPU 机（`epyc-cpu-64g`）上它直接 `no_gpu` ⇒ **比对集合恒为空**。

症状是这条期望永远 `unmeasurable`，而 `unmeasurable` 看起来像"数据不够"，
不像"判据写错了"。

⇒ 改写成 **`hybrid_trades_throughput_for_vram`**：跟**纯 GPU** 后端比，
验证"显存更低 **且** 产能更低"。这条**可测**（9 组同框格子），
而且它比原来那条**更有信息量**——原版只说"谁更快"，新版说出了**代价结构**。

⇒ **一个恒为 `unmeasurable` 的判据，比一个被推翻的判据更没用**：
它占着一个位置，却连"这条假设对不对"都没回答。

### 395. 判据把「持平」算成「下降」⇒ 把真结论说成假结论

`higher_arrival_raises_throughput` 第一版：

```python
if b > a: up += 1
else:     down += 1          # ← 持平也进这里
```

于是"产能与负载无关"这条**真结论**，被报成"**36 段下降**"——
一个完全相反、而且听起来很有信息量的假结论。

⇒ 必须**三分类**（升 / 持平 / 降）。二分类在"相等"这个常见结果上，
**默认值就是错的**。

### 396. `tokens_per_s = eb/tpot` 与交付吞吐自相矛盾

第一版吞吐用 `eb / tpot`（只算 decode 阶段的服务台数 × 每 token 时间）。
但交付吞吐（`delivered`）是 `59 请求 × 128 token / 7.4 s = 1024 tok/s`，
而读数印 **2500 tok/s** —— **同一个系统两个吞吐，差 2.4 倍**。

真因是 `eb/tpot` 只算了 decode，**没算 prefill**，而 prefill 在高 batch 下是大头。
改成**满批产能** `c·O / S(c)`（`S` 含 prefill + decode）后自洽。

⇒ **同一张表里的两个吞吐必须同一口径**。
"产能"（满批、benchmark 口径）与"交付"（`min(λ, 产能)·O`，在稳定区恒等于 `λ·O`）
是**两个数**，不能合成一个。

### 397. `protocol_digest_is_stable` 是个**弱判据**，被变异测试当场抓出

这条用例原来是「对同一份配置求两次摘要，两次相等」。
它**恒为真**——因为它比的是**同一个对象的两次调用**。

变异器用一个源码补丁证明它能红：把 `protocol_digest` 换成
`lambda c: f"unstable-{id(c)}"`（每次调用返回不同的值），用例**照样绿**
（因为它两次调用的其实是同一个常量对象）。

⇒ 改成**两份深拷贝 + 一份 dump→load 副本**三方比较。
现在它不仅验"稳定"，还验"**摘要真的读了配置内容**"。

⇒ 这是**变异测试最值钱的一次产出**：一个全绿的用例，
从头到尾什么都没验，而它占着一个"这里验过了"的位置。

### 398. 变异器自身的 3 个 bug：探针不共享 `touched`

`prove_cases_can_fail.py` 第一版跑出 **`generic 0`** ——
通用变异**一条都没命中**。三层原因，全在**工具**里，不在被测代码里：

1. **探针每层新建 `self.touched = []`** ⇒ 记录散在 N 个子实例里，
   外面只看根实例 ⇒ 每个用例只有 **1 条**记录、变异池**恒空**。
2. **`touched_files` 按 `.` 切一层**（`matrix.json` → `matrix`）
   与产物键对不上 ⇒ 靶集恒空、池恒 0 —— **产物名里含点**，
   必须先按**已知产物名**剥前缀再切下标。
3. **`walk_json_nums` 记的是 `int`/`float` 类型而不是值** ⇒ `abs(type)` 崩。

⇒ 三个 bug 的**共同症状**都是"用例恒真"。
**"打不中靶"会被误判成"用例恒真"** —— 这是变异测试最危险的假阳性，
因为它的结论方向是"没问题"。

### 399. 通用变异打不动的用例要有**另外三类靶**

修好探针后，还剩 **10 条**打不动 —— 它们读的是 `cfg` / 模型规格 / **纯函数**，
通用变异（只改产物）**结构上碰不到**。

⇒ 补三类靶：
- **配置变异**（改 `cfg`）：清空轴 / 塞重复 / 目录增删 / 改 seed / 改位宽 / 改 prompt；
- **源码补丁**（换纯函数）：Erlang-B/C、两条 layerwise、`protocol_digest`、`weight_bytes`；
- **定向变异**：每条写清对应哪种真缺陷。

⚠️ `cfg_mutations` 第一版还用**空 `touched`** 去筛自己 ⇒ 一条都生成不出来 ⇒
改成**无条件生成**。**用"读过的靶"来筛变异，本身就是一个循环依赖。**

### 400. `frozen=True` 的 dataclass 不能 `setattr`，且"换成常数"打不红是对的

给 `ModelSpec.weight_bytes` 打补丁时，`setattr(obj, name, bad)` 直接抛
`FrozenInstanceError`（`@dataclass(frozen=True)`）⇒ 只能**类级**
`setattr(cls, name, bad)`。

更值得记的是**另一半**：把 `weight_bytes(bits)` 换成**常数**，
用例**打不红是对的** —— 那条用例的判据是「**非降**」（位宽升 ⇒ 字节不降），
常数**不违反**非降。

⇒ 定向破坏必须**反单调**（把 `bits=4` 映射到比 `bits=16` **更大**的值），
才能打中"单调性"这个判据。**"打不红"有时是判据对，有时是变异弱 —— 必须能分清。**

### 401. `__pycache__` 与诊断脚本自身让补丁"看起来没生效"

多次判定"源码补丁没生效"，真因是 **`.scratch/__pycache__`**（诊断脚本自己也有缓存）
与被 patch 模块的字节码缓存干扰。

⇒ 之后一律 **`rm -rf .scratch/__pycache__` + `python -B`**。
**"补丁没生效"与"补丁生效了但打不红"是两种结论，必须先排除前者。**

### 402. `prefill` 的 `lm_head` 被算 `T` 遍

双实现互校是**唯一**发现这个错的地方：`analytic_prefill` 把 `lm_head`
按 `T`（token 数）累加，而 `layerwise_prefill` 按一次算 ⇒
flops 相对误差不为 0。

修成 `2·W_mat·B·T` 口径后两路一致。
⇒ **双实现互校的价值不在于"一致时很放心"，而在于"不一致时指得出位置"。**
如果只有一条实现，这个错会一直静默存在，而所有读数看起来都合理。

### 403. `visualize` 报告里的"-2"：渲染器读了什么本身就是一条不变量

（这是本项目沿用工作区既有约定时**再次撞上**的坑。）
`len(run.products) - 2` 在 `products_of` 调用它时 `products` 还是空的
⇒ 报告印「其余 **-2** 个文件」。

⇒ **负数比报错更危险，因为它看起来像个数字。**
对抗用例 `renderers_are_pure_functions_of_the_products` 现在是常驻的。
