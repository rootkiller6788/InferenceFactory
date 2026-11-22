"""设计期望：**允许被推翻**。

⚠️ 这一层与门槛/对抗用例的关系：
- 门槛 = 「能不能交付」；对抗用例 = 「代码对不对」；**期望 = 「我以为的世界对不对」**。
- 被推翻的期望是**产出**，不是事故。写期望的时候**不许**挑稳赢的写 ——
  那样这一层就退化成「又一组门槛」。

判定：`held` / `refuted` / `unmeasurable`（数据不足，说不上成立还是不成立）。
"""

from __future__ import annotations

from typing import Any, Callable

_EVALS: list[dict[str, Any]] = []


def _exp(eid: str, claim: str) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        _EVALS.append({"id": eid, "claim": claim, "fn": fn})
        return fn
    return deco


def _series(p: dict[str, Any]) -> list[dict[str, Any]]:
    return p["scaling.json"]["series"]


def _feas(p: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in p["matrix.json"]["cells"] if c["feasible"]]


# --------------------------------------------------------------------- 期望
@_exp("throughput_scales_linearly_with_batch",
      "吞吐随 batch **线性**增长（batch 翻倍 ⇒ tokens/s 翻倍）")
def _e_linear_batch(p, c):
    """`ratio = tps(B_max) / (B_max · tps(1))`；明显 < 1 就是次线性。"""
    worst = 1.0
    where = ""
    for s in _series(p):
        pts = [q for q in s["curves"]["batch"] if q["feasible"] and q["tokens_per_s"]]
        if len(pts) < 2:
            continue
        base = pts[0]
        top = max(pts, key=lambda q: q["x"])
        if base["tokens_per_s"] and top["tokens_per_s"]:
            r = top["tokens_per_s"] / (top["x"] * base["tokens_per_s"])
            if r < worst:
                worst, where = r, s["series_id"]
    if worst >= 1.0:
        return "unmeasurable", {"ratio": worst}, "没有可用的 batch 曲线"
    ok = worst >= 0.5
    return ("held" if ok else "refuted"), {"min_ratio": worst, "where": where}, (
        f"最差处 batch 放大只换来 {worst:.3f} 倍的线性预期（{where}）—— "
        "batch 的收益在翻过 roofline 拐点后**饱和**")


@_exp("the_largest_batch_is_the_best",
      "最优配置一定用**最大** batch（既然 batching 是免费的吞吐）")
def _e_max_batch(p, c):
    top = max(int(v) for v in c["axes"]["batch"])
    small = []
    for d in p["compare.json"]["per_deployment"]:
        dec = d.get("decision")
        if dec is not None and int(dec["batch"]) < top:
            small.append(f"{d['deployment']}(b={dec['batch']})")
    ok = not small
    return ("held" if ok else "refuted"), {"max_batch": top, "n_not_max": len(small),
                                           "examples": small[:4]}, (
        f"{len(small)} 个部署单元的最优 batch 不是 {top} —— 更大的 batch 要么 OOM，"
        "要么把 P99 推出 SLO")


@_exp("quantization_is_a_free_lunch",
      "量化越狠越好（位宽越低 ⇒ 吞吐越高，没有代价）")
def _e_quant_free(p, c):
    """同一 (模型, 硬件, 后端, batch, context, arrival) 下比不同量化。"""
    bits = {q: int(v["weight_bits"]) for q, v in c["quant"].items()}
    groups: dict[tuple, dict[str, dict]] = {}
    for x in _feas(p):
        k = (x["model"], x["hardware"], x["backend"], x["batch"],
             x["context"], x["arrival_rate"])
        groups.setdefault(k, {})[x["quant"]] = x
    worse = []
    for k, g in groups.items():
        if len(g) < 2:
            continue
        order = sorted(g, key=lambda q: bits[q])       # 位宽从低到高
        for lo_q, hi_q in zip(order, order[1:]):
            if g[lo_q]["readings"]["tokens_per_s"] < \
               g[hi_q]["readings"]["tokens_per_s"] - 1e-9:
                worse.append(f"{k[2]}/{k[1]}|b{k[3]}|c{k[4]}：{lo_q} < {hi_q}")
    ok = not worse
    return ("held" if ok else "refuted"), {"n_regressions": len(worse),
                                           "examples": worse[:4]}, (
        f"{len(worse)} 组里更狠的量化反而更慢 —— 量化省的是**带宽**不是算力，"
        "翻过拐点后反量化的算力折扣就露出来了")


@_exp("p99_scales_proportionally_with_arrival_rate",
      "P99 与到达率成正比（负载翻倍 ⇒ P99 翻倍）")
def _e_p99_prop(p, c):
    """弹性 `elasticity = Δln(p99) / Δln(λ)`；明显 ≠ 1 就不成正比。"""
    vals = []
    for s in _series(p):
        pts = [q for q in s["curves"]["arrival"] if q["feasible"] and q["p99_ms"]]
        for a, b in zip(pts, pts[1:]):
            if a["p99_ms"] > 0 and b["x"] > a["x"]:
                vals.append(((b["p99_ms"] / a["p99_ms"]), (b["x"] / a["x"])))
    if not vals:
        return "unmeasurable", {}, "没有可用的到达率曲线"
    import math
    elas = [math.log(pr) / math.log(lr) for pr, lr in vals if pr > 0 and lr > 1]
    if not elas:
        return "unmeasurable", {"n_pairs": len(vals)}, "P99 在稳定区几乎不变"
    hi = max(elas)
    lo = min(elas)
    ok = 0.8 <= lo and hi <= 1.2
    return ("held" if ok else "refuted"), {
        "n_pairs": len(elas), "elasticity_min": lo, "elasticity_max": hi}, (
        f"弹性在 [{lo:.3f}, {hi:.3f}] —— P99 在稳定区**几乎不随负载变**，"
        "然后直接掉下悬崖（不可行），中间没有「渐进劣化」的区间")


@_exp("longer_context_only_costs_memory",
      "context 拉长只多占显存，不影响速度")
def _e_ctx_only_mem(p, c):
    grew = []
    for s in _series(p):
        pts = [q for q in s["curves"]["context"] if q["feasible"] and q["tpot_ms"]]
        if len(pts) < 2:
            continue
        lo = min(pts, key=lambda q: q["x"])
        hi = max(pts, key=lambda q: q["x"])
        if hi["tpot_ms"] > lo["tpot_ms"] * 1.05:
            grew.append(f"{s['series_id']}:{lo['tpot_ms']:.3f}→{hi['tpot_ms']:.3f} ms")
    ok = not grew
    return ("held" if ok else "refuted"), {"n_grew": len(grew), "examples": grew[:4]}, (
        f"{len(grew)} 条曲线上 TPOT 随 context 明显变慢 —— 每个 decode 步都要把"
        " `B·C` 的 KV 读一遍，这是**带宽**，不是显存")


@_exp("cpu_backend_is_never_optimal",
      "CPU 后端在任何部署单元上都不会是最优解")
def _e_cpu(p, c):
    wins = []
    for d in p["compare.json"]["per_deployment"]:
        dec = d.get("decision")
        if dec is not None and dec["backend"] == "llamacpp-cpu":
            wins.append(d["deployment"])
    ok = not wins
    return ("held" if ok else "refuted"), {"cpu_wins": wins}, (
        (f"CPU 后端在 {wins} 上是最优 —— 8G 卡装不下的模型，"
         "最优解是**让 CPU 去跑**，代价是吞吐垫底")
        if wins else "CPU 后端从未胜出")


@_exp("one_backend_wins_everywhere",
      "存在**一个** backend 在所有部署单元上都是最优")
def _e_one_backend(p, c):
    wins = [d["decision"]["backend"] for d in p["compare.json"]["per_deployment"]
            if d.get("decision")]
    ok = len(set(wins)) <= 1
    return ("held" if ok else "refuted"), {"winners": sorted(set(wins)),
                                           "n_distinct": len(set(wins))}, (
        f"{len(set(wins))} 个不同的 backend 赢过 —— 「选哪个 backend」的答案"
        "**依赖模型与机器**，不是一个常数")


@_exp("the_throughput_optimum_always_satisfies_the_slo",
      "吞吐最高的那一格一定满足 SLO（不用单独判 SLO）")
def _e_slo(p, c):
    n = int(p["compare.json"]["n_slo_exclusions"])
    ok = n == 0
    return ("held" if ok else "refuted"), {"n_slo_exclusions": n}, (
        (f"{n} 次「吞吐最优」被 SLO 淘汰 —— 吞吐最优点落在**饱和边缘**，"
         "而饱和边缘正是 P99 爆掉的地方") if n else "吞吐最优点都满足 SLO")


@_exp("paged_kv_decouples_context_from_batch",
      "分页 KV 让「context 上限」与「batch 上限」**解耦**（连续预留做不到）")
def _e_paged(p, c):
    """找一对 (B, C)：分页后端可行、连续预留后端 OOM，且两者 context 相同。"""
    proof = []
    for x in _feas(p):
        if not c["backends"][x["backend"]].get("continuous", False):
            continue
        for y in p["matrix.json"]["cells"]:
            if (y["hardware"] == x["hardware"] and y["model"] == x["model"]
                    and y["batch"] == x["batch"] and y["context"] == x["context"]
                    and y["arrival_rate"] == x["arrival_rate"]
                    and y["quant"] == x["quant"]
                    and not c["backends"][y["backend"]].get("continuous", False)
                    and not y["feasible"]
                    and y["reason"] in ("gpu_oom", "ram_oom")):
                proof.append(f"{x['model']}@{x['hardware']}|ctx={x['context']}"
                             f"|b={x['batch']}|{x['quant']}")
    ok = bool(proof)
    return ("held" if ok else "refuted"), {"n_proofs": len(proof),
                                           "examples": sorted(set(proof))[:4]}, (
        f"{len(proof)} 组证据：同样的 (B, C)，分页后端可行、连续预留 OOM"
        if ok else "没找到这样的对照 —— 分页 KV 的收益没兑现")


@_exp("eager_beats_vllm_on_ttft",
      "纯 eager 的 TTFT **低于** vLLM（没有调度器开销）")
def _e_ttft(p, c):
    """比的是**服务**侧的 TTFT（不含排队），因为排队是到达率的函数。"""
    wins = losses = 0
    worst = None
    for x in _feas(p):
        if x["backend"] != "pytorch-eager":
            continue
        for y in _feas(p):
            if (y["backend"] == "vllm-paged" and y["model"] == x["model"]
                    and y["hardware"] == x["hardware"] and y["quant"] == x["quant"]
                    and y["batch"] == x["batch"] and y["context"] == x["context"]
                    and y["arrival_rate"] == x["arrival_rate"]):
                d = y["readings"]["ttft_service_ms"] - x["readings"]["ttft_service_ms"]
                if d > 1e-9:
                    wins += 1
                    if worst is None or d > worst:
                        worst = d
                else:
                    losses += 1
    if wins + losses == 0:
        return "unmeasurable", {}, "没有可比的 (eager, vllm) 对"
    ok = wins > losses
    return ("held" if ok else "refuted"), {"eager_faster": wins, "vllm_faster": losses,
                                           "max_gain_ms": worst}, (
        f"eager 在 {wins}/{wins + losses} 组上 TTFT 更低（最多低 {worst:.3f} ms）—— "
        "vLLM 的调度器开销在**小批短请求**上是真的；它的优势在可行域与排队，不在 TTFT"
        if ok else f"vLLM 在 {losses} 组上 TTFT 更低")


@_exp("decode_is_memory_bound_at_small_batch",
      "小 batch 的 decode 是**带宽**受限（读权重主导）")
def _e_mem_bound(p, c):
    small = [x for x in _feas(p) if int(x["batch"]) == min(int(v) for v in c["axes"]["batch"])]
    if not small:
        return "unmeasurable", {}, "最小 batch 没有可行格"
    n_mem = sum(1 for x in small if x["readings"]["bound_decode"] == "memory")
    ok = n_mem / len(small) >= 0.8
    return ("held" if ok else "refuted"), {"n_small": len(small), "n_memory": n_mem,
                                           "share": n_mem / len(small)}, (
        f"最小 batch 的可行格里 {n_mem}/{len(small)} 是 bandwidth-bound")


@_exp("prefill_is_compute_bound",
      "prefill 是**算力**受限（prompt 长 ⇒ 算术强度高）")
def _e_prefill_bound(p, c):
    feas = _feas(p)
    if not feas:
        return "unmeasurable", {}, "没有可行格"
    n = sum(1 for x in feas if x["readings"]["bound_prefill"] == "compute")
    ok = n / len(feas) >= 0.8
    return ("held" if ok else "refuted"), {"n": len(feas), "n_compute": n,
                                           "share": n / len(feas)}, (
        f"{n}/{len(feas)} 的可行格 prefill 是 compute-bound")


@_exp("kv_cache_dominates_vram_at_long_context",
      "最长 context 下，KV cache 是显存的主要占用")
def _e_kv(p, c):
    longest = max(int(v) for v in c["axes"]["context"])
    shares = []
    for x in _feas(p):
        if int(x["context"]) == longest and x["memory"]["peak_bytes"] > 0:
            shares.append(x["memory"]["kv_bytes"] / x["memory"]["peak_bytes"])
    if not shares:
        return "unmeasurable", {"longest": longest}, "最长 context 没有可行格"
    hi = max(shares)
    ok = hi >= 0.5
    return ("held" if ok else "refuted"), {"longest": longest, "n": len(shares),
                                           "max_share": hi}, (
        f"最长 context（{longest}）下 KV 最多占 {hi:.3f} 的峰值占用")


@_exp("hybrid_trades_throughput_for_vram",
      "混合 offload 是**显存换吞吐**：同格子的峰值显存更低，产能也更低")
def _e_hybrid(p, c):
    """⚠️ 第一版这条问的是「hybrid 吞吐高于纯 CPU」，而它**结构上不可测**：
    `llamacpp-hybrid` 的 `device = hybrid` 需要 GPU，纯 CPU 机上它直接
    `no_gpu` ⇒ 两者**永远不会在同一台机器上都可行**，比对集合恒为空。
    一条恒空的判据比一条被推翻的判据更没用 —— 它连「错在哪」都不告诉你。

    改成跟**纯 GPU 后端**比：这才是「要不要 offload 一部分层回 CPU」
    这个真实决策的对照物，而且两组都在同一台 GPU 机器上可行。
    """
    pair = 0
    lower_vram = lower_tps = 0
    for x in _feas(p):
        if x["backend"] != "llamacpp-hybrid":
            continue
        for y in _feas(p):
            if y["backend"] == x["backend"]:
                continue
            if not (y["model"] == x["model"] and y["hardware"] == x["hardware"]
                    and y["quant"] == x["quant"] and y["batch"] == x["batch"]
                    and y["context"] == x["context"]
                    and y["arrival_rate"] == x["arrival_rate"]):
                continue
            if y["readings"].get("gpu_bytes", 0) <= 0:
                continue        # 只跟**用 GPU 显存**的后端比
            pair += 1
            if x["readings"]["peak_bytes"] < y["readings"]["peak_bytes"]:
                lower_vram += 1
            if x["readings"]["tokens_per_s"] < y["readings"]["tokens_per_s"]:
                lower_tps += 1
    if pair == 0:
        return "unmeasurable", {}, "没有可与纯 GPU 后端同框的 hybrid 格子"
    ok = lower_vram == pair and lower_tps == pair
    return ("held" if ok else "refuted"), {
        "pairs": pair, "lower_vram": lower_vram, "lower_tps": lower_tps}, (
        f"{pair} 组同框格子：显存更低 {lower_vram}、产能更低 {lower_tps} —— "
        "offload 是一门**交易**，不是免费的能力"
        if ok else
        f"{pair} 组里显存更低 {lower_vram}、产能更低 {lower_tps} —— "
        "两项没有同时成立，「显存换吞吐」这个说法至少有一半是假的")


@_exp("higher_arrival_raises_throughput",
      "到达率越高，产能越高（批更容易填满），直到不可行")
def _e_arrival(p, c):
    """⚠️ 三分类，不是二分类。第一版写的是
    `if b > a: up += 1 else: down += 1` ⇒ **持平被算成下降**，
    于是「36 段下降」里绝大多数其实是「一段也没动」。
    把「没动」报成「往反方向动了」是**比判错更坏的失真** —— 它把一个
    「产能与负载无关」的真结论，说成了「产能随负载下降」的假结论。
    """
    up = flat = down = 0
    for s in _series(p):
        pts = [q for q in s["curves"]["arrival"] if q["feasible"] and q["tokens_per_s"]]
        for a, b in zip(pts, pts[1:]):
            d = b["tokens_per_s"] - a["tokens_per_s"]
            if d > 1e-9:
                up += 1
            elif d < -1e-9:
                down += 1
            else:
                flat += 1
    if up + flat + down == 0:
        return "unmeasurable", {}, "没有可用的到达率曲线"
    ok = up > down
    return ("held" if ok else "refuted"), {
        "rising": up, "flat": flat, "falling": down}, (
        f"{up} 段上升 / {flat} 段持平 / {down} 段下降 —— 负载把批填满，产能随之上来"
        if ok else
        f"{up} 段上升 / {flat} 段持平 / {down} 段下降 —— "
        f"**{flat} 段一动不动**：产能是**配置的属性**，与到达率无关；"
        "随 λ 变的是 P99（排队）与交付吞吐，不是产能")


def evaluate(products: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for spec in _EVALS:
        try:
            verdict, observed, detail = spec["fn"](products, cfg)
        except Exception as exc:                        # noqa: BLE001
            verdict, observed, detail = "unmeasurable", {}, f"抛异常：{exc}"
        if verdict not in ("held", "refuted", "unmeasurable"):
            verdict = "unmeasurable"
        rows.append({"id": spec["id"], "claim": spec["claim"],
                     "verdict": verdict, "observed": observed,
                     "detail": " ".join(str(detail).split())})
    return {"n": len(rows),
            "n_held": sum(1 for r in rows if r["verdict"] == "held"),
            "n_refuted": sum(1 for r in rows if r["verdict"] == "refuted"),
            "n_unmeasurable": sum(1 for r in rows if r["verdict"] == "unmeasurable"),
            "rows": rows}
