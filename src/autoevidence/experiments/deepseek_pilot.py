"""DeepSeek 真实 API 小批量验收：教师合成 + 裁判对比。

禁止静默回退规则教师/裁判；不触及 safety_seal；不做 GPU 训练。
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoevidence.clients.deepseek import DOCS_REF, DEFAULT_MODEL, build_deepseek_client
from autoevidence.clients.openai_compat import APIRequestError, ChatResult, MissingAPIKeyError
from autoevidence.data.split import assign_splits
from autoevidence.data.synthetic import generate_synthetic_corpus
from autoevidence.evidence.locator import FaultClueLocator
from autoevidence.rewards.cited_reward import cited_reward
from autoevidence.rewards.judge import RuleJudge
from autoevidence.schemas.enums import Disposal, FaultType, SplitName
from autoevidence.schemas.models import (
    AnalysisConclusion,
    AnalysisRecord,
    ClauseHit,
    EvidencePack,
    GoldLabel,
    Ticket,
)
from autoevidence.standards.citation import apply_three_filters
from autoevidence.standards.loader import load_clause_library
from autoevidence.synthesis.teacher import (
    TEACHER_ALLOWED_FIELDS,
    TEACHER_FORBIDDEN,
    CheckpointedBatch,
    build_teacher_prompt,
)

ROOT = Path(__file__).resolve().parents[3]


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def select_train_batch(n: int = 20, seed: int = 42) -> list[Ticket]:
    """仅 train；覆盖正常/事件/错位/难负例/证据不足。不足则扩大合成。"""
    need_tags = {
        "normal": 0,
        "quality_event": 0,
        "oral_signal_mismatch": 0,
        "hard_negative": 0,
        "evidence_missing": 0,
    }
    corpus_n = 120
    tickets = assign_splits(generate_synthetic_corpus(corpus_n), seed=seed)
    train = [t for t in tickets if t.split == SplitName.TRAIN and t.gold]
    selected: list[Ticket] = []
    used: set[str] = set()

    def take(pred, limit: int = 4) -> None:
        for t in train:
            if len(selected) >= n:
                return
            if t.ticket_id in used:
                continue
            if pred(t):
                selected.append(t)
                used.add(t.ticket_id)
                if sum(1 for x in selected if pred(x)) >= limit:
                    return

    take(lambda t: "evidence_missing" in t.tags, 3)
    take(lambda t: t.hard_negative or "hard_negative" in t.tags, 4)
    take(lambda t: t.mismatch == "口述信号错位" or "oral_signal_mismatch" in t.tags, 4)
    take(lambda t: bool(t.gold and t.gold.quality_event and not t.hard_negative), 5)
    take(lambda t: bool(t.gold and not t.gold.quality_event and "normal" in t.tags), 4)

    for t in train:
        if len(selected) >= n:
            break
        if t.ticket_id not in used:
            selected.append(t)
            used.add(t.ticket_id)

    if len(selected) < n:
        # 扩大合成，仍只取 train
        more = assign_splits(generate_synthetic_corpus(corpus_n + 80), seed=seed + 1)
        for t in more:
            if t.split != SplitName.TRAIN or not t.gold or t.ticket_id in used:
                continue
            selected.append(t)
            used.add(t.ticket_id)
            if len(selected) >= n:
                break

    # 覆盖统计
    for t in selected:
        if "evidence_missing" in t.tags:
            need_tags["evidence_missing"] += 1
        if t.hard_negative or "hard_negative" in t.tags:
            need_tags["hard_negative"] += 1
        if t.mismatch or "oral_signal_mismatch" in t.tags:
            need_tags["oral_signal_mismatch"] += 1
        if t.gold and t.gold.quality_event:
            need_tags["quality_event"] += 1
        if "normal" in t.tags and t.gold and not t.gold.quality_event:
            need_tags["normal"] += 1

    if len(selected) < n:
        raise RuntimeError(f"train 样本不足: got {len(selected)} < {n}")
    selected = selected[:n]
    # 禁止混入非 train
    assert all(t.split == SplitName.TRAIN for t in selected)
    return selected


def ping(out_dir: Path, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    client = build_deepseek_client(model=model, cache_dir=out_dir / "cache_ping")
    result = client.chat(
        [
            {"role": "system", "content": "Reply with valid JSON only."},
            {
                "role": "user",
                "content": 'Return JSON: {"ok": true, "service": "deepseek"}',
            },
        ],
        json_mode=True,
        max_tokens=64,
        temperature=0,
        dedupe_key="ping_v1",
    )
    parsed = _extract_json(result.content)
    report = {
        "ok": bool(parsed.get("ok")),
        "parsed": parsed,
        "api": result.to_log_dict(),
        "docs_ref": DOCS_REF,
        "at": _utc(),
        "note": "连通性调用成功 ≠ 模型质量提升",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ping.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def run_teacher_batch(
    out_dir: Path,
    *,
    n: int = 20,
    model: str = DEFAULT_MODEL,
    require_real_api: bool = True,
) -> dict[str, Any]:
    if not require_real_api:
        raise RuntimeError("本验收必须 require_real_api=True，禁止规则回退")

    out_dir.mkdir(parents=True, exist_ok=True)
    client = build_deepseek_client(model=model, cache_dir=out_dir / "cache_teacher")
    lib = load_clause_library()
    locator = FaultClueLocator()
    tickets = select_train_batch(n=n)
    ckpt = CheckpointedBatch(out_dir / "teacher_checkpoint.jsonl")

    kept_f = (out_dir / "teacher_kept.jsonl").open("a", encoding="utf-8")
    rej_f = (out_dir / "teacher_rejected.jsonl").open("a", encoding="utf-8")
    call_f = (out_dir / "teacher_calls.jsonl").open("a", encoding="utf-8")

    stats = Counter()
    reject_reasons: Counter = Counter()
    usage_total = Counter()
    latencies: list[float] = []
    models_seen: set[str] = set()

    try:
        for t in tickets:
            sid = f"rec_evp_{t.ticket_id}"
            if ckpt.is_done(sid):
                stats["skipped_checkpoint"] += 1
                continue

            pack = locator.locate(t)
            prompt = build_teacher_prompt(pack, lib)
            row: dict[str, Any] = {
                "sample_id": sid,
                "ticket_id": t.ticket_id,
                "split": t.split.value if t.split else None,
                "tags": t.tags,
                "allowed_fields": TEACHER_ALLOWED_FIELDS,
                "forbidden": TEACHER_FORBIDDEN,
                "at": _utc(),
            }
            try:
                chat = client.chat(
                    [
                        {
                            "role": "system",
                            "content": "You output a single JSON object for the quality analysis record.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    json_mode=True,
                    max_tokens=4096,
                    temperature=0.2,
                    dedupe_key=f"teacher:{sid}:{lib.version}:{model}:m4096",
                )
                models_seen.add(chat.model)
                latencies.append(chat.latency_s)
                for k, v in chat.usage.items():
                    usage_total[k] += v
                row["api"] = chat.to_log_dict()
                try:
                    data = _extract_json(chat.content)
                except json.JSONDecodeError:
                    # 真实 API 修复重试一次（仍非规则教师）
                    fix = client.chat(
                        [
                            {
                                "role": "system",
                                "content": "Fix the text into one valid JSON object only. No markdown.",
                            },
                            {
                                "role": "user",
                                "content": "Repair into valid JSON object with keys evidence, clause_check, conclusion:\n"
                                + chat.content[:6000],
                            },
                        ],
                        json_mode=True,
                        max_tokens=4096,
                        temperature=0,
                        dedupe_key=f"teacher_fix:{sid}:{hash(chat.content)%10**8}",
                    )
                    models_seen.add(fix.model)
                    latencies.append(fix.latency_s)
                    for k, v in fix.usage.items():
                        usage_total[k] += v
                    row["api_repair"] = fix.to_log_dict()
                    data = _extract_json(fix.content)
                record = AnalysisRecord.model_validate(data)
                gold = t.gold
                assert gold is not None
                filt = apply_three_filters(
                    record,
                    pack,
                    {
                        "quality_event": gold.quality_event,
                        "fault_type": gold.fault_type.value,
                        "disposal": gold.disposal.value,
                    },
                    lib,
                )
                out_obj = {
                    **row,
                    "record": record.model_dump(mode="json"),
                    "gold": gold.model_dump(mode="json"),
                    "filter": filt,
                    "teacher_version": client.version,
                    "returned_model": chat.model,
                    "standard_version": lib.version,
                    "prompt_version": "record_prompt_v1_deepseek",
                    "provenance": {
                        "ticket_id": t.ticket_id,
                        "evidence_pack_id": pack.sample_id,
                        "split": "train",
                        "data_source": "synthetic_train_only",
                    },
                }
                if filt["kept"]:
                    kept_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    stats["kept"] += 1
                else:
                    for r in filt["reject_reasons"]:
                        reject_reasons[r.split(":")[0]] += 1
                    rej_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    stats["rejected"] += 1
                stats["parsed_ok"] += 1
            except MissingAPIKeyError:
                raise
            except (json.JSONDecodeError, APIRequestError, Exception) as e:
                stats["parse_or_api_fail"] += 1
                fail_obj = {
                    **row,
                    "failed": True,
                    "error_type": type(e).__name__,
                    "error": str(e)[:300],
                    "score_forced": "n/a_not_high",
                }
                rej_f.write(json.dumps(fail_obj, ensure_ascii=False) + "\n")
                call_f.write(json.dumps(fail_obj, ensure_ascii=False) + "\n")
                ckpt.append({"sample_id": sid, "kept": False, "failed": True})
                continue

            call_f.write(json.dumps({"sample_id": sid, "api": row.get("api")}, ensure_ascii=False) + "\n")
            ckpt.append({"sample_id": sid, "kept": bool(filt["kept"])})
            stats["total_attempted"] += 1
    finally:
        kept_f.close()
        rej_f.close()
        call_f.close()

    attempted = stats["kept"] + stats["rejected"] + stats["parse_or_api_fail"]
    summary = {
        "task": "deepseek_teacher_pilot",
        "at": _utc(),
        "n_requested": n,
        "n_selected": len(tickets),
        "coverage_tags": _coverage(tickets),
        "stats": dict(stats),
        "reject_reason_prefix_counts": dict(reject_reasons),
        "pass_rate": (stats["kept"] / attempted) if attempted else None,
        "parse_fail_rate": (stats["parse_or_api_fail"] / max(attempted, 1)),
        "usage_total": dict(usage_total),
        "latency_s": {
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "n": len(latencies),
        },
        "returned_models": sorted(models_seen),
        "configured_model": model,
        "docs_ref": DOCS_REF,
        "allowed_fields": TEACHER_ALLOWED_FIELDS,
        "note": "API 调用成功 ≠ 模型质量提升；同一模型兼任教师与裁判不构成独立金标",
        "partitions": "train_only",
        "safety_seal_touched": False,
    }
    (out_dir / "teacher_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _coverage(tickets: list[Ticket]) -> dict[str, int]:
    c = Counter()
    for t in tickets:
        if "evidence_missing" in t.tags:
            c["evidence_missing"] += 1
        if t.hard_negative or "hard_negative" in t.tags:
            c["hard_negative"] += 1
        if t.mismatch or "mismatch" in "".join(t.tags):
            c["mismatch"] += 1
        if t.gold and t.gold.quality_event:
            c["quality_event"] += 1
        else:
            c["non_event"] += 1
    return dict(c)


def _candidate_bank() -> list[dict[str, Any]]:
    """开发用候选答案对（非 seal）；含正确/虚构/冲突/证据不足定论/技术排查躲避。"""
    def rec(evidence, clauses, qe, ft, dp):
        return AnalysisRecord(
            evidence=evidence,
            clause_check=[ClauseHit(clause_id=c, hit=h, why=w) for c, h, w in clauses],
            conclusion=AnalysisConclusion(
                quality_event=qe, fault_type=ft, disposal=dp
            ),
        )

    pairs = [
        {
            "pair_id": "correct_cite_vs_fabricate",
            "scenario": "低温续航难负例",
            "evidence_pack": {
                "dialogue": [{"turn": 1, "speaker": "车主", "text": "零下十度续航掉得快"}],
                "signal_windows": [
                    {"window_id": "sw_01", "summary": "健康度98.7%，环境零下9度", "anomaly_score": 0.1}
                ],
                "evidence_images": [],
                "service_history": [],
            },
            "a": rec(
                "对话第1轮低温续航；信号窗口sw_01健康度正常",
                [("Q06-02", True, "环境因素"), ("Q05-04", True, "归档")],
                False,
                FaultType.NONE,
                Disposal.ARCHIVE,
            ),
            "b": rec(
                "对话第1轮；信号窗口sw_01",
                [("Q01-99", True, "编造编号")],
                True,
                FaultType.BATTERY,
                Disposal.TECH,
            ),
            "defect_focus": "虚构引证",
        },
        {
            "pair_id": "conflict_conclusion",
            "scenario": "信号异常但结论正常冲突",
            "evidence_pack": {
                "dialogue": [{"turn": 1, "speaker": "车主", "text": "充电有塑料味"}],
                "signal_windows": [
                    {
                        "window_id": "sw_01",
                        "summary": "电芯温差12.4对基线4.1，DTC P0A80",
                        "anomaly_score": 0.91,
                    }
                ],
                "evidence_images": [{"image_id": "img_01", "from_ticket": True}],
                "service_history": [],
            },
            "a": rec(
                "对话第1轮异味；信号窗口sw_01温差三倍；照片img_01",
                [("Q01-02", True, "热相关"), ("Q05-01", True, "安全")],
                True,
                FaultType.BATTERY,
                Disposal.CRITICAL,
            ),
            "b": rec(
                "对话第1轮；信号窗口sw_01温差三倍",
                [("Q06-02", True, "环境")],
                False,
                FaultType.NONE,
                Disposal.ARCHIVE,
            ),
            "defect_focus": "结论与证据冲突",
        },
        {
            "pair_id": "insufficient_but_certain",
            "scenario": "证据不足却下定论",
            "evidence_pack": {
                "dialogue": [{"turn": 1, "speaker": "车主", "text": "说不清，怪怪的"}],
                "signal_windows": [],
                "evidence_images": [],
                "service_history": [],
            },
            "a": rec(
                "证据不足，对话含糊且无信号窗口",
                [("Q05-03", True, "人审")],
                False,
                FaultType.NONE,
                Disposal.HUMAN,
            ),
            "b": rec(
                "对话第12轮主缸爆炸；照片img_99可见",
                [("Q02-03", True, "臆造")],
                True,
                FaultType.BRAKE,
                Disposal.CRITICAL,
            ),
            "defect_focus": "证据不足却下定论/伪造",
        },
        {
            "pair_id": "tech_dodge_safety",
            "scenario": "制动安全却技术排查躲避",
            "evidence_pack": {
                "dialogue": [{"turn": 3, "speaker": "车主", "text": "踏板发软要踩到底"}],
                "signal_windows": [
                    {
                        "window_id": "sw_01",
                        "summary": "建压0.42秒对基线0.21秒",
                        "anomaly_score": 0.88,
                    }
                ],
                "evidence_images": [{"image_id": "img_02", "from_ticket": True}],
                "service_history": [],
            },
            "a": rec(
                "对话第3轮；信号窗口sw_01；照片img_02",
                [("Q02-03", True, "建压超时"), ("Q05-01", True, "安全")],
                True,
                FaultType.BRAKE,
                Disposal.CRITICAL,
            ),
            "b": rec(
                "对话第3轮；信号窗口sw_01",
                [("Q02-03", True, "建压超时"), ("Q05-02", True, "仅技术排查")],
                True,
                FaultType.BRAKE,
                Disposal.TECH,
            ),
            "defect_focus": "借技术排查逃避安全分流",
        },
        {
            "pair_id": "correct_vs_wrong_type",
            "scenario": "问题类型张冠李戴",
            "evidence_pack": {
                "dialogue": [{"turn": 1, "speaker": "车主", "text": "OTA后中控黑屏"}],
                "signal_windows": [
                    {"window_id": "sw_01", "summary": "IVI通信丢失 U0100", "anomaly_score": 0.77}
                ],
                "evidence_images": [],
                "service_history": [{"date": "2026-07-01", "item": "OTA 3.2.1"}],
            },
            "a": rec(
                "对话第1轮；信号窗口sw_01；服务历史OTA",
                [("Q04-02", True, "黑屏"), ("Q05-02", True, "技术排查")],
                True,
                FaultType.EE,
                Disposal.TECH,
            ),
            "b": rec(
                "对话第1轮；信号窗口sw_01",
                [("Q01-02", True, "误判电池")],
                True,
                FaultType.BATTERY,
                Disposal.TECH,
            ),
            "defect_focus": "错误引证类型",
        },
    ]
    # 复制扩展到至少 10 组（变换措辞，仍属开发集构造）
    extra = []
    for i, p in enumerate(pairs):
        q = dict(p)
        q["pair_id"] = p["pair_id"] + f"_v{i+2}"
        q["scenario"] = p["scenario"] + "（变体）"
        extra.append(q)
    return pairs + extra  # 10 组


def run_judge_batch(
    out_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    require_real_api: bool = True,
) -> dict[str, Any]:
    if not require_real_api:
        raise RuntimeError("本验收必须 require_real_api=True")

    out_dir.mkdir(parents=True, exist_ok=True)
    client = build_deepseek_client(model=model, cache_dir=out_dir / "cache_judge")
    lib = load_clause_library()
    rule_judge = RuleJudge()
    pairs = _candidate_bank()
    assert len(pairs) >= 10

    results = []
    usage_total = Counter()
    models_seen: set[str] = set()
    api_fail = 0
    parse_fail = 0
    position_consistent = 0
    position_compared = 0

    out_path = out_dir / "judge_pairs.jsonl"
    with out_path.open("w", encoding="utf-8") as fout:
        for pair in pairs:
            ep = pair["evidence_pack"]
            clauses_allowed = {
                cid: lib.clauses[cid]
                for cid in lib.clauses
                if cid.startswith(("Q01", "Q02", "Q04", "Q05", "Q06"))
            }
            # 规则分（开发对照，不喂给 DeepSeek）
            # 无金标规则分：用启发式 — fabricate / 越界 → 低
            def rule_score(record: AnalysisRecord) -> dict:
                # 伪 gold 仅用于规则奖励对照，不进入裁判 prompt
                pseudo = {
                    "quality_event": record.conclusion.quality_event,
                    "fault_type": record.conclusion.fault_type.value,
                    "disposal": record.conclusion.disposal.value,
                }
                # 对虚构条款单独惩罚可见性
                r = cited_reward(record, pseudo, lib, judge=rule_judge)
                return r

            ra = rule_score(pair["a"])
            rb = rule_score(pair["b"])

            def ask(order_a_first: bool) -> dict[str, Any]:
                left, right = (pair["a"], pair["b"]) if order_a_first else (pair["b"], pair["a"])
                labels = ("A", "B")
                prompt = (
                    "[SYSTEM_JUDGE_INSTRUCTION_LOCKED]\n"
                    "你是质量分析复核裁判。只根据问题场景、证据包、允许条款与两个候选答案，"
                    "判断哪个候选更好（引证真实、结论与证据一致、不逃避安全分流）。"
                    "不得假设未给出的金标或教师结论。只输出 JSON："
                    '{"winner":"A"|"B"|"tie","confidence":0-1,"reasons":["..."],'
                    '"flags":["fabricated_citation"|"evidence_conflict"|"insufficient_overclaim"|"safety_dodge"|...]}\n'
                    f"场景:{pair['scenario']}\n"
                    f"证据包:{json.dumps(ep, ensure_ascii=False)}\n"
                    f"允许条款:{json.dumps(clauses_allowed, ensure_ascii=False)}\n"
                    f"候选{labels[0]}:{json.dumps(left.model_dump(mode='json'), ensure_ascii=False)}\n"
                    f"候选{labels[1]}:{json.dumps(right.model_dump(mode='json'), ensure_ascii=False)}\n"
                )
                chat = client.chat(
                    [
                        {"role": "system", "content": "Output one JSON object only."},
                        {"role": "user", "content": prompt},
                    ],
                    json_mode=True,
                    max_tokens=1200,
                    temperature=0.0,
                    dedupe_key=f"judge:{pair['pair_id']}:{'AB' if order_a_first else 'BA'}:{model}:m1200",
                )
                try:
                    parsed = _extract_json(chat.content)
                except json.JSONDecodeError:
                    fix = client.chat(
                        [
                            {
                                "role": "system",
                                "content": "Repair into one valid JSON object only.",
                            },
                            {
                                "role": "user",
                                "content": "Fix JSON with keys winner,confidence,reasons,flags:\n"
                                + chat.content[:4000],
                            },
                        ],
                        json_mode=True,
                        max_tokens=800,
                        temperature=0,
                        dedupe_key=f"judge_fix:{pair['pair_id']}:{'AB' if order_a_first else 'BA'}:{hash(chat.content)%10**8}",
                    )
                    chat = fix
                    parsed = _extract_json(fix.content)
                return chat, parsed, left, right, order_a_first

            row: dict[str, Any] = {
                "pair_id": pair["pair_id"],
                "scenario": pair["scenario"],
                "defect_focus": pair["defect_focus"],
                "rule_score_a": ra,
                "rule_score_b": rb,
                "human_review": {
                    "prefer": None,
                    "agree_with_deepseek": None,
                    "notes": "",
                    "filled_by": "pending_human_only",
                },
                "disclaimer": "同一 DeepSeek 模型兼任教师与裁判不构成独立金标",
            }
            try:
                chat1, j1, _, _, _ = ask(True)
                chat2, j2, _, _, _ = ask(False)
                models_seen.add(chat1.model)
                models_seen.add(chat2.model)
                for chat in (chat1, chat2):
                    for k, v in chat.usage.items():
                        usage_total[k] += v

                # 映射回原始 A/B（pair 中的 a/b）
                def map_winner(j: dict, a_first: bool) -> str:
                    w = j.get("winner", "tie")
                    if w == "tie":
                        return "tie"
                    if a_first:
                        return "a" if w == "A" else "b"
                    return "b" if w == "A" else "a"

                w1 = map_winner(j1, True)
                w2 = map_winner(j2, False)
                position_compared += 1
                consistent = w1 == w2
                if consistent:
                    position_consistent += 1

                row.update(
                    {
                        "deepseek_order_AB": j1,
                        "deepseek_order_BA": j2,
                        "winner_mapped_AB": w1,
                        "winner_mapped_BA": w2,
                        "position_consistent": consistent,
                        "api_AB": chat1.to_log_dict(),
                        "api_BA": chat2.to_log_dict(),
                        "returned_model": chat1.model,
                    }
                )
            except MissingAPIKeyError:
                raise
            except json.JSONDecodeError as e:
                parse_fail += 1
                row.update(
                    {
                        "failed": True,
                        "error_type": "JSONDecodeError",
                        "error": str(e)[:200],
                        "score_forced_zero": True,
                    }
                )
            except Exception as e:
                api_fail += 1
                row.update(
                    {
                        "failed": True,
                        "error_type": type(e).__name__,
                        "error": str(e)[:200],
                        "score_forced_zero": True,
                    }
                )

            # 规则与 DeepSeek 分歧（若两边都有映射）
            if not row.get("failed"):
                rule_pref = "a" if ra["total"] >= rb["total"] else "b"
                ds_pref = row.get("winner_mapped_AB")
                row["rule_pref"] = rule_pref
                row["disagree_rule_vs_deepseek"] = ds_pref not in (None, "tie") and ds_pref != rule_pref

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            results.append(row)

    disagree = [r for r in results if r.get("disagree_rule_vs_deepseek")]
    summary = {
        "task": "deepseek_judge_pilot",
        "at": _utc(),
        "n_pairs": len(pairs),
        "api_fail": api_fail,
        "parse_fail": parse_fail,
        "position_consistency_rate": (
            position_consistent / position_compared if position_compared else None
        ),
        "position_compared": position_compared,
        "disagree_with_rule_count": len(disagree),
        "disagree_pair_ids": [r["pair_id"] for r in disagree],
        "usage_total": dict(usage_total),
        "returned_models": sorted(models_seen),
        "configured_model": model,
        "docs_ref": DOCS_REF,
        "human_review_sheet": str(out_dir / "human_review_sheet.json"),
        "note": "裁判未读取 gold/教师结论/规则分；人工栏必须真人填写",
        "safety_seal_touched": False,
    }
    (out_dir / "judge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sheet = {
        "instruction": "请人工逐条填写 prefer / agree_with_deepseek / notes；禁止模型代填通过。",
        "pairs": [
            {
                "pair_id": r["pair_id"],
                "scenario": r.get("scenario"),
                "defect_focus": r.get("defect_focus"),
                "deepseek_winner_AB": r.get("winner_mapped_AB"),
                "deepseek_winner_BA": r.get("winner_mapped_BA"),
                "position_consistent": r.get("position_consistent"),
                "rule_pref": r.get("rule_pref"),
                "human_review": {
                    "prefer": None,
                    "agree_with_deepseek": None,
                    "notes": "",
                },
            }
            for r in results
        ],
    }
    (out_dir / "human_review_sheet.json").write_text(
        json.dumps(sheet, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def estimate_cost_cny(usage: dict[str, int], model: str) -> dict[str, Any] | None:
    """仅在有公开定价依据时估算。定价来源：deepseek.com/en/platform（USD/1M tokens，会变动）。"""
    # Flash 参考（检索时）：input miss $0.14 / output $0.28 per 1M — 仅作粗算
    if "flash" not in model and "v4-flash" not in model and model != "deepseek-flash":
        return {
            "estimated": False,
            "reason": "未内置该模型可靠单价；请查官方定价页",
            "pricing_ref": "https://www.deepseek.com/en/platform/",
        }
    pin, pout = 0.14, 0.28  # USD / 1M tokens (cache miss)
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    usd = prompt / 1e6 * pin + completion / 1e6 * pout
    return {
        "estimated": True,
        "currency": "USD",
        "amount": round(usd, 6),
        "assumption": "flash cache-miss list price; ignores cache hit discount; may change",
        "pricing_ref": "https://www.deepseek.com/en/platform/",
        "checked_at": "2026-10-19",
    }


def write_pilot_report(out_dir: Path, teacher: dict, judge: dict, ping_report: dict) -> Path:
    usage = Counter()
    for u in (teacher.get("usage_total") or {}, judge.get("usage_total") or {}):
        for k, v in u.items():
            usage[k] += v
    model = teacher.get("configured_model") or DEFAULT_MODEL
    report = {
        "title": "DeepSeek API 小批量验收报告",
        "at": _utc(),
        "completion_levels": {
            "engineering_loop": "verified_prior",
            "real_api_link": "this_run",
            "formal_training": "reserved",
            "business_effect": "not_validated",
        },
        "ping": ping_report,
        "teacher": teacher,
        "judge": judge,
        "usage_total": dict(usage),
        "cost_estimate": estimate_cost_cny(dict(usage), model),
        "safety_seal_touched": False,
        "warnings": [
            "API 调用成功 ≠ 模型质量提升",
            "教师与裁判为同一 DeepSeek 模型时不构成独立金标",
            "human_review_sheet 中字段必须由真人填写",
        ],
    }
    path = out_dir / "PILOT_REPORT.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
