"""教师模型统一适配器：本地 / OpenAI 兼容 + 规则教师。"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Protocol

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from autoevidence.schemas.enums import Disposal, FaultType
from autoevidence.schemas.models import AnalysisRecord, EvidencePack, GoldLabel
from autoevidence.standards.loader import ClauseLibrary


class TeacherClient(Protocol):
    version: str

    def generate(self, prompt: str, images: list[Any] | None = None) -> str: ...


class RuleTeacher:
    """不调用外部 API 的确定性教师，用于 synthetic / smoke。"""

    version = "rule_teacher_v1"
    prompt_version = "record_prompt_v1"

    def generate(self, prompt: str, images: list[Any] | None = None) -> str:
        # 从 prompt 中解析关键字段（由 synthesizer 构造约定）
        raise RuntimeError("RuleTeacher 请通过 synthesize_with_gold 调用")

    def synthesize(
        self,
        pack: EvidencePack,
        gold: GoldLabel,
        lib: ClauseLibrary,
    ) -> AnalysisRecord:
        turns = [t for t in pack.dialogue if (t.symptom_score or 0) >= 0.55]
        turn_txt = "；".join(f"对话第{t.turn}轮{t.speaker}描述{t.text}" for t in turns[:2])
        if not turn_txt and pack.dialogue:
            turn_txt = f"对话第{pack.dialogue[0].turn}轮{pack.dialogue[0].text}"
        win_txt = "；".join(
            f"信号窗口{w.window_id}显示{w.summary}" for w in pack.signal_windows
        )
        img_txt = "；".join(
            f"照片{im.image_id}为{im.source.value}线索"
            for im in pack.evidence_images
            if im.from_ticket
        )
        hist = "；".join(f"服务历史{h.item}" for h in pack.service_history[:2])
        evidence = "；".join(x for x in [turn_txt, win_txt, img_txt, hist] if x)
        if not evidence:
            evidence = "证据不足，未见可复核线索"

        clause_check = []
        if gold.quality_event:
            # 按问题类型选主条款
            primary = {
                FaultType.BATTERY: "Q01-02",
                FaultType.BRAKE: "Q02-03",
                FaultType.POWERTRAIN: "Q03-01",
                FaultType.EE: "Q04-01",
            }.get(gold.fault_type, "Q05-02")
            clause_check.append(
                {
                    "clause_id": primary,
                    "hit": True,
                    "why": f"证据与{gold.fault_type.value}判定条件相符",
                }
            )
            if gold.disposal == Disposal.CRITICAL:
                clause_check.append(
                    {"clause_id": "Q05-01", "hit": True, "why": "安全相关功能性退化"}
                )
            else:
                clause_check.append(
                    {"clause_id": "Q05-02", "hit": True, "why": "可复核质量事件无即时失控风险"}
                )
            clause_check.append(
                {"clause_id": "Q06-03", "hit": False, "why": "豁免不适用或证据指向真实缺陷"}
            )
        else:
            # 难负例 / 正常
            if "hard_negative" in (pack.locate_reasons or []) or any(
                "低温" in (w.summary or "") or "手刹" in "" for w in pack.signal_windows
            ):
                pass
            exempt = "Q06-02"
            text_blob = evidence + "".join(w.summary for w in pack.signal_windows)
            if "手刹" in text_blob or "误操作" in text_blob:
                exempt = "Q06-04"
            elif "保养" in text_blob:
                exempt = "Q06-05"
            clause_check = [
                {"clause_id": exempt, "hit": True, "why": "命中豁免/正常解释"},
                {"clause_id": "Q05-04", "hit": True, "why": "不构成质量事件，归档"},
            ]

        from autoevidence.schemas.models import AnalysisConclusion, ClauseHit

        return AnalysisRecord(
            evidence=evidence,
            clause_check=[ClauseHit.model_validate(c) for c in clause_check],
            conclusion=AnalysisConclusion(
                quality_event=gold.quality_event,
                fault_type=gold.fault_type,
                disposal=gold.disposal,
            ),
        )


class OpenAICompatibleTeacher:
    version = "openai_compatible"
    prompt_version = "record_prompt_v1"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        cache_dir: str | Path | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.version = f"openai:{model}"

    def _cache_key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def generate(self, prompt: str, images: list[Any] | None = None) -> str:
        if self.cache_dir:
            ck = self._cache_key(prompt)
            cp = self.cache_dir / f"{ck}.txt"
            if cp.exists():
                return cp.read_text(encoding="utf-8")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        if self.cache_dir:
            (self.cache_dir / f"{self._cache_key(prompt)}.txt").write_text(text, encoding="utf-8")
        return text


TEACHER_ALLOWED_FIELDS = [
    "dialogue(turn,speaker,text,symptom_score,systems)",
    "signal_windows(window_id,system,dtc,start,end,anomaly_score,summary,anchor)",
    "evidence_images_metadata(image_id,source,sim,nearest_type,from_ticket) — 无像素，禁止臆造照片内容",
    "service_history(date,item,source)",
    "clause_library(id→text of current standard_version)",
]

TEACHER_FORBIDDEN = [
    "gold / 金标",
    "臆造未给出的图片像素或视觉细节",
    "臆造未出现的车辆事实、轮次、窗口、照片编号",
    "引用条款库以外的编号",
]


def build_teacher_prompt(pack: EvidencePack, lib: ClauseLibrary) -> str:
    image_meta = [
        {
            "image_id": im.image_id,
            "source": im.source.value if hasattr(im.source, "value") else im.source,
            "sim": im.sim,
            "nearest_type": im.nearest_type,
            "from_ticket": im.from_ticket,
            "note": "仅元数据，无图像像素；不得编造照片可见内容",
        }
        for im in pack.evidence_images
    ]
    return (
        "你是整车质量问题分析工程师。只根据下方【允许字段】写三段式分析笔录 JSON，"
        "不得编造未提供的事实。分三段:"
        "evidence 指出对话第几轮、哪个信号窗口、哪张照片编号里的线索（无像素时只能引用元数据与口述/信号）；"
        "clause_check 逐条引用条款库编号，写命中与否及理由，只能引用给出的编号；"
        "conclusion 给出 quality_event、fault_type、disposal。"
        "fault_type 取值：电池与高压系统/制动与转向/动力总成与传动/电子电气与软件/无；"
        "disposal 取值：紧急复核/技术排查/人审/归档。"
        "只输出 JSON 对象。不要输出隐藏思维链。\n"
        f"允许字段:{json.dumps(TEACHER_ALLOWED_FIELDS, ensure_ascii=False)}\n"
        f"禁止:{json.dumps(TEACHER_FORBIDDEN, ensure_ascii=False)}\n"
        f"条款库:{json.dumps(lib.as_prompt_dict(), ensure_ascii=False)}\n"
        f"对话:{json.dumps([t.model_dump() for t in pack.dialogue], ensure_ascii=False)}\n"
        f"信号窗口:{json.dumps([w.model_dump(mode='json') for w in pack.signal_windows], ensure_ascii=False)}\n"
        f"照片元数据:{json.dumps(image_meta, ensure_ascii=False)}\n"
        f"服务历史:{json.dumps([h.model_dump() for h in pack.service_history], ensure_ascii=False)}\n"
        'JSON 示例键: {"evidence":"...","clause_check":[{"clause_id":"Q02-03","hit":true,"why":"..."}],'
        '"conclusion":{"quality_event":true,"fault_type":"制动与转向","disposal":"紧急复核"}}'
    )


class CheckpointedBatch:
    """可恢复批处理。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[str] = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.done.add(json.loads(line)["sample_id"])

    def is_done(self, sample_id: str) -> bool:
        return sample_id in self.done

    def append(self, obj: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.done.add(obj["sample_id"])
