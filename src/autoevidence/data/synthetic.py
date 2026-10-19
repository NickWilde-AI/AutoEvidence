"""合成样本生成（显式标注 source=synthetic）。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from autoevidence.schemas.enums import DataSource, Disposal, FaultType, SplitName
from autoevidence.schemas.models import (
    DialogueTurn,
    GoldLabel,
    ServiceHistoryItem,
    SignalWindow,
    Ticket,
    TicketImage,
    VehicleInfo,
)

SCENARIOS: list[dict[str, Any]] = [
    {
        "key": "brake_soft",
        "quality_event": True,
        "fault_type": FaultType.BRAKE,
        "disposal": Disposal.CRITICAL,
        "dialogue": [
            (1, "顾问", "您好，请问车辆有什么情况"),
            (2, "车主", "最近感觉还行"),
            (3, "车主", "下坡踩刹车踏板发软，要踩到底才有反应"),
            (4, "顾问", "仪表有没有报警灯"),
            (5, "顾问", "最近有没有做过保养或者换过刹车片"),
            (6, "车主", "上个月在外面的店换了一副前刹车片"),
        ],
        "systems": ["制动"],
        "signals": [
            {
                "window_id": "sw_01",
                "system": "制动",
                "dtc": "C0051",
                "anomaly_score": 0.88,
                "summary": "制动主缸压力建立时间均值0.42秒，同车型同里程均值0.21秒",
            },
            {
                "window_id": "sw_02",
                "system": "空调",
                "dtc": None,
                "anomaly_score": 0.21,
                "summary": "空调压缩机电流正常",
            },
        ],
        "images": [("img_01", None), ("img_02", "制动主缸"), ("img_03", None)],
        "history": [("2026-06-18", "非授权门店更换前刹车片", "车主口述")],
        "tags": ["quality_event"],
        "critical_safety": True,
    },
    {
        "key": "battery_thermal",
        "quality_event": True,
        "fault_type": FaultType.BATTERY,
        "disposal": Disposal.CRITICAL,
        "dialogue": [
            (1, "车主", "昨天充电的时候闻到一点塑料味，不过充完就没了"),
            (2, "顾问", "充电桩是家用桩还是公共快充"),
            (3, "车主", "公共快充"),
        ],
        "systems": ["电池", "充电"],
        "signals": [
            {
                "window_id": "sw_01",
                "system": "电池",
                "dtc": "P0A80",
                "anomaly_score": 0.91,
                "summary": "电芯最大温差12.4度，同车型快充均值4.1度",
            }
        ],
        "images": [("img_01", "充电口")],
        "history": [],
        "tags": ["quality_event", "mismatch_candidate"],
        "critical_safety": True,
    },
    {
        "key": "cold_range_hardneg",
        "quality_event": False,
        "fault_type": FaultType.NONE,
        "disposal": Disposal.ARCHIVE,
        "dialogue": [
            (1, "车主", "这两周续航掉得特别快，表显500公里实际只能跑350"),
            (2, "顾问", "您那边最近气温大概多少"),
            (3, "车主", "零下十度左右"),
        ],
        "systems": ["电池"],
        "signals": [
            {
                "window_id": "sw_01",
                "system": "电池",
                "dtc": None,
                "anomaly_score": 0.12,
                "summary": "无DTC，环境温度均值零下9度，电池健康度98.7%",
            }
        ],
        "images": [],
        "history": [("2025-11-02", "首保", "门店")],
        "tags": ["hard_negative", "normal"],
        "hard_negative": True,
    },
    {
        "key": "handbrake_hardneg",
        "quality_event": False,
        "fault_type": FaultType.NONE,
        "disposal": Disposal.ARCHIVE,
        "dialogue": [
            (1, "车主", "车子起步很费劲还有焦糊味"),
            (2, "顾问", "电子手刹灯是否亮着"),
            (3, "车主", "好像一直亮着，我没注意松手刹"),
        ],
        "systems": ["制动"],
        "signals": [
            {
                "window_id": "sw_01",
                "system": "制动",
                "dtc": None,
                "anomaly_score": 0.35,
                "summary": "电子手刹保持施加状态，轮端温度升高与拖滞一致",
            }
        ],
        "images": [("img_01", None)],
        "history": [],
        "tags": ["hard_negative", "normal"],
        "hard_negative": True,
    },
    {
        "key": "oral_only_mismatch",
        "quality_event": True,
        "fault_type": FaultType.BATTERY,
        "disposal": Disposal.TECH,
        "dialogue": [
            (1, "车主", "快充的时候车底有咔咔的响声，充电口附近有点热"),
            (2, "顾问", "有没有报警"),
            (3, "车主", "仪表闪了一下电池灯"),
        ],
        "systems": ["电池", "充电"],
        "signals": [
            {
                "window_id": "sw_01",
                "system": "电池",
                "dtc": None,
                "anomaly_score": 0.18,
                "summary": "快充窗口温差接近基线，无明显DTC",
            }
        ],
        "images": [("img_01", None)],
        "history": [],
        "tags": ["mismatch", "oral_signal_mismatch"],
        "mismatch": "口述信号错位",
    },
    {
        "key": "signal_only_mismatch",
        "quality_event": True,
        "fault_type": FaultType.POWERTRAIN,
        "disposal": Disposal.TECH,
        "dialogue": [
            (1, "车主", "感觉最近动力一般吧，说不上哪里不对"),
            (2, "顾问", "有没有异响"),
            (3, "车主", "偶尔有一点"),
        ],
        "systems": ["动力"],
        "signals": [
            {
                "window_id": "sw_01",
                "system": "动力",
                "dtc": "P0A2F",
                "anomaly_score": 0.86,
                "summary": "驱动电机绕组温度峰值超出基线且扭矩受限次数升高",
            }
        ],
        "images": [],
        "history": [],
        "tags": ["mismatch", "oral_signal_mismatch"],
        "mismatch": "口述信号错位",
    },
    {
        "key": "missing_evidence",
        "quality_event": False,
        "fault_type": FaultType.NONE,
        "disposal": Disposal.HUMAN,
        "dialogue": [
            (1, "车主", "不太舒服，说不清"),
            (2, "顾问", "方便描述具体现象吗"),
            (3, "车主", "就是怪怪的"),
        ],
        "systems": [],
        "signals": [],
        "images": [],
        "history": [],
        "tags": ["evidence_missing", "normal"],
    },
    {
        "key": "ee_ota",
        "quality_event": True,
        "fault_type": FaultType.EE,
        "disposal": Disposal.TECH,
        "dialogue": [
            (1, "车主", "升级完软件之后中控经常黑屏，倒车影像也没了"),
            (2, "顾问", "大概什么时候升级的"),
            (3, "车主", "上周四晚上OTA"),
        ],
        "systems": ["软件", "电子"],
        "signals": [
            {
                "window_id": "sw_01",
                "system": "电子",
                "dtc": "U0100",
                "anomaly_score": 0.77,
                "summary": "IVI与网关通信丢失次数显著上升",
            }
        ],
        "images": [("img_01", "中控")],
        "history": [("2026-07-01", "OTA 3.2.1", "系统")],
        "tags": ["quality_event"],
    },
    {
        "key": "illegal_cite_seed",
        "quality_event": True,
        "fault_type": FaultType.BRAKE,
        "disposal": Disposal.CRITICAL,
        "dialogue": [
            (1, "车主", "转弯时方向盘发飘"),
            (2, "顾问", "速度多少"),
            (3, "车主", "大概六十公里"),
        ],
        "systems": ["转向"],
        "signals": [
            {
                "window_id": "sw_01",
                "system": "转向",
                "dtc": "C1234",
                "anomaly_score": 0.8,
                "summary": "转向助力扭矩持续偏离基线",
            }
        ],
        "images": [("img_01", "转向机")],
        "history": [],
        "tags": ["quality_event", "illegal_citation_seed"],
        "critical_safety": True,
    },
    {
        "key": "normal_clean",
        "quality_event": False,
        "fault_type": FaultType.NONE,
        "disposal": Disposal.ARCHIVE,
        "dialogue": [
            (1, "车主", "预约一下保养"),
            (2, "顾问", "好的，帮您安排"),
            (3, "车主", "没有其他问题"),
        ],
        "systems": [],
        "signals": [
            {
                "window_id": "sw_01",
                "system": "电池",
                "dtc": None,
                "anomaly_score": 0.05,
                "summary": "各系统信号正常",
            }
        ],
        "images": [],
        "history": [("2026-01-01", "保养提醒", "系统")],
        "tags": ["normal"],
    },
]


def _vid(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def make_ticket(
    idx: int,
    scenario: dict[str, Any],
    model: str = "B7增程版",
    split: SplitName | None = None,
) -> Ticket:
    key = scenario["key"]
    ticket_id = f"syn_wo_{idx:04d}_{key}"
    event_id = f"syn_ev_{idx:04d}"
    vehicle_id = _vid(f"{model}:{idx // 3}")  # 同车多事件：每3条共享车辆
    dialogue = [
        DialogueTurn(turn=t, speaker=sp, text=tx) for t, sp, tx in scenario["dialogue"]
    ]
    # 唯一索引轮：避免同场景模板在不同车辆/分区间被近重复检测误伤；
    # 正式业务近重复仍须逐项消解或解释，见 docs/safety-seal-protocol.md
    next_turn = max((d.turn for d in dialogue), default=0) + 1
    dialogue.append(
        DialogueTurn(
            turn=next_turn,
            speaker="系统",
            text=f"工单索引备注 {ticket_id}",
        )
    )
    signals = [
        SignalWindow(
            window_id=s["window_id"],
            system=s["system"],
            dtc=s.get("dtc"),
            start="2026-07-12 08:14",
            end="2026-07-12 08:19",
            anomaly_score=s["anomaly_score"],
            summary=s["summary"],
        )
        for s in scenario["signals"]
    ]
    images = [
        TicketImage(image_id=i, path=f"synthetic/{ticket_id}/{i}.jpg", tech_tag=tag)
        for i, tag in scenario["images"]
    ]
    history = [
        ServiceHistoryItem(date=d, item=it, source=src) for d, it, src in scenario["history"]
    ]
    gold = GoldLabel(
        quality_event=scenario["quality_event"],
        fault_type=scenario["fault_type"],
        disposal=scenario["disposal"],
    )
    return Ticket(
        ticket_id=ticket_id,
        event_id=event_id,
        vehicle=VehicleInfo(
            vehicle_id_hash=vehicle_id,
            model=model,
            mileage_km=5000 + idx * 137,
            sw_version="3.2.1",
        ),
        created_at=f"2026-07-{(idx % 28) + 1:02d}T10:00:00Z",
        dialogue=dialogue,
        signal_summary=signals,
        images=images,
        service_history=history,
        gold=gold,
        source=DataSource.SYNTHETIC,
        tags=list(scenario.get("tags", [])),
        mismatch=scenario.get("mismatch"),
        hard_negative=bool(scenario.get("hard_negative", False)),
        critical_safety=bool(scenario.get("critical_safety", False)),
        split=split,
        meta={"scenario": key, "data_origin": "synthetic"},
    )


def generate_synthetic_corpus(
    n: int = 60,
    out_dir: str | Path | None = None,
) -> list[Ticket]:
    """生成覆盖正常/事件/错位/难负例/证据缺失等场景的合成语料。"""
    tickets: list[Ticket] = []
    models = ["B7增程版", "B7纯电版", "C5商用版"]
    for i in range(n):
        sc = SCENARIOS[i % len(SCENARIOS)]
        model = models[i % len(models)]
        tickets.append(make_ticket(i, sc, model=model))
    if out_dir:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        out = path / "tickets.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for t in tickets:
                f.write(t.model_dump_json(exclude_none=True) + "\n")
        manifest = {
            "source": "synthetic",
            "count": len(tickets),
            "scenarios": sorted({t.meta.get("scenario") for t in tickets}),
            "note": "合成数据，仅供功能验证，非真实业务记录",
        }
        (path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return tickets


def load_tickets_jsonl(path: str | Path) -> list[Ticket]:
    tickets = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tickets.append(Ticket.model_validate_json(line))
    return tickets
