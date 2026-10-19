"""标签枚举与约束常量。"""

from __future__ import annotations

from enum import Enum


class FaultType(str, Enum):
    BATTERY = "电池与高压系统"
    BRAKE = "制动与转向"
    POWERTRAIN = "动力总成与传动"
    EE = "电子电气与软件"
    NONE = "无"


class Disposal(str, Enum):
    CRITICAL = "紧急复核"
    TECH = "技术排查"
    HUMAN = "人审"
    ARCHIVE = "归档"


class SplitName(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    SAFETY_GATE = "safety_gate"  # 兼容旧名；语义上默认视为 safety_dev 候选
    SAFETY_DEV = "safety_dev"  # 可迭代摸底，不得单独作为发布硬门槛
    SAFETY_SEAL = "safety_seal"  # 封存验收；失败阻断发布；禁止回流与按结果调参
    GENERALIZATION = "generalization"


class DataSource(str, Enum):
    SYNTHETIC = "synthetic"
    PRODUCTION = "production"  # 仅协议保留；本仓库不生成伪生产数据
    AUGMENT_SWAP = "augment_swap"
    TEACHER = "teacher"
    HARD_NEGATIVE = "hard_negative"
    REVIEW_REFLUX = "review_reflux"
    UNSTABLE = "unstable"
    HUMAN_DISAGREE = "human_disagree"


class EvidenceImageSource(str, Enum):
    TECHNICIAN = "技师标注"
    EXEMPLAR_LIB = "示例图库"  # 仅检索参考，不得冒充本车故障证据
    DEFAULT = "默认取法"
    TICKET = "工单现场"


class SignalAnchor(str, Enum):
    SYMPTOM = "症状关联"
    ANOMALY = "异常分数"
    DEFAULT = "默认取法"


class RefusalReason(str, Enum):
    UNKNOWN = "未知"
    INSUFFICIENT_EVIDENCE = "证据不足"
    REFUSE = "拒答"
    PARSE_FAIL = "解析失败"


FAULT_TYPES_EVENT = {
    FaultType.BATTERY,
    FaultType.BRAKE,
    FaultType.POWERTRAIN,
    FaultType.EE,
}

DISPOSAL_LEVELS = set(Disposal)

# 车辆系统 ↔ 问题类型映射（症状定位用）
SYSTEM_TO_FAULT: dict[str, FaultType] = {
    "电池": FaultType.BATTERY,
    "高压": FaultType.BATTERY,
    "充电": FaultType.BATTERY,
    "制动": FaultType.BRAKE,
    "转向": FaultType.BRAKE,
    "刹车": FaultType.BRAKE,
    "动力": FaultType.POWERTRAIN,
    "传动": FaultType.POWERTRAIN,
    "电机": FaultType.POWERTRAIN,
    "电子": FaultType.EE,
    "软件": FaultType.EE,
    "空调": FaultType.EE,
}
