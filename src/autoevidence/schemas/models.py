"""工单、证据、标签、模型输出与实验协议。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from autoevidence.schemas.enums import (
    DISPOSAL_LEVELS,
    FAULT_TYPES_EVENT,
    DataSource,
    Disposal,
    EvidenceImageSource,
    FaultType,
    RefusalReason,
    SignalAnchor,
    SplitName,
)


class DialogueTurn(BaseModel):
    turn: int
    speaker: str
    text: str
    symptom_score: Optional[float] = None
    systems: list[str] = Field(default_factory=list)


class SignalWindow(BaseModel):
    window_id: str
    system: str
    dtc: Optional[str] = None
    start: str
    end: str
    anomaly_score: float = 0.0
    summary: str
    anchor: Optional[SignalAnchor] = None


class TicketImage(BaseModel):
    image_id: str
    path: Optional[str] = None
    tech_tag: Optional[str] = None
    # 向量或像素不进 schema 序列化主路径；训练侧另传


class ServiceHistoryItem(BaseModel):
    date: str
    item: str
    source: str = "未知"


class VehicleInfo(BaseModel):
    vehicle_id_hash: str = Field(..., description="脱敏车辆 ID（哈希），禁止明文 VIN")
    model: str
    mileage_km: int
    sw_version: str = "unknown"


class GoldLabel(BaseModel):
    quality_event: bool
    fault_type: FaultType
    disposal: Disposal

    @model_validator(mode="after")
    def check_consistency(self) -> "GoldLabel":
        if not self.quality_event:
            if self.fault_type != FaultType.NONE:
                raise ValueError("正常事件的 fault_type 必须为「无」")
            if self.disposal not in (Disposal.ARCHIVE, Disposal.HUMAN):
                # 正常工单通常归档；人审允许用于边界样本
                pass
        else:
            if self.fault_type not in FAULT_TYPES_EVENT:
                raise ValueError(f"质量事件 fault_type 非法: {self.fault_type}")
            if self.disposal not in DISPOSAL_LEVELS:
                raise ValueError(f"disposal 非法: {self.disposal}")
        return self


class Ticket(BaseModel):
    ticket_id: str
    event_id: str
    vehicle: VehicleInfo
    created_at: str
    dialogue: list[DialogueTurn] = Field(default_factory=list)
    signal_summary: list[SignalWindow] = Field(default_factory=list)
    images: list[TicketImage] = Field(default_factory=list)
    service_history: list[ServiceHistoryItem] = Field(default_factory=list)
    gold: Optional[GoldLabel] = None
    source: DataSource = DataSource.SYNTHETIC
    parent_ids: list[str] = Field(default_factory=list, description="增广父样本 ID")
    tags: list[str] = Field(default_factory=list)
    mismatch: Optional[str] = None
    hard_negative: bool = False
    critical_safety: bool = False
    standard_version: str = "v4"
    split: Optional[SplitName] = None
    meta: dict[str, Any] = Field(default_factory=dict)


class EvidenceImage(BaseModel):
    image_id: str
    source: EvidenceImageSource
    sim: float = 0.0
    nearest_type: Optional[str] = None
    path: Optional[str] = None
    from_ticket: bool = True  # False 表示图库示例，禁止当作本车证据写入笔录事实


class EvidencePack(BaseModel):
    sample_id: str
    ticket_id: str
    vehicle: VehicleInfo
    dialogue: list[DialogueTurn]
    signal_windows: list[SignalWindow]
    evidence_images: list[EvidenceImage]
    service_history: list[ServiceHistoryItem]
    insufficient_evidence: bool = False
    locate_reasons: list[str] = Field(default_factory=list)
    source: DataSource = DataSource.SYNTHETIC
    standard_version: str = "v4"


class ClauseHit(BaseModel):
    clause_id: str
    hit: bool
    why: str


class AnalysisConclusion(BaseModel):
    quality_event: bool
    fault_type: FaultType
    disposal: Disposal
    status: Optional[RefusalReason] = None

    @model_validator(mode="after")
    def check_consistency(self) -> "AnalysisConclusion":
        if self.status in (
            RefusalReason.INSUFFICIENT_EVIDENCE,
            RefusalReason.REFUSE,
            RefusalReason.PARSE_FAIL,
            RefusalReason.UNKNOWN,
        ):
            return self
        if not self.quality_event and self.fault_type != FaultType.NONE:
            raise ValueError("非质量事件 fault_type 必须为「无」")
        if self.quality_event and self.fault_type == FaultType.NONE:
            raise ValueError("质量事件不可缺 fault_type")
        return self


class AnalysisRecord(BaseModel):
    evidence: str
    clause_check: list[ClauseHit]
    conclusion: AnalysisConclusion


class ModelOutput(BaseModel):
    """推理与评测统一输出。"""

    quality_event: bool
    fault_type: FaultType
    disposal: Disposal
    record: Optional[AnalysisRecord] = None
    cited_clauses: list[str] = Field(default_factory=list)
    evidence_summary: Optional[str] = None
    status: Optional[RefusalReason] = None
    raw_text: Optional[str] = None
    model_version: str = "unknown"
    request_id: Optional[str] = None


class SynthesizedSample(BaseModel):
    sample_id: str
    evidence_pack_id: str
    ticket_id: str
    standard_version: str
    record: AnalysisRecord
    gold: GoldLabel
    citation_valid: bool
    conclusion_match: bool
    ref_in_pack: bool
    kept: bool
    reject_reasons: list[str] = Field(default_factory=list)
    teacher_version: str
    prompt_version: str
    needs_semantic_review: bool = False
    source: DataSource = DataSource.TEACHER
    parent_ids: list[str] = Field(default_factory=list)
    split: Optional[SplitName] = None


class SplitManifest(BaseModel):
    version: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    strategy: str
    counts: dict[str, int]
    vehicle_groups: dict[str, list[str]] = Field(default_factory=dict)
    leakage_report: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    """训练条目 / 实验产物溯源。"""

    ticket_id: str
    evidence_pack_id: Optional[str] = None
    sample_id: Optional[str] = None
    standard_version: str = "v4"
    teacher_version: Optional[str] = None
    prompt_version: Optional[str] = None
    filter_kept: Optional[bool] = None
    reject_reasons: list[str] = Field(default_factory=list)
    split: Optional[SplitName] = None
    parent_ids: list[str] = Field(default_factory=list)
    data_version: str = "synthetic"
    source: DataSource = DataSource.SYNTHETIC


class ExperimentRecord(BaseModel):
    experiment_id: str
    config_path: str
    seed: int
    data_version: str
    model_version: str
    code_version: Optional[str] = None
    status: Literal["not_run", "running", "completed", "failed"] = "not_run"
    metrics_path: Optional[str] = None
    notes: str = ""
