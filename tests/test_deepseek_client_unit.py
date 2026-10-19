"""不调用真实 API 的单元测试。"""

import os

import pytest

from autoevidence.clients.openai_compat import MissingAPIKeyError, require_api_key
from autoevidence.experiments.deepseek_pilot import _candidate_bank, select_train_batch
from autoevidence.schemas.enums import SplitName


def test_require_api_key_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        require_api_key("DEEPSEEK_API_KEY")


def test_train_batch_train_only():
    tickets = select_train_batch(20)
    assert len(tickets) == 20
    assert all(t.split == SplitName.TRAIN for t in tickets)
    tags = set()
    for t in tickets:
        tags.update(t.tags)
        if t.hard_negative:
            tags.add("hard_negative")
        if t.mismatch:
            tags.add("mismatch")
    # 覆盖应尽量齐全
    assert any("evidence_missing" in t.tags for t in tickets)
    assert any(t.gold and t.gold.quality_event for t in tickets)


def test_candidate_bank_size():
    assert len(_candidate_bank()) >= 10
