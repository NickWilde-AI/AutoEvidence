from pathlib import Path

from autoevidence.data.seal import seal_tickets, verify_seal_integrity, mark_seal_failed
from autoevidence.data.split import assign_splits
from autoevidence.data.synthetic import generate_synthetic_corpus


def test_seal_and_verify(tmp_path: Path):
    tickets = assign_splits(generate_synthetic_corpus(40), seed=3)
    out = tmp_path / "seal_v_test"
    man = seal_tickets(tickets, out, seal_version="seal_v_test", max_samples=5)
    assert man["sample_count"] == 5
    assert man["clause_file_hash"]
    assert man["tickets_jsonl_sha256"]
    assert "不用于训练" in man["usage_rules"][0] or any("训练" in r for r in man["usage_rules"])
    v = verify_seal_integrity(out)
    assert v["ok"] is True


def test_mark_failed(tmp_path: Path):
    tickets = assign_splits(generate_synthetic_corpus(30), seed=4)
    out = tmp_path / "seal_fail"
    seal_tickets(tickets, out, seal_version="seal_fail", max_samples=3)
    rec = mark_seal_failed(out, reason="demo_fail", failed_ticket_ids=["x"])
    assert rec["block_release"] is True
    assert (out / "FAILURE_RECORD.json").exists()
