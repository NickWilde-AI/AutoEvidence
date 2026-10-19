"""AutoEvidence CLI。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

app = typer.Typer(help="AutoEvidence — 智能汽车质量分析模型", no_args_is_help=True)
ROOT = Path(__file__).resolve().parents[2]


@app.command()
def smoke() -> None:
    """CPU smoke：合成数据 → 定位 → 合成 → 规则评测（不下载大模型）。"""
    from autoevidence.data.augment import oversample_hard_negatives, swap_augment
    from autoevidence.data.split import assign_splits, leakage_check, save_split_artifacts
    from autoevidence.data.stats import distribution_report
    from autoevidence.data.synthetic import generate_synthetic_corpus
    from autoevidence.evaluation.runner import evaluate_tickets, save_report
    from autoevidence.evidence.locator import locate_batch
    from autoevidence.standards.loader import load_clause_library
    from autoevidence.synthesis.pipeline import expand_clause_variants, run_synthesis

    out = ROOT / "data" / "synthetic"
    tickets = generate_synthetic_corpus(60, out_dir=out)
    tickets = assign_splits(tickets)
    manifest = save_split_artifacts(tickets, out / "splits")
    leak = leakage_check(tickets)
    assert leak["ok"], leak

    packs = locate_batch(tickets)
    (out / "evidence_packs.jsonl").write_text(
        "\n".join(p.model_dump_json() for p in packs) + "\n", encoding="utf-8"
    )

    lib = load_clause_library()
    syn_stats = run_synthesis(tickets, packs, lib, out / "synthesis")

    # 增广仅 train
    swapped = swap_augment(tickets, lib, max_pairs=8)
    tickets_aug = tickets + swapped
    tickets_aug = oversample_hard_negatives(tickets_aug, ratio=0.2)
    (out / "augmented_count.json").write_text(
        json.dumps({"swapped": len(swapped), "total": len(tickets_aug)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # SFT entries from kept
    kept_path = out / "synthesis" / "kept.jsonl"
    entries = []
    pack_by_id = {p.sample_id: p for p in packs}
    if kept_path.exists():
        for line in kept_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            from autoevidence.schemas.models import SynthesizedSample

            sample = SynthesizedSample.model_validate_json(line)
            pack = pack_by_id.get(sample.evidence_pack_id)
            if pack is None:
                continue
            entries.extend(expand_clause_variants(sample, pack, lib))
    entries_path = out / "sft_entries.jsonl"
    with entries_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    test_tickets = [t for t in tickets if t.split and t.split.value == "test"]
    report = evaluate_tickets(test_tickets or tickets[:20], set_name="synthetic_test")
    save_report(report, ROOT / "outputs" / "reports" / "smoke_eval.json")

    dist = distribution_report(tickets)
    summary = {
        "tickets": len(tickets),
        "packs": len(packs),
        "synthesis": syn_stats,
        "sft_entries": len(entries),
        "leakage_ok": leak["ok"],
        "split_counts": manifest.counts,
        "distribution": dist,
        "report": str(ROOT / "outputs" / "reports" / "smoke_eval.json"),
    }
    (ROOT / "outputs" / "reports" / "smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rprint("[green]smoke OK[/green]", summary)


@app.command("build-dataset")
def build_dataset(synthetic: bool = typer.Option(True, "--synthetic/--no-synthetic")) -> None:
    smoke()


@app.command("train-sft")
def train_sft_cmd(
    config: Path = typer.Option(ROOT / "configs" / "sft_tiny.yaml", "--config"),
) -> None:
    from autoevidence.training.sft import train_sft

    meta = train_sft(config)
    rprint(meta)


@app.command("train-grpo")
def train_grpo_cmd(
    config: Path = typer.Option(ROOT / "configs" / "grpo_tiny.yaml", "--config"),
) -> None:
    from autoevidence.training.grpo import train_grpo

    meta = train_grpo(config)
    rprint(meta)


@app.command()
def eval(
    split: str = "test",
    out: Path = typer.Option(ROOT / "outputs" / "reports" / "eval.json", "--out"),
) -> None:
    from autoevidence.data.synthetic import load_tickets_jsonl
    from autoevidence.evaluation.runner import evaluate_tickets, save_report

    path = ROOT / "data" / "synthetic" / "splits" / f"{split}.jsonl"
    if not path.exists():
        smoke()
    tickets = load_tickets_jsonl(path)
    report = evaluate_tickets(tickets, set_name=f"synthetic_{split}")
    save_report(report, out)
    rprint(f"wrote {out}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("autoevidence.api.app:app", host=host, port=port, reload=False)


@app.command()
def ablations() -> None:
    from autoevidence.evaluation.ablation import ablation_registry

    rprint(ablation_registry())


@app.command("export-cases")
def export_cases(
    out: Path = typer.Option(ROOT / "outputs" / "reports" / "e2e_cases.json", "--out"),
) -> None:
    """导出端到端案例：正常/事件/难负例/证据不足/安全转人工。"""
    from autoevidence.evaluation.cases import export_e2e_cases

    path = export_e2e_cases(out)
    rprint(f"wrote {path}")


@app.command("seal-safety")
def seal_safety(
    version: str = typer.Option("seal_v0", "--version"),
    out: Path = typer.Option(None, "--out"),
    max_samples: int = typer.Option(8, "--max-samples"),
) -> None:
    """封存 safety_seal：样本 ID、内容哈希、标签/条款版本、使用规则。"""
    from autoevidence.data.seal import seal_tickets
    from autoevidence.data.split import assign_splits, leakage_check
    from autoevidence.data.synthetic import generate_synthetic_corpus

    tickets = assign_splits(generate_synthetic_corpus(80), seed=42)
    # 正式验收门槛：对全集报告 strict_ok（合成可能失败，写入清单解释）
    leak = leakage_check(tickets)
    dest = out or (ROOT / "data" / "seals" / version)
    manifest = seal_tickets(
        tickets,
        dest,
        seal_version=version,
        max_samples=max_samples,
        require_strict_ok=False,
    )
    manifest["full_corpus_leakage"] = {
        "ok": leak.get("ok"),
        "strict_ok": leak.get("strict_ok"),
        "near_dup_cross_split_count": leak.get("near_dup_cross_split_count"),
        "shared_image_cross_split_count": leak.get("shared_image_cross_split_count"),
        "formal_note": "正式验收要求 strict_ok=true；否则须逐项解释共图/近重复",
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rprint(
        {
            "seal_version": version,
            "sample_count": manifest["sample_count"],
            "out": str(dest),
            "strict_ok": leak.get("strict_ok"),
            "manifest_sha256": manifest.get("manifest_sha256"),
        }
    )


@app.command("verify-seal")
def verify_seal(
    dir: Path = typer.Option(ROOT / "data" / "seals" / "seal_v0", "--dir"),
) -> None:
    from autoevidence.data.seal import verify_seal_integrity

    rprint(verify_seal_integrity(dir))


@app.command("mark-seal-failed")
def mark_seal_failed_cmd(
    dir: Path = typer.Option(ROOT / "data" / "seals" / "seal_v0", "--dir"),
    reason: str = typer.Option(..., "--reason"),
) -> None:
    from autoevidence.data.seal import mark_seal_failed

    rprint(mark_seal_failed(dir, reason=reason))


@app.command("reward-spotcheck")
def reward_spotcheck(
    out: Path = typer.Option(ROOT / "outputs" / "reports" / "reward_spotcheck.json", "--out"),
) -> None:
    """导出奖励人工抽查对：方差≠质量优劣。"""
    from autoevidence.rewards.spotcheck import export_spotcheck

    path = export_spotcheck(out)
    rprint(f"wrote {path}")


@app.command("deepseek-ping")
def deepseek_ping(
    model: str = typer.Option("deepseek-flash", "--model"),
    out: Path = typer.Option(ROOT / "outputs" / "deepseek_pilot", "--out"),
) -> None:
    """DeepSeek 连通性调用（真实 API，无规则回退）。"""
    from autoevidence.experiments.deepseek_pilot import ping

    rprint(ping(out, model=model))


@app.command("deepseek-teacher")
def deepseek_teacher(
    n: int = typer.Option(20, "--n"),
    model: str = typer.Option("deepseek-flash", "--model"),
    out: Path = typer.Option(ROOT / "outputs" / "deepseek_pilot", "--out"),
) -> None:
    """train 分区教师合成小批量（真实 API）。"""
    from autoevidence.experiments.deepseek_pilot import run_teacher_batch

    rprint(run_teacher_batch(out, n=n, model=model, require_real_api=True))


@app.command("deepseek-judge")
def deepseek_judge(
    model: str = typer.Option("deepseek-flash", "--model"),
    out: Path = typer.Option(ROOT / "outputs" / "deepseek_pilot", "--out"),
) -> None:
    """裁判候选对比 + 顺序交换（真实 API，不读 gold）。"""
    from autoevidence.experiments.deepseek_pilot import run_judge_batch

    rprint(run_judge_batch(out, model=model, require_real_api=True))


@app.command("deepseek-pilot")
def deepseek_pilot(
    model: str = typer.Option("deepseek-flash", "--model"),
    n: int = typer.Option(20, "--n"),
    out: Path = typer.Option(ROOT / "outputs" / "deepseek_pilot", "--out"),
) -> None:
    """连通性 → 教师20条 → 裁判≥10对；写 PILOT_REPORT。"""
    from autoevidence.experiments.deepseek_pilot import (
        ping,
        run_judge_batch,
        run_teacher_batch,
        write_pilot_report,
    )

    p = ping(out, model=model)
    if not p.get("ok"):
        raise typer.Exit(code=2)
    t = run_teacher_batch(out, n=n, model=model, require_real_api=True)
    j = run_judge_batch(out, model=model, require_real_api=True)
    report = write_pilot_report(out, t, j, p)
    rprint({"report": str(report), "teacher_pass_rate": t.get("pass_rate"), "judge": j.get("position_consistency_rate")})


if __name__ == "__main__":
    app()
