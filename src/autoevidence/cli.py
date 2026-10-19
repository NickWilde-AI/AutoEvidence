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


if __name__ == "__main__":
    app()
