from autoevidence.data.synthetic import generate_synthetic_corpus
from autoevidence.inference.pipeline import Analyzer


def test_analyze_ignores_gold():
    t = generate_synthetic_corpus(1)[0]
    assert t.gold is not None
    gold_dump = t.gold.model_dump()
    result = Analyzer(mode="demo").analyze(t)
    # 结果结构完整
    assert "model_output" in result
    assert "routing" in result
    assert "evidence_pack" in result
    # 金标未写入输出依赖字段（routing 不引用 gold）
    assert "gold" not in result
    # 原始 ticket 金标未被函数需要（拷贝丢弃）
    assert t.gold.model_dump() == gold_dump


def test_insufficient_evidence_routes_human():
    t = generate_synthetic_corpus(20)
    miss = [x for x in t if "evidence_missing" in x.tags][0]
    result = Analyzer(mode="demo").analyze(miss)
    assert result["routing"]["queue"] in ("人审", "归档", "技术排查", "紧急复核")
