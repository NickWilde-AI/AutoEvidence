from autoevidence.rewards.spotcheck import build_spotcheck_pairs


def test_spotcheck_pairs_ordered():
    pairs = build_spotcheck_pairs()
    assert len(pairs) >= 2
    for p in pairs:
        assert p["high_reward"]["score"]["total"] > p["low_reward"]["score"]["total"]
        assert "human_prompt" in p
        assert p["reviewer"]["better_is_truly_better"] is None
