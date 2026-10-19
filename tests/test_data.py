from autoevidence.data.augment import swap_augment
from autoevidence.data.split import assign_splits, leakage_check
from autoevidence.data.synthetic import generate_synthetic_corpus
from autoevidence.schemas.enums import SplitName
from autoevidence.standards.loader import load_clause_library


def test_synthetic_and_isolation():
    tickets = generate_synthetic_corpus(30)
    tickets = assign_splits(tickets, seed=1)
    report = leakage_check(tickets)
    assert report["ok"]
    assert report["safety_train_overlap"] == []


def test_swap_only_train_parents():
    tickets = assign_splits(generate_synthetic_corpus(40), seed=2)
    lib = load_clause_library()
    swapped = swap_augment(tickets, lib, max_pairs=5)
    for s in swapped:
        assert s.split == SplitName.TRAIN
        assert len(s.parent_ids) == 2
        assert s.meta.get("label_confidence") == "ambiguous_needs_human"
