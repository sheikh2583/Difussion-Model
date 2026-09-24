from scripts.verify_thesis_progression import STAGES, main


def test_thesis_progression_has_ordered_unique_stages() -> None:
    assert [stage.number for stage in STAGES] == list(range(1, 10))
    assert len({stage.name for stage in STAGES}) == len(STAGES)


def test_local_thesis_progression_evidence_is_present() -> None:
    assert main() == 0
