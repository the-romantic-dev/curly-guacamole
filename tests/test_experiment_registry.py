import json

import pytest

from src.training.notes import notes_template, parse_notes


def card(name, extra=""):
    return f"---\nrun: {name}\nseries: test\nverdict: inconclusive\n{extra}---\n\nОписание.\n"


def test_parent_is_default_baseline_and_can_be_overridden():
    assert parse_notes(card("child", "parent: root\n")).baseline == "root"
    assert parse_notes(card("child", "parent: root\nbaseline:\n")).baseline == "root"
    assert parse_notes(card("child", "parent: root\nbaseline: other\n")).baseline == "other"
    assert parse_notes(card("old", "baseline: root\n")).baseline == "root"
    assert parse_notes(card("old")).parent is None


@pytest.mark.parametrize("value", ["0", "1", "0.834"])
def test_manual_leaderboard_score(value):
    assert parse_notes(card("a", f"leaderboard_score: {value}\n")).leaderboard_score == float(value)


@pytest.mark.parametrize("extra", ["leaderboard_score: .nan\n", "leaderboard_score: true\n", "leaderboard_score: 1.1\n", "parent: [a, b]\n", "baseline: [a, b]\n"])
def test_invalid_metadata_is_rejected(extra):
    with pytest.raises(ValueError):
        parse_notes(card("a", extra))


def test_template_contains_optional_fields():
    text = notes_template("new")
    assert "parent:" in text and "leaderboard_score:" in text and "change:" in text
    assert parse_notes(text).leaderboard_score is None


def write_run(root, name, extra=""):
    folder = root / name
    folder.mkdir()
    (folder / "notes.md").write_text(card(name, extra), encoding="utf-8")
    return folder


def test_registry_reads_tree_and_scores_without_writing(tmp_path):
    from src.training.experiments import ExperimentRegistry

    write_run(tmp_path, "root")
    child = write_run(tmp_path, "child", "parent: root\nleaderboard_score: 0.8\nchange: decoder\n")
    (child / "summary.json").write_text(json.dumps({"best_aic": 0.9}), encoding="utf-8")
    (tmp_path / "unrecorded").mkdir()
    registry = ExperimentRegistry(tmp_path)
    table = registry.table().set_index("run")
    assert table.loc["child", "baseline"] == "root"
    assert table.loc["child", "leaderboard_score"] == 0.8
    assert table.loc["child", "validation_aic"] == 0.9
    assert "root\n└── child" in registry.tree()
    assert "unrecorded" in registry.tree()
    assert not (tmp_path / "unrecorded" / "notes.md").exists()


def test_registry_reports_invalid_links_and_still_shows_all_runs(tmp_path):
    from src.training.experiments import ExperimentRegistry

    write_run(tmp_path, "a", "parent: b\n")
    write_run(tmp_path, "b", "parent: a\n")
    write_run(tmp_path, "c", "parent: missing\nbaseline: other\n")
    registry = ExperimentRegistry(tmp_path)
    issues = "\n".join(registry.issues)
    assert "цикл" in issues and "missing" in issues and "other" in issues
    assert all(name in registry.tree() for name in ["a", "b", "c"])
