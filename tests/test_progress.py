import pytest

from src.progress import ConsoleProgress


def test_eta_excludes_slow_first_batch(monkeypatch, capsys):
    clock = [0.0]
    monkeypatch.setattr("src.progress.time.monotonic", lambda: clock[0])

    for index in ConsoleProgress.iterate(range(6000), "training"):
        clock[0] += 47.4 if index == 0 else 30.0 / 177

    lines = capsys.readouterr().out.splitlines()
    first = next(line for line in lines if "1/6000," in line)
    assert "прошло 00:00:47" in first
    assert "осталось ~" not in first
    report = next(line for line in lines if "осталось ~" in line)
    # About 30 seconds for 177 batches, independent of the 47.4-second startup.
    hours, minutes, seconds = map(int, report.split("осталось ~")[1].split(":"))
    remaining = hours * 3600 + minutes * 60 + seconds
    assert 985 <= remaining <= 987
    assert "6000/6000," in lines[-2]
    assert "осталось ~00:00:00" in lines[-2]


@pytest.mark.parametrize("items", [[], [1], iter([1, 2])])
def test_progress_handles_short_and_unsized_iterables(items, capsys):
    result = list(ConsoleProgress.iterate(items, "work"))
    output = capsys.readouterr().out
    assert f"завершено, обработано {len(result)}" in output
    if result == [1]:
        assert "осталось ~00:00:00" in output
