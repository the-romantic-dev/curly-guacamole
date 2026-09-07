"""Обзор карточек и результатов без изменения файлов экспериментов."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from .notes import Notes, read_notes


class ExperimentRegistry:
    """Снимок папки runs. Создайте заново после редактирования карточек."""

    COLUMNS = (
        "run", "parent", "baseline", "series", "change", "validation_aic",
        "validation_resolution", "leaderboard_score", "verdict", "notes_status",
    )

    def __init__(self, runs_root: str | Path) -> None:
        self.root = Path(runs_root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.notes: dict[str, Notes | None] = {}
        self.issues: list[str] = []
        self._rows: list[dict] = []
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir() or folder.name.startswith((".", "__")):
                continue
            status = "missing"
            try:
                note = read_notes(folder)
                if note is not None:
                    status = "stub" if note.is_stub else "filled"
            except (ValueError, OSError, yaml.YAMLError) as exc:
                note = None
                status = "invalid"
                self.issues.append(f"{folder.name}/notes.md: {exc}")
            self.notes[folder.name] = note
            summary = self._read_summary(folder)
            self._rows.append(dict(
                run=folder.name,
                parent=note.parent if note else None,
                baseline=note.baseline if note else None,
                series=note.series if note else None,
                change=note.change if note else None,
                validation_aic=summary.get("best_aic"),
                validation_resolution=summary.get("validation_resolution"),
                leaderboard_score=note.leaderboard_score if note else None,
                verdict=note.verdict if note else None,
                notes_status=status,
            ))
        self._check_links()

    def _read_summary(self, folder: Path) -> dict:
        path = folder / "summary.json"
        if not path.exists():
            return {}
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(summary, dict):
                raise ValueError("ожидался JSON-объект")
            return summary
        except (ValueError, OSError) as exc:
            self.issues.append(f"{folder.name}/summary.json: {exc}")
            return {}

    def _check_links(self) -> None:
        for name, note in self.notes.items():
            if note is None:
                continue
            for field, target in (("parent", note.parent), ("baseline", note.baseline)):
                if target and target not in self.notes:
                    self.issues.append(f"{name}: неизвестный {field} {target!r}")
                if field == "baseline" and target == name:
                    self.issues.append(f"{name}: baseline ссылается на себя")
        checked: set[str] = set()
        for name in self.notes:
            path: list[str] = []
            current = name
            while current in self.notes and current not in checked:
                if current in path:
                    cycle = path[path.index(current):] + [current]
                    self.issues.append("цикл parent: " + " -> ".join(cycle))
                    break
                path.append(current)
                note = self.notes[current]
                current = note.parent if note else None
            checked.update(path)

    def table(self) -> pd.DataFrame:
        """Числа без автоматических дельт: сопоставимость проверяет исследователь."""
        return pd.DataFrame(self._rows, columns=self.COLUMNS)

    def tree(self) -> str:
        """Дерево parent; отсутствующие связи и циклы перечислены в issues."""
        children: dict[str, list[str]] = {name: [] for name in self.notes}
        roots = []
        for name, note in self.notes.items():
            parent = note.parent if note else None
            if parent in children:
                children[parent].append(name)
            else:
                roots.append(name)
        lines: list[str] = []
        visited: set[str] = set()
        for root in roots + list(self.notes):
            if root in visited:
                continue
            stack = [(root, "", "", "")]
            while stack:
                name, prefix, branch, suffix = stack.pop()
                if name in visited:
                    lines.append(prefix + branch + name + " [повторная связь]")
                    continue
                visited.add(name)
                lines.append(prefix + branch + name)
                siblings = children[name]
                for index in reversed(range(len(siblings))):
                    last = index == len(siblings) - 1
                    stack.append((siblings[index], prefix + suffix,
                                  "└── " if last else "├── ",
                                  "    " if last else "│   "))
        return "\n".join(lines)
