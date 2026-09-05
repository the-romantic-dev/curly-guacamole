"""Карточка прогона: смысл эксперимента рядом с его числами.

Метрик здесь нет намеренно. Единственный источник чисел — `summary.json`;
карточка отвечает за «что проверяли» и «что из этого следует». Разъехаться им
негде: карточка лежит внутри папки прогона, а папка после завершения не меняется.

Модуль ничего не знает ни про torch, ни про способ обучения: он разбирает
markdown с YAML-шапкой и проверяет, что обязательные поля на месте.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: единственный файл папки прогона, который попадает под git
NOTES_NAME = "notes.md"

#: чем закончился эксперимент. `broken` — прогон не обучился и мерять нечего
VERDICTS = ("proved", "refuted", "inconclusive", "diagnostic", "broken")

#: без этих полей карточка бесполезна для сборки RESULTS.md
REQUIRED = ("run", "series", "verdict")

#: абзац, из которого берётся строка в RESULTS.md
HEADLINE_MARKER = "**Что проверяли.**"

#: слово, которым заготовка отмечает незаполненное место. Пока оно в теле,
#: карточки фактически нет, и RESULTS.md обязан считать это долгом, а не
#: документацией — иначе заготовка молча закрывала бы прогон от учёта
STUB_MARKER = "ЗАПОЛНИТЬ"

_FRONT_MATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?(.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class Notes:
    """Разобранная карточка. Числа сюда не попадают и попасть не могут."""

    run: str
    series: str
    verdict: str
    baseline: str | None
    source: str | None
    covers: tuple[str, ...]
    body: str

    @property
    def headline(self) -> str:
        """Одна строка для RESULTS.md: абзац «что проверяли» без переносов."""
        return _paragraph(self.body, HEADLINE_MARKER) or _first_paragraph(self.body)

    @property
    def is_stub(self) -> bool:
        """Заготовка, которую ещё не заполнили. Для учёта это отсутствие карточки."""
        return STUB_MARKER in self.body


def series_of(run_name: str) -> str:
    """Догадка о серии по имени папки: всё до первого разделителя.

    Догадка, а не правило: заготовка ставит её автоматически, человек правит.
    """
    return re.split(r"[-_]", run_name, maxsplit=1)[0]


def parse_notes(text: str, *, expected_run: str | None = None) -> Notes:
    """Разобрать карточку. Любая неполнота — `ValueError` с внятным текстом."""
    match = _FRONT_MATTER.match(text)
    if match is None:
        raise ValueError(
            "в карточке нет YAML-шапки: файл обязан начинаться со строки '---' "
            "и закрывать шапку второй такой же строкой"
        )

    head_text, body = match.group(1), match.group(2)
    head = yaml.safe_load(head_text) or {}
    if not isinstance(head, Mapping):
        raise ValueError("шапка карточки — не словарь полей")

    missing = [key for key in REQUIRED if not head.get(key)]
    if missing:
        raise ValueError(f"в шапке карточки нет обязательных полей: {', '.join(missing)}")

    verdict = str(head["verdict"])
    if verdict not in VERDICTS:
        raise ValueError(
            f"неизвестный verdict {verdict!r}; допустимы: {', '.join(VERDICTS)}"
        )

    run = str(head["run"])
    if expected_run is not None and run != expected_run:
        raise ValueError(
            f"в шапке run={run!r}, а карточка лежит в папке {expected_run!r}: "
            f"одно из двух переименовано"
        )

    return Notes(
        run=run,
        series=str(head["series"]),
        verdict=verdict,
        baseline=_optional(head.get("baseline")),
        source=_optional(head.get("source")),
        covers=tuple(str(item) for item in (head.get("covers") or ())),
        body=body.strip(),
    )


def read_notes(run_dir: str | Path) -> Notes | None:
    """Карточка из папки прогона или None, если её там нет."""
    path = Path(run_dir) / NOTES_NAME
    if not path.exists():
        return None
    return parse_notes(path.read_text(encoding="utf-8"), expected_run=Path(run_dir).name)


def notes_template(run_name: str) -> str:
    """Заготовка: шапка заполнена, тело пустое и заметно пустое."""
    return (
        "---\n"
        f"run: {run_name}\n"
        f"series: {series_of(run_name)}\n"
        "baseline:\n"
        "verdict: inconclusive\n"
        "source:\n"
        "---\n"
        "\n"
        "**Что проверяли.** ЗАПОЛНИТЬ: ровно одно отличие от baseline, одним предложением.\n"
        "\n"
        "**Что получилось.** ЗАПОЛНИТЬ: числа берутся из summary.json и приводятся\n"
        "как часть объяснения, а не как данные.\n"
        "\n"
        "**Чего это не доказывает.** ЗАПОЛНИТЬ: границы вывода.\n"
    )


def _optional(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _paragraph(body: str, marker: str) -> str | None:
    """Абзац, начинающийся с маркера, без маркера и без переносов строк."""
    for chunk in body.split("\n\n"):
        stripped = chunk.strip()
        if stripped.startswith(marker):
            return " ".join(stripped[len(marker):].split())
    return None


def _first_paragraph(body: str) -> str:
    for chunk in body.split("\n\n"):
        stripped = chunk.strip()
        if stripped:
            return " ".join(stripped.split())
    return ""
