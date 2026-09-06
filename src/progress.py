"""Immediate console progress for long experiment operations."""

import time
from collections.abc import Callable, Iterable, Iterator, Sized
from typing import TypeVar

T = TypeVar("T")


class ConsoleProgress:
    @staticmethod
    def format_duration(seconds: float) -> str:
        hours, remainder = divmod(max(0, round(seconds)), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def info(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    @classmethod
    def iterate(
        cls, items: Iterable[T], label: str, *, image_count: Callable[[T], int] | None = None
    ) -> Iterator[T]:
        total = len(items) if isinstance(items, Sized) else None
        started = last_report = time.monotonic()
        first_completed = None
        count = 0
        images_processed = 0
        cls.info(f"{label}: начало, всего {total if total is not None else '?'}; ожидание первого элемента")
        for count, item in enumerate(items, 1):
            yield item
            now = time.monotonic()
            if image_count is not None:
                images_processed += image_count(item)
            if count == 1:
                # Startup and the first item are not representative of steady throughput.
                first_completed = now
            if count == 1 or count == total or now - last_report >= 30:
                elapsed = now - started
                status = f"{count}/{total}" if total is not None else str(count)
                eta = ""
                if total is not None:
                    if count == total:
                        eta = ", осталось ~00:00:00"
                    elif count > 1 and first_completed is not None:
                        seconds_per_item = (now - first_completed) / (count - 1)
                        eta = f", осталось ~{cls.format_duration(seconds_per_item * (total - count))}"
                throughput = ""
                if images_processed > 0 and elapsed > 0:
                    throughput = (
                        f", {elapsed * 1000 / images_processed:.2f} мс/изобр."
                        f", {images_processed / elapsed:.2f} изобр./с"
                    )
                cls.info(f"{label}: {status}, прошло {cls.format_duration(elapsed)}{throughput}{eta}")
                last_report = now
        cls.info(f"{label}: завершено, обработано {count}, {cls.format_duration(time.monotonic() - started)}")
