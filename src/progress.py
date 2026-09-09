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
        cls, items: Iterable[T], label: str, *, image_count: Callable[[T], int] | None = None,
        measure_wait: bool = False,
    ) -> Iterator[T]:
        total = len(items) if isinstance(items, Sized) else None
        started = last_report = time.monotonic()
        first_completed = None
        count = 0
        images_processed = 0
        wait_seconds = work_seconds = 0.0
        measured_batches = 0
        cls.info(f"{label}: начало, всего {total if total is not None else '?'}; ожидание первого элемента")
        waiting_since = time.monotonic()
        for count, item in enumerate(items, 1):
            received = time.monotonic()
            yield item
            now = time.monotonic()
            if count > 1:
                wait_seconds += received - waiting_since
                work_seconds += now - received
                measured_batches += 1
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
                waiting = cls._wait_report(wait_seconds, work_seconds, measured_batches) if measure_wait else ""
                cls.info(f"{label}: {status}, прошло {cls.format_duration(elapsed)}{throughput}{waiting}{eta}")
                last_report = now
            waiting_since = time.monotonic()
        waiting = cls._wait_report(wait_seconds, work_seconds, measured_batches) if measure_wait else ""
        cls.info(f"{label}: завершено, обработано {count}, {cls.format_duration(time.monotonic() - started)}{waiting}")

    @staticmethod
    def _wait_report(wait_seconds, work_seconds, count):
        # Host-side next(loader) latency, not GPU idle time: CUDA may still work.
        if not count:
            return ""
        fraction = wait_seconds / max(wait_seconds + work_seconds, 1e-9)
        return f", ожидание данных={1000 * wait_seconds / count:.1f} мс/батч ({100 * fraction:.1f}%)"
