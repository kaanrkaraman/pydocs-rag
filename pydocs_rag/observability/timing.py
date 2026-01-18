"""Timing instrumentation for pipeline stages."""

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TimingStats:
    stage_times: dict[str, list[float]] = field(default_factory=dict)

    def record(self, stage: str, duration_ms: float) -> None:
        if stage not in self.stage_times:
            self.stage_times[stage] = []
        self.stage_times[stage].append(duration_ms)

    def get_stats(self, stage: str) -> dict[str, float]:
        times = self.stage_times.get(stage, [])
        if not times:
            return {"count": 0, "mean": 0, "min": 0, "max": 0, "p50": 0, "p95": 0}

        sorted_times = sorted(times)
        n = len(sorted_times)

        return {
            "count": n,
            "mean": sum(times) / n,
            "min": sorted_times[0],
            "max": sorted_times[-1],
            "p50": sorted_times[n // 2],
            "p95": sorted_times[int(n * 0.95)] if n > 1 else sorted_times[-1],
            "total": sum(times),
        }

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        return {stage: self.get_stats(stage) for stage in self.stage_times}

    def reset(self) -> None:
        self.stage_times.clear()


class Timer:
    def __init__(self, stats: TimingStats | None = None) -> None:
        self.stats = stats
        self.current_timings: dict[str, float] = {}

    @contextmanager
    def time_stage(self, stage: str) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.current_timings[stage] = duration_ms
            if self.stats:
                self.stats.record(stage, duration_ms)

    def get_current_timings(self) -> dict[str, float]:
        return self.current_timings.copy()

    def get_total_time(self) -> float:
        return sum(self.current_timings.values())

    def reset_current(self) -> None:
        self.current_timings.clear()


_global_stats = TimingStats()


def get_global_stats() -> TimingStats:
    return _global_stats


@contextmanager
def timed_stage(
    stage: str, stats: TimingStats | None = None
) -> Generator[dict[str, float], None, None]:
    use_stats = stats or _global_stats
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        result["duration_ms"] = duration_ms
        use_stats.record(stage, duration_ms)
