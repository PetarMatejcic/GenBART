from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from collections import defaultdict

@dataclass
class ProfileStats:
    totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @contextmanager
    def section(self, name: str):
        t0 = perf_counter()
        try:
            yield
        finally:
            self.totals[name] += perf_counter() - t0
            self.counts[name] += 1

    def report(self, sort_by: str = "total") -> str:
        rows = []
        for name in self.totals:
            total = self.totals[name]
            count = self.counts[name]
            mean = total / count if count else 0.0
            rows.append((name, total, count, mean))

        key = 1 if sort_by == "total" else 3
        rows.sort(key=lambda x: x[key], reverse=True)

        lines = ["name                           total_s   count   mean_s"]
        for name, total, count, mean in rows:
            lines.append(f"{name:30s} {total:8.4f} {count:7d} {mean:8.6f}")
        return "\n".join(lines)

    def to_rows(self):
        rows = []
        for name in self.totals:
            total = self.totals[name]
            count = self.counts[name]
            mean = total / count if count else 0.0
            rows.append((name, total, count, mean))
        return rows