"""Feature definitions, written once and used by both the offline and the online store.

A feature view is declarative: an aggregation over a window of a customer's events. The registry
refuses to change a definition in place; a change is a new version, so a model trained on v1 is never
silently served v2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Feature:
    name: str
    agg: str  # count | sum | mean | max | distinct | days_since
    window: float  # days
    column: str = "amount"
    version: int = 1


FEATURES = [
    Feature("txn_count_7d", "count", 7),
    Feature("txn_count_30d", "count", 30),
    Feature("amount_sum_7d", "sum", 7),
    Feature("amount_sum_30d", "sum", 30),
    Feature("amount_mean_30d", "mean", 30),
    Feature("amount_max_30d", "max", 30),
    Feature("categories_30d", "distinct", 30, column="category"),
    Feature("days_since_last", "days_since", 60),
]
MAX_WINDOW = max(f.window for f in FEATURES)


def aggregate(f: Feature, values: np.ndarray, times: np.ndarray, now: float) -> float:
    """The one definition of every aggregation. `values`/`times` are the events already in the window."""
    if f.agg == "count":
        return float(len(values))
    if f.agg == "days_since":
        return float(now - times.max()) if len(times) else float(f.window)
    if not len(values):
        return 0.0
    if f.agg == "sum":
        return float(values.sum())
    if f.agg == "mean":
        return float(values.mean())
    if f.agg == "max":
        return float(values.max())
    if f.agg == "distinct":
        return float(len(np.unique(values)))
    raise ValueError(f.agg)


@dataclass
class Registry:
    features: dict[str, Feature] = field(default_factory=dict)

    def register(self, f: Feature) -> None:
        old = self.features.get(f.name)
        if old is not None and old != f and old.version >= f.version:
            raise ValueError(f"{f.name}: changing a definition needs a new version (registered v{old.version})")
        self.features[f.name] = f

    @classmethod
    def default(cls) -> Registry:
        r = cls()
        for f in FEATURES:
            r.register(f)
        return r
