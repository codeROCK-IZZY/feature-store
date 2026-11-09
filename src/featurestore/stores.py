"""The offline store (training sets) and the online store (serving), built independently.

Offline, four ways to attach features to labelled moments:
  point in time   events that had *arrived* by the moment and happened inside the window: what serving sees
  event time      events that *happened* inside the window, even if they arrived later (late data leaks in)
  label date      windows ending when the label became known (moment + 14 days): the label period leaks in,
                  a common bug when labels are keyed by the date the outcome was recorded
  snapshot        today's feature values joined to every past label (the classic join on customer id)

Online: events are applied in arrival order to a per-customer buffer; a request computes the features from
the buffer at that moment. If the offline point-in-time values don't equal what the online store returned at
the same moment, training and serving disagree.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from .data import HORIZON, Events, Labels
from .features import MAX_WINDOW, Feature, aggregate


def _by_user(ev: Events) -> dict[int, np.ndarray]:
    order = np.argsort(ev.user, kind="stable")
    users = ev.user[order]
    starts = np.searchsorted(users, np.arange(users.max() + 2))
    return {u: order[starts[u] : starts[u + 1]] for u in range(len(starts) - 1)}


def offline(ev: Events, labels: Labels, features: list[Feature], mode: str = "point_in_time") -> np.ndarray:
    out = np.zeros((len(labels.user), len(features)))
    idx = _by_user(ev)
    snapshot: dict[int, list[float]] = {}
    end = float(ev.ingest_time.max())
    for row, (u, ts) in enumerate(zip(labels.user, labels.ts, strict=True)):
        e = idx.get(int(u), np.zeros(0, int))
        et, it = ev.event_time[e], ev.ingest_time[e]
        if mode == "snapshot" and int(u) in snapshot:
            out[row] = snapshot[int(u)]
            continue
        for k, f in enumerate(features):
            now = {"snapshot": end, "label_date": float(ts) + HORIZON}.get(mode, float(ts))
            m = (et > now - f.window) & (et <= now)
            if mode == "point_in_time":
                m &= it <= now
            col = ev.amount[e] if f.column == "amount" else ev.category[e]
            out[row, k] = aggregate(f, col[m], et[m], now)
        if mode == "snapshot":
            snapshot[int(u)] = list(out[row])
    return out


class OnlineStore:
    """Per-customer buffers of recent events; old ones are evicted as time moves on."""

    def __init__(self, features: list[Feature], amount_scale: float = 1.0) -> None:
        self.features = features
        self.buf: dict[int, deque[tuple[float, float, int]]] = defaultdict(deque)
        self.amount_scale = amount_scale  # 1.0 normally; 100.0 reproduces a cents-vs-pounds bug

    def ingest(self, user: int, event_time: float, amount: float, category: int) -> None:
        self.buf[user].append((event_time, amount * self.amount_scale, category))

    def get(self, user: int, now: float) -> list[float]:
        b = self.buf.get(user)
        if b is None:
            b = deque()
        while b and b[0][0] <= now - MAX_WINDOW and len(b) > 0:
            b.popleft()
        arr = np.array(list(b)) if b else np.zeros((0, 3))
        out = []
        for f in self.features:
            m = (arr[:, 0] > now - f.window) & (arr[:, 0] <= now) if len(arr) else np.zeros(0, bool)
            col = arr[:, 1] if f.column == "amount" else arr[:, 2]
            out.append(aggregate(f, col[m] if len(arr) else np.zeros(0), arr[m, 0] if len(arr) else np.zeros(0), now))
        return out


def replay(ev: Events, labels: Labels, store: OnlineStore) -> np.ndarray:
    """Interleave event arrivals and feature requests in time order, as production would see them."""
    timeline = [(float(t), 0, i) for i, t in enumerate(ev.ingest_time)] + [
        (float(t), 1, i) for i, t in enumerate(labels.ts)
    ]
    timeline.sort()
    out = np.zeros((len(labels.user), len(store.features)))
    for _, kind, i in timeline:
        if kind == 0:
            store.ingest(int(ev.user[i]), float(ev.event_time[i]), float(ev.amount[i]), int(ev.category[i]))
        else:
            out[i] = store.get(int(labels.user[i]), float(labels.ts[i]))
    return out


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index between a training and a serving distribution of one feature."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(np.clip(expected, edges[0], edges[-1]), edges)[0] / len(expected)
    a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / len(actual)
    e, a = np.maximum(e, 1e-4), np.maximum(a, 1e-4)
    return float(((a - e) * np.log(a / e)).sum())
