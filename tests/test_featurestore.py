from __future__ import annotations

import numpy as np
import pytest

from featurestore import data, stores
from featurestore.features import FEATURES, Feature, Registry, aggregate


def ev_of(rows: list[tuple[int, float, float, float, int]]) -> data.Events:
    a = np.array(rows, dtype=float)
    return data.Events(a[:, 0].astype(int), a[:, 1], a[:, 2], a[:, 3], a[:, 4].astype(int))


EVENTS = ev_of(
    [
        # user, event time, ingest time, amount, category
        (0, 1.0, 1.0, 10.0, 1),
        (0, 9.0, 9.0, 20.0, 2),
        (0, 9.5, 12.0, 99.0, 3),  # happened on day 9.5 but arrived on day 12
        (0, 15.0, 15.0, 5.0, 1),  # after the moment: must never count
    ]
)
MOMENT = data.Labels(np.array([0]), np.array([10.0]), np.array([True]))


def test_aggregations() -> None:
    f = Feature("x", "mean", 7)
    assert aggregate(f, np.array([2.0, 4.0]), np.array([1.0, 2.0]), 3.0) == 3.0
    assert aggregate(Feature("d", "days_since", 60), np.zeros(0), np.zeros(0), 5.0) == 60.0
    assert aggregate(Feature("c", "distinct", 30, "category"), np.array([1, 1, 2]), np.zeros(3), 0.0) == 2.0


def test_point_in_time_uses_only_what_had_arrived() -> None:
    x = dict(zip([f.name for f in FEATURES], stores.offline(EVENTS, MOMENT, FEATURES, "point_in_time")[0], strict=True))
    assert x["txn_count_30d"] == 2  # the late event and the future event are excluded
    assert x["amount_sum_7d"] == 20.0  # day 1 is outside the 7-day window
    assert x["days_since_last"] == pytest.approx(1.0)


def test_event_time_join_leaks_late_data_and_label_date_leaks_the_future() -> None:
    ev_t = dict(zip([f.name for f in FEATURES], stores.offline(EVENTS, MOMENT, FEATURES, "event_time")[0], strict=True))
    assert ev_t["txn_count_30d"] == 3  # counts the event that had not arrived yet
    lab = dict(zip([f.name for f in FEATURES], stores.offline(EVENTS, MOMENT, FEATURES, "label_date")[0], strict=True))
    assert lab["txn_count_30d"] == 4  # includes day 15, inside the label period


def test_online_store_matches_point_in_time_exactly() -> None:
    online = stores.replay(EVENTS, MOMENT, stores.OnlineStore(FEATURES))
    offline = stores.offline(EVENTS, MOMENT, FEATURES, "point_in_time")
    assert np.array_equal(online, offline)


def test_online_matches_offline_on_generated_data() -> None:
    ev, lb = data.generate(n_users=150, seed=9)
    assert np.allclose(stores.replay(ev, lb, stores.OnlineStore(FEATURES)), stores.offline(ev, lb, FEATURES))


def test_late_events_exist_and_labels_look_forward() -> None:
    ev, lb = data.generate(n_users=200, seed=1)
    late = ev.ingest_time - ev.event_time
    assert (late >= 0).all()
    assert 0.02 < np.mean(late > 0.5) < 0.12
    assert lb.ts.max() <= data.DAYS - data.HORIZON


def test_registry_requires_a_new_version_to_change_a_feature() -> None:
    r = Registry.default()
    with pytest.raises(ValueError, match="new version"):
        r.register(Feature("amount_sum_7d", "sum", 14))
    r.register(Feature("amount_sum_7d", "sum", 14, version=2))
    assert r.features["amount_sum_7d"].window == 14


def test_psi_flags_a_shift_and_ignores_noise() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.lognormal(3, 0.5, 5000), rng.lognormal(3, 0.5, 5000)
    assert stores.psi(a, b) < 0.05
    assert stores.psi(a, b * 100) > 1.0
