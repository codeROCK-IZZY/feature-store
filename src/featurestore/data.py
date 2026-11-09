"""Card transactions for a few thousand customers over 120 days, and the question a model must answer.

The question (label): at a given moment, will this customer buy anything in the next 14 days?

Each customer has a purchase rate; some start to lapse at a random moment and their rate decays, which
is what makes recent activity predictive. Events reach the warehouse late: most within minutes, but 6%
arrive one to three days after they happened. A feature computed "as of" a moment must only use what
had *arrived* by then, or the training data knows things serving never will.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DAYS = 120
HORIZON = 14
CATEGORIES = 8


@dataclass(frozen=True)
class Events:
    user: np.ndarray
    event_time: np.ndarray  # days, float
    ingest_time: np.ndarray  # when the event became available
    amount: np.ndarray
    category: np.ndarray


@dataclass(frozen=True)
class Labels:
    user: np.ndarray
    ts: np.ndarray  # the moment of prediction
    y: np.ndarray  # bought anything in (ts, ts + HORIZON]


def generate(n_users: int = 4000, seed: int = 3) -> tuple[Events, Labels]:
    rng = np.random.default_rng(seed)
    base_rate = rng.lognormal(-2.3, 0.9, n_users)  # purchases per day
    lapse_at = np.where(rng.random(n_users) < 0.5, rng.uniform(20, DAYS + 30, n_users), np.inf)
    lapse_speed = rng.uniform(0.03, 0.15, n_users)
    spend = rng.lognormal(3.2, 0.6, n_users)
    fav = rng.integers(0, CATEGORIES, n_users)
    us, et, it, am, ca = [], [], [], [], []
    for u in range(n_users):
        # thinning: candidate events at the base rate, kept with the decayed rate's share
        n = rng.poisson(base_rate[u] * (DAYS + HORIZON))
        t = np.sort(rng.uniform(0, DAYS + HORIZON, n))
        decay = np.where(t > lapse_at[u], np.exp(-lapse_speed[u] * (t - lapse_at[u])), 1.0)
        t = t[rng.random(n) < decay]
        delay = np.where(rng.random(len(t)) < 0.06, rng.uniform(1, 3, len(t)), rng.exponential(0.002, len(t)))
        us.append(np.full(len(t), u))
        et.append(t)
        it.append(t + delay)
        am.append(np.round(spend[u] * rng.lognormal(0, 0.5, len(t)), 2))
        ca.append(np.where(rng.random(len(t)) < 0.6, fav[u], rng.integers(0, CATEGORIES, len(t))))
    ev = Events(np.concatenate(us), np.concatenate(et), np.concatenate(it), np.concatenate(am), np.concatenate(ca))
    # one prediction moment per customer per week, from day 35 (enough history) to the last full horizon
    lu, lt, ly = [], [], []
    order = np.argsort(ev.user, kind="stable")
    starts = np.searchsorted(ev.user[order], np.arange(n_users + 1))
    for u in range(n_users):
        times = np.sort(ev.event_time[order][starts[u] : starts[u + 1]])
        for ts in np.arange(35.0, DAYS - HORIZON + 1e-9, 7.0) + rng.uniform(0, 1):
            lu.append(u)
            lt.append(ts)
            ly.append(bool(((times > ts) & (times <= ts + HORIZON)).any()))
    return ev, Labels(np.array(lu), np.array(lt), np.array(ly))
