# ADR 2: Compare what serving produces with what training saw, feature by feature

## Context

Even with one definition per feature, serving can drift from training: a client sends amounts in
pence instead of pounds, an upstream service changes a default, or a feature's freshness slips.
A model doesn't fail on such inputs. It quietly returns worse scores.

## Decision

For every feature, compute the population stability index (PSI) between the training distribution
(point-in-time values for the training period) and the values serving produced. Bins come from the
training deciles. Alert at PSI ≥ 0.2, a common rule of thumb for a shift that needs action.

## Consequences

- Healthy serving stays well below the alert level on every feature, even with natural drift
  between the training and serving periods.
- A pence-for-pounds bug flags every amount feature, while counts and recency stay quiet, which
  points straight at the cause.
- PSI is unsupervised and cheap, but it can't tell a harmful shift from a harmless one. Pairing it
  with live model metrics (once labels arrive) is still needed.
