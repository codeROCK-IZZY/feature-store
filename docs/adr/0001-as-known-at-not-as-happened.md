# ADR 1: Training rows are built "as known at" the moment, not "as happened by" it

## Context

A training row pairs a label ("did the customer buy in the next 14 days?") with the features a model
would have seen at the moment of prediction. Three common shortcuts break that:

- **Joining today's feature table to historical labels.** Every past moment gets features computed
  with knowledge of everything since.
- **Keying labels by the date the outcome was recorded.** The label resolves 14 days after the
  prediction moment. If the join uses that date, the feature windows cover the label period itself.
  The model learns to read the answer.
- **Filtering on event time only.** An event that happened before the moment but arrived after it
  is counted, although serving could not have seen it.

## Decision

The offline store filters every window on both clocks: `event_time` inside the window **and**
`ingest_time` at or before the moment. Features are defined once (`features.py`) and evaluated by the
same `aggregate` function offline and online. The online store applies events in arrival order and
answers requests from what it holds at that moment.

## Consequences

- Replaying the stream through the online store reproduces every one of the 44,000 offline training
  rows exactly, so offline AUC equals production AUC.
- The label-date join scores a perfect 1.000 offline and 0.70 in production. That's worse than the
  honest model, and a team would only find out after launch.
- In this data, ignoring arrival time changes few rows (late events are 6% of the total), so it
  barely moves AUC. The rule still matters wherever late data is common or concentrated in the
  signal.
