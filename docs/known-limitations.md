# Known limitations

- **In-memory stores.** The offline store scans events in Python and the online store keeps buffers
  in a dictionary. Production versions would use a columnar warehouse (with an as-of join) and a
  key-value store such as Redis or DynamoDB. The semantics shown here are what those systems must
  reproduce.
- **One entity type and one event stream.** Features over several entities (customer x merchant) and
  joins across streams are not shown.
- **Window aggregations only.** Embeddings, model-derived features and on-demand transformations at
  request time are not covered.
- **Synthetic transactions.** Customers, lapses and arrival delays are generated, so the size of
  each leak is specific to this data. The direction (leaky joins look better offline and do worse
  in production) is not.
- **Skew alerts use a fixed PSI threshold.** They don't account for sample size or seasonality and
  can't tell a harmful shift from a harmless one.
