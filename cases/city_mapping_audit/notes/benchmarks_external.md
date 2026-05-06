# External benchmarks — context only, not used in this audit

We've occasionally been asked "what do external benchmarks say about
hotel-name matching?" Answer: the closest public benchmarks are

* **Booking.com 2023 property-matching evaluation** — 3.1M hotels,
  standardised address-inclusive inputs. Accuracy on name-only subset
  reported at ~0.71 top-1 for their in-house system; they are not
  limited to name-only retrieval so the comparison is loose.

* **Expedia 2022 canonicalisation paper** — reports 82% top-1 but
  evaluates on a GT derived from their own property database, which
  makes the measurement circular.

* **OpenTravelData** — public hotel dataset with ~2M entries. Not a
  retrieval benchmark per se but often used to bootstrap fuzzy
  scorers.

None of these are directly comparable to our setup. They're included
here only to head off the "why aren't you at 80%?" question.
