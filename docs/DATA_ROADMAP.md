# Data roadmap — what an external join would unlock

The shipped listing cannot measure audience success. This document specifies
exactly which external sources would change that, how they would be keyed,
and what would still be missing.

## Current file

| field | present | notes |
|---|---|---|
| show_id, type, title, director, country | yes | `director` / `country` use `"Not Given"` as a missing token (2,588 / 287 rows) |
| date_added, release_year, rating, duration, listed_in | yes | `date_added` arrives as `M/D/YYYY` (and a few datetime cells from the xlsx) |
| cast | **no** | present on the public 8,807-row Kaggle listing (~9% null) |
| description | **no** | present on the public 8,807-row Kaggle listing |
| views / watch time / completion | **no** | requires internal telemetry |
| IMDb / RT / TMDB score | **no** | requires an external join |
| budget / revenue | **no** | requires TMDB / OMDb |
| awards / renewals | **no** | requires Wikidata / internal |

## Join 1 — TMDB or OMDb (ratings, votes, popularity, budget, revenue)

- **Key.** `(normalised title, release_year)`, with a fallback fuzzy match
  (token-sort ratio ≥ 90) when the exact key misses.
- **Expected coverage.** 80–90% of titles. The remaining 10–20% miss on
  non-English titles, alternate titles, and year mismatches. That miss rate
  **must be reported** on every run that enables enrichment; it is a
  selection bias (misses concentrate in non-English and older titles).
- **What it unlocks.** A real, if noisy, success-adjacent target:
  `log1p(vote_count)`, `vote_average`, `popularity`. Budget and revenue are
  sparser (often < 40% even on TMDB) and should not be used as a primary
  target without an explicit missingness model.
- **Implementation.** Behind `ENRICHMENT_ENABLED`. API key from
  `TMDB_API_KEY` / `OMDB_API_KEY`. Responses cached under
  `data/external/tmdb_cache/`. The pipeline runs end-to-end with the flag
  off. CI never sets the flag.

## Join 2 — Wikipedia / Wikidata (awards)

- **Key.** Wikidata sitelink on title + year, or a SPARQL query for
  `P345` (IMDb ID) after Join 1.
- **What it unlocks.** A binary `has_major_award` and a count of award
  nominations. Useful as a rare-event classification head, not as a
  continuous success score.
- **Caveat.** Awards are a lagging, prestige-biased signal. Children's
  content and non-English catalogue are systematically under-awarded.

## Join 3 — internal telemetry (the only true success signal)

In a real Netflix setting the join key is the title identifier already in
the warehouse. Required fields:

- starts, unique viewers, completion rate, average watch time
- row / country / device breakdowns
- first-28-day and lifetime windows
- subscriber-acquisition and retention lift where attributed

Without at least starts and completion, "success" remains undefined.

## Feature-flag contract

| flag | default | CI | effect |
|---|---|---|---|
| `ENRICHMENT_ENABLED` | false | must be false | optional TMDB/OMDb join |
| `INSIGHT_LLM_POLISH` | false | must be false | optional rephrase of templated insights |

The platform is complete without either flag. Enabling enrichment adds
columns to the feature store and, if coverage ≥ a configured floor, a
seventh model head against `log1p(vote_count)`. That head is absent when
the flag is off.
