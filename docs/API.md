# API

Base URL: `$API_BASE_URL` (default `http://127.0.0.1:8000`).
Interactive docs: `/docs`.

Proxy fields always carry `is_proxy` and `proxy_note` in the body.

## Health

```bash
curl -s $API_BASE_URL/health
# {"status":"ok"}
```

```bash
curl -s $API_BASE_URL/ready
# {"status":"ready","models":["type","rating",...]}
```

## Meta

```bash
curl -s $API_BASE_URL/meta
```

Returns version, git SHA, feature schema hash, and the promoted pointer per head.

## Predict type (Head A)

```bash
curl -s -X POST $API_BASE_URL/predict/type \
  -H 'content-type: application/json' \
  -d '{"title":"Dick Johnson Is Dead","listed_in":"Documentaries","release_year":2020,"country":"United States"}'
```

Sending `duration` returns **422** naming the banned-feature list.

## Predict rating (Head B)

```bash
curl -s -X POST $API_BASE_URL/predict/rating \
  -H 'content-type: application/json' \
  -d '{"title":"Midnight Mass","listed_in":"TV Dramas, TV Horror","release_year":2021}'
```

Probabilities are in ladder order: Kids → Older Kids → Teens → Mature.

## Predict acquisition (Heads C/D)

```bash
curl -s -X POST $API_BASE_URL/predict/acquisition \
  -H 'content-type: application/json' \
  -d '{"title":"The Irishman","listed_in":"Dramas","release_year":2019,"country":"United States"}'
```

Response includes a point estimate, an 80% residual-quantile interval, the
bucket, and `is_proxy: true`. Sending `date_added` or `year_added` returns **422**.

## Analyze (flagship)

```bash
curl -s -X POST $API_BASE_URL/analyze \
  -H 'content-type: application/json' \
  -d '{"title":"Squid Game","listed_in":"International TV Shows, TV Dramas","release_year":2021,"country":"South Korea"}'
```

Every head plus relevant insights and a templated summary. Acquisition is labelled as a proxy in the payload.

## Insights

```bash
curl -s "$API_BASE_URL/insights?theme=composition&limit=5&min_effect=0.2"
curl -s $API_BASE_URL/insights/0
```

## Stats / forecast / drift

```bash
curl -s $API_BASE_URL/stats/overview
curl -s "$API_BASE_URL/stats/timeseries?metric=titles_added"
curl -s "$API_BASE_URL/stats/segments?by=primary_country"
curl -s "$API_BASE_URL/forecast?horizon=6"
curl -s $API_BASE_URL/drift
curl -s $API_BASE_URL/metrics
```

Validation errors are 422 naming the field. Unexpected errors are 500 with a request ID; the traceback stays server-side.
