"""Executive dashboard. Talks to the API over HTTP — does not import models."""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
CLUSTER_COLORS = {
    0: "#8B0000",
    1: "#0B3D2E",
    2: "#1F4E79",
    3: "#C4A35A",
    4: "#5C4D7A",
    5: "#B85C38",
    6: "#2F3E46",
    7: "#6B705C",
}


def api_get(path: str, params: dict | None = None) -> tuple[dict | list | None, str | None]:
    try:
        r = httpx.get(f"{API}{path}", params=params, timeout=8.0)
        r.raise_for_status()
        return r.json(), None
    except Exception as exc:
        return None, str(exc)


def api_post(path: str, payload: dict) -> tuple[dict | None, str | None]:
    try:
        r = httpx.post(f"{API}{path}", json=payload, timeout=12.0)
        if r.status_code == 422:
            return None, r.json().get("detail", r.text)
        r.raise_for_status()
        return r.json(), None
    except Exception as exc:
        return None, str(exc)


def err(msg: str) -> None:
    st.error(
        f"The API at `{API}` did not respond cleanly. {msg} "
        "Start it with `make api` or `docker compose up`."
    )


st.set_page_config(page_title="Netflix Catalogue Analytics", layout="wide", page_icon="🎬")
st.markdown(
    """
    <style>
    .stApp { background: #f7f3ee; }
    h1, h2, h3 { color: #3b1f1f; font-family: Georgia, serif; }
    div[data-testid="stMetric"] { background: #fff; padding: 0.6rem 0.8rem; border: 1px solid #e6ddd0; }
    </style>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive overview",
        "Insight explorer",
        "Title analyzer",
        "Catalogue explorer",
        "Segments",
        "Model performance",
        "About & limitations",
    ],
)

if page == "Executive overview":
    st.title("Netflix catalogue analytics")
    st.warning(
        "This platform analyses **catalogue composition and acquisition behaviour**, "
        "not audience success. The listing has no views, watch time, revenue, ratings "
        "or subscriber data. Every acquisition number is a **proxy**."
    )
    overview, e = api_get("/stats/overview")
    if e or not overview:
        err(e or "empty overview")
    else:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Titles", f"{overview['n_titles']:,}")
        c2.metric("Movies", f"{overview['n_movies']:,}")
        c3.metric("TV shows", f"{overview['n_tv']:,}")
        c4.metric("Countries", f"{overview['n_countries']:,}")
        c5.metric("Median acquisition lag (proxy)", f"{overview.get('median_acquisition_lag')}")
        c6.metric("Share added last 3y", f"{overview.get('share_added_last_3y', 0):.0%}")
        ts, _ = api_get("/stats/timeseries", {"metric": "titles_added"})
        if ts and ts.get("points"):
            chart = pd.DataFrame(ts["points"]).rename(columns={"t": "year", "v": "titles added"})
            st.subheader("Titles added by year")
            st.line_chart(chart.set_index("year"))
        ins, _ = api_get("/insights", {"limit": 5})
        st.subheader("Top automated insights")
        if ins:
            st.caption(
                f"{ins.get('n_candidates')} candidates tested; {ins.get('n_survivors')} survived BH-FDR."
            )
            for item in ins.get("items") or []:
                st.markdown(f"- {item.get('text')}")
                st.caption(
                    f"effect={item.get('effect')} ({item.get('effect_name')}) "
                    f"q={item.get('q')} n={item.get('n')}"
                )

elif page == "Insight explorer":
    st.title("Insight explorer")
    ins, e = api_get("/insights", {"limit": 100})
    if e or not ins:
        err(e or "no insights")
    else:
        st.info(
            f"Methodology: {ins.get('n_candidates')} candidates → {ins.get('n_survivors')} survivors "
            "after Benjamini–Hochberg FDR and an effect-size floor. A finding without its "
            "effect size, adjusted p and n is a claim, not a finding."
        )
        themes = sorted({i.get("theme") for i in ins.get("items") or []} - {None})
        theme = st.selectbox("Theme", ["all", *themes])
        min_e = st.slider("Minimum |effect|", 0.0, 1.0, 0.0)
        items = ins.get("items") or []
        if theme != "all":
            items = [i for i in items if i.get("theme") == theme]
        items = [i for i in items if abs(float(i.get("effect") or 0)) >= min_e]
        for i, item in enumerate(items):
            with st.expander(item.get("text", f"finding {i}")):
                st.json(item)

elif page == "Title analyzer":
    st.title("Title analyzer")
    st.caption("All acquisition outputs are proxies. Intervals are residual-quantile 80% bands.")
    examples = {
        "Dick Johnson Is Dead": {
            "title": "Dick Johnson Is Dead",
            "listed_in": "Documentaries",
            "release_year": 2020,
            "country": "United States",
            "rating": "PG-13",
        },
        "Ganglands": {
            "title": "Ganglands",
            "type": "TV Show",
            "listed_in": "Crime TV Shows, International TV Shows",
            "release_year": 2021,
            "country": "France",
            "rating": "TV-MA",
        },
        "Kota Factory": {
            "title": "Kota Factory",
            "type": "TV Show",
            "listed_in": "International TV Shows, Romantic TV Shows, TV Comedies",
            "release_year": 2021,
            "country": "India",
            "rating": "TV-MA",
        },
        "Odd: 1925 silent from nowhere": {
            "title": "Odd: 1925 silent from nowhere",
            "listed_in": "Classic Movies",
            "release_year": 1925,
            "country": "",
            "rating": "NR",
        },
    }
    choice = st.selectbox("Prefill", list(examples))
    base = examples[choice]
    title = st.text_input("Title", base["title"])
    listed = st.text_input("Genres (listed_in)", base["listed_in"])
    year = st.number_input("Release year", 1900, 2026, int(base["release_year"]))
    country = st.text_input("Country", base.get("country", ""))
    rating = st.text_input("Rating", base.get("rating", ""))
    typ = st.selectbox("Type (optional, unused by Head A features)", [None, "Movie", "TV Show"])
    if st.button("Analyze"):
        payload = {
            "title": title,
            "listed_in": listed,
            "release_year": int(year),
            "country": country,
            "rating": rating,
            "type": typ,
        }
        data, e = api_post("/analyze", payload)
        if e or not data:
            err(e or "analyze failed")
        else:
            st.write(data.get("summary"))
            st.json(data)

elif page == "Catalogue explorer":
    st.title("Catalogue explorer")
    by = st.selectbox("Segment", ["primary_country", "primary_genre", "maturity_level", "type"])
    segs, e = api_get("/stats/segments", {"by": by})
    if e or not segs:
        err(e or "no segments")
    else:
        df = pd.DataFrame(segs["counts"])
        st.bar_chart(df.set_index("key")["n"])
        st.caption(
            "Counts respond to the selected segment. This is catalogue composition, not demand."
        )

elif page == "Segments":
    st.title("Catalogue segments")
    st.warning(
        "Projection distances (when shown) are **not interpretable** as similarity in the original "
        "metadata space. Cluster names come from genre lift, not from an editorial taxonomy."
    )
    meta, e = api_get("/models/cluster/card")
    drift, _ = api_get("/drift")
    if e:
        err(e)
    else:
        st.write("Promoted clustering incumbent:")
        st.json((meta or {}).get("incumbent"))

elif page == "Model performance":
    st.title("Model performance")
    models, e = api_get("/models")
    if e or models is None:
        err(e or "no models")
    else:
        st.dataframe(pd.DataFrame(models))
    drift, _ = api_get("/drift")
    if drift:
        st.subheader("Temporal drift")
        st.json(drift.get("performance_drift") or drift)

elif page == "About & limitations":
    st.title("About & limitations")
    st.markdown(
        """
This platform analyses **catalogue composition and platform acquisition behaviour**.
It does **not** measure audience success.

The shipped listing contains no views, watch time, completion, revenue,
subscriber impact, IMDb/RT score, awards, budget or renewal status.
**Success is not measurable from this file.**

### Proxies (labelled `is_proxy: true` in every API response)

- **Acquisition priority** (`years_to_platform`) — add-year minus release-year.
  Strongest genuine signal of platform intent. Not demand.
- **`is_likely_original`** — heuristic: added in or before release year.
  Real false positives and negatives.
- **`catalogue_prominence`** — constructed metadata-richness index. Weights in `config/features.yaml`.
- **`portfolio_fit`** — typicality vs the clustering head. Gap/duplicate story, not quality.

### What would unlock a real success target

TMDB/OMDb ratings and vote counts (expected 10–20% miss rate), Wikidata awards,
and internal telemetry. See `docs/DATA_ROADMAP.md`.
"""
    )
