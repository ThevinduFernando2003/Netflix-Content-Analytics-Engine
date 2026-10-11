"""Executive dashboard. Talks to the API over HTTP — does not import models."""

from __future__ import annotations

import os

import altair as alt
import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
PAGES = ["Overview", "Catalogue", "Insights", "Analyzer", "Models", "About"]
SEGMENT_FIELDS = {
    "Country": "primary_country",
    "Genre": "primary_genre",
    "Maturity": "maturity_level",
    "Type": "type",
}

st.set_page_config(page_title="Netflix Catalogue Analytics", layout="wide", page_icon="🎬")
st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: #f7f3ee;
        color: #141414;
    }
    h1, h2, h3, p, span, label, li, td, th, caption,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] p,
    [data-testid="stWidgetLabel"] p {
        color: #141414 !important;
    }
    .banner {
        background: #2c1818; color: #f6f1ea !important; border-radius: 12px;
        padding: 0.7rem 1rem; margin: 0.4rem 0 0.8rem; font-size: 0.95rem;
    }
    .banner, .banner * { color: #f6f1ea !important; }
    div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e4d8c8; border-radius: 12px;
        padding: 0.7rem 0.85rem;
    }
    div[data-testid="stMetric"] * { color: #141414 !important; }
    div[data-testid="stAlert"], div[data-testid="stAlert"] * {
        color: #142016 !important;
        background-color: #e7f6ea;
    }
    div[data-testid="stRadio"] label, div[data-testid="stRadio"] p {
        color: #141414 !important;
        white-space: nowrap !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        flex-wrap: nowrap !important;
        gap: 0.6rem;
    }
    [data-testid="stDataFrame"], [data-testid="stDataFrame"] * {
        color: #141414 !important;
    }
    input, textarea { color: #141414 !important; background: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300, show_spinner=False)
def api_get(path: str, params_key: str = "") -> tuple[dict | list | None, str | None]:
    params = dict(part.split("=", 1) for part in params_key.split("&") if part)
    try:
        response = httpx.get(f"{API}{path}", params=params or None, timeout=8.0)
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:
        return None, str(exc)


def api_post(path: str, payload: dict) -> tuple[dict | None, str | None]:
    try:
        response = httpx.post(f"{API}{path}", json=payload, timeout=20.0)
        if response.status_code == 422:
            detail = response.json().get("detail", response.text)
            return None, detail if isinstance(detail, str) else str(detail)
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:
        return None, str(exc)


def fail(message: str | None) -> None:
    st.error(f"The API at `{API}` did not respond. {message or ''} Start it, then refresh.")


def banner() -> None:
    st.markdown(
        '<div class="banner">Catalogue composition and acquisition lag only. '
        "This listing has no views, watch time, or revenue, so every score is a proxy.</div>",
        unsafe_allow_html=True,
    )


def navigate() -> str:
    if "page" not in st.session_state:
        st.session_state.page = "Overview"
    choice = st.radio(
        "Go to",
        PAGES,
        index=PAGES.index(st.session_state.page),
        horizontal=True,
        key="nav_radio",
    )
    st.session_state.page = choice
    return choice


def _readable_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(background="#ffffff")
        .configure_view(strokeWidth=0, fill="#ffffff")
        .configure_axis(
            labelColor="#141414",
            titleColor="#141414",
            labelFontSize=13,
            titleFontSize=15,
            gridColor="#e6dfd6",
            domainColor="#141414",
            tickColor="#141414",
        )
        .configure_title(color="#141414", fontSize=16)
    )


def bar_chart(frame: pd.DataFrame, label: str) -> None:
    if frame.empty:
        st.info("Nothing matches that search.")
        return
    named = frame.rename(columns={"key": "name"})
    chart = (
        alt.Chart(named)
        .mark_bar(color="#b11226", cornerRadiusEnd=4)
        .encode(
            x=alt.X("n:Q", title="Number of titles"),
            y=alt.Y("name:N", sort="-x", title=label),
            tooltip=[
                alt.Tooltip("name:N", title=label),
                alt.Tooltip("n:Q", title="Number of titles", format=","),
            ],
        )
        .properties(height=max(240, 32 * len(named)), background="#ffffff")
    )
    st.altair_chart(_readable_chart(chart), use_container_width=True)


def year_chart(points: list[dict]) -> None:
    frame = pd.DataFrame(points).rename(columns={"t": "Year", "v": "Titles added"})
    chart = (
        alt.Chart(frame)
        .mark_line(color="#b11226", point=alt.OverlayMarkDef(color="#141414", size=50))
        .encode(
            x=alt.X("Year:O", title="Year added to Netflix"),
            y=alt.Y("Titles added:Q", title="Titles added that year"),
            tooltip=["Year", "Titles added"],
        )
        .properties(height=280, background="#ffffff")
    )
    st.altair_chart(_readable_chart(chart), use_container_width=True)


def page_overview() -> None:
    st.title("Netflix catalogue")
    banner()
    overview, error = api_get("/stats/overview")
    if error or not isinstance(overview, dict):
        fail(error)
        return
    lag = overview.get("median_acquisition_lag")
    lag_txt = f"{lag:.1f}y" if isinstance(lag, int | float) else "—"
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Titles", f"{overview['n_titles']:,}")
    c2.metric("Movies", f"{overview['n_movies']:,}")
    c3.metric("TV shows", f"{overview['n_tv']:,}")
    c4.metric("Countries", f"{overview['n_countries']:,}")
    c5.metric("Median lag (proxy)", lag_txt)
    c6.metric("Added in last 3y", f"{overview.get('share_added_last_3y', 0):.0%}")

    series, _ = api_get("/stats/timeseries", "metric=titles_added")
    if isinstance(series, dict) and series.get("points"):
        st.subheader("Titles added by year")
        year_chart(series["points"])

    insights, _ = api_get("/insights", "limit=5")
    st.subheader("Strongest findings")
    if isinstance(insights, dict):
        st.caption(
            f"{insights.get('n_candidates')} tested · {insights.get('n_survivors')} survived "
            "BH-FDR and the effect-size floor."
        )
        for item in insights.get("items") or []:
            st.markdown(f"**{item.get('text')}**")
            st.caption(
                f"{item.get('effect_name')} {item.get('effect')} · q={item.get('q')} · n={item.get('n')}"
            )


def page_catalogue() -> None:
    st.title("Catalogue bars")
    st.caption("Each bar is a group of titles in the listing. Taller is not more popular.")
    choice = st.radio("Show bars for", list(SEGMENT_FIELDS), horizontal=True)
    how_many = st.radio("How many bars", ["Top 8", "Top 15", "All"], horizontal=True, index=0)
    query = st.text_input("Jump to a name", placeholder="Type a country or genre")
    payload, error = api_get("/stats/segments", f"by={SEGMENT_FIELDS[choice]}")
    if error or not isinstance(payload, dict):
        fail(error)
        return
    frame = pd.DataFrame(payload.get("counts") or [])
    if frame.empty:
        st.info("No rows for that breakdown.")
        return
    if query:
        frame = frame[frame["key"].str.contains(query, case=False, na=False)]
    if how_many == "Top 8":
        frame = frame.head(8)
    elif how_many == "Top 15":
        frame = frame.head(15)
    left, right = st.columns([3, 1])
    with left:
        bar_chart(frame, choice)
    with right:
        st.metric("Bars shown", f"{len(frame)}")
        st.metric("Titles in view", f"{int(frame['n'].sum()):,}" if len(frame) else "0")
        picked = st.selectbox("Read one bar", frame["key"].tolist()) if len(frame) else None
        if picked:
            count = int(frame.loc[frame["key"] == picked, "n"].iloc[0])
            st.markdown(f"**{picked}**")
            st.markdown(f"{count:,} titles")


def page_insights() -> None:
    st.title("Findings")
    insights, error = api_get("/insights", "limit=27")
    if error or not isinstance(insights, dict):
        fail(error)
        return
    st.caption(
        f"{insights.get('n_candidates')} candidates tested, {insights.get('n_survivors')} survived. "
        "Each row has an effect size, an adjusted p-value, and n."
    )
    items = insights.get("items") or []
    themes = sorted({item.get("theme") for item in items if item.get("theme")})
    theme = st.radio("Theme", ["All", *themes], horizontal=True)
    rows = items if theme == "All" else [item for item in items if item.get("theme") == theme]
    table = pd.DataFrame(
        [
            {
                "Finding": item.get("text"),
                "Effect": item.get("effect"),
                "Measure": item.get("effect_name"),
                "q": item.get("q"),
                "n": item.get("n"),
            }
            for item in rows
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True, height=360)
    if rows:
        pick = st.selectbox("Open one finding", [item.get("text") for item in rows])
        chosen = next(item for item in rows if item.get("text") == pick)
        st.json(chosen)


def page_analyzer() -> None:
    st.title("Try a title")
    st.caption("Acquisition lag is a proxy. The interval is an 80% residual band.")
    examples = {
        "Ganglands": {
            "title": "Ganglands",
            "listed_in": "Crime TV Shows, International TV Shows",
            "release_year": 2021,
            "country": "France",
            "rating": "TV-MA",
        },
        "Squid Game": {
            "title": "Squid Game",
            "listed_in": "International TV Shows, TV Dramas, TV Thrillers",
            "release_year": 2021,
            "country": "South Korea",
            "rating": "TV-MA",
        },
        "Dick Johnson Is Dead": {
            "title": "Dick Johnson Is Dead",
            "listed_in": "Documentaries",
            "release_year": 2020,
            "country": "United States",
            "rating": "PG-13",
        },
        "Kota Factory": {
            "title": "Kota Factory",
            "listed_in": "International TV Shows, Romantic TV Shows, TV Comedies",
            "release_year": 2021,
            "country": "India",
            "rating": "TV-MA",
        },
    }
    choice = st.selectbox("Example", list(examples))
    base = examples[choice]
    with st.form(f"analyze_{choice}"):
        title = st.text_input("Title", base["title"])
        listed = st.text_input("Genres", base["listed_in"])
        year = st.number_input("Release year", 1920, 2026, int(base["release_year"]))
        country = st.text_input("Country", base["country"])
        rating = st.text_input("Rating", base["rating"])
        submitted = st.form_submit_button("Analyze", type="primary")
    if not submitted:
        return
    with st.spinner("Scoring this title…"):
        data, error = api_post(
            "/analyze",
            {
                "title": title,
                "listed_in": listed,
                "release_year": int(year),
                "country": country,
                "rating": rating,
            },
        )
    if error or not data:
        fail(error)
        return
    st.success(data.get("summary") or "Scored.")
    acquired = data.get("acquisition_priority") or {}
    segment = data.get("segment") or {}
    kind = data.get("type") or {}
    rated = data.get("rating") or {}
    a, b, c, d = st.columns(4)
    a.metric("Type", kind.get("label") or "—")
    b.metric("Maturity band", rated.get("label") or "—")
    c.metric("Lag (proxy, years)", _fmt(acquired.get("years_to_platform")))
    d.metric("Bucket", acquired.get("bucket") or "—")
    st.caption(
        f"Interval {_fmt(acquired.get('interval_lo'))} to {_fmt(acquired.get('interval_hi'))} years. "
        f"Segment: {segment.get('name') or 'unassigned'}. "
        f"is_proxy={acquired.get('is_proxy')}."
    )
    with st.expander("Full response"):
        st.json(data)


def page_models() -> None:
    st.title("Model scoreboard")
    models, error = api_get("/models")
    if error or not isinstance(models, list):
        fail(error)
        return
    rows = []
    for rec in models:
        metrics = rec.get("metrics") or {}
        primary = next(iter(metrics.values()), None) if metrics else None
        rows.append(
            {
                "Head": rec.get("head"),
                "Candidate": rec.get("candidate"),
                "Baseline": rec.get("is_baseline"),
                "Promoted": rec.get("promoted"),
                "First metric": primary,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=420)
    drift, _ = api_get("/drift")
    perf = (drift or {}).get("performance_drift") if isinstance(drift, dict) else None
    if isinstance(perf, dict):
        st.subheader("Later years vs a random split")
        left, right = st.columns(2)
        type_row = perf.get("type") or {}
        lag_row = perf.get("acquisition_reg") or {}
        left.metric(
            "Type ROC-AUC drop",
            _fmt(type_row.get("degradation")),
            delta=f"{_fmt(type_row.get('random_split'))} → {_fmt(type_row.get('temporal'))}",
            delta_color="off",
        )
        right.metric(
            "Acquisition MAE increase (years)",
            _fmt(lag_row.get("degradation")),
            delta=f"{_fmt(lag_row.get('random_split'))} → {_fmt(lag_row.get('temporal'))}",
            delta_color="off",
        )


def page_about() -> None:
    st.title("What this can and cannot say")
    banner()
    st.markdown(
        """
**Success is not measurable from this file.** There is no view count, watch time,
completion, revenue, subscriber impact, review score, award, budget, or renewal flag.

| Proxy | What it actually is |
|---|---|
| Acquisition lag | Year added minus release year. Platform timing, not demand. |
| Likely original | Added in or before the release year. A heuristic, with false positives. |
| Catalogue prominence | A weighted metadata-richness index. Not an editorial outcome. |
| Portfolio fit | Distance to a cluster centroid. Not quality. |
"""
    )


def _fmt(value: object) -> str:
    if isinstance(value, int | float):
        return f"{value:.2f}"
    return "—"


page = navigate()
{
    "Overview": page_overview,
    "Catalogue": page_catalogue,
    "Insights": page_insights,
    "Analyzer": page_analyzer,
    "Models": page_models,
    "About": page_about,
}[page]()
