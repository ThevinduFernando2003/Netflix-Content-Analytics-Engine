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
    header[data-testid="stHeader"] { background: #141414; height: 0; }
    [data-testid="stToolbar"] { display: none; }
    section.main > div.block-container { padding-top: 4.6rem; }
    .topbar {
        position: fixed; top: 0; left: 0; right: 0; z-index: 1000000;
        display: flex; align-items: center; justify-content: space-between; gap: 1.25rem;
        background: #141414; padding: 0.75rem 1.4rem;
    }
    .topbar, .topbar * { color: #f5f5f5 !important; }
    .topbar .brand {
        font-size: 1.2rem; font-weight: 700; white-space: nowrap; letter-spacing: -0.02em;
    }
    .topbar nav { display: flex; flex-wrap: nowrap; gap: 0.25rem; overflow-x: auto; }
    .topbar a {
        text-decoration: none; white-space: nowrap; padding: 0.4rem 0.75rem; border-radius: 8px;
    }
    .topbar a.on { background: #e50914; color: #ffffff !important; }
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


PLAIN = {
    "title_length": "title length, counted in characters",
    "years_to_platform": "years from the release year until Netflix added it",
    "catalogue_prominence": "how much detail the listing includes",
    "is_likely_original": "whether it was added in the release year or earlier",
    "is_movie": "whether it is a movie",
    "n_genres": "how many genres are listed",
    "n_countries": "how many countries are listed",
    "n_directors": "how many directors are listed",
    "primary_country": "country",
    "primary_genre": "genre",
    "type": "movie or TV show",
    "maturity_level": "age-rating band",
    "acquisition_bucket": "how quickly it was added",
    "country": "country",
    "genre": "genre",
}

THEME_LABELS = {
    "composition": "What is listed",
    "genre": "By genre",
    "geography": "By country",
    "concentration": "How uneven",
    "acquisition": "How fast titles were added",
    "gaps": "Missing combinations",
}


def _num(value: object, digits: int = 1) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.{digits}f}"
    return "—"


def _gap(effect: object) -> str:
    try:
        size = abs(float(effect))
    except (TypeError, ValueError):
        return "a real difference"
    if size >= 0.43:
        return "a large difference"
    if size >= 0.28:
        return "a clear difference"
    return "a modest difference"


def plain_finding(item: dict) -> str:
    """One sentence a non-specialist can read. Numbers stay; field names do not."""
    kind = item.get("kind")
    values = item.get("values") or {}
    n = item.get("n")
    n_txt = f"{int(n):,}" if isinstance(n, int | float) else "the titles in this comparison"
    effect = item.get("effect")
    if kind == "segment_difference":
        what = PLAIN.get(str(values.get("metric")), "this measure")
        left = values.get("segment_a") or "one group"
        right = values.get("segment_b") or "another group"
        direction = values.get("direction") or "different"
        return (
            f"{left} and {right} are not the same on {what}. "
            f"{left} is {direction}: about {_num(values.get('mean_a'))}, "
            f"against about {_num(values.get('mean_b'))} for {right}. "
            f"That is {_gap(effect)}. The score is {_num(effect, 2)} on a scale from 0 "
            f"(no difference) to 1 (the groups do not overlap). "
            f"This uses {n_txt} titles. It describes the listing, not what people watched."
        )
    if kind == "association":
        left = PLAIN.get(str(values.get("var_a")), str(values.get("var_a") or "one label"))
        right = PLAIN.get(str(values.get("var_b")), str(values.get("var_b") or "another label"))
        return (
            f"{left.capitalize()} and {right} tend to show up together. "
            f"The link scores {_num(effect, 2)} on a scale from 0 (no link) to 1 "
            f"(they always go together). This uses {n_txt} titles. "
            "It does not mean one causes the other."
        )
    if kind == "concentration":
        dim = PLAIN.get(str(values.get("dimension")), "group")
        share = values.get("top_share")
        share_txt = f"{float(share):.0%}" if isinstance(share, int | float) else "most"
        return (
            f"The catalogue is uneven by {dim}. The five largest groups hold {share_txt} "
            f"of the titles, and many other groups are tiny. This uses {n_txt} titles."
        )
    if kind == "composition_shift":
        share_a = values.get("start_share")
        share_b = values.get("end_share")
        a_txt = f"{float(share_a):.1%}" if isinstance(share_a, int | float) else "—"
        b_txt = f"{float(share_b):.1%}" if isinstance(share_b, int | float) else "—"
        return (
            f"The share of {values.get('segment')} {values.get('direction') or 'changed'} "
            f"from {a_txt} to {b_txt} between {values.get('start_period')} and "
            f"{values.get('end_period')}. That is about {_num(values.get('delta_pp'))} "
            f"percentage points across {n_txt} titles. This is what was added, not what was watched."
        )
    return (
        "This comparison passed the checks, but it does not have a plain-language sentence yet. "
        f"It uses {n_txt} titles."
    )


def fail(message: str | None) -> None:
    st.error(f"The API at `{API}` did not respond. {message or ''} Start it, then refresh.")


def banner() -> None:
    st.markdown(
        '<div class="banner">Catalogue composition and acquisition lag only. '
        "This listing has no views, watch time, or revenue, so every score is a proxy.</div>",
        unsafe_allow_html=True,
    )


def navigate() -> str:
    raw = st.query_params.get("page", "Overview")
    current = raw if isinstance(raw, str) and raw in PAGES else "Overview"
    links = "".join(
        f'<a class="{"on" if name == current else ""}" href="?page={name}">{name}</a>'
        for name in PAGES
    )
    st.markdown(
        f'<div class="topbar"><span class="brand">Netflix Catalogue</span><nav>{links}</nav></div>',
        unsafe_allow_html=True,
    )
    return current


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
    lag_txt = f"{lag:.1f} years" if isinstance(lag, int | float) else "—"
    top = st.columns(3)
    top[0].metric("Titles", f"{overview['n_titles']:,}")
    top[1].metric("Movies", f"{overview['n_movies']:,}")
    top[2].metric("TV shows", f"{overview['n_tv']:,}")
    bottom = st.columns(3)
    bottom[0].metric("Countries", f"{overview['n_countries']:,}")
    bottom[1].metric("Typical wait after release", lag_txt)
    bottom[2].metric("Added in the last 3 years", f"{overview.get('share_added_last_3y', 0):.0%}")

    series, _ = api_get("/stats/timeseries", "metric=titles_added")
    if isinstance(series, dict) and series.get("points"):
        st.subheader("Titles added by year")
        year_chart(series["points"])

    insights, _ = api_get("/insights", "limit=5")
    st.subheader("What stands out")
    if isinstance(insights, dict):
        st.caption(
            f"The engine compared {insights.get('n_candidates')} patterns and kept "
            f"{insights.get('n_survivors')}. A pattern is kept only when the gap is large "
            "and it still looks real after many comparisons. These are not viewing figures."
        )
        for item in insights.get("items") or []:
            st.markdown(plain_finding(item))


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
    st.title("What the catalogue shows")
    insights, error = api_get("/insights", "limit=27")
    if error or not isinstance(insights, dict):
        fail(error)
        return
    st.markdown(
        "Each note below is a difference inside the listing. "
        f"Out of {insights.get('n_candidates')} comparisons, "
        f"{insights.get('n_survivors')} were large enough to keep. "
        "None of them measure views, watch time, or revenue."
    )
    items = insights.get("items") or []
    themes = [theme for theme in THEME_LABELS if any(item.get("theme") == theme for item in items)]
    labels = ["All", *[THEME_LABELS[theme] for theme in themes]]
    picked = st.radio("Show", labels, horizontal=True)
    if picked == "All":
        rows = items
    else:
        theme = next(key for key, label in THEME_LABELS.items() if label == picked)
        rows = [item for item in items if item.get("theme") == theme]
    for index, item in enumerate(rows, start=1):
        st.markdown(f"**{index}. {plain_finding(item)}**")


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
    left = st.columns(2)
    left[0].metric("Movie or TV show", kind.get("label") or "—")
    left[1].metric("Age-rating band", rated.get("label") or "—")
    right = st.columns(2)
    right[0].metric("Years until Netflix added it", _fmt(acquired.get("years_to_platform")))
    right[1].metric("How fast it was added", acquired.get("bucket") or "—")
    st.markdown(
        f"A typical range for that wait is {_fmt(acquired.get('interval_lo'))} to "
        f"{_fmt(acquired.get('interval_hi'))} years. "
        f"The closest catalogue group is {segment.get('name') or 'not assigned'}. "
        "The wait is a timing proxy. It is not a popularity score."
    )
    with st.expander("Full response"):
        st.json(data)


HEAD_HELP = {
    "type": (
        "Movie or TV show",
        "Reads the title and the other listing fields and guesses whether the row is a movie or a TV show.",
        "roc_auc",
        "Share of pairs it ranks correctly. 0.50 is a coin flip. 1.00 is perfect.",
    ),
    "rating": (
        "Age-rating band",
        "Guesses the maturity band, such as kids, teen, or mature, from the listing.",
        "quadratic_weighted_kappa",
        "How close the guess is to the real band. 1.00 is an exact match. 0 is no better than chance.",
    ),
    "acquisition_reg": (
        "Years until added",
        "Guesses how many years passed between the release year and the year Netflix added the title.",
        "mae",
        "Average miss, in years. Lower is closer.",
    ),
    "acquisition_clf": (
        "Speed bucket",
        "Puts that wait into a bucket: day-one, fast, recent, or back catalogue.",
        "quadratic_weighted_kappa",
        "How close the bucket is to the real one. 1.00 is exact.",
    ),
    "cluster": (
        "Similar listings",
        "Groups titles that look alike in the catalogue. The group is a neighbourhood, not a quality mark.",
        "silhouette",
        "How cleanly the groups separate. Higher means tighter groups.",
    ),
    "forecast": (
        "Titles added per month",
        "Guesses how many titles arrive in a future month. On this file the simple seasonal guess won.",
        "mae",
        "Average miss, in titles per month. Lower is closer.",
    ),
}


def page_models() -> None:
    st.title("What the models are")
    st.markdown(
        "A model here is a repeatable guess about a field that is already in the listing. "
        "The scoreboard compares each guess with a simple baseline, such as always picking "
        "the most common answer. **Served** means that guess was good enough to use in the app. "
        "None of these scores are views, ratings from audiences, or revenue."
    )
    models, error = api_get("/models")
    if error or not isinstance(models, list):
        fail(error)
        return
    for head, (title, sentence, metric_key, metric_help) in HEAD_HELP.items():
        rows = [rec for rec in models if rec.get("head") == head]
        if not rows:
            continue
        st.subheader(title)
        st.markdown(sentence)
        shown: set[tuple[str, bool, bool]] = set()
        for rec in rows:
            key = (
                str(rec.get("candidate")),
                bool(rec.get("is_baseline")),
                bool(rec.get("promoted")),
            )
            if key in shown:
                continue
            shown.add(key)
            metrics = rec.get("metrics") or {}
            score = metrics.get(metric_key)
            role = "Served in the app" if rec.get("promoted") else "Tried, not served"
            if rec.get("is_baseline"):
                role = "Simple baseline"
            st.markdown(
                f"**{rec.get('candidate')}** — {role}. "
                f"Score {_fmt(score) if score is not None else 'not recorded'}. {metric_help}"
            )
    drift, _ = api_get("/drift")
    perf = (drift or {}).get("performance_drift") if isinstance(drift, dict) else None
    if isinstance(perf, dict):
        st.subheader("Do the guesses still work on later years?")
        st.markdown(
            "One check trains on a random mix of years. The other trains on earlier years "
            "and scores later years. A bigger miss on later years means the pattern shifted."
        )
        left, right = st.columns(2)
        type_row = perf.get("type") or {}
        lag_row = perf.get("acquisition_reg") or {}
        left.metric(
            "Movie-or-TV score change",
            _fmt(type_row.get("degradation")),
            delta=f"{_fmt(type_row.get('random_split'))} to {_fmt(type_row.get('temporal'))}",
            delta_color="off",
        )
        right.metric(
            "Wait-time miss, extra years",
            _fmt(lag_row.get("degradation")),
            delta=f"{_fmt(lag_row.get('random_split'))} to {_fmt(lag_row.get('temporal'))}",
            delta_color="off",
        )


def page_about() -> None:
    st.title("What this can and cannot say")
    banner()
    st.markdown(
        "**This file cannot tell you what succeeded.** It has no view count, watch time, "
        "completion, revenue, subscriber impact, review score, award, budget, or renewal flag. "
        "Every score below is a stand-in built from the listing itself."
    )
    notes = [
        (
            "Years until added",
            "The year Netflix added the title, minus the release year. It says when the title arrived, not whether people wanted it.",
        ),
        (
            "Looks like an original",
            "The title was added in the release year or earlier. That is a rough rule and it is sometimes wrong.",
        ),
        (
            "How much detail the listing has",
            "A weighted count of filled-in fields, such as director and country. A fuller listing is not a better show.",
        ),
        (
            "How close it sits to similar titles",
            "Distance from a group of lookalike listings. Close means similar metadata, not higher quality.",
        ),
    ]
    for title, body in notes:
        st.markdown(f"**{title}**")
        st.markdown(body)


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
