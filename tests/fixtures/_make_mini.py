"""One-shot generator for tests/fixtures/mini_netflix.csv."""

from __future__ import annotations

import csv
from pathlib import Path

COLS = [
    "show_id",
    "type",
    "title",
    "director",
    "country",
    "date_added",
    "release_year",
    "rating",
    "duration",
    "listed_in",
]


def main() -> None:
    rows: list[dict[str, object]] = []

    def add(**kw: object) -> None:
        rows.append({c: kw.get(c, "") for c in COLS})

    for i in range(1, 21):
        add(
            show_id=f"s{i}",
            type="Movie",
            title=f"US Doc {i}",
            director="Alex Rivera",
            country="United States",
            date_added=f"3/{(i % 28) + 1}/2016",
            release_year=2015,
            rating="PG-13",
            duration=f"{80 + i} min",
            listed_in="Documentaries",
        )
    for i in range(21, 41):
        add(
            show_id=f"s{i}",
            type="Movie",
            title=f"India Drama {i}",
            director="Rajiv Mehta",
            country="India",
            date_added=f"6/{(i % 28) + 1}/2017",
            release_year=2014,
            rating="TV-14",
            duration=f"{110 + i % 20} min",
            listed_in="Dramas, International Movies",
        )
    for i in range(41, 61):
        add(
            show_id=f"s{i}",
            type="TV Show",
            title=f"Kids Club {i}",
            director="Not Given",
            country="United States, Canada",
            date_added=f"9/{(i % 28) + 1}/2018",
            release_year=2018,
            rating="TV-Y",
            duration="1 Season",
            listed_in="Kids' TV, TV Comedies",
        )
    for i, rat in enumerate(["G", "G", "TV-G", "TV-G"], start=61):
        add(
            show_id=f"s{i}",
            type="Movie",
            title=f"Kids Extra {i}",
            director="Sam Lee",
            country="United Kingdom",
            date_added=f"1/{i - 60}/2019",
            release_year=2018,
            rating=rat,
            duration="90 min",
            listed_in="Children and Family Movies",
        )
    for i, rat in enumerate(["TV-Y7", "TV-Y7", "PG", "TV-PG", "TV-Y7-FV"], start=65):
        add(
            show_id=f"s{i}",
            type="Movie",
            title=f"Older Kids {i}",
            director="Pat Kim",
            country="Canada",
            date_added=f"2/{i - 64}/2019",
            release_year=2017,
            rating=rat,
            duration="95 min",
            listed_in="Children and Family Movies, Comedies",
        )
    for i in range(70, 75):
        add(
            show_id=f"s{i}",
            type="TV Show",
            title=f"Teen Show {i}",
            director="Not Given",
            country="United Kingdom",
            date_added=f"4/{i - 69}/2020",
            release_year=2019,
            rating="TV-14",
            duration="2 Seasons",
            listed_in="British TV Shows, TV Dramas",
        )
    for i in range(75, 85):
        add(
            show_id=f"s{i}",
            type="Movie",
            title=f"Mature Film {i}",
            director="Jordan Blake",
            country="United States, France",
            date_added=f"5/{i - 74}/2020",
            release_year=2016,
            rating="TV-MA" if i % 2 == 0 else "R",
            duration=f"{100 + i % 15} min",
            listed_in="Dramas, Thrillers",
        )
    add(
        show_id="s200",
        type="Movie",
        title="Contaminated Rating",
        director="Ada Lovelace",
        country="United States",
        date_added="7/1/2021",
        release_year=2020,
        rating="74 min",
        duration="",
        listed_in="Documentaries",
    )
    add(
        show_id="s201",
        type="Movie",
        title="Missing Rating Row",
        director="Not Given",
        country="Not Given",
        date_added="8/1/2021",
        release_year=2019,
        rating="",
        duration="88 min",
        listed_in="Dramas",
    )
    add(
        show_id="s202",
        type="Movie",
        title="NR Picture",
        director="Chris Nolan",
        country="United States",
        date_added="8/2/2015",
        release_year=2012,
        rating="NR",
        duration="120 min",
        listed_in="Dramas, Independent Movies",
    )
    add(
        show_id="s203",
        type="Movie",
        title="Early Bird Premiere",
        director="Greta Gerwig",
        country="United States",
        date_added="1/1/2018",
        release_year=2019,
        rating="PG-13",
        duration="105 min",
        listed_in="Comedies, Dramas",
    )
    add(
        show_id="s204",
        type="Movie",
        title="Amelie, the extra comma",
        director="Jean-Pierre Jeunet",
        country="France, Germany",
        date_added="3/15/2016",
        release_year=2001,
        rating="R",
        duration="122 min",
        listed_in="Comedies, Dramas, Independent Movies, International Movies, Romantic Movies",
    )
    add(
        show_id="s205",
        type="Movie",
        title="Cafes and Pinatas: A Year",
        director="Sofia Reyes",
        country="Mexico, Spain",
        date_added="11/11/2014",
        release_year=2010,
        rating="TV-MA",
        duration="99 min",
        listed_in="Documentaries, International Movies",
    )
    add(
        show_id="s206",
        type="Movie",
        title="Cafés & Piñatas",
        director="Sofía Reyes",
        country="Mexico",
        date_added="12/1/2014",
        release_year=2011,
        rating="TV-MA",
        duration="91 min",
        listed_in="Documentaries",
    )
    for i in range(86, 150):
        yr_add = 2014 + (i % 8)
        rel = yr_add - (i % 7)
        typ = "TV Show" if i % 5 == 0 else "Movie"
        rat = ["TV-MA", "TV-14", "TV-PG", "PG", "TV-Y"][i % 5]
        dur = "1 Season" if typ == "TV Show" else f"{90 + i % 30} min"
        genre = "TV Dramas, International TV Shows" if typ == "TV Show" else "Comedies, Dramas"
        country = ["United States", "India", "Japan", "Nigeria", "South Korea"][i % 5]
        add(
            show_id=f"s{i}",
            type=typ,
            title=f"Filler Title {i}",
            director="Not Given" if i % 4 == 0 else f"Director {i}",
            country=country,
            date_added=f"10/1/{yr_add}",
            release_year=rel,
            rating=rat,
            duration=dur,
            listed_in=genre,
        )

    out = Path(__file__).with_name("mini_netflix.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} n={len(rows)}")


if __name__ == "__main__":
    main()
