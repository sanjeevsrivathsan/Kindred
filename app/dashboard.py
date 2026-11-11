"""Kindred dashboard.

    streamlit run app/dashboard.py

Reads the files written by `python -m kindred.pipeline`. No Spark needed here.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kindred.recommender import Recommender  # noqa: E402

ARTIFACTS = os.environ.get("KINDRED_ARTIFACTS", "artifacts")
BLUE = "#2a78d6"      # the one data colour used across charts
GRAY = "#9a9994"      # context / baselines

st.set_page_config(page_title="Kindred", page_icon="🎬", layout="wide")


# ----------------------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading model...")
def load_recommender(path: str) -> Recommender:
    return Recommender(path)


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(ARTIFACTS, "analytics", name))


if not os.path.exists(os.path.join(ARTIFACTS, "summary.json")):
    st.title("Kindred")
    st.error(
        f"No model found in `{ARTIFACTS}/`. Run the pipeline first:\n\n"
        "```\npython scripts/download_data.py\npython -m kindred.pipeline\n```"
    )
    st.stop()

rec = load_recommender(ARTIFACTS)
summary = rec.summary
DEFAULT_MIN_RATINGS = int(np.quantile(rec.movies["n_ratings"], 0.75))
MAX_MIN_RATINGS = max(int(np.quantile(rec.movies["n_ratings"], 0.99)), DEFAULT_MIN_RATINGS + 1)


# -------------------------------------------------------------------------- helpers
def bar(x, y, x_title, y_title, hover, horizontal=False, colors=None, height=340):
    fig = go.Figure(
        go.Bar(
            x=y if horizontal else x,
            y=x if horizontal else y,
            orientation="h" if horizontal else "v",
            marker=dict(color=colors or BLUE, cornerradius=4),
            hovertemplate=hover + "<extra></extra>",
            customdata=None,
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=x_title,
        yaxis_title=y_title,
        bargap=0.25,
        showlegend=False,
    )
    if horizontal:
        fig.update_yaxes(autorange="reversed")
    return fig


def line(x, y, x_title, y_title, hover, height=340):
    fig = go.Figure(
        go.Scatter(
            x=x, y=y, mode="lines+markers",
            line=dict(color=BLUE, width=2), marker=dict(size=6),
            hovertemplate=hover + "<extra></extra>",
        )
    )
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=x_title, yaxis_title=y_title, showlegend=False, hovermode="x",
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def show_movies(df: pd.DataFrame, score_col: str | None = None, score_label: str = "", key: str = ""):
    """Render a movie list as a table."""
    if df.empty:
        st.info("No movies match these filters. Try lowering the minimum number of ratings.")
        return
    out = pd.DataFrame(
        {
            "Title": df["title"],
            "Genres": df["genres"].str.replace("|", ", ", regex=False),
        }
    )
    config = {}
    if score_col:
        out[score_label] = df[score_col].astype(float).round(2)
        if score_col in ("predicted_rating", "rating", "weighted_score"):
            config[score_label] = st.column_config.ProgressColumn(
                score_label, min_value=0, max_value=5, format="%.2f"
            )
        else:
            config[score_label] = st.column_config.ProgressColumn(
                score_label, min_value=0, max_value=1, format="%.2f"
            )
    out["Avg rating"] = df["mean_rating"].astype(float).round(2)
    out["Ratings"] = df["n_ratings"].astype(int)
    if df["imdbId"].notna().any():
        out["IMDb"] = df["imdbId"].map(
            lambda i: f"https://www.imdb.com/title/tt{i}/" if isinstance(i, str) and i else None
        )
        config["IMDb"] = st.column_config.LinkColumn("IMDb", display_text="open")
    out.index = np.arange(1, len(out) + 1)
    st.dataframe(out, width="stretch", column_config=config, key=key or None)


def filters(key: str, with_genres: bool = True):
    """Shared filter row: how many, genres, minimum ratings."""
    c1, c2, c3 = st.columns([1, 2, 2])
    n = c1.number_input("How many", 5, 50, 10, step=5, key=f"{key}_n")
    genres = c2.multiselect("Genres", rec.genres, key=f"{key}_genres") if with_genres else None
    min_ratings = c3.slider(
        "Minimum number of ratings",
        0, MAX_MIN_RATINGS, DEFAULT_MIN_RATINGS, key=f"{key}_min",
        help="Hides obscure movies. Movies with only a handful of ratings get unreliable predictions.",
    )
    return int(n), genres, int(min_ratings)


@st.cache_data
def movie_options(limit: int = 15000) -> pd.DataFrame:
    """Movies offered in search boxes: the most-rated ones."""
    return rec.movies.nlargest(limit, "n_ratings")[["movieId", "title"]].sort_values("title")


def movie_picker(label: str, key: str):
    options = movie_options()
    titles = dict(zip(options["movieId"], options["title"]))
    return st.selectbox(
        label, options["movieId"].tolist(), index=None, key=key,
        format_func=lambda m: titles[m], placeholder="Type to search for a movie",
    )


# --------------------------------------------------------------------------- layout
st.sidebar.title("🎬 Kindred")
st.sidebar.caption("Movie recommendations from people with kindred taste.")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Recommend for a user", "Rate and discover", "Similar movies", "Model"],
    label_visibility="collapsed",
)
d, m = summary["data"], summary["model"]
st.sidebar.divider()
st.sidebar.caption(
    f"{d['n_ratings']:,} ratings · {d['n_users']:,} users · {d['n_movies']:,} movies\n\n"
    f"ALS rank {m['rank']} · Spark {summary.get('spark_version', '')}"
)


# ------------------------------------------------------------------------- overview
if page == "Overview":
    st.title("Dataset overview")
    st.caption(f"Ratings from {d['first_rating']} to {d['last_rating']}, aggregated with Spark SQL.")

    c = st.columns(5)
    c[0].metric("Ratings", f"{d['n_ratings']:,}")
    c[1].metric("Users", f"{d['n_users']:,}")
    c[2].metric("Movies", f"{d['n_movies']:,}")
    c[3].metric("Average rating", f"{d['mean_rating']:.2f}")
    c[4].metric(
        "Matrix density", f"{d['density']:.2%}",
        help="Share of all possible user-movie pairs that have a rating. The rest is what the model predicts.",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("How people rate")
        dist = load_csv("rating_distribution.csv")
        st.plotly_chart(
            bar(dist["rating"].astype(str), dist["count"], "Rating (stars)", "Ratings",
                "%{x} stars<br>%{y:,} ratings"),
            width="stretch",
        )
    with right:
        st.subheader("Ratings per year")
        over = load_csv("ratings_over_time.csv")
        st.plotly_chart(
            line(over["year"], over["count"], "Year", "Ratings", "%{x}<br>%{y:,} ratings"),
            width="stretch",
        )

    left, right = st.columns(2)
    genres = load_csv("genre_stats.csv")
    with left:
        st.subheader("Most rated genres")
        st.plotly_chart(
            bar(genres["genre"], genres["count"], "Ratings", "", "%{y}<br>%{x:,} ratings",
                horizontal=True, height=480),
            width="stretch",
        )
    with right:
        st.subheader("Best rated genres")
        best = genres.sort_values("mean_rating", ascending=False)
        fig = bar(best["genre"], best["mean_rating"], "Average rating", "",
                  "%{y}<br>average %{x:.2f}", horizontal=True, height=480)
        fig.update_xaxes(range=[0, 5])
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Average rating by release decade")
        dec = load_csv("decade_stats.csv")
        fig = bar(dec["decade"].astype(str) + "s", dec["mean_rating"], "Release decade",
                  "Average rating", "%{x}<br>average %{y:.2f}")
        fig.update_yaxes(range=[0, 5])
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("The long tail")
        tail = load_csv("movie_popularity.csv")
        st.plotly_chart(
            bar(tail["label"], tail["n"], "Ratings received by a movie", "Movies",
                "%{x} ratings<br>%{y:,} movies"),
            width="stretch",
        )
        top_share = tail.tail(3)
        st.caption(
            f"The {int(top_share['n'].sum()):,} most-rated movies collect "
            f"{top_share['ratings'].sum() / tail['ratings'].sum():.0%} of all ratings."
        )

    st.subheader("Top rated movies")
    st.caption("Ranked by a Bayesian weighted score, so a movie needs many ratings as well as a high average.")
    show_movies(rec.popular(25), "weighted_score", "Weighted score", key="top")
    with st.expander("View chart data as tables"):
        for name, frame in [("Rating distribution", dist), ("Ratings per year", over),
                            ("Genres", genres), ("Release decades", dec), ("Movie popularity", tail)]:
            st.caption(name)
            st.dataframe(frame, width="stretch", hide_index=True)


# ----------------------------------------------------------------- recommend (user)
elif page == "Recommend for a user":
    st.title("Recommend for a user")
    st.caption("Pick any user from the dataset to see what they liked and what the model suggests next.")

    if "user_id" not in st.session_state:
        st.session_state.user_id = int(rec.user_ids[0])

    def pick_random():
        st.session_state.user_id = int(np.random.choice(rec.user_ids))

    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    user_id = c1.number_input(
        "User ID", int(rec.user_ids.min()), int(rec.user_ids.max()), key="user_id", step=1
    )
    c2.button("Random user", on_click=pick_random, width="stretch")

    if not rec.has_user(user_id):
        st.warning("This user is not in the model (too few ratings). Showing popular movies instead.")
        n, genres, min_ratings = filters("cold")
        show_movies(rec.popular(n, genres, min_ratings), "weighted_score", "Weighted score")
    else:
        history = rec.user_history(user_id)
        c = st.columns(3)
        c[0].metric("Movies rated", f"{len(history):,}")
        c[1].metric("Their average rating", f"{history['rating'].mean():.2f}")
        fav = pd.Series([g for s in history[history["rating"] >= 4]["genres"] for g in s.split("|") if g])
        c[2].metric("Favourite genre", fav.mode().iat[0] if len(fav) else "n/a")

        n, genres, min_ratings = filters("user")
        left, right = st.columns(2)
        with left:
            st.subheader("Recommended")
            show_movies(
                rec.recommend_for_user(user_id, n, genres, min_ratings),
                "predicted_rating", "Predicted rating", key="user_recs",
            )
        with right:
            st.subheader("Their highest rated")
            show_movies(history.head(max(n, 15)), "rating", "Their rating", key="user_hist")


# ---------------------------------------------------------------- rate and discover
elif page == "Rate and discover":
    st.title("Rate and discover")
    st.caption(
        "Rate a few movies and get recommendations right away. Your ratings are folded into the "
        "trained model with one least-squares step, so nothing is retrained."
    )
    my = st.session_state.setdefault("my_ratings", {})

    c1, c2, c3 = st.columns([4, 2, 1], vertical_alignment="bottom")
    with c1:
        picked = movie_picker("Movie", "rate_pick")
    stars = c2.select_slider("Your rating", [x / 2 for x in range(1, 11)], value=4.0)
    if c3.button("Add", type="primary", width="stretch", disabled=picked is None):
        my[int(picked)] = float(stars)

    if not my:
        st.info("Add at least 3 to 5 movies for useful results. Until then, here are the crowd favourites.")
        n, genres, min_ratings = filters("new_cold")
        show_movies(rec.popular(n, genres, min_ratings), "weighted_score", "Weighted score")
    else:
        left, right = st.columns([2, 3])
        with left:
            st.subheader(f"Your ratings ({len(my)})")
            for movie_id, rating in list(my.items()):
                title = rec.movies.loc[rec._item_pos[movie_id], "title"]
                a, b = st.columns([5, 1], vertical_alignment="center")
                a.write(f"**{rating:g}** · {title}")
                if b.button("✕", key=f"rm_{movie_id}", help="Remove"):
                    del my[movie_id]
                    st.rerun()
            if st.button("Clear all"):
                my.clear()
                st.rerun()
        with right:
            st.subheader("Recommended for you")
            n, genres, min_ratings = filters("new")
            show_movies(
                rec.recommend_from_ratings(my, n, genres, min_ratings),
                "predicted_rating", "Predicted rating", key="new_recs",
            )


# ------------------------------------------------------------------- similar movies
elif page == "Similar movies":
    st.title("Similar movies")
    st.caption(
        "Movies that the same kinds of people enjoy. Similarity is the cosine between the movie "
        "vectors ALS learned from ratings alone: no genres, cast or plot were used."
    )
    picked = movie_picker("Movie", "similar_pick")
    if picked is None:
        st.info("Pick a movie to see its neighbours.")
    else:
        row = rec.movies.loc[rec._item_pos[int(picked)]]
        c = st.columns(3)
        c[0].metric("Average rating", f"{row['mean_rating']:.2f}")
        c[1].metric("Ratings", f"{int(row['n_ratings']):,}")
        c[2].metric("Genres", row["genres"].replace("|", ", ") or "n/a")
        n, _, min_ratings = filters("sim", with_genres=False)
        show_movies(rec.similar_movies(int(picked), n, min_ratings), "similarity", "Similarity", key="sim")


# ---------------------------------------------------------------------------- model
elif page == "Model":
    st.title("Model")
    met = summary["metrics"]
    st.caption(
        "Alternating Least Squares matrix factorisation (Spark MLlib), evaluated on a held-out "
        f"{summary['config']['test_fraction']:.0%} of ratings."
    )

    c = st.columns(4)
    c[0].metric("Rank (latent factors)", m["rank"])
    c[1].metric("Regularisation", m["reg_param"])
    c[2].metric("Iterations", m["max_iter"])
    c[3].metric("Pipeline run time", f"{summary['total_seconds'] / 60:.1f} min")

    left, right = st.columns(2)
    with left:
        st.subheader("Prediction error (RMSE)")
        names = ["Global mean", "User + movie bias", "ALS"]
        values = [met["global_mean"]["rmse"], met["bias"]["rmse"], met["als"]["rmse"]]
        fig = bar(names, values, "", "RMSE", "%{x}<br>RMSE %{y:.4f}", colors=[GRAY, GRAY, BLUE])
        fig.update_traces(text=[f"{v:.3f}" for v in values], textposition="outside", cliponaxis=False)
        fig.update_layout(margin=dict(t=30))
        st.plotly_chart(fig, width="stretch")
        gain = 1 - met["als"]["rmse"] / met["global_mean"]["rmse"]
        st.caption(
            f"ALS is off by {met['als']['rmse']:.2f} stars on a typical prediction, {gain:.0%} better "
            f"than always guessing the average. MAE: {met['als']['mae']:.3f}."
        )
    with right:
        st.subheader("Ranking quality")
        r = met.get("ranking") or {}
        if r:
            k = r["k"]
            a, b = st.columns(2)
            a.metric(f"Precision@{k}", f"{r['precision_at_k']:.3f}",
                     help=f"Share of the top {k} recommendations the user rated highly in the test set.")
            b.metric(f"NDCG@{k}", f"{r['ndcg_at_k']:.3f}",
                     help="Like precision, but hits near the top of the list count for more.")
            a.metric(f"Hit rate@{k}", f"{r['hit_rate_at_k']:.3f}",
                     help=f"Share of users with at least one hit in their top {k}.")
            b.metric("Catalogue coverage", f"{r['catalog_coverage']:.1%}",
                     help="Share of all movies that appear in at least one evaluated user's top list.")
            st.caption(
                f"Measured on {r['users_evaluated']:,} sampled users. A hit is a held-out movie the user "
                f"rated {summary['config']['relevance_threshold']:g} or more. These numbers look low for every "
                "recommender on this kind of test, because most good movies were simply never rated by the user."
            )
        else:
            st.info("Ranking metrics were not computed for this run.")

    if summary.get("tuning"):
        st.subheader("Hyperparameter search")
        st.dataframe(pd.DataFrame(summary["tuning"]), width="stretch", hide_index=True)

    st.subheader("Pipeline stages")
    timings = pd.DataFrame(
        {"Stage": list(summary["timings_seconds"]), "Seconds": list(summary["timings_seconds"].values())}
    )
    st.dataframe(timings, width="content", hide_index=True)
