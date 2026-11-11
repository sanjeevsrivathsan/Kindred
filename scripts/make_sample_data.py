"""Generate a small synthetic dataset in MovieLens format.

Useful for tests and for trying the pipeline with no download. The ratings
come from hidden user/movie taste vectors plus noise, so ALS has a real
pattern to learn. The movies are made up; use real MovieLens data for a demo.

    python scripts/make_sample_data.py --out data/sample
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Animation"]


def generate(out: str, n_users: int = 2000, n_movies: int = 800, per_user: int = 60, seed: int = 7) -> str:
    rng = np.random.default_rng(seed)
    k = len(GENRES)

    # Each movie leans towards one or two genres; each user likes a few genres.
    movie_vecs = rng.normal(0, 0.25, (n_movies, k))
    main = rng.integers(0, k, n_movies)
    second = rng.integers(0, k, n_movies)
    movie_vecs[np.arange(n_movies), main] += 1.0
    has_second = rng.random(n_movies) < 0.4
    movie_vecs[has_second, second[has_second]] += 0.6
    user_vecs = rng.normal(0, 0.7, (n_users, k))

    movie_bias = rng.normal(0, 0.35, n_movies)
    user_bias = rng.normal(0, 0.3, n_users)
    popularity = 1.0 / np.arange(1, n_movies + 1) ** 0.8
    popularity /= popularity.sum()

    rows = []
    for u in range(n_users):
        n = int(np.clip(rng.poisson(per_user), 8, n_movies))
        picked = rng.choice(n_movies, size=n, replace=False, p=popularity)
        raw = 3.4 + user_bias[u] + movie_bias[picked] + movie_vecs[picked] @ user_vecs[u]
        raw += rng.normal(0, 0.45, n)
        rating = np.clip(np.round(raw * 2) / 2, 0.5, 5.0)
        ts = rng.integers(946_684_800, 1_577_836_800, n)  # 2000-01-01 .. 2020-01-01
        rows.append(pd.DataFrame({"userId": u + 1, "movieId": picked + 1, "rating": rating, "timestamp": ts}))
    ratings = pd.concat(rows, ignore_index=True)

    years = rng.integers(1960, 2020, n_movies)
    genres = [
        GENRES[main[i]] if not has_second[i] or second[i] == main[i] else f"{GENRES[main[i]]}|{GENRES[second[i]]}"
        for i in range(n_movies)
    ]
    movies = pd.DataFrame(
        {
            "movieId": np.arange(1, n_movies + 1),
            "title": [f"Sample {GENRES[main[i]]} Movie {i + 1:04d} ({years[i]})" for i in range(n_movies)],
            "genres": genres,
        }
    )

    os.makedirs(out, exist_ok=True)
    ratings.to_csv(os.path.join(out, "ratings.csv"), index=False)
    movies.to_csv(os.path.join(out, "movies.csv"), index=False)
    print(f"Wrote {len(ratings):,} ratings, {n_users:,} users, {n_movies:,} movies to {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sample")
    p.add_argument("--users", type=int, default=2000)
    p.add_argument("--movies", type=int, default=800)
    args = p.parse_args()
    generate(args.out, args.users, args.movies)
