"""Descriptive analytics over the full ratings table (all computed in Spark).

Each function returns a small pandas DataFrame, ready to save and chart.
"""
from __future__ import annotations

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .config import Config


def summary(ratings: DataFrame) -> dict:
    row = ratings.agg(
        F.count("*").alias("n_ratings"),
        F.countDistinct("userId").alias("n_users"),
        F.countDistinct("movieId").alias("n_movies"),
        F.avg("rating").alias("mean_rating"),
        F.min("timestamp").alias("first_ts"),
        F.max("timestamp").alias("last_ts"),
    ).first()
    n_ratings, n_users, n_movies = row["n_ratings"], row["n_users"], row["n_movies"]
    return {
        "n_ratings": int(n_ratings),
        "n_users": int(n_users),
        "n_movies": int(n_movies),
        "mean_rating": float(row["mean_rating"]),
        "density": n_ratings / (n_users * n_movies),
        "first_rating": pd.to_datetime(row["first_ts"], unit="s").strftime("%Y-%m-%d"),
        "last_rating": pd.to_datetime(row["last_ts"], unit="s").strftime("%Y-%m-%d"),
    }


def rating_distribution(ratings: DataFrame) -> pd.DataFrame:
    return ratings.groupBy("rating").count().orderBy("rating").toPandas()


def ratings_over_time(ratings: DataFrame) -> pd.DataFrame:
    """Ratings count and average per calendar year."""
    return (
        ratings.withColumn("year", F.year(F.from_unixtime("timestamp")))
        .groupBy("year")
        .agg(
            F.count("*").alias("count"),
            F.avg("rating").alias("mean_rating"),
            F.countDistinct("userId").alias("active_users"),
        )
        .orderBy("year")
        .toPandas()
    )


def genre_stats(ratings: DataFrame, movies: DataFrame) -> pd.DataFrame:
    """Rating volume and average per genre (a movie counts once per genre)."""
    exploded = movies.select("movieId", F.explode(F.split("genres", r"\|")).alias("genre")).filter(
        F.col("genre") != "(no genres listed)"
    )
    return (
        ratings.join(F.broadcast(exploded), "movieId")
        .groupBy("genre")
        .agg(
            F.count("*").alias("count"),
            F.avg("rating").alias("mean_rating"),
            F.countDistinct("movieId").alias("n_movies"),
        )
        .orderBy(F.desc("count"))
        .toPandas()
    )


def decade_stats(ratings: DataFrame, movies: DataFrame) -> pd.DataFrame:
    """Rating volume and average by the decade a movie was released."""
    by_decade = movies.filter(F.col("year").isNotNull()).select(
        "movieId", (F.floor(F.col("year") / 10) * 10).cast("int").alias("decade")
    )
    return (
        ratings.join(F.broadcast(by_decade), "movieId")
        .groupBy("decade")
        .agg(F.count("*").alias("count"), F.avg("rating").alias("mean_rating"))
        .orderBy("decade")
        .toPandas()
    )


def activity_buckets(ratings: DataFrame, key: str) -> pd.DataFrame:
    """How many users (or movies) fall in each ratings-count bucket.

    Shows the long tail: most movies have very few ratings.
    """
    edges = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 20000]
    counts = ratings.groupBy(key).count()
    bucket = F.lit(None).cast("int")
    for i in reversed(range(len(edges))):
        bucket = F.when(F.col("count") >= edges[i], F.lit(i)).otherwise(bucket)
    pdf = (
        counts.withColumn("bucket", bucket)
        .groupBy("bucket")
        .agg(F.count("*").alias("n"), F.sum("count").alias("ratings"))
        .orderBy("bucket")
        .toPandas()
    )
    labels = []
    for i in pdf["bucket"]:
        lo = edges[int(i)]
        hi = edges[int(i) + 1] - 1 if int(i) + 1 < len(edges) else None
        labels.append(f"{lo:,}+" if hi is None else (f"{lo}" if lo == hi else f"{lo:,}-{hi:,}"))
    pdf["label"] = labels
    return pdf[["bucket", "label", "n", "ratings"]]


def movie_table(ratings: DataFrame, movies: DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per movie with rating count, mean, and a Bayesian weighted score.

    score = (v / (v + m)) * R + (m / (v + m)) * C
      v = ratings for the movie, R = its mean, C = global mean,
      m = prior strength (how many ratings before we trust the movie's own mean).
    The score doubles as the popularity fallback for cold-start users.
    """
    stats = ratings.groupBy("movieId").agg(
        F.count("*").alias("n_ratings"), F.avg("rating").alias("mean_rating")
    )
    global_mean = ratings.agg(F.avg("rating")).first()[0]
    m = cfg.bayes_prior_votes or int(stats.approxQuantile("n_ratings", [0.9], 0.01)[0])
    m = max(m, 1)
    v, r = F.col("n_ratings"), F.col("mean_rating")
    scored = stats.withColumn("score", (v / (v + m)) * r + (m / (v + m)) * F.lit(global_mean))
    pdf = (
        movies.join(scored, "movieId", "inner")
        .select("movieId", "title", "genres", "year", "imdbId", "n_ratings", "mean_rating", "score")
        .toPandas()
    )
    pdf["year"] = pdf["year"].astype("Int64")
    return pdf.sort_values("score", ascending=False).reset_index(drop=True)
