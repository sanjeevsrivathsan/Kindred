"""Loading and cleaning the MovieLens files with Spark."""
from __future__ import annotations

import os

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from .config import Config

RATINGS_SCHEMA = T.StructType(
    [
        T.StructField("userId", T.IntegerType(), False),
        T.StructField("movieId", T.IntegerType(), False),
        T.StructField("rating", T.FloatType(), False),
        T.StructField("timestamp", T.LongType(), False),
    ]
)

MOVIES_SCHEMA = T.StructType(
    [
        T.StructField("movieId", T.IntegerType(), False),
        T.StructField("title", T.StringType(), True),
        T.StructField("genres", T.StringType(), True),
    ]
)

LINKS_SCHEMA = T.StructType(
    [
        T.StructField("movieId", T.IntegerType(), False),
        T.StructField("imdbId", T.StringType(), True),
        T.StructField("tmdbId", T.StringType(), True),
    ]
)


def _path(cfg: Config, name: str) -> str:
    path = os.path.join(cfg.data_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first, "
            f"or pass --data-dir pointing at a MovieLens folder."
        )
    return os.path.abspath(path)


def load_ratings(spark: SparkSession, cfg: Config) -> DataFrame:
    """Read ratings.csv, drop bad rows and duplicates, apply activity filters."""
    raw = spark.read.csv(_path(cfg, "ratings.csv"), header=True, schema=RATINGS_SCHEMA)

    ratings = (
        raw.dropna(subset=["userId", "movieId", "rating"])
        .filter((F.col("rating") >= 0.5) & (F.col("rating") <= 5.0))
        # one rating per (user, movie): keep the most recent
        .groupBy("userId", "movieId")
        .agg(F.max(F.struct("timestamp", "rating")).alias("latest"))
        .select("userId", "movieId", "latest.rating", "latest.timestamp")
    )

    if cfg.sample_fraction < 1.0:
        # Sample whole users, so every kept user still has their full history.
        users = ratings.select("userId").distinct().sample(False, cfg.sample_fraction, cfg.seed)
        ratings = ratings.join(users, "userId", "left_semi")

    if cfg.min_movie_ratings > 1:
        ok = ratings.groupBy("movieId").count().filter(F.col("count") >= cfg.min_movie_ratings)
        ratings = ratings.join(ok.select("movieId"), "movieId", "left_semi")
    if cfg.min_user_ratings > 1:
        ok = ratings.groupBy("userId").count().filter(F.col("count") >= cfg.min_user_ratings)
        ratings = ratings.join(ok.select("userId"), "userId", "left_semi")

    return ratings


def load_movies(spark: SparkSession, cfg: Config) -> DataFrame:
    """Read movies.csv and pull the release year out of the title."""
    movies = spark.read.csv(
        _path(cfg, "movies.csv"), header=True, schema=MOVIES_SCHEMA, quote='"', escape='"'
    )
    year = F.regexp_extract("title", r"\((\d{4})\)\s*$", 1)
    movies = movies.withColumn("year", F.when(year != "", year.cast("int")))

    links_path = os.path.join(cfg.data_dir, "links.csv")
    if os.path.exists(links_path):
        links = spark.read.csv(os.path.abspath(links_path), header=True, schema=LINKS_SCHEMA)
        movies = movies.join(links.select("movieId", "imdbId"), "movieId", "left")
    else:
        movies = movies.withColumn("imdbId", F.lit(None).cast("string"))
    return movies


def split(ratings: DataFrame, cfg: Config) -> tuple[DataFrame, DataFrame]:
    """Random train/test split."""
    train, test = ratings.randomSplit([1.0 - cfg.test_fraction, cfg.test_fraction], seed=cfg.seed)
    return train, test
