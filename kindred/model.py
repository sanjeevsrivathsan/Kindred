"""ALS collaborative filtering: train, evaluate, tune."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .config import Config


def train_als(train: DataFrame, rank: int, reg_param: float, max_iter: int, seed: int) -> ALSModel:
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=rank,
        regParam=reg_param,
        maxIter=max_iter,
        nonnegative=False,
        implicitPrefs=False,
        coldStartStrategy="drop",   # skip test rows whose user/movie was unseen in training
        seed=seed,
    )
    return als.fit(train)


def regression_metrics(predictions: DataFrame, col: str = "prediction") -> dict:
    out = {}
    for metric in ("rmse", "mae"):
        ev = RegressionEvaluator(metricName=metric, labelCol="rating", predictionCol=col)
        out[metric] = float(ev.evaluate(predictions))
    return out


def baseline_metrics(train: DataFrame, test: DataFrame) -> dict:
    """Two simple baselines to show how much ALS actually adds.

    global_mean: predict the overall average for everything.
    bias: global mean + movie offset + user offset (damped averages).
    """
    mu = train.agg(F.avg("rating")).first()[0]
    damp = 10.0

    movie_bias = train.groupBy("movieId").agg(
        (F.sum(F.col("rating") - mu) / (F.count("*") + damp)).alias("b_i")
    )
    user_bias = (
        train.join(movie_bias, "movieId")
        .groupBy("userId")
        .agg((F.sum(F.col("rating") - mu - F.col("b_i")) / (F.count("*") + damp)).alias("b_u"))
    )
    preds = (
        test.join(movie_bias, "movieId", "left")
        .join(user_bias, "userId", "left")
        .fillna(0.0, subset=["b_i", "b_u"])
        .withColumn("global_pred", F.lit(mu).cast("float"))
        .withColumn(
            "bias_pred",
            F.least(F.lit(5.0), F.greatest(F.lit(0.5), mu + F.col("b_i") + F.col("b_u"))).cast("float"),
        )
    ).cache()
    out = {
        "global_mean": regression_metrics(preds, "global_pred"),
        "bias": regression_metrics(preds, "bias_pred"),
    }
    preds.unpersist()
    return out


def factors_to_numpy(factors: DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Collect an ALS factor table as (ids, matrix), sorted by id."""
    pdf = factors.select("id", "features").toPandas().sort_values("id")
    ids = pdf["id"].to_numpy(dtype=np.int32)
    matrix = np.asarray(pdf["features"].tolist(), dtype=np.float32)
    return ids, matrix


def ranking_metrics(
    model: ALSModel, train: DataFrame, test: DataFrame, cfg: Config
) -> dict:
    """Precision@K, Recall@K, NDCG@K and hit rate on a sample of test users.

    For each sampled user we score every movie, remove the ones they rated in
    training, take the top K and compare against the test movies they rated
    at or above the relevance threshold.
    """
    k = cfg.eval_k
    relevant = test.filter(F.col("rating") >= cfg.relevance_threshold)
    candidates = relevant.select("userId").distinct()
    n_candidates = candidates.count()
    if n_candidates == 0:
        return {}
    fraction = min(1.0, cfg.eval_users / n_candidates * 1.2)
    users = candidates.sample(False, fraction, cfg.seed).limit(cfg.eval_users).cache()

    seen = train.join(F.broadcast(users), "userId").select("userId", "movieId").toPandas()
    truth = relevant.join(F.broadcast(users), "userId").select("userId", "movieId").toPandas()
    user_ids, U = factors_to_numpy(model.userFactors.join(F.broadcast(users), F.col("id") == F.col("userId"), "left_semi"))
    item_ids, V = factors_to_numpy(model.itemFactors)
    users.unpersist()

    item_pos = pd.Series(np.arange(len(item_ids)), index=item_ids)
    seen = seen[seen["movieId"].isin(item_pos.index)]
    seen_by_user = seen.groupby("userId")["movieId"].apply(lambda s: item_pos.loc[s].to_numpy())
    truth_by_user = truth.groupby("userId")["movieId"].apply(set)

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    precision, recall, ndcg, hits, recommended = [], [], [], [], set()
    for row, uid in enumerate(user_ids):
        rel = truth_by_user.get(int(uid))
        if not rel:
            continue
        scores = V @ U[row]
        if uid in seen_by_user.index:
            scores[seen_by_user.loc[uid]] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        top_ids = item_ids[top]
        recommended.update(top_ids.tolist())
        gains = np.fromiter((mid in rel for mid in top_ids), dtype=np.float32, count=k)
        n_hit = gains.sum()
        precision.append(n_hit / k)
        recall.append(n_hit / len(rel))
        ideal = discounts[: min(len(rel), k)].sum()
        ndcg.append(float((gains * discounts).sum() / ideal))
        hits.append(1.0 if n_hit > 0 else 0.0)

    if not precision:
        return {}
    return {
        "k": k,
        "users_evaluated": len(precision),
        "precision_at_k": float(np.mean(precision)),
        "recall_at_k": float(np.mean(recall)),
        "ndcg_at_k": float(np.mean(ndcg)),
        "hit_rate_at_k": float(np.mean(hits)),
        "catalog_coverage": len(recommended) / len(item_ids),
    }


def tune(ratings: DataFrame, cfg: Config, log=print) -> tuple[dict, pd.DataFrame]:
    """Grid search over rank x regParam on a sample of users."""
    users = ratings.select("userId").distinct().sample(False, cfg.tune_fraction, cfg.seed)
    sample = ratings.join(users, "userId", "left_semi")
    train, valid = sample.randomSplit([0.8, 0.2], seed=cfg.seed)
    train.cache()
    valid.cache()
    log(f"  tuning on {train.count():,} train / {valid.count():,} validation ratings")

    rows = []
    for rank in cfg.tune_ranks:
        for reg in cfg.tune_regs:
            start = time.time()
            model = train_als(train, rank, reg, cfg.tune_max_iter, cfg.seed)
            rmse = regression_metrics(model.transform(valid))["rmse"]
            took = time.time() - start
            rows.append({"rank": rank, "reg_param": reg, "rmse": rmse, "seconds": round(took, 1)})
            log(f"  rank={rank:<4} reg={reg:<5} rmse={rmse:.4f}  ({took:.0f}s)")
    train.unpersist()
    valid.unpersist()

    results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    best = results.iloc[0]
    return {"rank": int(best["rank"]), "reg_param": float(best["reg_param"])}, results
