"""End-to-end pipeline: load -> analyse -> (tune) -> train -> evaluate -> export.

    python -m kindred.pipeline --data-dir data/ml-25m
    python -m kindred.pipeline --data-dir data/ml-latest-small --rank 20
    python -m kindred.pipeline --tune
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import fields

from . import analytics, data, export, model
from .config import Config
from .spark import build_session


def parse_args(argv=None) -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="Kindred: train the ALS recommender and export artifacts.")
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        default = getattr(cfg, f.name)
        if isinstance(default, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif isinstance(default, tuple):
            p.add_argument(flag, nargs="+", type=type(default[0]), default=default)
        else:
            p.add_argument(flag, type=type(default), default=default)
    args = p.parse_args(argv)
    values = vars(args)
    for name in ("tune_ranks", "tune_regs"):
        values[name] = tuple(values[name])
    return Config(**values)


def run(cfg: Config, log=print) -> dict:
    started = time.time()
    timings: dict[str, float] = {}

    def step(name):
        log(f"\n[{time.time() - started:6.0f}s] {name}")
        return time.time()

    spark = build_session(cfg)
    try:
        # ---------------------------------------------------------------- load
        t = step("Loading and cleaning ratings")
        ratings = data.load_ratings(spark, cfg).cache()
        movies = data.load_movies(spark, cfg).cache()
        stats = analytics.summary(ratings)
        log(
            f"  {stats['n_ratings']:,} ratings | {stats['n_users']:,} users | "
            f"{stats['n_movies']:,} movies | density {stats['density']:.4%}"
        )
        timings["load"] = time.time() - t

        # ----------------------------------------------------------- analytics
        t = step("Computing analytics")
        adir = os.path.join(cfg.out_dir, "analytics")
        export.save_table(analytics.rating_distribution(ratings), os.path.join(adir, "rating_distribution.csv"))
        export.save_table(analytics.ratings_over_time(ratings), os.path.join(adir, "ratings_over_time.csv"))
        export.save_table(analytics.genre_stats(ratings, movies), os.path.join(adir, "genre_stats.csv"))
        export.save_table(analytics.decade_stats(ratings, movies), os.path.join(adir, "decade_stats.csv"))
        export.save_table(analytics.activity_buckets(ratings, "userId"), os.path.join(adir, "user_activity.csv"))
        export.save_table(analytics.activity_buckets(ratings, "movieId"), os.path.join(adir, "movie_popularity.csv"))
        export.save_table(analytics.movie_table(ratings, movies, cfg), os.path.join(cfg.out_dir, "movies.parquet"))
        timings["analytics"] = time.time() - t

        # --------------------------------------------------------------- tune
        rank, reg = cfg.rank, cfg.reg_param
        tuning = None
        if cfg.tune:
            t = step("Tuning rank and regParam")
            best, results = model.tune(ratings, cfg, log)
            rank, reg = best["rank"], best["reg_param"]
            export.save_table(results, os.path.join(cfg.out_dir, "tuning.csv"))
            tuning = results.to_dict(orient="records")
            log(f"  best: rank={rank} reg={reg}")
            timings["tune"] = time.time() - t

        # ------------------------------------------------------ train + evaluate
        t = step(f"Training ALS (rank={rank}, reg={reg}, iterations={cfg.max_iter})")
        train, test = data.split(ratings, cfg)
        train.cache()
        test.cache()
        log(f"  train {train.count():,} | test {test.count():,}")
        als = model.train_als(train, rank, reg, cfg.max_iter, cfg.seed)
        timings["train"] = time.time() - t

        t = step("Evaluating")
        predictions = als.transform(test).cache()
        metrics = {"als": model.regression_metrics(predictions)}
        metrics["als"]["test_rows_scored"] = predictions.count()
        predictions.unpersist()
        metrics.update(model.baseline_metrics(train, test))
        metrics["ranking"] = model.ranking_metrics(als, train, test, cfg)
        log(
            f"  RMSE  ALS {metrics['als']['rmse']:.4f} | bias baseline {metrics['bias']['rmse']:.4f} | "
            f"global mean {metrics['global_mean']['rmse']:.4f}"
        )
        if metrics["ranking"]:
            r = metrics["ranking"]
            log(
                f"  Precision@{r['k']} {r['precision_at_k']:.4f} | NDCG@{r['k']} {r['ndcg_at_k']:.4f} | "
                f"hit rate {r['hit_rate_at_k']:.4f}  ({r['users_evaluated']:,} users)"
            )
        timings["evaluate"] = time.time() - t
        train.unpersist()
        test.unpersist()

        # --------------------------------------------------------------- export
        if cfg.refit_full:
            t = step("Refitting on all ratings for serving")
            als = model.train_als(ratings, rank, reg, cfg.max_iter, cfg.seed)
            timings["refit"] = time.time() - t

        t = step("Exporting factors and rating history")
        shape = export.save_model(als, ratings, cfg.out_dir)
        timings["export"] = time.time() - t

        summary = {
            "data": stats,
            "model": {"rank": rank, "reg_param": reg, "max_iter": cfg.max_iter, **shape},
            "metrics": metrics,
            "tuning": tuning,
            "timings_seconds": {k: round(v, 1) for k, v in timings.items()},
            "total_seconds": round(time.time() - started, 1),
            "config": cfg.to_dict(),
            "spark_version": spark.version,
        }
        export.save_json(summary, os.path.join(cfg.out_dir, "summary.json"))
        log(f"\nDone in {summary['total_seconds']:.0f}s. Artifacts written to {os.path.abspath(cfg.out_dir)}")
        return summary
    finally:
        spark.stop()


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
