"""Write pipeline outputs as plain numpy / parquet / CSV / JSON files.

Everything is collected to the driver and written with pandas and numpy
instead of Spark's own writers. That keeps the project runnable on Windows
without installing winutils.exe / hadoop.dll.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import DataFrame

from .model import factors_to_numpy


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_table(pdf: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".parquet"):
        pdf.to_parquet(path, index=False)
    else:
        pdf.to_csv(path, index=False)


def save_model(model: ALSModel, ratings: DataFrame, out_dir: str) -> dict:
    """Save factor matrices and each user's rating history (CSR layout)."""
    os.makedirs(out_dir, exist_ok=True)
    item_ids, item_factors = factors_to_numpy(model.itemFactors)
    user_ids, user_factors = factors_to_numpy(model.userFactors)
    np.save(os.path.join(out_dir, "item_ids.npy"), item_ids)
    np.save(os.path.join(out_dir, "item_factors.npy"), item_factors)
    np.save(os.path.join(out_dir, "user_ids.npy"), user_ids)
    np.save(os.path.join(out_dir, "user_factors.npy"), user_factors)

    hist = ratings.select("userId", "movieId", "rating").toPandas()
    user_row = np.searchsorted(user_ids, hist["userId"].to_numpy())
    item_row = np.searchsorted(item_ids, hist["movieId"].to_numpy())
    # ids are sorted, so searchsorted is an exact lookup for known ids
    known = (user_row < len(user_ids)) & (item_row < len(item_ids))
    known &= user_ids[np.minimum(user_row, len(user_ids) - 1)] == hist["userId"].to_numpy()
    known &= item_ids[np.minimum(item_row, len(item_ids) - 1)] == hist["movieId"].to_numpy()
    user_row, item_row = user_row[known], item_row[known]
    rating = hist["rating"].to_numpy(dtype=np.float32)[known]

    order = np.argsort(user_row, kind="stable")
    counts = np.bincount(user_row, minlength=len(user_ids))
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    np.save(os.path.join(out_dir, "hist_indptr.npy"), indptr)
    np.save(os.path.join(out_dir, "hist_items.npy"), item_row[order].astype(np.int32))
    np.save(os.path.join(out_dir, "hist_ratings.npy"), rating[order])

    return {"n_users": int(len(user_ids)), "n_items": int(len(item_ids)), "rank": int(item_factors.shape[1])}
