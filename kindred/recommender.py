"""Serving layer: recommendations from the exported ALS factors.

This module only needs numpy and pandas, so the dashboard starts in a second
and never has to boot Spark. Scores are plain dot products of the user and
movie factor vectors, exactly what ALSModel.transform computes.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


class Recommender:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.dir = artifacts_dir
        with open(os.path.join(artifacts_dir, "summary.json"), encoding="utf-8") as f:
            self.summary = json.load(f)

        self.item_ids = np.load(self._p("item_ids.npy"))
        self.item_factors = np.load(self._p("item_factors.npy"))
        self.user_ids = np.load(self._p("user_ids.npy"))
        self.user_factors = np.load(self._p("user_factors.npy"), mmap_mode="r")

        # Rating history in CSR form, one block per user (same order as user_ids).
        self.hist_indptr = np.load(self._p("hist_indptr.npy"))
        self.hist_items = np.load(self._p("hist_items.npy"), mmap_mode="r")
        self.hist_ratings = np.load(self._p("hist_ratings.npy"), mmap_mode="r")

        movies = pd.read_parquet(self._p("movies.parquet"))
        # Re-order so that row i of `movies` describes item_factors[i].
        self.movies = movies.set_index("movieId").reindex(self.item_ids).reset_index()
        self.movies["title"] = self.movies["title"].fillna("(unknown title)")
        self.movies["genres"] = self.movies["genres"].fillna("")
        self.movies["n_ratings"] = self.movies["n_ratings"].fillna(0).astype(int)

        self._item_pos = {int(m): i for i, m in enumerate(self.item_ids)}
        self._user_pos = {int(u): i for i, u in enumerate(self.user_ids)}
        norms = np.linalg.norm(self.item_factors, axis=1, keepdims=True)
        self._unit_items = self.item_factors / np.maximum(norms, 1e-9)

        self.reg_param = float(self.summary["model"]["reg_param"])
        self.genres = sorted(
            {g for s in self.movies["genres"] for g in s.split("|") if g and not g.startswith("(")}
        )

    def _p(self, name: str) -> str:
        return os.path.join(self.dir, name)

    # ------------------------------------------------------------------ lookups
    def has_user(self, user_id: int) -> bool:
        return int(user_id) in self._user_pos

    def search(self, query: str, limit: int = 20) -> pd.DataFrame:
        """Case-insensitive title search, most-rated matches first."""
        mask = self.movies["title"].str.contains(query.strip(), case=False, regex=False)
        return self.movies[mask].sort_values("n_ratings", ascending=False).head(limit)

    def user_history(self, user_id: int) -> pd.DataFrame:
        """Movies a user has rated, best first."""
        row = self._user_pos[int(user_id)]
        lo, hi = self.hist_indptr[row], self.hist_indptr[row + 1]
        out = self.movies.iloc[np.asarray(self.hist_items[lo:hi])].copy()
        out["rating"] = np.asarray(self.hist_ratings[lo:hi])
        return out.sort_values(["rating", "n_ratings"], ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ ranking
    def _filter_mask(
        self,
        genres: Iterable[str] | None,
        min_ratings: int,
        year_range: tuple[int, int] | None,
    ) -> np.ndarray:
        mask = self.movies["n_ratings"].to_numpy() >= min_ratings
        if genres:
            wanted = set(genres)
            mask &= self.movies["genres"].map(lambda s: bool(wanted & set(s.split("|")))).to_numpy()
        if year_range:
            year = self.movies["year"].astype("float").to_numpy()
            mask &= (year >= year_range[0]) & (year <= year_range[1])
        return mask

    def _top(self, scores: np.ndarray, n: int, column: str) -> pd.DataFrame:
        n = min(n, int(np.isfinite(scores).sum()))
        if n <= 0:
            return self.movies.iloc[[]].assign(**{column: []})
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        out = self.movies.iloc[top].copy()
        out[column] = scores[top]
        return out.reset_index(drop=True)

    def _rank_for_vector(self, user_vec, exclude, n, genres, min_ratings, year_range):
        scores = self.item_factors @ np.asarray(user_vec, dtype=np.float32)
        scores[~self._filter_mask(genres, min_ratings, year_range)] = -np.inf
        if len(exclude):
            scores[np.asarray(exclude)] = -np.inf
        out = self._top(scores, n, "predicted_rating")
        out["predicted_rating"] = out["predicted_rating"].clip(0.5, 5.0)
        return out

    def recommend_for_user(
        self, user_id: int, n: int = 10, genres=None, min_ratings: int = 0, year_range=None
    ) -> pd.DataFrame:
        """Top-N unseen movies for a user who was in the training data."""
        row = self._user_pos[int(user_id)]
        seen = self.hist_items[self.hist_indptr[row] : self.hist_indptr[row + 1]]
        return self._rank_for_vector(self.user_factors[row], seen, n, genres, min_ratings, year_range)

    def fold_in(self, ratings: Mapping[int, float]) -> np.ndarray:
        """Compute a factor vector for a brand-new user from a few ratings.

        This is one ALS half-step with the movie factors held fixed:
            u = (V_s^T V_s + lambda * n * I)^-1  V_s^T r
        where V_s are the factors of the n movies the user rated. No retraining.
        """
        pos = [self._item_pos[int(m)] for m in ratings if int(m) in self._item_pos]
        r = np.array([ratings[m] for m in ratings if int(m) in self._item_pos], dtype=np.float64)
        if not pos:
            raise ValueError("None of the rated movies are known to the model.")
        V = self.item_factors[pos].astype(np.float64)
        k = V.shape[1]
        A = V.T @ V + self.reg_param * len(pos) * np.eye(k)
        return np.linalg.solve(A, V.T @ r).astype(np.float32)

    def recommend_from_ratings(
        self, ratings: Mapping[int, float], n: int = 10, genres=None, min_ratings: int = 0, year_range=None
    ) -> pd.DataFrame:
        """Top-N for a new user described only by {movieId: rating}."""
        vec = self.fold_in(ratings)
        exclude = [self._item_pos[int(m)] for m in ratings if int(m) in self._item_pos]
        return self._rank_for_vector(vec, exclude, n, genres, min_ratings, year_range)

    def similar_movies(self, movie_id: int, n: int = 10, min_ratings: int = 0) -> pd.DataFrame:
        """Movies whose factor vectors point the same way (cosine similarity)."""
        pos = self._item_pos[int(movie_id)]
        scores = self._unit_items @ self._unit_items[pos]
        scores[~self._filter_mask(None, min_ratings, None)] = -np.inf
        scores[pos] = -np.inf
        return self._top(scores, n, "similarity")

    def popular(self, n: int = 10, genres=None, min_ratings: int = 0, year_range=None) -> pd.DataFrame:
        """Cold-start fallback: best Bayesian weighted score."""
        scores = self.movies["score"].fillna(-np.inf).to_numpy(dtype=np.float32).copy()
        scores[~self._filter_mask(genres, min_ratings, year_range)] = -np.inf
        return self._top(scores, n, "weighted_score")
