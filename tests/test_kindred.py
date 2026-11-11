import os

import numpy as np


def test_pipeline_outputs(artifacts):
    out_dir, summary = artifacts
    for name in ["summary.json", "movies.parquet", "item_factors.npy", "user_factors.npy",
                 "hist_indptr.npy", "analytics/genre_stats.csv", "analytics/ratings_over_time.csv"]:
        assert os.path.exists(os.path.join(out_dir, name)), name
    assert summary["data"]["n_users"] == 600
    assert 0 < summary["data"]["density"] < 1


def test_als_beats_baselines(artifacts):
    metrics = artifacts[1]["metrics"]
    assert metrics["als"]["rmse"] < metrics["bias"]["rmse"] < metrics["global_mean"]["rmse"]
    assert 0 <= metrics["ranking"]["precision_at_k"] <= 1


def test_history_matches_source(rec, artifacts):
    assert rec.hist_indptr[-1] == artifacts[1]["data"]["n_ratings"]
    history = rec.user_history(int(rec.user_ids[0]))
    assert len(history) >= 5
    assert history["rating"].between(0.5, 5.0).all()


def test_user_recommendations_exclude_seen(rec):
    user = int(rec.user_ids[3])
    seen = set(rec.user_history(user)["movieId"])
    recs = rec.recommend_for_user(user, n=20)
    assert len(recs) == 20
    assert not seen & set(recs["movieId"])
    assert recs["predicted_rating"].is_monotonic_decreasing


def test_filters(rec):
    user = int(rec.user_ids[0])
    recs = rec.recommend_for_user(user, n=10, genres=["Comedy"], min_ratings=20)
    assert recs["genres"].str.contains("Comedy").all()
    assert (recs["n_ratings"] >= 20).all()


def test_fold_in_agrees_with_trained_user(rec):
    """Folding in a real user's ratings should land near their trained vector."""
    user = int(rec.user_ids[10])
    history = rec.user_history(user)
    vec = rec.fold_in(dict(zip(history["movieId"], history["rating"])))
    trained = np.asarray(rec.user_factors[rec._user_pos[user]])
    cosine = vec @ trained / (np.linalg.norm(vec) * np.linalg.norm(trained))
    assert cosine > 0.9


def test_new_user_recommendations(rec):
    liked = rec.search("Horror", limit=5)["movieId"].tolist()
    ratings = {m: 5.0 for m in liked}
    recs = rec.recommend_from_ratings(ratings, n=10)
    assert len(recs) == 10
    assert not set(liked) & set(recs["movieId"])


def test_similar_movies(rec):
    movie = int(rec.movies.nlargest(1, "n_ratings")["movieId"].iat[0])
    sims = rec.similar_movies(movie, n=10)
    assert movie not in set(sims["movieId"])
    assert sims["similarity"].between(-1, 1.0001).all()
    assert sims["similarity"].is_monotonic_decreasing


def test_popular_and_search(rec):
    assert len(rec.popular(5)) == 5
    assert rec.search("zzz-no-such-movie").empty


def test_dashboard_pages_render(artifacts, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("KINDRED_ARTIFACTS", artifacts[0])
    app = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "dashboard.py")
    at = AppTest.from_file(app, default_timeout=60).run()
    assert not at.exception
    for page in at.sidebar.radio[0].options[1:]:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
