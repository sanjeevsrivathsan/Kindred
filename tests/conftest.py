import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    """Generate a small dataset and run the whole Spark pipeline on it once."""
    from make_sample_data import generate

    from kindred.config import Config
    from kindred.pipeline import run

    base = tmp_path_factory.mktemp("kindred")
    data_dir = generate(str(base / "data"), n_users=600, n_movies=300, per_user=50)
    cfg = Config(
        data_dir=data_dir, out_dir=str(base / "artifacts"), rank=8, max_iter=8,
        driver_memory="1g", shuffle_partitions=4, eval_users=200,
    )
    summary = run(cfg, log=lambda *_: None)
    return cfg.out_dir, summary


@pytest.fixture(scope="session")
def rec(artifacts):
    from kindred.recommender import Recommender

    return Recommender(artifacts[0])
