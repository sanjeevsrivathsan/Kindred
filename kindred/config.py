"""Pipeline configuration. Every field can be overridden from the CLI."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Config:
    # --- paths -----------------------------------------------------------
    data_dir: str = "data/ml-25m"       # folder holding ratings.csv, movies.csv
    out_dir: str = "artifacts"          # where the pipeline writes its outputs

    # --- Spark -----------------------------------------------------------
    master: str = "local[*]"
    driver_memory: str = "6g"
    shuffle_partitions: int = 0         # 0 = 2 x CPU cores

    # --- data ------------------------------------------------------------
    sample_fraction: float = 1.0        # < 1.0 trains on a random sample of users
    min_user_ratings: int = 5           # drop users with fewer ratings than this
    min_movie_ratings: int = 5          # drop movies with fewer ratings than this
    test_fraction: float = 0.2
    seed: int = 42

    # --- ALS -------------------------------------------------------------
    rank: int = 50
    reg_param: float = 0.05
    max_iter: int = 15
    refit_full: bool = True             # retrain on all ratings before export

    # --- tuning ----------------------------------------------------------
    tune: bool = False
    tune_ranks: tuple = (20, 50, 100)
    tune_regs: tuple = (0.03, 0.05, 0.1)
    tune_fraction: float = 0.25         # share of users used during the grid search
    tune_max_iter: int = 10

    # --- evaluation ------------------------------------------------------
    eval_users: int = 2000              # users sampled for ranking metrics
    eval_k: int = 10
    relevance_threshold: float = 4.0    # a test rating >= this counts as "relevant"

    # --- analytics -------------------------------------------------------
    bayes_prior_votes: int = 0          # 0 = use the 90th percentile of rating counts

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tune_ranks"] = list(self.tune_ranks)
        d["tune_regs"] = list(self.tune_regs)
        return d
