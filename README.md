# Kindred

A movie recommender built as a big data analytics project. Kindred processes the
MovieLens 25M dataset (25 million ratings, 162,000 users, 59,000 movies) with
Apache Spark, trains a collaborative filtering model with Spark MLlib's ALS, and
serves the results in an interactive dashboard.

## What it does

| Stage | What happens | Code |
|---|---|---|
| Ingest and clean | Reads the CSVs with explicit schemas, removes invalid and duplicate ratings, filters inactive users and movies | `kindred/data.py` |
| Analytics | Spark SQL aggregations: rating distribution, activity per year, genre and decade statistics, the long tail, Bayesian weighted top movies | `kindred/analytics.py` |
| Model | ALS matrix factorisation, optional grid search over rank and regularisation | `kindred/model.py` |
| Evaluation | RMSE and MAE against two baselines, plus Precision@K, Recall@K, NDCG@K, hit rate and catalogue coverage | `kindred/model.py` |
| Export | User and movie factor matrices and rating history saved as numpy, parquet and CSV | `kindred/export.py` |
| Serving | Top-N for existing users, instant recommendations for new users (fold-in), similar movies, popularity fallback | `kindred/recommender.py` |
| Dashboard | Streamlit app with five pages | `app/dashboard.py` |

## Requirements

- Python 3.10 to 3.13
- Java 17 or 21 (Spark 4 needs it). Check with `java -version`, and make sure `JAVA_HOME` is set.
- About 8 GB of free RAM for the full 25M run

## Quick start (Windows)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python scripts\download_data.py
python -m kindred.pipeline
streamlit run app\dashboard.py
```

Or double-click `setup.bat` once, then `run_pipeline.bat`, then `run_dashboard.bat`.

For a fast first run, use the small dataset (100,000 ratings, finishes in about a minute):

```bat
python scripts\download_data.py --small
python -m kindred.pipeline --data-dir data\ml-latest-small --rank 20
```

No internet? `python scripts\make_sample_data.py` writes a synthetic dataset to
`data\sample` that the pipeline accepts in the same way.

## Pipeline options

Every setting in `kindred/config.py` is also a command-line flag:

```bat
python -m kindred.pipeline --rank 100 --reg-param 0.05 --max-iter 20
python -m kindred.pipeline --tune                    :: grid search first
python -m kindred.pipeline --sample-fraction 0.2     :: train on 20% of users
python -m kindred.pipeline --driver-memory 10g
python -m kindred.pipeline --no-refit-full           :: skip the final retrain
python -m kindred.pipeline --help
```

## How the model works

ALS learns one vector of `rank` numbers for every user and every movie, so that
the dot product of a user vector and a movie vector approximates the rating.
It alternates between two steps: fix the movie vectors and solve for the user
vectors, then fix the user vectors and solve for the movie vectors. Each step is
a set of independent least-squares problems, which is why it parallelises well
in Spark.

- **Existing user:** score every movie with a dot product, drop the ones already rated, keep the top N.
- **New user:** run one of those least-squares steps for just that user, using the few movies they rated (`Recommender.fold_in`). No retraining.
- **Similar movies:** cosine similarity between movie vectors.
- **No ratings at all:** fall back to the Bayesian weighted score, `(v/(v+m))·R + (m/(v+m))·C`.

## Outputs

`artifacts/` after a run:

```
summary.json            dataset stats, model settings, all metrics, timings
movies.parquet          one row per movie with counts, mean and weighted score
item_factors.npy        movie vectors          item_ids.npy   their movieIds
user_factors.npy        user vectors           user_ids.npy   their userIds
hist_*.npy              every user's rating history (CSR layout)
analytics/*.csv         the aggregates behind the dashboard charts
tuning.csv              grid search results (only with --tune)
```

## Tests

```bat
pytest
```

The tests generate a small synthetic dataset, run the whole Spark pipeline on it,
and check the outputs, the recommender logic and that every dashboard page renders.

## Troubleshooting

- **`JAVA_HOME is not set` or `Java gateway process exited`:** install JDK 17 or 21 and set `JAVA_HOME` to its folder.
- **`HADOOP_HOME and hadoop.home.dir are unset` warning:** safe to ignore. Kindred writes all outputs with pandas and numpy, so Spark never needs `winutils.exe`. If reading fails with a winutils error on your machine, download `winutils.exe` and `hadoop.dll` for Hadoop 3.4, put them in `C:\hadoop\bin`, and set `HADOOP_HOME=C:\hadoop`.
- **Out of memory:** lower `--rank`, pass `--sample-fraction 0.5`, or raise `--driver-memory`.

## Dataset

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and
Context. ACM Transactions on Interactive Intelligent Systems 5, 4.
https://grouplens.org/datasets/movielens/25m/
