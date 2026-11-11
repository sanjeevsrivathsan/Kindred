"""Spark session factory."""
from __future__ import annotations

import os
import sys
import warnings

from pyspark.sql import SparkSession

from .config import Config


def build_session(cfg: Config, app_name: str = "Kindred") -> SparkSession:
    # On Windows, Spark looks for an executable called "python3" unless told
    # otherwise. Point it at the interpreter that is running this script.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    warnings.filterwarnings("ignore", category=FutureWarning, module="pyspark")

    partitions = cfg.shuffle_partitions or 2 * (os.cpu_count() or 4)
    spark = (
        SparkSession.builder.appName(app_name)
        .master(cfg.master)
        .config("spark.driver.memory", cfg.driver_memory)
        .config("spark.driver.maxResultSize", "0")
        .config("spark.sql.shuffle.partitions", str(partitions))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
