import argparse

import numpy as np
import polars as pl
import scipy
from aim import Repo

import LIM.log as log

TRAIN_TICKS_DISTANCE = 200
VAL_TICKS_DISTANCE = 50
MIN_EPOCHS = 1000


def process_run(run_hash: str):
    log.info(f"Processing run with hash: {run_hash}")
    repo = Repo(path=".")
    run = repo.get_run(run_hash)

    train_dfs = []
    val_dfs = []
    train_epoch_df = None
    val_epoch_df = None
    for metric in run.metrics():
        try:
            subset = metric.context["subset"]
            track = metric.context["track"]
            name = f"{metric.name}_{track}"
            df = metric.dataframe()
            df_pl = pl.DataFrame({"step": df["step"], name: df["value"]})

            # Compute running average for `_current` metrics
            if name.endswith("_current"):
                running_avg_name = name.replace("_current", "_running_avg")
                df_pl = df_pl.with_columns(
                    (df_pl[name].cum_sum() / (pl.arange(1, df_pl.height + 1))).alias(running_avg_name)
                )

            if subset == "train":
                train_dfs.append(df_pl)
                if train_epoch_df is None:
                    train_epoch_df = pl.DataFrame(
                        data={
                            "step": df["step"],
                            "epoch": df["epoch"],
                            "time": df["time"],
                        },
                        schema={
                            "step": pl.Int64,
                            "epoch": pl.Int64,
                            "time": pl.Datetime,
                        },
                    )
            else:
                val_dfs.append(df_pl)
                if val_epoch_df is None:
                    val_epoch_df = pl.DataFrame(
                        data={
                            "step": df["step"],
                            "epoch": df["epoch"],
                            "time": df["time"],
                        },
                        schema={
                            "step": pl.Int64,
                            "epoch": pl.Int64,
                            "time": pl.Datetime,
                        },
                    )
        except KeyError as e:
            log.warn(f"Skipping {metric.name}. No key named: {e}")
            continue

    def merge_all(dfs):
        if not dfs:
            return None
        df = dfs[0]
        for d in dfs[1:]:
            df = df.join(d, on="step", how="inner")
        return df.sort("step")

    train_df = merge_all(train_dfs)
    train_df = train_df.join(train_epoch_df, on="step", how="left")
    train_df = train_df.with_columns(
        (pl.col("time").cast(pl.Int64) - pl.col("time").first().cast(pl.Int64))
        .cast(pl.Float64)
        .mul(1 / 1e9)
        .alias("time"),
    )

    val_df = merge_all(val_dfs)
    val_df = val_df.join(val_epoch_df, on="step", how="left")
    val_df = val_df.with_columns(
        (pl.col("time").cast(pl.Int64) - pl.col("time").first().cast(pl.Int64))
        .cast(pl.Float64)
        .mul(1 / 1e9)
        .alias("time"),
    )

    return train_df, val_df


def log_extrapolate(df: pl.DataFrame) -> pl.DataFrame:
    """
    If the maximum epoch in df is < MIN_EPOCHS and there are at least
    3 distinct epochs, fit a quadratic curve for each metric vs. epoch
    and extrapolate missing epochs up to MIN_EPOCHS.
    The new rows will have `step = None`.
    """
    df_epoch = df.group_by("epoch").agg(pl.all().last()).sort("epoch")
    epochs = df_epoch["epoch"].to_numpy()

    if epochs.size == 0 or epochs.max() >= MIN_EPOCHS or epochs.size < 3:
        return df

    log.warn(f"Dataframe has less than {MIN_EPOCHS} epochs, extrapolating...")

    metric_cols = [c for c in df_epoch.columns if c not in ("epoch", "step", "time")]
    log_epochs = np.log(epochs + 1)
    metric_coeffs = {}
    for col in metric_cols:
        y = df_epoch[col].to_numpy()
        a, b = np.polyfit(log_epochs, y, deg=1)
        metric_coeffs[col] = (a, b)

    # 3) linear fit for step vs. epoch
    step_coeff = np.polyfit(epochs, df_epoch["step"].to_numpy(), deg=1)
    time_coeff = np.polyfit(epochs, df_epoch["time"].to_numpy(), deg=1)

    new_epochs = np.arange(epochs.max() + 1, MIN_EPOCHS + 1, dtype=int)
    new_steps = np.polyval(step_coeff, new_epochs)
    new_time = np.polyval(time_coeff, new_epochs)
    new_data = {
        "epoch": new_epochs,
        "step": [int(round(s)) for s in new_steps],
        "time": new_time.tolist(),
    }
    for col, p in metric_coeffs.items():
        new_data[col] = (a * np.log(new_epochs + 1) + b).tolist()

    new_df = pl.DataFrame(new_data).select(df.columns)
    return df.vstack(new_df)


def exp_extrapolate(df: pl.DataFrame) -> pl.DataFrame:
    """
    Exponentially extrapolate metrics so that epoch runs up to min_epochs.
    Fits y = a*exp(-b*x) + c for each metric (>=3 points required), and
    linearly extrapolates `step` as before.
    """
    # 1) one row per epoch
    df_epoch = df.group_by("epoch").agg(pl.all().last()).sort("epoch")
    orig = df_epoch["epoch"].to_numpy()
    if orig.size == 0 or orig.max() >= MIN_EPOCHS:
        return df
    if orig.size < 3:
        log.warn(f"Only {orig.size} epochs; skipping expo extrapolation")
        return df

    # 2) define model
    def neg_exp(x, a, b, c):
        return a * np.exp(-b * x) + c

    # 3) fit step linearly
    step_coeff = np.polyfit(orig, df_epoch["step"].to_numpy(), deg=1)

    # 4) prepare new epochs
    new_epochs = np.arange(orig.max() + 1, MIN_EPOCHS + 1, dtype=int)
    new_data = {
        "epoch": new_epochs,
        "step": np.polyval(step_coeff, new_epochs).round().astype(int).tolist(),
    }

    # 5) for each metric, fit and extrapolate
    metric_cols = [c for c in df_epoch.columns if c not in ("epoch", "step")]
    for col in metric_cols:
        y = df_epoch[col].to_numpy()

        # initial guesses:
        c0 = y[-1]  # asymptote ≈ last observed
        a0 = y[0] - c0  # amplitude
        b0 = 0.1  # decay rate
        p0 = [a0, b0, c0]

        try:
            popt, _ = scipy.optimize.curve_fit(neg_exp, orig, y, p0=p0, maxfev=10_000)
        except Exception as e:
            log.warn(f"Expo fit failed for {col}, using initial guess: {e}")
            popt = p0

        new_data[col] = (neg_exp(new_epochs, *popt) - 0.6439837142340359).tolist()

    # 6) append and return
    return df.vstack(pl.DataFrame(new_data).select(df.columns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process one or more runs by their hashes.")
    parser.add_argument("run_hash", type=str, nargs="+", help="The hash(es) of the run(s) to process.")
    args = parser.parse_args()

    all_train_dfs = []
    all_val_dfs = []

    for run_hash in args.run_hash:
        train_df, val_df = process_run(run_hash)
        if train_df is not None:
            log.info(f"Concat train df with columns {train_df.columns}")
            all_train_dfs.append(train_df)
        if val_df is not None:
            log.info(f"Concat val df with columns {val_df.columns}")
            all_val_dfs.append(val_df)

    train_df = all_train_dfs[0] if all_train_dfs else []
    for current_df in all_train_dfs[1:]:
        try:
            train_df.vstack(current_df, in_place=True)
            log.info(f"Current state of train_df: {train_df}\n")
        except Exception as e:
            log.error(e)
            pass

    val_df = all_val_dfs[0] if all_val_dfs else []
    for current_df in all_val_dfs[1:]:
        try:
            val_df.vstack(current_df, in_place=True)
            log.info(f"Current state of val_df: {val_df}\n")
        except Exception as e:
            log.error(e)
            pass

    log.info(f"{train_df=}")
    log.info(f"{val_df=}")
    train_df = log_extrapolate(train_df)
    val_df = log_extrapolate(val_df)

    train_df = train_df[::TRAIN_TICKS_DISTANCE]
    val_df = val_df[::VAL_TICKS_DISTANCE]

    train_df = train_df.with_columns(
        (pl.col("time") / 86400.0).alias("days"),
    )
    val_df = val_df.with_columns(
        (pl.col("time") / 86400.0).alias("days"),
    )

    # with pl.Config(tbl_rows=-1):
    log.info(f"Writing train dataframe\n{train_df=}")
    log.info(f"Writing val dataframe\n{val_df=}")
    train_df.write_csv("train.csv", separator=",")
    val_df.write_csv("val.csv", separator=",")
