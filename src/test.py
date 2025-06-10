import argparse
from aim import Repo
import polars as pl


def process_run(run_hash: str):
    print(f"Processing run with hash: {run_hash}")
    repo = Repo(path=".")
    run = repo.get_run(run_hash)

    train_dfs = []
    val_dfs = []

    for metric in run.metrics():
        try:
            subset = metric.context["subset"]
            track = metric.context["track"]
            name = f"{metric.name}_{track}"
            df = metric.dataframe()
            print(name, subset, track)

            df_pl = pl.DataFrame({"step": df["step"], name: df["value"]})

            # Compute running average for `_current` metrics
            if name.endswith("_current"):
                running_avg_name = name.replace("_current", "_running_avg")
                df_pl = df_pl.with_columns(
                    (df_pl[name].cum_sum() / (pl.arange(1, df_pl.height + 1))).alias(running_avg_name)
                )

            if subset == "train":
                train_dfs.append(df_pl)
            else:
                val_dfs.append(df_pl)
        except KeyError:
            continue

    def merge_all(dfs):
        if not dfs:
            return None
        df = dfs[0]
        for d in dfs[1:]:
            df = df.join(d, on="step", how="inner")
        return df.sort("step")

    train_df = merge_all(train_dfs)
    val_df = merge_all(val_dfs)

    return train_df, val_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process one or more runs by their hashes.")
    parser.add_argument("run_hash", type=str, nargs="+", help="The hash(es) of the run(s) to process.")
    args = parser.parse_args()

    all_train_dfs = []
    all_val_dfs = []

    for run_hash in args.run_hash:
        train_df, val_df = process_run(run_hash)
        if train_df is not None:
            print(f"Concat train df with columns {train_df.columns}")
            all_train_dfs.append(train_df)
        if val_df is not None:
            print(f"Concat val df with columns {val_df.columns}")
            all_val_dfs.append(val_df)

    train_df = all_train_dfs[0] if all_train_dfs else []
    for current_df in all_train_dfs[1:]:
        try:
            train_df.vstack(current_df, in_place=True)
            print(f"Current state of train_df: {train_df}\n")
        except Exception as e:
            print(e)
            pass
    train_df[::10_000].write_csv("train.csv", separator=",")

    val_df = all_val_dfs[0] if all_val_dfs else []
    for current_df in all_val_dfs[1:]:
        try:
            val_df.vstack(current_df, in_place=True)
            print(f"Current state of val_df: {val_df}\n")
        except Exception as e:
            print(e)
            pass
    val_df[::].write_csv("val.csv", separator=",")
