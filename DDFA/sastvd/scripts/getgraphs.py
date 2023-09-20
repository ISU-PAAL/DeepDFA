import functools
from multiprocessing import Pool
import os
import traceback

import numpy as np
import tqdm
import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.helpers.joern as svdj
import sastvd.helpers.joern_session as svdjs


def write_file(row):
    # Write C Files
    savedir_before = svd.get_dir(svd.processed_dir() / row["dataset"] / "before")
    fpath1 = savedir_before / f"{row['id']}.c"
    with open(fpath1, "w") as f:
        f.write(row["before"])

    if row["dataset"] == "bigvul":
        savedir_after = svd.get_dir(svd.processed_dir() / row["dataset"] / "after")
        fpath2 = savedir_after / f"{row['id']}.c"
        if len(row["diff"]) > 0:
            with open(fpath2, "w") as f:
                f.write(row["after"])
    else:
        fpath2 = None

    return fpath1, fpath2


def preprocess(row, fn):
    """Parallelise svdj functions.

    Example:
    df = svdd.bigvul()
    row = df.iloc[180189]  # PAPER EXAMPLE
    row = df.iloc[177860]  # EDGE CASE 1
    preprocess(row)
    """
    try:
        # if row["dataset"] == "bigvul":
        fpath1, fpath2 = write_file(row)

        # Run Joern on "before" code
        if args.overwrite or not os.path.exists(f"{fpath1}.edges.json"):
            fn(filepath=fpath1, verbose=args.verbose)
        elif args.verbose > 0:
            print("skipping", fpath1)

        # Run Joern on "after" code
        if args.overwrite or (row["dataset"] == "bigvul" and len(row["diff"]) > 0 and not os.path.exists(f"{fpath2}.edges.json")):
            fn(filepath=fpath2, verbose=args.verbose)
        elif args.verbose > 0:
            print("skipping", fpath2)
    except Exception:
        with open("failed_joern.txt", "a") as f:
            print(f"ERROR {row['id']}: {traceback.format_exc()}\ndata={row}", file=f)


def test_preprocess():
    """
    test that preprocessing progresses alright
    """
    row = {}
    result = preprocess(row)
    print(f"{result}")


def preprocess_whole_df_split(t):
    """
    preprocess one split of the dataframe
    """
    i, split = t
    with open(f"hpc/logs/getgraphs_output_{i}.joernlog", "wb") as lf:
        sess = svdjs.JoernSession(f"getgraphs/{i}", logfile=lf, clean=True)
        sess.import_script("get_func_graph")
        try:
            fn = functools.partial(
                svdj.run_joern_sess,
                sess=sess,
                verbose=args.verbose,
                export_json=True,
                export_cpg=True,
                export_dataflow=True,
            )
            items = split.to_dict("records")
            position = 0 if not isinstance(i, int) else int(i)
            for row in tqdm.tqdm(items, desc=f"(worker {i})", position=position):
                preprocess(row, fn)
        finally:
            sess.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",
        # choices=["bigvul", "devign", "sard"]
    )
    parser.add_argument("--job_array_number", type=int)
    parser.add_argument("--num_jobs", default=100, type=int)
    parser.add_argument("--partition")
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--run_sast", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--sess", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--file_only", action="store_true")
    args = parser.parse_args()

    df = svdd.ds(args.dataset, sample=args.sample)
    if args.partition is not None:
        df = svdd.ds_partition(df, args.partition, args.dataset)

    if args.file_only:

        def write_file_pair(row):
            i, row = t
            write_file(row)

        with Pool(args.workers) as pool:
            for _ in tqdm.tqdm(
                pool.imap_unordered(write_file, df.iterrows()), total=len(df)
            ):
                pass

    # Read Data
    if args.sample:
        args.verbose = 4

    if args.job_array_number is None:
        if args.workers == 1:
            preprocess_whole_df_split(("all", df))
        else:
            splits = np.array_split(df, args.workers)
            svd.dfmp(enumerate(splits), preprocess_whole_df_split, ordr=False, workers=args.workers, cs=1)

    elif args.sess:
        splits = np.array_split(df, args.num_jobs)
        my_split = splits[args.job_array_number]
        print("processing", my_split)
        preprocess_whole_df_split((args.job_array_number, my_split))
    else:
        splits = np.array_split(df, args.num_jobs)
        split_number = args.job_array_number
        df = splits[split_number]
        svd.dfmp(
            df,
            functools.partial(preprocess, fn=svdj.run_joern),
            ordr=False,
            workers=args.workers,
        )
