import argparse
import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.helpers.evaluate as ivde


def bigvul():
    """Run preperation scripts for BigVul dataset."""
    print(svdd.bigvul(sample=args.sample))
    ivde.get_dep_add_lines_bigvul("bigvul", sample=args.sample)
    # svdglove.generate_glove("bigvul", sample=args.sample)
    # svdd2v.generate_d2v("bigvul", sample=args.sample)
    print("success")


def devign():
    raise NotImplementedError
    print(svdd.devign(sample=args.sample))
    ivde.get_dep_add_lines("devign", sample=args.sample)
    svdglove.generate_glove("devign", sample=args.sample)
    svdd2v.generate_d2v("devign", sample=args.sample)
    print("success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare master dataframe")
    parser.add_argument("--sample", action="store_true", help="Extract a sample only")
    parser.add_argument("--global_workers", type=int, help="Number of workers to use")
    parser.add_argument("--dataset")
    args = parser.parse_args()

    if args.global_workers is not None:
        svd.DFMP_WORKERS = args.global_workers

    if args.dataset == "bigvul":
        bigvul()
    if args.dataset == "devign":
        devign()
