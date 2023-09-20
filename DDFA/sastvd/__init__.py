"""Set up project paths."""
import hashlib
import inspect
import os
import random
import string
import subprocess
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
import traceback

import pandas as pd
from tqdm import tqdm


def print_memory_mb():
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError as e:
        print("Error getting memory profile: {ex}".format(ex=e))


def threadsafe_mkdir(path):
    """
    https://python-security.readthedocs.io/vuln/os-makedirs-not-thread-safe.html
    https://bugs.python.org/issue21082#msg215346
    """
    try:
        path.mkdir(exist_ok=True, parents=True)
    except FileExistsError:
        traceback.print_exc()


def project_dir() -> Path:
    """Get project path."""
    return Path(__file__).parent.parent


def storage_dir() -> Path:
    """Get storage path."""
    storage = os.getenv("SINGSTORAGE")
    if storage:
        return Path(storage) / "storage"
    return Path(__file__).parent.parent / "storage"


def external_dir() -> Path:
    """Get storage external path."""
    path = storage_dir() / "external"
    threadsafe_mkdir(path)
    return path


def interim_dir() -> Path:
    """Get storage interim path."""
    path = storage_dir() / "interim"
    threadsafe_mkdir(path)
    return path


def processed_dir() -> Path:
    """Get storage processed path."""
    path = storage_dir() / "processed"
    threadsafe_mkdir(path)
    return path


def outputs_dir() -> Path:
    """Get output path."""
    path = storage_dir() / "outputs"
    threadsafe_mkdir(path)
    return path


def cache_dir() -> Path:
    """Get storage cache path."""
    path = storage_dir() / "cache"
    threadsafe_mkdir(path)
    return path


def get_dir(path) -> Path:
    """Get path, if exists. If not, create it."""
    threadsafe_mkdir(path)
    return path


def debug(msg, noheader=False, sep="\t"):
    """Print to console with debug information."""
    caller = inspect.stack()[1]
    file_name = caller.filename
    ln = caller.lineno
    now = datetime.now()
    time = now.strftime("%m/%d/%Y - %H:%M:%S")
    if noheader:
        print("\t\x1b[94m{}\x1b[0m".format(msg), end="")
        return
    print(
        '\x1b[40m[{}] File "{}", line {}\x1b[0m\n\t\x1b[94m{}\x1b[0m'.format(
            time, file_name, ln, msg
        )
    )


def gitsha():
    """Get current git commit sha for reproducibility."""
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode()
    )


def gitmessage():
    """Get current git commit sha for reproducibility."""
    m = subprocess.check_output(["git", "log", "-1", "--format=%s"]).strip().decode()
    return "_".join(m.lower().split())


def subprocess_cmd(
    command: str, verbose: int = 0, force_shell: bool = False, check_diff: bool = False
):
    """Run command line process.

    Example:
    subprocess_cmd('echo a; echo b', verbose=1)
    >>> a
    >>> b
    """
    singularity = os.getenv("SINGULARITY")
    if singularity != "true" and not force_shell:
        command = f"singularity exec {project_dir() / 'main.sif'} " + command
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    output = process.communicate()
    if verbose > 1:
        debug(output[0].decode())
        debug(output[1].decode())
    if check_diff:
        assert process.returncode in (
            1,
            0,
        ), f"git exited with code {process.returncode}"
    return output


def watch_subprocess_cmd(command: str, force_shell: bool = True):
    """Run subprocess and monitor output. Used for debugging purposes."""
    singularity = os.getenv("SINGULARITY")
    if singularity != "true" and not force_shell:
        command = f"singularity exec {project_dir() / 'main.sif'} " + command
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    # Poll process for new output until finished
    noheader = False
    while True:
        nextline = process.stdout.readline()
        if nextline == b"" and process.poll() is not None:
            break
        debug(nextline.decode(), noheader=noheader)
        noheader = True


def genid():
    """Generate random string."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=10))


def get_run_id(args=None):
    """Generate run ID."""
    if not args:
        ID = datetime.now().strftime("%Y%m%d%H%M_{}".format(gitsha()))
        return ID + "_" + gitmessage()
    ID = datetime.now().strftime(
        "%Y%m%d%H%M_{}_{}".format(
            gitsha(), "_".join([f"{v}" for _, v in vars(args).items()])
        )
    )
    return ID


def hashstr(s):
    """Hash a string."""
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10**8)


DFMP_WORKERS = 6


def dfmp(
    df, function, columns=None, ordr=True, workers=DFMP_WORKERS, cs=10, desc="Run: ", generator=False, total=None
):
    """Parallel apply function on dataframe.

    Example:
    def asdf(x):
        return x

    dfmp(list(range(10)), asdf, ordr=False, workers=6, cs=1)
    """
    if isinstance(columns, str):
        items = df[columns].tolist()
    elif isinstance(columns, list):
        items = df[columns].to_dict("records")
    elif isinstance(df, pd.DataFrame):
        items = df.to_dict("records")
    # elif isinstance(df, list):
    else:
        items = df
    # else:
    #     raise ValueError("First argument of dfmp should be pd.DataFrame or list.")

    it = _dfmp(function, items, ordr, workers, cs, desc, total)
    if generator:
        return it
    else:
        processed = []
        processed.extend(it)
        return processed

def _dfmp(function, items, ordr, workers, cs, desc, total):
    if desc is not None:
        desc = f"({workers} Workers) {desc}"
    with Pool(processes=workers) as p:
        map_func = getattr(p, "imap" if ordr else "imap_unordered")
        it = map_func(function, items, cs)
        if desc is not None:
            try:
                items_len = len(items)
            except:
                if total is not None:
                    items_len = total
                else:
                    items_len = None
            it = tqdm(it, total=items_len, desc=desc)
        yield from it


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
