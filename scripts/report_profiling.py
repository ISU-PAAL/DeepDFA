"""Print reports from profiling the inference time and MACs"""

#%%
import json
import jsonlines
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--profiledata', type=str)
parser.add_argument('--timedata', type=str)
args = parser.parse_args()

profiledata = args.profiledata
timedata = args.timedata

#%%
if profiledata is not None:
    if profiledata.endswith(".jsonl"):
        data = list(jsonlines.open(profiledata))
    else:
        with open(profiledata) as f:
            data = json.load(f)


    flops = []
    macs = []
    for d in data:
        flopstr = d["flops"]
        count, unit = flopstr.split(" ")
        count = float(count)
        if unit == "G" or unit == "GMACS":
            count *= 1e9
        elif unit == "M" or unit == "MMACS":
            count *= 1e6
        elif unit == "K" or unit == "KMACS":
            count *= 1e6
        else:
            raise NotImplementedError
        flops.append(count)
        
        macstr = d["flops"]
        count, unit = macstr.split(" ")
        count = float(count)
        if unit == "G":
            count *= 1e9
        if unit == "M":
            count *= 1e6
        macs.append(count)
    flops = np.array(flops)
    macs = np.array(macs)
    examples = np.array([d["batch_size"] for d in data])

    print("gflops:", np.sum(flops)/1e9, "average:", np.sum(flops) / np.sum(examples)/1e9)
    print("gmacs:", np.sum(macs)/1e9, "average:", np.sum(macs) / np.sum(examples)/1e9)

#%%
if timedata is not None:
    if timedata.endswith(".jsonl"):
        time_data = list(jsonlines.open(timedata))
    else:
        with open(timedata) as f:
            time_data = json.load(f)
    runtime = np.array([d["runtime"] for d in time_data])
    examples = np.array([d["batch_size"] for d in time_data])
    print("average runtime:", np.sum(runtime) / np.sum(examples), "ms")
