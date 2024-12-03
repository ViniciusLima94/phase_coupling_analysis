import os
import argparse
import numpy as np
import xarray as xr

from src.metrics.phase import hilbert_decomposition
from src.util import get_dates
from util import load_session_data
from mne.filter import filter_data

# from src.signal.surrogates import trial_swap_surrogates

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to be run", type=int)
parser.add_argument("ALIGN", help="wheter to align data to cue or match", type=str)
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument("SURR", help="whether to run for surrogates", type=int)
args = parser.parse_args()

# Index of the session to be load
idx = args.SIDX
at = args.ALIGN
monkey = args.MONKEY
surrogate = bool(args.SURR)

session_number = get_dates(monkey)[idx]
print(session_number)

# Root directory
_ROOT = os.path.expanduser("~/funcog/gda")
_SAVE = os.path.expanduser("~/Documents/phaseanalysis")

###########################################################################
# Loading session
###########################################################################

data = load_session_data(session_number, monkey, at)

print(data.shape)

# channels = ["a8M_17", "a1_103", "a7B_121", "a2_125", "a5_172",
# "a7A_181", "a7B_121", "a5_172"]

# idx = [roi in channels for roi in data.roi]

# data = data.isel(roi=idx)

# Shuffle data if needed
# if surrogate:

# channel_pairs = np.random.choice(range(data.sizes["roi"]), size=(n_boot, 2))
# trial_pairs = np.random.choice(range(data.sizes["trials"]), size=(n_boot, 2))
# shuffled = np.concatenate((channel_pairs, trial_pairs), axis=1)

#    data_surr = []
#
#    for i in range(10):
#        data_surr += [trial_swap_surrogates(data, seed=0, verbose=False)]

# for i, j, ti, tj in tqdm(shuffled):
# temp = xr.concat(
# (
# data.isel(trials=ti, roi=i).drop_vars("trials").drop_vars("roi"),
# data.isel(trials=tj, roi=j).drop_vars("trials").drop_vars("roi"),
# ),
# "roi",
# )
# data_surr += [
# temp
# ]

#    data = xr.concat(data_surr, "trials").transpose("trials", "roi", "time")
#
#    data.attrs["t_match_on"] = np.array(
#        [data.attrs["t_match_on"].mean()] * n_boot
#    )  # np.random.choice(data.attrs["t_match_on"], size=n_boot)
#    data.attrs["t_cue_on"] = np.array(
#        [data.attrs["t_cue_on"].mean()] * n_boot
#    )  # np.random.choice(data.attrs["t_cue_on"], size=n_boot)
#
#    del data_surr


###########################################################################
# Filter data
###########################################################################

# f_low = np.arange(0, 80, 10, dtype=np.int_)
# f_high = f_low + 10

# bands = np.c_[f_low, f_high]
# freqs = bands.mean(axis=1).astype(int)

# bands = np.array([[0, 10], [5, 15], [10, 20], [15, 25]])
bands = np.array([[0, 10], [5, 15]])
freqs = bands.mean(axis=1).astype(int)

temp = []

for f_l, f_h in bands:

    temp += [filter_data(data.values, data.fsample, f_l, f_h, n_jobs=10, verbose=False)]


data = xr.DataArray(
    np.stack(temp, axis=2),
    dims=("trials", "roi", "freqs", "times"),
    coords={
        "trials": data.trials,
        "roi": data.roi,
        "freqs": freqs,
        "times": data.time.values,
    },
    attrs=data.attrs,
)

del temp


_dims = ("trials", "roi", "freqs", "times")

power, phase, phase_diff = [], [], []

# for s in range(epoch_data.sizes["epochs"]):
power, phase, phase_diff = hilbert_decomposition(
    data,
    sfreq=data.fsample,
    decim=5,
    times="times",
    roi="roi",
    n_jobs=10,
    verbose=None,
)

power = power.transpose(*_dims)
phase = phase.transpose(*_dims)
phase_diff = phase_diff.transpose(*_dims)


###########################################################################
# Create epoched data
###########################################################################


# def create_epoched_data(data):

# t_match_on = (data.attrs["t_match_on"] - data.attrs["t_cue_on"]) / data.fsample
# t_match_on = np.round(t_match_on, 1)

# epoch_data = []

# for i in range(data.sizes["trials"]):
# stages = [
# [-0.4, 0.0],
# [0, 0.4],
# [0.5, 0.9],
# [0.9, 1.3],
# [t_match_on[i] - 0.4, t_match_on[i]],
# ]

# temp = []

# for t_i, t_f in stages:
# temp += [data[i].sel(times=slice(t_i, t_f)).values]


# # for _temp in temp:
# # print(_temp.shape)

# epoch_data += [np.stack(temp, axis=-3)]

# _dims = ("trials", "roi", "epochs", "freqs", "times")

# epoch_data = xr.DataArray(
# np.stack(epoch_data),
# dims=_dims,
# coords={
# "trials": data.trials,
# "roi": data.roi,
# "freqs": freqs,
# },
# attrs=data.attrs,
# )

# stim_labels = data.attrs["stim"]

# return epoch_data

# power = create_epoched_data(power)
# phase = create_epoched_data(phase)
# phase_diff = create_epoched_data(phase_diff)

###########################################################################
# Saves file
###########################################################################

# Path in which to save coherence data
results_path = os.path.join(_SAVE, "Results", monkey, session_number)

if not os.path.exists(results_path):
    os.makedirs(results_path)

file_name = f"power_time_series_surr_{surrogate}.nc"
path_pow = os.path.join(results_path, file_name)
power.to_netcdf(path_pow)

file_name = f"phase_time_series_surr_{surrogate}.nc"
path_pow = os.path.join(results_path, file_name)
phase.to_netcdf(path_pow)

file_name = f"phase_difference_time_series_surr_{surrogate}.nc"
path_pow = os.path.join(results_path, file_name)
phase_diff.to_netcdf(path_pow)
