import os
import argparse
import numpy as np
import xarray as xr

from src.metrics.phase import hilbert_decomposition
from src.util import get_dates
from util import load_session_data
from mne.filter import filter_data
from config import bands, freqs

# from src.signal.surrogates import trial_swap_surrogates

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to be run", type=int)
parser.add_argument("ALIGN", help="wheter to align data to cue or match", type=str)
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument("BAND", help="which band to use", type=int)
parser.add_argument("SURR", help="whether to run for surrogates", type=int)
args = parser.parse_args()

# Index of the session to be load
idx = args.SIDX
at = args.ALIGN
monkey = args.MONKEY
band_id = args.BAND
surrogate = bool(args.SURR)

session_number = get_dates(monkey)[idx]
print(session_number)

# Root directory
_ROOT = os.path.expanduser("~/funcog/gda")
_SAVE = os.path.expanduser("~/funcog/phaseanalysis")

###########################################################################
# Loading session
###########################################################################

data = load_session_data(session_number, monkey, at)

print(data.shape)


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


# Shuffle data if needed
if surrogate:
    attrs = data.attrs
    data_surr = data.values.copy()
    data_surr = shuffle_along_axis(data_surr, 0)
    data = xr.DataArray(data_surr, dims=data.dims, coords=data.coords).drop_vars(
        "trials"
    )
    data.attrs = attrs
    del data_surr


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

f_l, f_h = bands[band_id]

temp = []

temp = filter_data(data.values, data.fsample, f_l, f_h, n_jobs=10, verbose=False)
temp = np.expand_dims(temp, 2)
print(temp.shape)
print(freqs)

data = xr.DataArray(
    temp,
    # np.stack(temp, axis=2),
    dims=("trials", "roi", "freqs", "times"),
    coords={
        "trials": data.trials,
        "roi": data.roi,
        "freqs": freqs[band_id],
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
    decim=1,
    times="times",
    roi="roi",
    n_jobs=10,
    verbose=None,
)

power = power.transpose(*_dims)
phase = phase.transpose(*_dims)
phase_diff = phase_diff.transpose(*_dims)


###########################################################################
# Saves file
###########################################################################

# Path in which to save coherence data
results_path = os.path.join(_SAVE, "Results", monkey, session_number)

if not os.path.exists(results_path):
    os.makedirs(results_path)

file_name = f"power_time_series_band_{band_id}_surr_{surrogate}.nc"
path_pow = os.path.join(results_path, file_name)
power.to_netcdf(path_pow)

file_name = f"phase_time_series_band_{band_id}_surr_{surrogate}.nc"
path_pow = os.path.join(results_path, file_name)
phase.to_netcdf(path_pow)

file_name = f"phase_difference_time_series_band_{band_id}_surr_{surrogate}.nc"
path_pow = os.path.join(results_path, file_name)
phase_diff.to_netcdf(path_pow)
