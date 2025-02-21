import os
import argparse
import numpy as np
import xarray as xr

from src.metrics.phase import hilbert_decomposition
from src.util import get_dates
from tqdm import tqdm
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
# parser.add_argument("SURR", help="whether to run for surrogates", type=int)
args = parser.parse_args()

# Index of the session to be load
idx = args.SIDX
at = args.ALIGN
monkey = args.MONKEY
band_id = args.BAND
# surrogate = bool(args.SURR)

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


def create_generalized_surrogate(data, n_boot):
    count = 0
    trials_surr = []
    while count < n_boot:
        out = np.random.choice(
            range(data.sizes["trials"]),
            size=2,
            replace=True,
        )

        if out[0] == out[1]:
            continue
        else:
            trials_surr += [out]
            count = count + 1
    trials_surr = np.stack(trials_surr)

    count = 0
    channels_surr = []
    while count < n_boot:
        out = np.random.choice(
            range(data.sizes["roi"]),
            size=2,
            replace=True,
        )

        if out[0] == out[1]:
            continue
        else:
            channels_surr += [out]
            count = count + 1
    channels_surr = np.stack(channels_surr)

    data_surr = []
    for c_i, c_j, trial_i, trial_j in np.concatenate(
        (channels_surr, trials_surr), axis=1
    ):
        x = data[trial_i, c_i]
        y = data[trial_j, c_j][..., ::-1]
        data_surr += [np.stack((x, y))]

    data_surr = np.stack(data_surr)

    data_surr = xr.DataArray(
        data_surr, dims=data.dims, coords={"time": data.time.values}
    )

    data_surr.attrs["fsample"] = data.attrs["fsample"]

    return data_surr


# Shuffle data if needed
# if surrogate:
n_boot = 100
data_surr = [
    create_generalized_surrogate(data, data.sizes["trials"])
    for i in tqdm(range(n_boot))
]
# attrs = data.attrs
# data_surr = data.values.copy()
# data_surr = shuffle_along_axis(data_surr, 0)
# data = xr.DataArray(data_surr, dims=data.dims, coords=data.coords).drop_vars(
#    "trials"
# )
# data.attrs = attrs
# del data_surr


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


def get_filtered_data(data, bands, band_id):

    f_l, f_h = bands[band_id]

    temp = []

    temp = filter_data(data.values, data.fsample, f_l, f_h, n_jobs=10, verbose=False)
    temp = np.expand_dims(temp, 2)

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

    return data


# Original data
data = get_filtered_data(data, bands, band_id)
# Surrogates
data_surr_filt = [
    get_filtered_data(data_surr_, bands, band_id) for data_surr_ in tqdm(data_surr)
]


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

# For the surrogate save only the phase differences

phase_diff_surr = []

for data_surr_ in tqdm(data_surr_filt):
    # for s in range(epoch_data.sizes["epochs"]):
    _, _, temp = hilbert_decomposition(
        data_surr_,
        sfreq=data.fsample,
        decim=1,
        times="times",
        roi="roi",
        n_jobs=1,
        verbose=False,
    )

    phase_diff_surr += [temp.transpose(*_dims)]

phase_diff_surr = xr.concat(phase_diff_surr, "boot")


###########################################################################
# Saves file
###########################################################################

# Path in which to save coherence data
results_path = os.path.join(_SAVE, "Results", monkey, session_number)

if not os.path.exists(results_path):
    os.makedirs(results_path)

file_name = f"power_time_series_band_{band_id}_surr_False.nc"
path_pow = os.path.join(results_path, file_name)
power.to_netcdf(path_pow)

file_name = f"phase_time_series_band_{band_id}_surr_False.nc"
path_pow = os.path.join(results_path, file_name)
phase.to_netcdf(path_pow)

file_name = f"phase_difference_time_series_band_{band_id}_surr_False.nc"
path_pow = os.path.join(results_path, file_name)
phase_diff.to_netcdf(path_pow)

file_name = f"phase_difference_time_series_band_{band_id}_surr_True.nc"
path_pow = os.path.join(results_path, file_name)
phase_diff_surr.to_netcdf(path_pow)
