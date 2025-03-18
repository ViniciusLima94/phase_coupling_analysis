import os
import argparse
import numpy as np
import xarray as xr

# from src.session import session
from src.util import get_dates
from src.metrics.spectral import conn_spec_average
from util import load_session_data

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to be run", type=int)
parser.add_argument("ALIGN", help="wheter to align data to cue or match", type=str)
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument("SURR", help="whether to compute surrogate or not", type=str)
args = parser.parse_args()

# Index of the session to be load
idx = args.SIDX
at = args.ALIGN
monkey = args.MONKEY
surr = bool(args.SURR)

session = get_dates(monkey)[idx]
print(session)

# Root directory
_ROOT = os.path.expanduser("~/funcog/gda")
_SAVE = os.path.expanduser("~/funcog/phaseanalysis")

###########################################################################
# Loading session
###########################################################################

data = load_session_data(session, monkey, at)

###########################################################################
# Surrogate data if needed
###########################################################################


def resample_trials(X):
    T, R, N = X.shape  # Extract dimensions
    # Generate a shuffled set of trial indices for each row
    sampled_trials = np.array([np.random.permutation(T) for _ in range(T)])

    # Create the new array where each ROI gets data from a different trial
    resampled_X = np.zeros((T, R, N))  # Placeholder for sampled data

    for row in range(T):
        trial_indices = sampled_trials[row]  # Get shuffled trials for this row
        for roi in range(R):
            resampled_X[row, roi, :] = X[trial_indices[roi], roi, :]

    return resampled_X


if surr:

    data_surr = xr.DataArray(
        resample_trials(data),
        dims=("trials", "roi", "time"),
        coords={"time": data.time.values, "roi": data.roi.values},
    )

    data_surr.attrs["fsample"] = data.attrs["fsample"]

    data = data_surr

    del data_surr

###########################################################################
# Create epoched data
###########################################################################

t_match_on = (data.attrs["t_match_on"] - data.attrs["t_cue_on"]) / data.fsample
t_match_on = np.round(t_match_on, 1)

epoch_data = []

for i in range(data.sizes["trials"]):
    stages = [
        [-0.4, 0.0],
        [0, 0.4],
        [0.5, 0.9],
        [0.9, 1.3],
        [t_match_on[i] - 0.4, t_match_on[i]],
    ]

    temp = []

    for t_i, t_f in stages:
        temp += [data[i].sel(time=slice(t_i, t_f)).data]

    epoch_data += [np.stack(temp, axis=-2)]

epoch_data = xr.DataArray(
    np.stack(epoch_data),
    dims=("trials", "roi", "epochs", "time"),
    coords={
        "trials": data.trials,
        "roi": data.roi,
    },
    attrs=data.attrs,
)

stim_labels = epoch_data.attrs["stim"]

coh = []
for i in range(epoch_data.sizes["epochs"]):
    coh_stim = []
    for stim in np.unique(stim_labels):
        coh_stim += [
            conn_spec_average(
                epoch_data.sel(epochs=i).isel(trials=stim_labels == stim),
                fmin=0.1,
                fmax=80,
                roi="roi",
                n_jobs=10,
                bandwidth=5,
            )
        ]
    coh += [xr.concat(coh_stim, "stim")]

coh = xr.concat(coh, "epochs")
coh.attrs = data.attrs

###########################################################################
# Saves file
###########################################################################

# Path in which to save coherence data
results_path = os.path.join(_SAVE, "Results", monkey, session)

if not os.path.exists(results_path):
    os.makedirs(results_path)

file_name = "average_coherence.nc"
path_coh = os.path.join(results_path, file_name)
coh.to_netcdf(path_coh)
