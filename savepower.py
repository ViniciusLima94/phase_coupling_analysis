import os
import argparse
import numpy as np
import xarray as xr

# from src.session import session
from src.util import get_dates
from src.metrics.spectral import xr_psd_array_multitaper
from util import load_session_data

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to be run", type=int)
parser.add_argument("ALIGN", help="wheter to align data to cue or match", type=str)
parser.add_argument("MONKEY", help="which monkey to use", type=str)
args = parser.parse_args()

# Index of the session to be load
idx = args.SIDX
at = args.ALIGN
monkey = args.MONKEY

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

sxx = []
for i in range(epoch_data.sizes["epochs"]):
    sxx_stim = []
    for stim in np.unique(stim_labels):
        sxx_stim += [
            xr_psd_array_multitaper(
                epoch_data.sel(epochs=i).isel(trials=stim_labels == stim),
                n_jobs=20,
                bandwidth=5,
            )
        ]
    sxx += [xr.concat(sxx_stim, "stim")]

sxx = xr.concat(sxx, "epochs")
sxx.attrs = data.attrs

###########################################################################
# Saves file
###########################################################################

# Path in which to save coherence data
results_path = os.path.join(_SAVE, "Results", monkey, session)

if not os.path.exists(results_path):
    os.makedirs(results_path)

file_name = "average_power.nc"
path_pow = os.path.join(results_path, file_name)
sxx.to_netcdf(path_pow)
