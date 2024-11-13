import os
import argparse
import numpy as np
import xarray as xr

from src.session import session
from tqdm import tqdm
from src.util import get_dates
from src.metrics.spectral import xr_psd_array_multitaper
from util import load_session_data
import scipy

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

sxx = xr_psd_array_multitaper(data.sel(time=slice(-.5, 1.5)), n_jobs=20, bandwidth=5)

###########################################################################
# Saves file
###########################################################################

# Path in which to save coherence data
results_path = os.path.join(_SAVE, "Results", monkey, session)

if not os.path.exists(results_path):
    os.makedirs(results_path)

file_name = f"average_power_whole_trial.nc"
path_pow = os.path.join(results_path, file_name)
sxx.to_netcdf(path_pow)
