import os
import argparse

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

sid = get_dates(monkey)[idx]
print(sid)

# Root directory
_ROOT = os.path.expanduser("~/funcog/gda")
_SAVE = os.path.expanduser("~/Documents/phaseanalysis")

###########################################################################
# Loading session
###########################################################################

data = load_session_data(sid, monkey, at)

###########################################################################
# Create epoched data
###########################################################################

sxx = xr_psd_array_multitaper(data.sel(time=slice(-0.5, 1.5)), n_jobs=20, bandwidth=5)

###########################################################################
# Saves file
###########################################################################

# Path in which to save coherence data
results_path = os.path.join(_SAVE, "Results", monkey, sid)

if not os.path.exists(results_path):
    os.makedirs(results_path)

file_name = "average_power_whole_trial.nc"
path_pow = os.path.join(results_path, file_name)
sxx.to_netcdf(path_pow)
