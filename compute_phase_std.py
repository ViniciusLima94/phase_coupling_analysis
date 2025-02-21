import os
import argparse
import xarray as xr
import numpy as np
import scipy

from src.util import get_dates
from config import freqs

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to be run", type=int)
parser.add_argument("ALIGN", help="wheter to align data to cue or match", type=str)
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument("QLOW", help="lower quantile used to threshold", type=int)
parser.add_argument("QUP", help="upper quantile used to threshold", type=int)
parser.add_argument("SURR", help="upper quantile used to threshold", type=int)
args = parser.parse_args()

# Index of the session to be load
idx = args.SIDX
at = args.ALIGN
monkey = args.MONKEY
q_l = args.QLOW
q_u = args.QUP
surr = bool(args.SURR)

n_bands = len(freqs)

session_number = get_dates(monkey)[idx]
print(session_number)


DATA_PATH = os.path.expanduser(
    f"/home/vinicius/funcog/phaseanalysis/Results/{monkey}/{session_number}"
)


# Root directory
_ROOT = os.path.expanduser("~/funcog/gda")
_SAVE = os.path.expanduser("~/funcog/phaseanalysis")


###############################################################################
# Load data
###############################################################################

# Path in which to save coherence data
results_path = os.path.join(_SAVE, "Results", monkey, session_number)

data_path = os.path.join(_SAVE, "Results", monkey, session_number)


for band, freq in enumerate(freqs):

    print(f"Band {band + 1} of {n_bands} (f_c = {freq} Hz)")

    pec_file_name = f"phase_std_band_{band}_{q_l}_{q_u}_surr_{surr}.nc"

    pec = xr.load_dataarray(
        os.path.join(data_path, f"burst_trains_band_{band}_{q_l}_{q_u}_surr_{surr}.nc")
    )

    # Load time series of phase differences for data and surrogate
    phi_series = xr.load_dataarray(
        os.path.join(
            DATA_PATH, f"phase_difference_time_series_band_{band}_surr_{surr}.nc"
        )
    )

    print(phi_series.dtype)

    # Get phase only for coincident events
    filtered_phi_series = xr.DataArray(
        np.where(~pec, np.nan, phi_series),
        dims=phi_series.dims,
        coords=phi_series.coords,
    )

    size = (~(filtered_phi_series.squeeze().isnull())).sum(("times", "trials"))

    phi_max = np.sqrt(np.log(size))

    std = scipy.stats.circstd(filtered_phi_series, axis=(0, 3), nan_policy="omit")

    std = xr.DataArray(std, dims=("roi", "freqs"), coords=(pec.roi.values, freq))

    # phi = 1 - std / phi_max
    # phi_shuffle = 1 - std_shuffle / phi_max_shuffle

    ###########################################################################
    # Saves file
    ###########################################################################

    std.to_netcdf(
        os.path.join(results_path, pec_file_name),
        # mode="a",
        # unlimited_dims=["freqs"],
        engine="h5netcdf",
    )

    del pec, phi_series
