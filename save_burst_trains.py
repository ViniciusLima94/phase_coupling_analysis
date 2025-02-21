import os
import argparse
import xarray as xr
import numpy as np

from src.util import get_dates
from config import freqs
from frites.conn.conn_utils import conn_links
from frites.utils import parallel_func

###############################################################################
# Argument parsing
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("SIDX", help="index of the session to be run", type=int)
parser.add_argument("ALIGN", help="wheter to align data to cue or match", type=str)
parser.add_argument("MONKEY", help="which monkey to use", type=str)
parser.add_argument("QLOW", help="lower quantile used to threshold", type=int)
parser.add_argument("QUP", help="upper quantile used to threshold", type=int)
args = parser.parse_args()

# Index of the session to be load
idx = args.SIDX
at = args.ALIGN
monkey = args.MONKEY
q_l = args.QLOW
q_u = args.QUP

session_number = get_dates(monkey)[idx]
print(session_number)


DATA_PATH = os.path.expanduser(
    f"/home/vinicius/funcog/phaseanalysis/Results/{monkey}/{session_number}"
)


# Root directory
_ROOT = os.path.expanduser("~/funcog/gda")
_SAVE = os.path.expanduser("~/funcog/phaseanalysis")


###############################################################################
# Function to compute power events coincidence
###############################################################################


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def _int(w, x_s, x_t, kw_para):

    # define the power envelope correlations
    def pairwise_int(w_x, w_y):
        # computes fraciton of events above threshold that intersect
        x = w[:, w_x, :, :]
        y = w[:, w_y, :, :]
        prod = x * y
        norm = np.max([x.sum(-1), y.sum(-1)], axis=0)
        norm = np.where(norm == 0, 1, norm)
        return prod

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(pairwise_int, **kw_para)

    # compute the single trial power envelope correlations
    return parallel(p_fun(s, t) for s, t in zip(x_s, x_t))


def power_events_coincidence(
    power, q_l, q_u=None, n_jobs=10, verbose=False, shuffle=False
):

    # Extract dimensions
    dims = power.dims
    trials, roi, freqs = power.trials.data, power.roi.data, power.freqs.data
    ntrials, nroi, nfreqs, ntimes = power.shape

    roi_gp = roi
    (x_s, x_t), roi_p = conn_links(roi_gp, {})
    n_pairs = len(x_s)

    quantiles = power.quantile(q_l, "times")

    z_power = (power >= quantiles).values

    if isinstance(q_u, float):
        quantiles = power.quantile(q_u, "times")
        z_power = np.logical_and(z_power, power < quantiles).values

    if shuffle:
        z_power = shuffle_along_axis(z_power, 0)

    kw_para = dict(n_jobs=n_jobs, verbose=verbose, total=n_pairs)

    pec = _int(z_power, x_s, x_t, kw_para)
    pec = np.stack(pec, axis=1)

    # conversion
    pec = xr.DataArray(
        pec,
        dims=dims,
        name="pec",
        coords={"trials": trials, "roi": roi_p, "freqs": freqs},
    )

    return pec


###############################################################################
# Compute for each band
###############################################################################


# Path in which to save coherence data
results_path = os.path.join(_SAVE, "Results", monkey, session_number)

if not os.path.exists(results_path):
    os.makedirs(results_path)


n_bands = len(freqs)

pec, pec_shuffle = [], []

# freqs = freqs[:3]

for band, freq in enumerate(freqs):

    print(f"Band {band + 1} of {n_bands} (f_c = {freq} Hz)")

    pec_file_name = f"burst_trains_band_{band}_{q_l}_{q_u}_surr_False.nc"
    pec_shuffle_file_name = f"burst_trains_band_{band}_{q_l}_{q_u}_surr_True.nc"

    power_time_series = xr.load_dataarray(
        os.path.join(DATA_PATH, f"power_time_series_band_{band}_surr_False.nc")
    )

    pec = power_events_coincidence(
        power_time_series, q_l / 100, q_u / 100, verbose=False
    )
    pec_shuffle = power_events_coincidence(
        power_time_series, q_l / 100, q_u / 100, shuffle=True, verbose=False
    )

    # print(pec.dims)
    # print(pec.freqs)
    # Concat frequencies
    # pec = pec.assign_coords({"freqs": freq})
    # pec_shuffle = pec_shuffle.assign_coords({"freqs": freq})

    ###########################################################################
    # Saves file
    ###########################################################################

    pec.to_netcdf(
        os.path.join(results_path, pec_file_name),
        # mode="a",
        # unlimited_dims=["freqs"],
        engine="h5netcdf",
    )
    pec_shuffle.to_netcdf(
        os.path.join(results_path, pec_shuffle_file_name),
        # mode="a",
        # unlimited_dims=["freqs"],
        engine="h5netcdf",
    )
