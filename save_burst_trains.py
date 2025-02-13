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
parser.add_argument("QLOW", help="lower quantile used to threshold", type=float)
parser.add_argument("QUP", help="upper quantile used to threshold", type=float)
args = parser.parse_args()

# Index of the session to be load
idx = args.SIDX
at = args.ALIGN
monkey = args.MONKEY
band_id = args.BAND
q_l = bool(args.QLOW)
q_u = bool(args.QU)

session_number = get_dates(monkey)[idx]
print(session_number)


DATA_PATH = os.path.expanduser(
    f"/home/vinicius/funcog/phaseanalysis/Results/{monkey}/{session_number}"
)


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
    power, q_l, q_u=None, n_jobs=1, verbose=False, shuffle=False
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


for band, freq in enumerate(freqs):

    power_time_series = xr.load_dataarray(
        os.path.join(DATA_PATH, f"power_time_series_band_{band}_surr_False.nc")
    )

    pec = power_events_coincidence(power_time_series, q_l, q_u)  # noqa
    pec_shuffle = power_events_coincidence(power_time_series, q_l, q_u)  # noqa
