import os import argparse
import numpy as np
import xarray as xr

from scipy.signal import find_peaks
from src.util import get_dates
from tqdm import tqdm
from frites.conn import conn_links

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

session_number = get_dates(monkey)[idx]
print(session_number)

# Root directory
_ROOT = os.path.expanduser("~/funcog/gda")
_SAVE = os.path.expanduser("~/funcog/phaseanalysis")


###############################################################################
# Functions
###############################################################################
def detect_peak_frequencies(power=None, prominence=0.01, verbose=False):

    assert power.ndim == 2
    assert isinstance(power, xr.DataArray)

    roi, freqs = power.roi.data, power.freqs.data
    n_roi = len(roi)

    rois = []
    peak_freqs = []
    peak_prominences = []

    __iter = range(n_roi)
    for i in tqdm(__iter) if verbose else __iter:
        peak_index, peak_info = find_peaks(power[i, :], prominence=prominence)
        peak_freqs += [freqs[peak_index]]
        peak_prominences += [peak_info["prominences"]]
        rois += [[roi[i]] * len(peak_index)]

    return peak_freqs, peak_prominences, rois


def check_peaks(peak_freqs, peak_prominences, rois):

    has_peak = np.zeros((average_power_norm.sizes["roi"], len(bands)), dtype=bool)

    for i in tqdm(range(average_power_norm.sizes["roi"])):
        for peak in peak_freqs[i]:
            for n_band, band in enumerate(bands.keys()):
                if not has_peak[i, n_band]:
                    has_peak[i, n_band] = bands[band][0] <= peak <= bands[band][1]

    has_peak = xr.DataArray(
        has_peak,
        dims=("roi", "bands"),
        coords=(average_power_norm.roi, list(bands.keys())),
    )

    peak_freqs = xr.DataArray(
        np.hstack(peak_freqs),
        dims="roi",
        coords={"roi": np.hstack(rois)},
        name="peak_freq",
    )

    peak_prominences = xr.DataArray(
        np.hstack(peak_prominences),
        dims="roi",
        coords={"roi": np.hstack(rois)},
        name="peak_prom",
    )

    return has_peak, peak_freqs, peak_prominences


###############################################################################
# Load data
###############################################################################
DATA_PATH = os.path.join(_SAVE, "Results", monkey, session_number)
print(_SAVE)
print(DATA_PATH)


average_power_epochs = xr.load_dataarray(os.path.join(DATA_PATH, "average_power.nc"))

# Define bands
bands = {
    "theta": [0, 6],
    "alpha": [6, 14],
    "beta_1": [14, 26],
    "beta_2": [26, 43],
    "gamma": [43, 80],
} ###############################################################################
# Find spectral peaks
###############################################################################
# Remove fixation trials
average_power_epochs = average_power_epochs.isel(stim=slice(0, 5)).mean("stim")

average_power_norm = average_power_epochs / average_power_epochs.max("freqs")

unique_rois = average_power_norm.roi.values

peak_freqs, peak_prominences, rois = [], [], []

for average_power_norm_ in average_power_norm:
    out1, out2, out3 = detect_peak_frequencies(average_power_norm_, prominence=0.01)
    peak_freqs += [out1]
    peak_prominences += [out2]
    rois += [out3]

# Do it for each epoch
has_peaks = []
freqs_array = []
prominences_array = []

for i in range(5):
    out = check_peaks(peak_freqs[i], peak_prominences[i], rois[i])
    has_peaks += [out[0]]
    freqs_array += [out[1]]
    prominences_array += [out[2]]


has_peaks = xr.DataArray(
    np.stack(has_peaks, axis=1).squeeze(),
    dims=("roi", "epochs", "bands"),
    coords={"roi": average_power_norm.roi.data},
)

# Apply criteria for peaks in pars of channels
roi = has_peaks.roi.values
roi_gp, roi_idx = roi, np.arange(len(roi)).reshape(-1, 1)

kw_links = {}
(x_s, x_t), roi_p = conn_links(roi_gp, **kw_links)

rois_s, rois_t = roi[x_s], roi[x_t]

has_peaks_pairs = []

for roi_s, roi_t in zip(rois_s, rois_t):

    has_peaks_pairs += [has_peaks.sel(roi=[roi_s, roi_t]).all("roi").data]

has_peaks_pairs = xr.DataArray(
    np.stack(has_peaks_pairs), dims=("roi", "epochs", "bands"), coords={"roi": roi_p}
).any("epochs")


###########################################################################
# Concatenate peak freq and prominences for each epoch
###########################################################################

freqs_array = xr.concat(freqs_array, "epochs")
prominences_array = xr.concat(prominences_array, "epochs")

peak_freqs = np.zeros((5, len(unique_rois))) 

for i in range(5):
    for j, roi_ in enumerate(unique_rois):
        index = prominences_array[i].sel(roi=roi_).argmax()[0]



###########################################################################
# Saves file
###########################################################################

# Path in which to save coherence data
results_path = os.path.join(_SAVE, "Results", monkey, session_number)

if not os.path.exists(results_path):
    os.makedirs(results_path)

file_name = "has_peak.nc"
path_has_peaks = os.path.join(results_path, file_name)
has_peaks.to_netcdf(path_has_peaks)

file_name = "peak_freqs.nc"
path_has_peaks = os.path.join(results_path, file_name)
freqs_array.to_netcdf(path_has_peaks)

file_name = "peak_prominences.nc"
path_has_peaks = os.path.join(results_path, file_name)
prominences_array.to_netcdf(path_has_peaks)

file_name = "has_peaks_pairs.nc"
path_has_peaks_pairs = os.path.join(results_path, file_name)
has_peaks_pairs.to_netcdf(path_has_peaks_pairs)
