import os

import numpy as np
import pandas as pd
import xarray as xr
from frites.conn import conn_reshape_undirected
from frites.conn.conn_utils import conn_links
from sklearn.metrics import euclidean_distances
from tqdm import tqdm

from src.util import _extract_roi, get_dates

# Define the matrix using ranges where necessary
xy_grid_lucy = [
    [0, 0, 0, 0, 0, 0] + list(range(248, 253)),
    [0, 0, 0, 0, 0] + list(range(242, 248)),
    [0, 0, 0] + list(range(234, 242)),
    [0, 0] + list(range(225, 234)),
    [0, 0] + list(range(216, 225)),
    [0] + list(range(206, 216)),
    list(range(195, 206)),
    list(range(184, 195)),
    list(range(173, 184)),
    list(range(162, 173)),
    list(range(151, 162)),
    list(range(140, 151)),
    list(range(129, 140)),
    list(range(118, 129)),
    list(range(107, 118)),
    list(range(96, 107)),
    list(range(85, 96)),
    list(range(74, 85)),
    [0] + list(range(64, 74)),
    [0, 0] + list(range(55, 64)),
    [0, 0] + list(range(46, 55)),
    [0, 0] + list(range(37, 46)),
    [0, 0, 0] + list(range(29, 37)),
    [0, 0, 0] + list(range(21, 29)),
    [0, 0, 0, 0, 0] + list(range(15, 21)),
    [0, 0, 0, 0, 0] + list(range(9, 15)),
    [0, 0, 0, 0, 0, 0, 0] + list(range(5, 9)),
    [0, 0, 0, 0, 0, 0, 0] + list(range(1, 5)),
]

# Convert to NumPy array and transpose
xy_grid_lucy = np.array(xy_grid_lucy).T

dates = get_dates("lucy")

nsessions = 0
peak_freqs_corrected = []

for date in tqdm(dates):

    _get_data_path = lambda date: os.path.expanduser(
        f"/home/vinicius/funcog/phaseanalysis/Results/lucy/{date}"
    )

    DATA_PATH = _get_data_path(date)

    average_power = xr.load_dataarray(os.path.join(DATA_PATH, "average_power.nc"))
    peak_freqs = xr.load_dataarray(os.path.join(DATA_PATH, "peak_freqs.nc"))

    roi = average_power.roi.values

    roi_gp, roi_idx = roi, np.arange(len(roi)).reshape(-1, 1)

    _, pairs = conn_links(roi_gp)

    ##################################################################### Compute white matters distances #####################################################################

    wmd = pd.read_excel(
        "/home/vinicius/Documents/WhiteMatterDistance.xlsx", index_col=0
    ).fillna(0)

    r, c = wmd.shape

    for i in range(r):
        for j in range(c):
            if isinstance(wmd.values[i, j], str):
                try:
                    wmd.iloc[i, j] = float(wmd.values[i, j].replace(",", "."))
                except:
                    wmd.values[i, j] = 0.0

    # White matter distances

    rois_ch_s, rois_ch_t = _extract_roi(pairs, "-")

    _, rois_s = _extract_roi(rois_ch_s, "_")
    _, rois_t = _extract_roi(rois_ch_t, "_")

    wmd_pairs = []
    for s, t in zip(rois_s, rois_t):
        temp = wmd.loc[wmd.index == s, wmd.columns == t].values.squeeze()
        if temp.any():
            wmd_pairs += [temp]
        else:
            wmd_pairs += [np.nan]

    wmd_pairs = xr.DataArray(
        np.hstack(wmd_pairs).squeeze(), dims=("roi"), coords={"roi": pairs}, name="wmd"
    )

    average_power.attrs["xp"] = np.zeros(average_power.sizes["roi"])
    average_power.attrs["yp"] = np.zeros(average_power.sizes["roi"])

    for pos, c in enumerate(average_power.attrs["channels_labels"]):
        x, y = np.where(xy_grid_lucy == c)
        average_power.attrs["xp"][pos] = x[0] * 2.3
        average_power.attrs["yp"][pos] = y[0] * 2.3

    xyz = np.stack(
        (
            average_power.attrs["xp"],
            average_power.attrs["yp"],
            average_power.attrs["z"][average_power.attrs["indch"]] / 1000,
        ),
        axis=1,
    )

    ed = pd.DataFrame(
        euclidean_distances(xyz),
        index=average_power.roi.data,
        columns=average_power.roi.data,
    )

    rois_ch_s, rois_ch_t = _extract_roi(pairs, "-")
    ed_pairs = []

    for s, t in zip(rois_ch_s, rois_ch_t):
        ed_pairs += [ed.loc[ed.index == s, ed.columns == t].values]

    ed_pairs = xr.DataArray(
        np.hstack(ed_pairs).squeeze(), dims=("roi"), coords={"roi": pairs}, name="ed"
    )

    temp = xr.concat((wmd_pairs, ed_pairs), "metrics")

    distances = temp.max("metrics")

    mask = (ed_pairs < wmd_pairs).values

    distances[mask] = temp.isel(roi=mask).mean("metrics")

    distances = conn_reshape_undirected(distances)

    ##################################################################### Correct peak frequenciess #####################################################################

    for i in range(5):

        # Indexes of channels with undetected peaks
        indexes = np.where(peak_freqs[i] == -1)[0]
        if len(indexes) > 0:
            for ind in indexes:
                roi = peak_freqs[i, ind].roi.values
                peak_freq_neighbor = peak_freqs[i].sel(
                    roi=distances.sel(sources=roi).argsort()[::-1].targets[0].values
                )
                j = 1
                while peak_freq_neighbor == -1:
                    peak_freq_neighbor = peak_freqs[i].sel(
                        roi=distances.sel(sources=roi).argsort()[::-1].targets[j].values
                    )
                    j = j + 1
                peak_freqs[i, ind] = peak_freq_neighbor

    peak_freqs_corrected += [peak_freqs]

xr.concat(peak_freqs_corrected, "sessions").mean("sessions").to_netcdf(
    "notebooks/data/peak_freqs.nc"
)
