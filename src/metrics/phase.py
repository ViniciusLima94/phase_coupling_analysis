import numpy as np
import xarray as xr
import scipy

from frites.conn import conn_io
from frites.io import logger
from frites.utils import parallel_func


def _phase_diff(w, x_s, x_t, kw_para):
    def pairwise_phase_diff(w_x, w_y):
        return np.unwrap(w[:, w_x, :, :] - w[:, w_y, :, :])

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(pairwise_phase_diff, **kw_para)

    # compute the single trial coherence
    return parallel(p_fun(s, t) for s, t in zip(x_s, x_t))


def hilbert_decomposition(
    data,
    sfreq=None,
    times=None,
    decim=None,
    roi=None,
    bands=None,
    n_jobs=1,
    verbose=None,
    dtype=np.float32,
    **kw_links,
):
    """
    Docstring
    """
    # ________________________________ INPUTS _________________________________
    # inputs conversion
    kw_links.update({"directed": False, "net": False})
    data, cfg = conn_io(
        data,
        times=times,
        roi=roi,
        agg_ch=False,
        win_sample=None,
        sfreq=sfreq,
        verbose=verbose,
        name="Hilbert Decomposition",
        kw_links=kw_links,
    )

    # Extract variables
    x, trials, attrs = data.data, data["y"].data, cfg["attrs"]
    times = data["times"].data
    x_s, x_t, roi_p, roi = cfg["x_s"], cfg["x_t"], cfg["roi_p"], data["roi"].data
    sfreq = cfg["blocks"], cfg["sfreq"]
    n_pairs, f_vec = len(x_s), data.freqs.values
    f_vec = np.atleast_1d(f_vec)

    if isinstance(decim, int):
        times = times[::decim]

    _dims = ("trials", "roi", "freqs", "times")
    _coord_nodes = (trials, roi, f_vec, times)
    _coord_links = (trials, roi_p, f_vec, times)

    # Hilbert coefficients
    h = scipy.signal.hilbert(x, axis=-1)

    if isinstance(decim, int):
        h = h[..., ::decim]

    # Power and phase time-series
    power = (h * np.conj(h)).real
    phase = np.angle(h)

    # Compute phase-differences in parellel
    # show info
    logger.info(f"Computing pairwise phase difference (n_pairs={n_pairs})")
    kw_para = dict(n_jobs=n_jobs, verbose=verbose, total=n_pairs)
    delta_phase = np.stack(_phase_diff(phase, x_s, x_t, kw_para), axis=1)

    # Wrapp to xrray
    power = xr.DataArray(
        power, dims=_dims, coords=_coord_nodes, attrs=attrs, name="power"
    ).astype(dtype)
    phase = xr.DataArray(
        phase, dims=_dims, coords=_coord_nodes, attrs=attrs, name="phase"
    ).astype(dtype)

    delta_phase = xr.DataArray(
        delta_phase, dims=_dims, coords=_coord_links, attrs=attrs, name="phase_diff"
    ).astype(dtype)

    return power, phase, delta_phase
