import numpy as np
import xarray as xr
import scipy
from frites.conn import conn_io
from frites.io import check_attrs, logger, set_log_level
from frites.utils import parallel_func
from mne.filter import filter_data
from mne.time_frequency import psd_array_multitaper


def xr_psd_array_multitaper(data, bandwidth=1.0, n_jobs=1, fmin=0.1, fmax=80):
    _, roi, _ = data.trials.values, data.roi.values, data.time.values

    psds_c, freqs, _ = psd_array_multitaper(
        data,
        data.fsample,
        fmin=fmin,
        fmax=fmax,
        n_jobs=n_jobs,
        bandwidth=bandwidth,
        output="complex",
    )

    # Spectra
    sxx = (psds_c * np.conj(psds_c)).mean((0, 2)).real

    sxx = xr.DataArray(sxx, dims=("roi", "freqs"), coords=(roi, freqs))

    return sxx


def _coh(w, x_s, x_t, kw_para):
    """Pairwise coherence."""
    # auto spectra (faster that w * w.conj())
    s_auto = (w.real**2 + w.imag**2).mean((0, 2))

    # define the pairwise coherence
    def pairwise_coh(w_x, w_y):
        # computes the coherence
        s_xy = (w[:, w_y, :, :] * np.conj(w[:, w_x, :, :])).mean((0, 1))
        s_xx = s_auto[w_x]
        s_yy = s_auto[w_y]
        return np.abs(s_xy) ** 2 / (s_xx * s_yy)

    # define the function to compute in parallel
    parallel, p_fun = parallel_func(pairwise_coh, **kw_para)

    # compute the single trial coherence
    return parallel(p_fun(s, t) for s, t in zip(x_s, x_t))


def conn_spec_average(
    data,
    fmin=None,
    fmax=None,
    roi=None,
    sfreq=None,
    n_cycles=7.0,
    bandwidth=None,
    decim=1,
    kw_mt={},
    block_size=None,
    n_jobs=-1,
    verbose=None,
    dtype=np.float32,
    **kw_links,
):
    set_log_level(verbose)

    # _________________________________ INPUTS ________________________________
    # inputs conversion
    kw_links.update({"directed": False, "net": False})
    data, cfg = conn_io(
        data,
        times=None,
        roi=roi,
        agg_ch=False,
        win_sample=None,
        block_size=block_size,
        sfreq=sfreq,
        freqs=None,
        foi=None,
        sm_times=None,
        sm_freqs=None,
        verbose=verbose,
        name="Spectral connectivity (metric = coh)",
        kw_links=kw_links,
    )

    # extract variables
    x, trials, attrs = data.data, data["y"].data, cfg["attrs"]
    times, _ = data["times"].data, len(trials)
    x_s, x_t, roi_p = cfg["x_s"], cfg["x_t"], cfg["roi_p"]
    _, sfreq = cfg["blocks"], cfg["sfreq"]
    n_pairs = len(x_s)

    kw_para = dict(n_jobs=n_jobs, verbose=verbose, total=n_pairs)

    # temporal decimation
    times = times[::decim]

    # define arguments for parallel computing
    # mesg = "Estimating pairwise coh for trials %s"
    kw_para = dict(n_jobs=n_jobs, verbose=verbose, total=n_pairs)

    # show info
    logger.info(f"Computing pairwise coh (n_pairs={n_pairs}, " f"decim={decim}")

    # --------------------------- TIME-FREQUENCY --------------------------
    # time-frequency decomposition
    w, f_vec, _ = psd_array_multitaper(
        x[..., ::decim],
        sfreq,
        fmin=fmin,
        fmax=fmax,
        n_jobs=n_jobs,
        bandwidth=bandwidth,
        output="complex",
        **kw_mt,
    )
    # ______________________ CONTAINER FOR CONNECTIVITY _______________________
    dims = ("roi", "freqs")
    coords = (roi_p, f_vec)

    conn = _coh(w, x_s, x_t, kw_para)

    # configuration
    cfg = dict(
        sfreq=sfreq,
        n_cycles=n_cycles,
        mt_bandwidth=bandwidth,
        decim=decim,
    )

    # conversion
    conn = xr.DataArray(
        conn, dims=dims, name="coh", coords=coords, attrs=check_attrs({**attrs, **cfg})
    )
    return conn


def _phase_diff(w, x_s, x_t, kw_para):
    """
    Compute the pairwise phase difference between two sets of time-series data.

    Parameters
    ----------
    w : ndarray
        The phase time-series array with shape (n_trials, n_rois, n_freqs, n_times),
        representing the instantaneous phase values.
    x_s : list of int
        List of source indices specifying which regions of interest (ROIs) to compute phase differences from.
    x_t : list of int
        List of target indices specifying which regions of interest (ROIs) to compute phase differences to.
    kw_para : dict
        Dictionary of keyword arguments for parallel computation (e.g., number of jobs, verbosity).

    Returns
    -------
    ndarray
        Array of pairwise phase differences between each source-target pair.
        The shape of the output will be (n_trials, n_pairs, n_freqs, n_times).

    Notes
    -----
    The function computes phase differences between the time-series from the specified ROIs,
    then uses `np.unwrap` to handle phase wrapping across 2π discontinuities.
    Parallel computation is utilized for efficiency.
    """

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
    roi=None,
    bands=None,
    n_jobs=1,
    verbose=None,
    dtype=np.float32,
    **kw_links,
):
    """
    Perform Hilbert decomposition on time-series data to extract power, phase, and phase differences.

    Parameters
    ----------
    data : xarray.DataArray or ndarray
        The input time-series data. Expected to be in the shape (n_trials, n_rois, n_times).
    sfreq : float, optional
        The sampling frequency of the data in Hz. Required for band-pass filtering.
    times : ndarray, optional
        The time points corresponding to the data samples.
    roi : list or ndarray, optional
        The regions of interest (ROIs) to analyze. Can be indices or names corresponding to the data.
    bands : list of tuple, optional
        The frequency bands to filter the data. Each tuple should contain the low and high frequency (in Hz) of the band.
    n_jobs : int, optional
        The number of parallel jobs to use for computations (default is 1).
    verbose : bool or int, optional
        Verbosity level for logging.
    dtype : data-type, optional
        The data type of the returned arrays (default is np.float32).
    **kw_links : dict
        Additional arguments for connection analysis.

    Returns
    -------
    power : xarray.DataArray
        Power time-series of the filtered signals across the specified frequency bands.
        Dimensions: (n_trials, n_rois, n_freqs, n_times).
    phase : xarray.DataArray
        Phase time-series of the filtered signals across the specified frequency bands.
        Dimensions: (n_trials, n_rois, n_freqs, n_times).
    delta_phase : xarray.DataArray
        Pairwise phase differences between the ROIs across frequency bands.
        Dimensions: (n_trials, n_roi_pairs, n_freqs, n_times).

    Notes
    -----
    The Hilbert decomposition is applied after band-pass filtering the data to extract the analytic signal,
    from which power and phase are derived. Pairwise phase differences are computed in parallel
    for specified pairs of ROIs.
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
    times, _ = data["times"].data, len(trials)
    x_s, x_t, roi_p, roi = cfg["x_s"], cfg["x_t"], cfg["roi_p"], data["roi"].data
    _, sfreq = cfg["blocks"], cfg["sfreq"]
    n_pairs, f_vec, n_freqs = len(x_s), np.mean(bands, axis=1), len(bands)
    # If no bands are passed use broadband signal

    _dims = ("trials", "roi", "freqs", "times")
    _coord_nodes = (trials, roi, f_vec, times)
    _coord_links = (trials, roi_p, f_vec, times)

    # Filter data in the specified bands
    x_filt = []

    for f_low, f_high in bands:
        x_filt += [
            xr.DataArray(
                filter_data(x, sfreq, f_low, f_high, n_jobs=n_jobs, verbose=verbose),
                dims=data.dims,
                coords=data.coords,
                attrs=attrs,
            )
        ]

    x_filt = xr.concat(x_filt, "freqs").transpose("trials", "roi", "freqs", "times")

    # Hilbert coefficients
    h = scipy.signal.hilbert(x_filt, axis=3)

    # Power and phase time-series
    power = (h * np.conj(h)).real
    phase = np.angle(h)

    # Compute phase-differences in parellel
    # show info
    logger.info(
        f"Computing pairwise phase difference (n_pairs={n_pairs}, " f"n_bands={n_freqs}"
    )
    kw_para = dict(n_jobs=n_jobs, verbose=verbose, total=n_pairs)
    delta_phase = np.stack(_phase_diff(phase, x_s, x_t, kw_para), axis=1)

    # Wrapp to xrray
    power = xr.DataArray(
        power, dims=_dims, coords=_coord_nodes, attrs=attrs, name="power"
    )
    phase = xr.DataArray(
        phase, dims=_dims, coords=_coord_nodes, attrs=attrs, name="phase"
    )
    delta_phase = xr.DataArray(
        delta_phase, dims=_dims, coords=_coord_links, attrs=attrs, name="phase_diff"
    )

    return power.astype(dtype), phase.astype(dtype), delta_phase.astype(dtype)
