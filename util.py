import os
import numpy as np
import xarray as xr
from src.session import session


def load_session_data(sid, monkey, align):
    # Instantiate class
    ses = session(
        raw_path=os.path.expanduser("~/funcog/gda/GrayLab/"),
        monkey=monkey,
        date=sid,
        session=1,
        slvr_msmod=False,
        only_unique_recordings=False,
        align_to=align,
        evt_dt=[-0.65, 2.],
    )

    # Read data from .mat files
    ses.read_from_mat()
    print(ses.data.shape)

    # Load XYZ coordinates
    coords = np.concatenate(
        (ses.get_xy_coords(), ses.recording_info["depth"][:, None]), axis=1
    )

    # Filtering by trials
    data_task = ses.filter_trials(trial_type=[1], behavioral_response=[1])
    data_fixation = ses.filter_trials(trial_type=[2], behavioral_response=None)

    attrs_task, attrs_fixation = data_task.attrs, data_fixation.attrs

    stim = np.hstack((attrs_task["stim"], attrs_fixation["stim"]))
    t_cue_on = np.hstack((attrs_task["t_cue_on"], attrs_fixation["t_cue_on"]))
    t_cue_off = np.hstack((attrs_task["t_cue_off"], attrs_fixation["t_cue_off"]))
    t_match_on = np.hstack((attrs_task["t_match_on"], attrs_fixation["t_match_on"]))

    np.nan_to_num(stim, nan=6, copy=False)

    data = xr.concat((data_task, data_fixation), "trials")
    data.attrs = attrs_task
    data.attrs["stim"] = stim
    data.attrs["t_cue_on"] = t_cue_on
    data.attrs["t_cue_off"] = t_cue_off
    data.attrs["t_match_on"] = t_match_on
    data.attrs["x"] = coords[:, 0]
    data.attrs["y"] = coords[:, 1]
    data.attrs["z"] = coords[:, 2]

    # ROIs with channels
    rois = [
        f"{roi}_{channel}" for roi, channel in zip(data.roi.data, data.channels_labels)
    ]
    data = data.assign_coords({"roi": rois})
    # data.attrs = attrs
    data.values *= 1e6

    # return node_xr_remove_sca(data)
    return data


def z_score(x, dim=-1):
    return (x - x.mean(dim)[:, None]) / x.std(dim)[:, None]


def WrapToPi(x):
    xwrap = np.remainder(x, 2 * np.pi)
    mask = np.abs(xwrap) > np.pi
    xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask])
    return xwrap


def flatten(xss):
    return [x for xs in xss for x in xs]
