import numpy as np
import os

# Define default return type
_DEFAULT_TYPE = np.float32
# Defining default paths
_ROOT = "~/storage1/projects/phase_coupling_analysis/src/"
_ROOT_NAS = "~/funcog/gda/"
# _COORDS_PATH = os.path.expanduser(_ROOT+'Brain Areas/lucy_brainsketch_xy.mat')
_COORDS_PATH = os.path.expanduser(
    _ROOT + "Brain Areas/lucy_xy_coordinates_wholebrain.mat"
)
_DATA_PATH = os.path.expanduser(_ROOT_NAS + "GrayLab")
_COH_PATH = os.path.expanduser(_ROOT_NAS + "PhaseAnalysis")
