import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd # needed for reading coords df as inv
import obspy

eps = 1e-12

def get_stream_coordinates(x, y = None):
    if type(x) is obspy.Stream:
        try:
            staLoc = obspy.signal.array_analysis.get_geometry(x) # https://docs.obspy.org/packages/autogen/obspy.signal.array_analysis.get_geometry.html
        except:
            #staLoc = get_geometry(x, 'xy') # re-centers array (bad)
            xx = np.array([tr.stats.coordinates['x'] for tr in x])
            yy = np.array([tr.stats.coordinates['y'] for tr in x])
            staLoc = np.array([xx,yy]).transpose()
    else:
        staLoc = np.array([x,y]).transpose()
    return staLoc

def _az_dist(a, b):
    """
    Find angular distance between two azimuths. Result is always between -180 and 180.
    """
    return ((a - b + 180) % 360) - 180

def _polar_transform(spec, sx_list, sy_list, az_start = 0, backazimuth = False):
    az_list = np.arange(36) * 10
    az_list = ((az_list - az_start) % 360) + az_start

    r_list = np.concatenate([sx_list[sx_list >= 0], [sx_list.max() + eps]])
    if len(spec.shape) == 3:
        polar_spec = np.zeros([spec.shape[0], len(az_list), len(r_list)])
    elif len(spec.shape) == 4:
        polar_spec = np.zeros([spec.shape[0], spec.shape[1], len(az_list), len(r_list)])
    for i, sx in enumerate(sx_list):
        for j, sy in enumerate(sy_list):
            if backazimuth:
                az = np.arctan2(-sx, -sy) * 180/np.pi
            else:
                az = np.arctan2(sx, sy) * 180/np.pi
            r = np.sqrt(sx**2 + sy**2)
            m = np.argmin(np.abs(_az_dist(az, az_list)))
            n = np.argmin(np.abs(r_list - r))
            if len(spec.shape) == 3:
                polar_spec[:,m,n] += spec[:,i,j]
            elif len(spec.shape) == 4:
                polar_spec[:,:,m,n] += spec[:,:,i,j]
    # drop the last r (unphysical excessive slowness)
    r_list = r_list[:-1]
    if len(spec.shape) == 3:
        polar_spec = polar_spec[:,:,:-1]
    elif len(spec.shape)==4:
        polar_spec = polar_spec[:,:,:,:-1]
       
    return (polar_spec, az_list, r_list)
